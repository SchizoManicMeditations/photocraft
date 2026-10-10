//! Liquid Metal: a PhotoCraft filter plug-in (plug-in ABI v1, see `docs/plugins.md`).
//!
//! Two passes over each band:
//!
//! 1. **Liquid distortion.** Every pixel is pulled from `p + d`, where `d` mixes a domain-warped
//!    noise flow (marbled, molten swirls) with the tangent of the image's own blurred luminance
//!    (so the flow runs *along* edges, smearing them like poured metal). Several samples along `d`
//!    are averaged into streaks. Noise is evaluated in document coordinates, so bands join
//!    seamlessly.
//! 2. **Metal emboss.** The distorted luminance (fine detail plus a broad blur) becomes a height
//!    field. Its normals are lit (diffuse + a tight specular) and reflect a banded chrome
//!    environment; the original colour is kept as much as `color` asks.
//!
//! `no_std`, no allocator, no imports. Scratch buffers come straight from `memory.grow`; each band
//! runs in a fresh instance, so nothing is ever freed.
#![no_std]

use core::arch::wasm32;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    wasm32::unreachable()
}

const MANIFEST: &str = r#"{
  "id": "org.photocraft.example.liquid-metal",
  "name": "Liquid Metal…",
  "version": "1.0.0",
  "kind": "filter",
  "author": "PhotoCraft contributors",
  "description": "Melts the image into flowing liquid metal: a contour-following liquid distortion, then a chrome emboss.",
  "params": {
    "distortion": {"type": "number", "min": 0, "max": 80, "default": 36},
    "scale":      {"type": "number", "min": 8, "max": 400, "default": 90},
    "swirl":      {"type": "number", "min": 0, "max": 100, "default": 55},
    "smear":      {"type": "number", "min": 0, "max": 100, "default": 60},
    "relief":     {"type": "number", "min": 0, "max": 100, "default": 60},
    "chrome":     {"type": "number", "min": 0, "max": 100, "default": 55},
    "bands":      {"type": "int",    "min": 1, "max": 8,   "default": 3},
    "lightAngle": {"type": "number", "min": 0, "max": 360, "default": 135},
    "color":      {"type": "number", "min": 0, "max": 100, "default": 60},
    "mix":        {"type": "number", "min": 0, "max": 100, "default": 100},
    "seed":       {"type": "int",    "min": 0, "max": 9999, "default": 1}
  },
  "overlap": 128,
  "area": "content"
}"#;

/// Bit 8 of `format`: the last channel is alpha.
const HAS_ALPHA: u32 = 1 << 8;
/// Most channels handled (Multichannel documents can have many).
const MAX_CHANNELS: usize = 16;
/// Smear samples per pixel.
const SMEAR_TAPS: usize = 4;
/// Spacing in pixels of the grid the noise flow is evaluated on.
const GRID: f32 = 4.0;

const PI: f32 = core::f32::consts::PI;
const TAU: f32 = core::f32::consts::TAU;

#[unsafe(no_mangle)]
pub extern "C" fn pc_abi_version() -> u32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn pc_manifest() -> u64 {
    ((MANIFEST.len() as u64) << 32) | MANIFEST.as_ptr() as u64
}

#[unsafe(no_mangle)]
pub extern "C" fn pc_alloc(size: u32) -> u32 {
    grow(size as usize).map_or(0, |p| p as u32)
}

/// Fresh zeroed pages for `bytes` bytes, or `None` at the host's memory cap.
fn grow(bytes: usize) -> Option<usize> {
    let old = wasm32::memory_grow(0, bytes.div_ceil(65536).max(1));
    if old == usize::MAX { None } else { Some(old * 65536) }
}

/// A scratch buffer of `len` floats.
fn scratch(len: usize) -> Option<&'static mut [f32]> {
    let p = grow(len.checked_mul(4)?)?;
    // SAFETY: `p..p + len * 4` was just grown, is zeroed, 4-aligned and owned by nobody else.
    Some(unsafe { core::slice::from_raw_parts_mut(p as *mut f32, len) })
}

// ---- math (no libm in no_std) ---------------------------------------------------------------

fn floor(x: f32) -> f32 {
    // Exact for |x| < 2^31, far beyond any coordinate here.
    let t = x as i32 as f32;
    if t > x { t - 1.0 } else { t }
}
/// Square root by Newton's method from a bit-trick guess (relative error < 1e-6).
fn sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut r = f32::from_bits((x.to_bits() >> 1) + 0x1fbd_1df5);
    r = 0.5 * (r + x / r);
    r = 0.5 * (r + x / r);
    0.5 * (r + x / r)
}
fn abs(x: f32) -> f32 {
    x.abs()
}
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.min(hi).max(lo)
}
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
/// Parabolic sine, max error ~0.001: plenty for shading.
fn sin(x: f32) -> f32 {
    let x = x - TAU * floor((x + PI) / TAU);
    let y = (4.0 / PI) * x - (4.0 / (PI * PI)) * x * abs(x);
    0.225 * (y * abs(y) - y) + y
}
fn cos(x: f32) -> f32 {
    sin(x + 0.5 * PI)
}

// ---- noise ----------------------------------------------------------------------------------

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^= h >> 15;
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Smooth value noise in 0..1.
fn noise(x: f32, y: f32, seed: u32) -> f32 {
    let (fx, fy) = (floor(x), floor(y));
    let (ix, iy) = (fx as i32, fy as i32);
    let (tx, ty) = (x - fx, y - fy);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let a = hash(ix, iy, seed);
    let b = hash(ix.wrapping_add(1), iy, seed);
    let c = hash(ix, iy.wrapping_add(1), seed);
    let d = hash(ix.wrapping_add(1), iy.wrapping_add(1), seed);
    lerp(lerp(a, b, sx), lerp(c, d, sx), sy)
}

/// Two octaves, centred on 0 (roughly -0.5..0.5).
fn fbm(x: f32, y: f32, seed: u32) -> f32 {
    noise(x, y, seed) * 0.65 + noise(x * 2.03 + 5.1, y * 2.03 + 1.7, seed ^ 0x9e37_79b9) * 0.35 - 0.5
}

// ---- params (tiny JSON key lookup; the host has already validated and clamped them) --------

/// Index just past `"key":` at or after `from`.
fn find(hay: &[u8], key: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i + key.len() + 3 <= hay.len() {
        if hay[i] == b'"' && hay[i + 1..].starts_with(key) && hay[i + 1 + key.len()] == b'"' {
            let mut j = i + 2 + key.len();
            while j < hay.len() && hay[j] == b' ' {
                j += 1;
            }
            if j < hay.len() && hay[j] == b':' {
                return Some(j + 1);
            }
        }
        i += 1;
    }
    None
}

/// A JSON number at `pos` (sign, digits, fraction, exponent).
fn number(hay: &[u8], mut pos: usize) -> Option<f32> {
    while pos < hay.len() && hay[pos] == b' ' {
        pos += 1;
    }
    let neg = hay.get(pos) == Some(&b'-');
    if neg {
        pos += 1;
    }
    let (mut v, mut any) = (0.0f64, false);
    while let Some(d @ b'0'..=b'9') = hay.get(pos) {
        v = v * 10.0 + (d - b'0') as f64;
        pos += 1;
        any = true;
    }
    if hay.get(pos) == Some(&b'.') {
        pos += 1;
        let mut f = 0.1;
        while let Some(d @ b'0'..=b'9') = hay.get(pos) {
            v += (d - b'0') as f64 * f;
            f *= 0.1;
            pos += 1;
            any = true;
        }
    }
    if matches!(hay.get(pos), Some(b'e' | b'E')) {
        pos += 1;
        let eneg = hay.get(pos) == Some(&b'-');
        if matches!(hay.get(pos), Some(b'-' | b'+')) {
            pos += 1;
        }
        let mut e = 0i32;
        while let Some(d @ b'0'..=b'9') = hay.get(pos) {
            e = (e * 10 + (d - b'0') as i32).min(300);
            pos += 1;
        }
        for _ in 0..e {
            if eneg { v /= 10.0 } else { v *= 10.0 }
        }
    }
    if !any {
        return None;
    }
    Some(if neg { -v } else { v } as f32)
}

fn param(hay: &[u8], key: &[u8], default: f32) -> f32 {
    find(hay, key, 0).and_then(|p| number(hay, p)).unwrap_or(default)
}

// ---- image helpers --------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Model {
    Rgb,
    Cmyk,
    Lab,
    /// Gray, Duotone, Multichannel, alpha channels: every channel is a brightness.
    Gray,
}

/// Perceived brightness 0..1 of one pixel.
fn luma(p: &[f32], model: Model, colour: usize) -> f32 {
    match model {
        Model::Rgb => 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2],
        Model::Cmyk => clamp(1.0 - (0.3 * p[0] + 0.59 * p[1] + 0.11 * p[2]) - p[3], 0.0, 1.0),
        Model::Lab => p[0],
        Model::Gray => {
            let mut s = 0.0;
            for v in &p[..colour] {
                s += *v;
            }
            s / colour as f32
        }
    }
}

/// Box blur of radius `r` (horizontal then vertical), edges clamped; `t` is scratch.
fn box_blur(v: &mut [f32], t: &mut [f32], w: usize, h: usize, r: usize) {
    if r == 0 || w == 0 || h == 0 {
        return;
    }
    let inv = 1.0 / (2 * r + 1) as f32;
    for y in 0..h {
        let row = &v[y * w..y * w + w];
        let out = &mut t[y * w..y * w + w];
        let mut acc = 0.0;
        for k in 0..=2 * r {
            acc += row[k.saturating_sub(r).min(w - 1)];
        }
        for x in 0..w {
            out[x] = acc * inv;
            acc += row[(x + r + 1).min(w - 1)] - row[x.saturating_sub(r)];
        }
    }
    for x in 0..w {
        let mut acc = 0.0;
        for k in 0..=2 * r {
            acc += t[k.saturating_sub(r).min(h - 1) * w + x];
        }
        for y in 0..h {
            v[y * w + x] = acc * inv;
            acc += t[(y + r + 1).min(h - 1) * w + x] - t[y.saturating_sub(r) * w + x];
        }
    }
}

/// Streak: the average of samples from `(1 - smear)·d` to `d` past `(x, y)`, into `acc`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn streak(src: &[f32], w: usize, h: usize, n: usize, x: f32, y: f32, dx: f32, dy: f32, smear: f32, acc: &mut [f32; MAX_CHANNELS]) {
    acc[..n.min(MAX_CHANNELS)].fill(0.0);
    for s in 0..SMEAR_TAPS {
        let f = 1.0 - smear * s as f32 / (SMEAR_TAPS - 1) as f32;
        sample_add(src, w, h, n, x + dx * f, y + dy * f, 1.0 / SMEAR_TAPS as f32, acc);
    }
}

/// Adds `weight ×` the bilinear sample of `src` at `(x, y)` (clamped to the buffer) to `acc`.
#[inline(always)]
fn sample_add(src: &[f32], w: usize, h: usize, n: usize, x: f32, y: f32, weight: f32, acc: &mut [f32; MAX_CHANNELS]) {
    let x = clamp(x, 0.0, (w - 1) as f32);
    let y = clamp(y, 0.0, (h - 1) as f32);
    let (x0, y0) = (x as usize, y as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let (w00, w10, w01, w11) = ((1.0 - fx) * (1.0 - fy) * weight, fx * (1.0 - fy) * weight, (1.0 - fx) * fy * weight, fx * fy * weight);
    let (a, b, c, d) = ((y0 * w + x0) * n, (y0 * w + x1) * n, (y1 * w + x0) * n, (y1 * w + x1) * n);
    if src.len() < w * h * n || n > MAX_CHANNELS {
        return;
    }
    for k in 0..n {
        // SAFETY: x0, x1 < w and y0, y1 < h (clamped above), k < n <= MAX_CHANNELS, and
        // src.len() >= w * h * n (checked above), so every index is in bounds. The interpreter
        // charges for each bounds check, and this is the innermost loop.
        unsafe {
            *acc.get_unchecked_mut(k) += src.get_unchecked(a + k) * w00
                + src.get_unchecked(b + k) * w10
                + src.get_unchecked(c + k) * w01
                + src.get_unchecked(d + k) * w11;
        }
    }
}

/// Central-difference gradient of `v` at `(x, y)`, edges clamped.
fn gradient(v: &[f32], w: usize, h: usize, x: usize, y: usize) -> (f32, f32) {
    let (xl, xr) = (x.saturating_sub(1), (x + 1).min(w - 1));
    let (yu, yd) = (y.saturating_sub(1), (y + 1).min(h - 1));
    ((v[y * w + xr] - v[y * w + xl]) * 0.5, (v[yd * w + x] - v[yu * w + x]) * 0.5)
}

// ---- the filter -----------------------------------------------------------------------------

/// # Safety
/// The host guarantees `buf..buf + len` and `params..params + params_len` are blocks it got from
/// `pc_alloc` and filled.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pc_filter(buf: *mut f32, len: u32, width: u32, height: u32, channels: u32, format: u32, params: *const u8, params_len: u32) -> i32 {
    let (w, h, n) = (width as usize, height as usize, channels as usize);
    let Some(count) = w.checked_mul(h) else { return 1 };
    let Some(samples) = count.checked_mul(n) else { return 1 };
    if n == 0 || n > MAX_CHANNELS || samples.checked_mul(4).is_none_or(|b| b > len as usize) {
        return 1;
    }
    if count == 0 {
        return 0;
    }
    // SAFETY: see the function's contract (sizes checked above).
    let px = unsafe { core::slice::from_raw_parts_mut(buf, samples) };
    // SAFETY: see the function's contract.
    let p = unsafe { core::slice::from_raw_parts(params, params_len as usize) };

    let alpha = format & HAS_ALPHA != 0;
    let colour = if alpha { n - 1 } else { n };
    if colour == 0 {
        return 0;
    }
    let model = match (format >> 16) & 0xff {
        2 | 3 if colour >= 3 => Model::Rgb,
        4 if colour >= 4 => Model::Cmyk,
        9 if colour >= 3 => Model::Lab,
        _ => Model::Gray,
    };

    let distortion = param(p, b"distortion", 36.0);
    let scale = clamp(param(p, b"scale", 90.0), 4.0, 1000.0);
    let swirl = param(p, b"swirl", 55.0) / 100.0;
    let smear = param(p, b"smear", 60.0) / 100.0;
    let relief = param(p, b"relief", 60.0) / 100.0;
    let chrome = param(p, b"chrome", 55.0) / 100.0;
    let bands = param(p, b"bands", 3.0);
    let light = param(p, b"lightAngle", 135.0) * (PI / 180.0);
    let keep = param(p, b"color", 60.0) / 100.0;
    let mix = param(p, b"mix", 100.0) / 100.0;
    let seed = param(p, b"seed", 1.0) as u32;
    // Document position of the buffer's top-left sample (seamless noise across bands).
    let img = find(p, b"_image", 0).unwrap_or(0);
    let ox = find(p, b"x", img).and_then(|i| number(p, i)).unwrap_or(0.0);
    let oy = find(p, b"y", img).and_then(|i| number(p, i)).unwrap_or(0.0);

    let (Some(src), Some(a), Some(b), Some(t)) = (scratch(samples), scratch(count), scratch(count), scratch(count)) else {
        return 2;
    };
    src.copy_from_slice(px);

    // Broad height field: blurred luminance, sized by the flow scale.
    let r_big = clamp(scale / 10.0, 2.0, 12.0) as usize;
    for i in 0..count {
        b[i] = luma(&src[i * n..i * n + n], model, colour);
    }
    box_blur(b, t, w, h, r_big);
    box_blur(b, t, w, h, r_big);

    if distortion > 0.0 && distort(src, px, b, w, h, n, ox, oy, distortion, scale, swirl, smear, seed).is_none() {
        return 2;
    }

    if mix <= 0.0 {
        return 0;
    }

    // Pass 2: metal emboss of the distorted image.
    emboss(px, a, b, t, w, h, n, model, colour, alpha, r_big, relief, chrome, bands, light, keep, mix);
    0
}

/// Pass 1: liquid distortion of `src` into `px`, along a noise flow mixed with the tangent of
/// the height field `height`. `None` when scratch memory runs out.
#[allow(clippy::too_many_arguments)]
fn distort(
    src: &[f32],
    px: &mut [f32],
    height: &[f32],
    w: usize,
    h: usize,
    n: usize,
    ox: f32,
    oy: f32,
    distortion: f32,
    scale: f32,
    swirl: f32,
    smear: f32,
    seed: u32,
) -> Option<()> {
    let b = height;
    // Domain-warped noise flow (molten marbling) on a coarse grid aligned to document
    // coordinates, so it is seamless across bands; pixels interpolate it.
    let inv_scale = 1.0 / scale;
    let gx0 = floor(ox / GRID);
    let gy0 = floor(oy / GRID);
    let gw = (((ox + w as f32) / GRID) - gx0) as usize + 2;
    let gh = (((oy + h as f32) / GRID) - gy0) as usize + 2;
    let flow = scratch(gw.checked_mul(gh)?.checked_mul(2)?)?;
    for j in 0..gh {
        for i in 0..gw {
            let nx = (gx0 + i as f32) * GRID * inv_scale;
            let ny = (gy0 + j as f32) * GRID * inv_scale;
            let wx = fbm(nx, ny, seed);
            let wy = fbm(nx + 31.4, ny + 17.2, seed);
            let o = (j * gw + i) * 2;
            flow[o] = clamp(fbm(nx + 1.8 * wx, ny + 1.8 * wy, seed ^ 0x5bd1_e995) * 3.0, -1.0, 1.0);
            flow[o + 1] = clamp(fbm(nx + 1.8 * wx + 9.2, ny + 1.8 * wy + 4.4, seed ^ 0x5bd1_e995) * 3.0, -1.0, 1.0);
        }
    }

    // Pass 1: liquid distortion.
    let mut acc = [0.0f32; MAX_CHANNELS];
    for y in 0..h {
        let v = (oy + y as f32) / GRID - gy0;
        let j0 = (v as usize).min(gh - 2);
        let fv = v - j0 as f32;
        for x in 0..w {
            let i = y * w + x;
            // Flow along the image's contours (tangent of the height gradient).
            let (gx, gy) = gradient(b, w, h, x, y);
            let gm = sqrt(gx * gx + gy * gy) + 1e-6;
            let k = clamp(gm * 60.0, 0.0, 1.0) / gm;
            let (tx, ty) = (-gy * k, gx * k);
            let u = (ox + x as f32) / GRID - gx0;
            let i0 = (u as usize).min(gw - 2);
            let fu = u - i0 as f32;
            let (o00, o01) = ((j0 * gw + i0) * 2, ((j0 + 1) * gw + i0) * 2);
            let fx = lerp(lerp(flow[o00], flow[o00 + 2], fu), lerp(flow[o01], flow[o01 + 2], fu), fv);
            let fy = lerp(lerp(flow[o00 + 1], flow[o00 + 3], fu), lerp(flow[o01 + 1], flow[o01 + 3], fu), fv);
            let dx = distortion * lerp(fx, tx, swirl);
            let dy = distortion * lerp(fy, ty, swirl);
            // Streak: average samples from (1 - smear)·d to d.
            // A literal channel count lets the compiler unroll the per-channel loops.
            let (fx, fy) = (x as f32, y as f32);
            match n {
                4 => streak(src, w, h, 4, fx, fy, dx, dy, smear, &mut acc),
                3 => streak(src, w, h, 3, fx, fy, dx, dy, smear, &mut acc),
                1 => streak(src, w, h, 1, fx, fy, dx, dy, smear, &mut acc),
                _ => streak(src, w, h, n, fx, fy, dx, dy, smear, &mut acc),
            }
            px[i * n..i * n + n].copy_from_slice(&acc[..n]);
        }
    }
    Some(())
}

/// Pass 2: metal emboss of `px` in place; `a`, `b` and `t` are scratch planes.
#[allow(clippy::too_many_arguments)]
fn emboss(
    px: &mut [f32],
    a: &mut [f32],
    b: &mut [f32],
    t: &mut [f32],
    w: usize,
    h: usize,
    n: usize,
    model: Model,
    colour: usize,
    alpha: bool,
    r_big: usize,
    relief: f32,
    chrome: f32,
    bands: f32,
    light: f32,
    keep: f32,
    mix: f32,
) {
    let count = w * h;
    for i in 0..count {
        a[i] = luma(&px[i * n..i * n + n], model, colour);
    }
    b.copy_from_slice(a);
    box_blur(a, t, w, h, 2);
    box_blur(a, t, w, h, 2);
    box_blur(b, t, w, h, r_big);
    box_blur(b, t, w, h, r_big);
    // Height: soft detail riding on broad molten bulges.
    for i in 0..count {
        a[i] = 0.3 * a[i] + 0.7 * b[i];
    }

    let strength = relief * 70.0;
    let (lx, ly, lz) = {
        let elev = 0.6f32; // ~35°
        let (ce, se) = (cos(elev), sin(elev));
        (cos(light) * ce, -sin(light) * ce, se)
    };
    // Blinn half vector with the viewer straight on (0, 0, 1).
    let (hx, hy, hz) = {
        let (x, y, z) = (lx, ly, lz + 1.0);
        let m = 1.0 / sqrt(x * x + y * y + z * z);
        (x * m, y * m, z * m)
    };
    // A flat surface keeps its tone: lighting and reflections are relative to facing the viewer.
    let flat_lit = 1.0 / (0.3 + 1.05 * lz);
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let o = i * n;
            if alpha && px[o + n - 1] <= 0.0 {
                continue;
            }
            let (gx, gy) = gradient(a, w, h, x, y);
            let (mut nx, mut ny, mut nz) = (-gx * strength, -gy * strength, 1.0);
            let m = 1.0 / sqrt(nx * nx + ny * ny + 1.0);
            nx *= m;
            ny *= m;
            nz *= m;
            let diffuse = clamp(nx * lx + ny * ly + nz * lz, 0.0, 1.0);
            let mut spec = clamp(nx * hx + ny * hy + nz * hz, 0.0, 1.0);
            for _ in 0..6 {
                spec *= spec; // ^64
            }
            // Reflected view ray against a banded studio environment, phase-shifted by height.
            let (rx, ry) = (2.0 * nz * nx, 2.0 * nz * ny);
            let phase = TAU * bands * 0.5 * a[i];
            let env = sin(PI * bands * (0.85 * ry + 0.35 * rx) + phase) - sin(phase);
            let lum = luma(&px[o..o + n], model, colour);
            let lit = lum * (0.3 + 1.05 * diffuse) * flat_lit;
            let metal = clamp(lit + env * chrome * 0.45 + spec * 0.9 * relief, 0.0, 4.0);

            let pix = &mut px[o..o + colour];
            let inv = 1.0 / (lum + 0.02);
            match model {
                Model::Lab => {
                    let l = lerp(pix[0], metal, mix);
                    pix[0] = l;
                    for v in &mut pix[1..] {
                        *v = lerp(*v, 0.5 + (*v - 0.5) * keep, mix);
                    }
                }
                Model::Cmyk => {
                    for v in &mut pix[..3] {
                        let bright = 1.0 - *v;
                        let tint = 1.0 - keep + keep * clamp(bright * inv, 0.0, 3.0);
                        *v = lerp(*v, clamp(1.0 - metal * tint, 0.0, 1.0), mix);
                    }
                    pix[3] = lerp(pix[3], pix[3] * keep, mix);
                }
                Model::Rgb | Model::Gray => {
                    for v in pix.iter_mut() {
                        let tint = 1.0 - keep + keep * clamp(*v * inv, 0.0, 3.0);
                        *v = lerp(*v, metal * tint, mix);
                    }
                }
            }
        }
    }
}
