# Liquid Metal: an example PhotoCraft plug-in

A WebAssembly filter plug-in (`no_std` Rust, no imports, ~10 KiB built) that melts an image into
flowing liquid metal:

1. **Liquid distortion**: pixels are pulled along a domain-warped noise flow mixed with the tangent
   of the image's own contours, then smeared into streaks.
2. **Metal emboss**: the distorted luminance becomes a height field, lit with diffuse + specular
   light and a banded chrome reflection.

The ABI is documented in [`docs/plugins.md`](../../../docs/plugins.md).

## Parameters

| Param | Range | Default | Effect |
|---|---|---|---|
| `distortion` | 0–80 px | 36 | How far pixels flow |
| `scale` | 8–400 px | 90 | Size of the molten swirls |
| `swirl` | 0–100 | 55 | 0 = pure noise flow, 100 = flow along the image's edges |
| `smear` | 0–100 | 60 | Length of the streaks |
| `relief` | 0–100 | 60 | Emboss depth and specular strength |
| `chrome` | 0–100 | 55 | Strength of the chrome reflection bands |
| `bands` | 1–8 | 3 | Number of chrome reflection bands |
| `lightAngle` | 0–360° | 135 | Light direction (135 = top left) |
| `color` | 0–100 | 60 | How much of the original colour survives (0 = bare steel) |
| `mix` | 0–100 | 100 | Blend of the metal pass over the distorted image |
| `seed` | 0–9999 | 1 | Varies the flow pattern |

The plug-in declares `overlap: 128`, so bands join seamlessly. Its interpreter cost is roughly
1–2 k instructions per pixel: large images take a while.

## Build and run

```sh
cargo build --release --target wasm32-unknown-unknown
photocraft-cli run in.jpg \
  --cmd plugin.install --params '{"path": "target/wasm32-unknown-unknown/release/photocraft_plugin_liquid_metal.wasm"}' \
  --cmd plugin.run --params '{"id": "org.photocraft.example.liquid-metal", "distortion": 40}' \
  --out out.png
```

In the app: Filter › Plug-ins › Install Plug-in…, then Filter › Plug-ins › Liquid Metal….
