# Branding assets

**`diaugeia.png`** — the master logo (1254×1254, full resolution). This is the
source of truth; keep it untouched. It is intentionally **not** under `public/`,
so it is never shipped to the public site.

Derived assets (regenerate from this file if the logo changes):

```bash
# Web logo (loaded in the nav/footer — small & sharp at ~32px)
magick branding/diaugeia.png -resize 256x256 -strip public/diaugeia.png

# Favicons / app icons
magick branding/diaugeia.png -resize 48x48   -strip app/icon.png
magick branding/diaugeia.png -resize 180x180 -strip app/apple-icon.png
magick branding/diaugeia.png -define icon:auto-resize=16,32,48 app/favicon.ico
magick branding/diaugeia.png -resize 192x192 -strip public/icon-192.png
magick branding/diaugeia.png -resize 512x512 -strip public/icon-512.png
```

> The logo is a soft light-beam image (gradients + glow), so it is **not**
> suitable for SVG — vector tracing would destroy the glow, and embedding the
> raster in SVG only inflates the size. A small PNG/WebP is the right format.
> If a true vector source (`.ai`/`.svg`) ever exists, prefer that.
