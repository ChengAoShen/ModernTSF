/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Static export to `out/`, served by Cloudflare Workers Static Assets.
  // A thin worker (worker.js) handles POST /api/submit; everything else is a file.
  output: "export",
  // Directory-style URLs (/zh/news/) → .../index.html, served cleanly as assets.
  trailingSlash: true,
  // No image-optimization server; serve images as-is.
  images: { unoptimized: true },
};

export default nextConfig;
