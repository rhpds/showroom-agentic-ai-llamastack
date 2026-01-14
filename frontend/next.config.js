/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  distDir: process.env.NEXT_BUILD_DIR || 'out',
  images: {
    unoptimized: true,
  },
  // Base path configured for Antora integration - app will be served from /ai-assistant/
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || '/ai-assistant',
  assetPrefix: process.env.NEXT_PUBLIC_ASSET_PREFIX || '/ai-assistant',
}

module.exports = nextConfig
