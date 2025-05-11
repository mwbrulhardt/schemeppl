import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Other config options...
  webpack(config) {
    // Enable async WebAssembly support
    config.experiments = { ...config.experiments, asyncWebAssembly: true };
    return config;
  },
};

export default nextConfig;

