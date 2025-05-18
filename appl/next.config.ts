import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Other config options...
  webpack(config, { isServer }) {
    if (!isServer) {
      config.resolve.alias['hammerjs'] = false;
    }
    // Enable async WebAssembly support
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };
    // config.module.rules.push({
    //   test: /\.wasm$/,
    //   type: 'asset/resource',
    // });
    return config;
  },
};

export default nextConfig;
