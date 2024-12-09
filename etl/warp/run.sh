#!/usr/bin/env bash

# Create a Docker network if needed
docker network create warp_network || true

# Run the Docker container
docker run --rm \
  --device /dev/net/tun:/dev/net/tun \
  -p 1080:1080 \
  -e WARP_SLEEP=2 \
  --cap-add MKNOD \
  --cap-add AUDIT_WRITE \
  --cap-add NET_ADMIN \
  --sysctl net.ipv6.conf.all.disable_ipv6=0 \
  --sysctl net.ipv4.conf.all.src_valid_mark=1 \
  -v ./data:/var/lib/cloudflare-warp \
  --network warp_network \
  caomingjun/warp
