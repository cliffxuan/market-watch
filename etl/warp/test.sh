#!/usr/bin/env bash
if curl --socks5-hostname 127.0.0.1:1080 https://cloudflare.com/cdn-cgi/trace | grep -q "warp=on"; then
    echo "Success: WARP is enabled."
else
    echo "Failure: WARP is not enabled."
fi
