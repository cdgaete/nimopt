#!/bin/sh
set -eu

model=${MODEL:-qwen3.5-4b-instruct-Q4_K_M.gguf}
host=${HOST:-127.0.0.1}
port=${PORT:-8080}
ctx=${CTX:-73728}
threads=${THREADS:-16}
cache=$(dirname "$0")/cache
mkdir -p "$cache"

exec llama-server \
  -m "$model" \
  -c "$ctx" \
  -t "$threads" \
  --flash-attn on \
  -b 2048 -ub 2048 \
  -np 1 \
  --slot-save-path "$cache" \
  --swa-full \
  -ctk q8_0 -ctv q8_0 \
  --host "$host" \
  --port "$port"
