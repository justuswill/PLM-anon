"""Engine settings every decoding shares, so speeds compare like for like: one request at a time, a fixed KV cache, and
temperature 1 over the 128 likeliest tokens wherever a decoding samples."""

KV_BLOCK_SIZE = 64
KV_CACHE_BYTES = 1 << 30
GPU_MEMORY_UTILIZATION = 0.25
TOP_K = 128
