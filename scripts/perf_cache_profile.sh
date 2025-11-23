#!/bin/bash
# Script to profile cache hits/misses using Linux perf
# Usage: ./scripts/perf_cache_profile.sh [benchmark_name] [benchmark_function]

set -e

BENCHMARK_NAME="${1:-lookup_kernel_bench}"
BENCHMARK_FUNC="${2:-single_vocab_lookup}"
PROFILE_TIME="${3:-10}"

echo "=== Cache Profiling with Linux Perf ==="
echo "Benchmark: $BENCHMARK_NAME"
echo "Function: $BENCHMARK_FUNC"
echo "Profile time: ${PROFILE_TIME}s"
echo ""

# Find the benchmark binary
BENCH_BINARY=$(find target/release/deps -name "${BENCHMARK_NAME}-*" -type f -executable | head -1)

if [ -z "$BENCH_BINARY" ]; then
    echo "Error: Benchmark binary not found. Building..."
    cargo build --release --benches
    BENCH_BINARY=$(find target/release/deps -name "${BENCHMARK_NAME}-*" -type f -executable | head -1)
fi

if [ -z "$BENCH_BINARY" ]; then
    echo "Error: Could not find benchmark binary after building"
    exit 1
fi

echo "Using binary: $BENCH_BINARY"
echo ""

# Cache events to profile
# L1 = Level 1 cache (fastest, smallest)
# LLC = Last Level Cache (usually L3, largest)
CACHE_EVENTS=(
    "cache-references"              # Total cache references
    "cache-misses"                  # Total cache misses
    "L1-dcache-loads"              # L1 data cache loads
    "L1-dcache-load-misses"        # L1 data cache load misses
    "L1-dcache-stores"              # L1 data cache stores
    "L1-dcache-store-misses"        # L1 data cache store misses
    "L1-icache-load-misses"         # L1 instruction cache misses
    "LLC-loads"                     # Last level cache loads
    "LLC-load-misses"               # Last level cache load misses
    "LLC-stores"                    # Last level cache stores
    "LLC-store-misses"              # Last level cache store misses
    "dTLB-loads"                    # Data TLB loads
    "dTLB-load-misses"              # Data TLB load misses
    "iTLB-loads"                    # Instruction TLB loads
    "iTLB-load-misses"              # Instruction TLB load misses
)

# Join events with commas
EVENTS_STR=$(IFS=,; echo "${CACHE_EVENTS[*]}")

echo "Running perf stat with cache events..."
echo ""

# Run perf stat
perf stat -e "$EVENTS_STR" \
    "$BENCH_BINARY" \
    --bench "$BENCHMARK_FUNC" \
    --profile-time "$PROFILE_TIME" \
    2>&1 | tee perf_cache_output.txt

echo ""
echo "Results saved to perf_cache_output.txt"
echo ""
echo "=== Key Metrics to Watch ==="
echo "  - cache-misses / cache-references = Cache miss rate"
echo "  - L1-dcache-load-misses / L1-dcache-loads = L1 data cache miss rate"
echo "  - LLC-load-misses / LLC-loads = L3 cache miss rate (most important for your 15MB tables)"

