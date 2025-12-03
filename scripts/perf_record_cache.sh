#!/bin/bash
# Script to record detailed cache profiling data with perf record
# This creates a perf.data file that can be analyzed with perf report
# Usage: ./scripts/perf_record_cache.sh [benchmark_name] [benchmark_function]

set -e

BENCHMARK_NAME="${1:-lookup_kernel_bench}"
BENCHMARK_FUNC="${2:-single_table_lookup}"
PROFILE_TIME="${3:-10}"

echo "=== Recording Cache Profile with perf record ==="
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

# Record cache events with call graph
echo "Recording cache events (this may take a while)..."
perf record \
    -e cache-misses,cache-references,L1-dcache-load-misses,L1-dcache-loads,LLC-load-misses,LLC-loads \
    -g \
    --call-graph dwarf \
    -- \
    "$BENCH_BINARY" \
    --bench "$BENCHMARK_FUNC" \
    --profile-time "$PROFILE_TIME"

echo ""
echo "Recording complete. Analyzing..."
echo ""

# Generate report
echo "=== Top functions by cache misses ==="
perf report --stdio --sort=symbol,cache-misses | head -50

echo ""
echo "=== Detailed cache analysis ==="
echo "Run 'perf report' to explore interactively"
echo "Or 'perf report --stdio' for full text output"
echo ""
echo "To see cache miss rates by function:"
echo "  perf report --stdio --sort=symbol,cache-misses,cache-references"


