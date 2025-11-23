#!/bin/bash
# Script to run cache profiling in Docker with perf
#
# Usage:
#   ./scripts/docker_perf.sh [benchmark_function] [profile_time]
#   ./scripts/docker_perf.sh interactive  # Enter container for manual use
#
# This script runs on your HOST machine (macOS) and handles Docker automatically.
# You do NOT need to manually enter the container.

set -e

BENCHMARK_FUNC="${1:-single_vocab_lookup}"
PROFILE_TIME="${2:-10}"

IMAGE_NAME="simd-lookup-perf"
CONTAINER_NAME="simd-lookup-perf-$$"

# Check if user wants interactive mode
if [ "$1" = "interactive" ]; then
    echo "=== Entering Docker Container ==="
    echo "You can now run perf commands manually, or use:"
    echo "  ./scripts/perf_cache_profile.sh lookup_kernel_bench single_vocab_lookup 10"
    echo ""

    # Check if Docker image exists, build if not
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        echo "Building Docker image with perf support (ARM64)..."
        docker build --platform linux/arm64 -f Dockerfile.perf -t "$IMAGE_NAME" .
        echo ""
    fi

    docker run --rm -it \
        --platform linux/arm64 \
        --privileged \
        -v "$(pwd):/workspace" \
        -w /workspace \
        "$IMAGE_NAME" \
        bash
    exit 0
fi

echo "=== Docker Cache Profiling Setup ==="
echo "Running from HOST machine - Docker is handled automatically"
echo ""

# Check if Docker image exists, build if not
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Building Docker image with perf support (ARM64 for native performance)..."
    docker build --platform linux/arm64 -f Dockerfile.perf -t "$IMAGE_NAME" .
    echo ""
fi

echo "Running benchmark in Docker container (ARM64)..."
echo "Benchmark function: $BENCHMARK_FUNC"
echo "Profile time: ${PROFILE_TIME}s"
echo ""

# Run container with privileged access (required for perf)
# Use ARM64 platform for native performance and proper perf counters
# Mount the entire workspace so we can build if needed
docker run --rm \
    --platform linux/arm64 \
    --privileged \
    --name "$CONTAINER_NAME" \
    -v "$(pwd):/workspace" \
    -w /workspace \
    "$IMAGE_NAME" \
    bash -c "
        # Always rebuild inside Docker to ensure we have Linux ARM64 binaries
        # Use a separate target directory to avoid macOS artifacts
        export CARGO_TARGET_DIR=\"/tmp/cargo-target-docker\"

        echo 'Building benchmark for Linux ARM64 (using separate target directory)...'
        cargo build --release --benches

        # Update binary search path
        BENCH_BINARY=\$(find \"\$CARGO_TARGET_DIR/release/deps\" -name 'lookup_kernel_bench-*' -type f -executable | head -1)

        if [ -z \"\$BENCH_BINARY\" ]; then
            # Fallback to regular target directory
            BENCH_BINARY=\$(find target/release/deps -name 'lookup_kernel_bench-*' -type f -executable | head -1)
        fi

        if [ -z \"\$BENCH_BINARY\" ]; then
            echo 'Error: Could not find benchmark binary after building'
            exit 1
        fi

        echo \"Using binary: \$BENCH_BINARY\"
        # Verify it's a Linux ELF binary (file command might not be available)
        if command -v file &>/dev/null; then
            file \"\$BENCH_BINARY\" | grep -q ELF || echo 'WARNING: Binary does not appear to be a Linux ELF file'
        fi
        echo ''

        # Detect CPU architecture and available events
        echo '=== Detecting CPU Architecture and Available Events ==='
        ARCH=\$(uname -m)
        CPUINFO=\$(cat /proc/cpuinfo 2>/dev/null | grep -m1 'model name' || echo 'Unknown CPU')
        echo \"Architecture: \$ARCH\"
        echo \"CPU: \$CPUINFO\"
        echo ''

        # Check if we're in emulation (common on M1 Macs running x86_64 Docker)
        if [ \"\$ARCH\" = \"x86_64\" ] && grep -q 'hypervisor' /proc/cpuinfo 2>/dev/null; then
            echo 'WARNING: Running in emulated x86_64 mode. Cache events may not be available.'
            echo 'For accurate cache profiling, use a native Linux host or ARM64 container.'
            echo ''
        fi

        # List available cache events
        echo 'Available cache-related perf events:'
        perf list cache 2>&1 | head -50 || echo 'No cache events found'
        echo ''

        # For ARM64, check what's actually available
        if [ \"\$ARCH\" = \"aarch64\" ] || [ \"\$ARCH\" = \"arm64\" ]; then
            echo 'ARM64 hardware events:'
            perf list hardware 2>&1 | head -30 || true
            echo ''
            echo 'ARM64 cache events (detailed):'
            perf list | grep -iE '(cache|l1|l2|l3|llc)' | head -30 || true
            echo ''
        fi

        # ARM64 uses different perf event names than x86_64
        # Try ARM64-specific events first, then fall back to generic
        if [ \"\$ARCH\" = \"aarch64\" ] || [ \"\$ARCH\" = \"arm64\" ]; then
            echo 'ARM64 detected - using ARM64-specific perf events...'
            # ARM64 PMU events (these are the actual hardware counters)
            # Try common ARM64 cache events
            ARM64_EVENTS=(
                'L1-dcache-loads'
                'L1-dcache-load-misses'
                'L1-dcache-stores'
                'L1-dcache-store-misses'
                'L1-icache-load-misses'
                'LLC-loads'
                'LLC-load-misses'
                'LLC-stores'
                'LLC-store-misses'
                'dTLB-loads'
                'dTLB-load-misses'
            )

            # Test events with actual benchmark binary (more reliable)
            # Use perf stat with --verbose to see what's actually happening
            WORKING_EVENTS=()
            echo 'Testing ARM64 cache events (this may take a moment)...'
            for event in \"\${ARM64_EVENTS[@]}\"; do
                # Test with a quick run and check if we get actual counter values (not zeros or errors)
                TEST_OUTPUT=\$(timeout 3 perf stat -e \"\$event\" \"\$BENCH_BINARY\" --bench \"$BENCHMARK_FUNC\" --profile-time 0.1 2>&1)
                if echo \"\$TEST_OUTPUT\" | grep -qE '(not supported|Error|Permission)'; then
                    echo \"  ✗ \$event (not supported or permission denied)\"
                elif echo \"\$TEST_OUTPUT\" | grep -qE \"\$event.*[0-9]+\"; then
                    # Event exists and has a counter value
                    WORKING_EVENTS+=(\"\$event\")
                    echo \"  ✓ \$event\"
                else
                    echo \"  ✗ \$event (no counter output)\"
                fi
            done

            if [ \${#WORKING_EVENTS[@]} -eq 0 ]; then
                echo ''
                echo 'WARNING: No ARM64 cache events supported. Trying generic events...'
                # Fall back to generic events that might work
                GENERIC_EVENTS=('cache-references' 'cache-misses' 'cycles' 'instructions')
                for event in \"\${GENERIC_EVENTS[@]}\"; do
                    if timeout 2 perf stat -e \"\$event\" \"\$BENCH_BINARY\" --bench \"$BENCHMARK_FUNC\" --profile-time 0.1 2>&1 | grep -qvE '(not supported|Error)'; then
                        WORKING_EVENTS+=(\"\$event\")
                    fi
                done
            fi

            if [ \${#WORKING_EVENTS[@]} -eq 0 ]; then
                echo 'No cache events available. Using basic CPU counters...'
                CACHE_EVENTS='cycles,instructions,cpu-clock,task-clock'
            else
                CACHE_EVENTS=\$(IFS=','; echo \"\${WORKING_EVENTS[*]}\")
                echo ''
                echo \"Using \${#WORKING_EVENTS[@]} working events: \$CACHE_EVENTS\"
            fi
        else
            # x86_64 events
            echo 'Testing x86_64 cache events...'
            TEST_EVENTS=(
                'cache-references'
                'cache-misses'
                'L1-dcache-loads'
                'L1-dcache-load-misses'
                'LLC-loads'
                'LLC-load-misses'
            )

            WORKING_EVENTS=()
            for event in \"\${TEST_EVENTS[@]}\"; do
                if timeout 2 perf stat -e \"\$event\" \"\$BENCH_BINARY\" --bench \"$BENCHMARK_FUNC\" --profile-time 0.1 2>&1 | grep -qvE '(not supported|Error)'; then
                    WORKING_EVENTS+=(\"\$event\")
                fi
            done

            if [ \${#WORKING_EVENTS[@]} -eq 0 ]; then
                CACHE_EVENTS='cycles,instructions,cpu-clock,task-clock'
            else
                CACHE_EVENTS=\$(IFS=','; echo \"\${WORKING_EVENTS[*]}\")
                echo \"Found \${#WORKING_EVENTS[@]} working cache events\"
            fi
        fi

        echo ''
        echo '=== Cache Profiling Results ==='
        echo \"Using events: \$CACHE_EVENTS\"
        echo ''

        # Check perf permissions and capabilities
        echo '=== Perf Configuration Check ==='
        if [ ! -r /proc/sys/kernel/perf_event_paranoid ]; then
            echo 'WARNING: Cannot check perf permissions. Some events may not work.'
        else
            PARANOID=\$(cat /proc/sys/kernel/perf_event_paranoid)
            echo \"perf_event_paranoid: \$PARANOID (should be -1 or 0 for full access)\"
            if [ \"\$PARANOID\" -gt 1 ]; then
                echo 'WARNING: perf_event_paranoid is too restrictive. Cache events may not work.'
            fi
        fi

        # Check if we're in a container (Docker limitation)
        if [ -f /.dockerenv ] || grep -qa docker /proc/1/cgroup 2>/dev/null; then
            echo 'Running in Docker container - hardware counters may be limited.'
            echo 'For best results, use a native Linux host with: ./scripts/perf_cache_profile.sh'
        fi
        echo ''

        # Try perf stat - if cache events fail, it's likely a Docker limitation
        # Hardware PMU events often don't work in containers even with --privileged
        OUTPUT=\$(perf stat -e \"\$CACHE_EVENTS\" \
            \"\$BENCH_BINARY\" \
            --bench \"$BENCHMARK_FUNC\" \
            --profile-time \"$PROFILE_TIME\" 2>&1)

        # Check if events actually worked
        if echo \"\$OUTPUT\" | grep -q '<not supported>'; then
            echo \"\$OUTPUT\"
            echo ''
            echo '=== IMPORTANT: Docker Limitation Detected ==='
            echo 'Hardware performance counters (cache events) are not accessible in Docker containers,'
            echo 'even with --privileged. This is a known limitation.'
            echo ''
            echo 'Solutions:'
            echo '1. Use a native Linux host: ./scripts/perf_cache_profile.sh'
            echo '2. Use software events (less accurate but available):'
            echo ''
            perf stat -e cycles,instructions,cpu-clock,task-clock,context-switches,cpu-migrations,page-faults \
                \"\$BENCH_BINARY\" \
                --bench \"$BENCHMARK_FUNC\" \
                --profile-time \"$PROFILE_TIME\"
        else
            echo \"\$OUTPUT\"
        fi
    "

echo ""
echo "=== Profiling Complete ==="
echo ""
echo "Tip: Run './scripts/docker_perf.sh interactive' to enter the container"
echo "     and use perf_cache_profile.sh or perf_record_cache.sh manually"

