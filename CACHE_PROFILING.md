# Cache Profiling Guide

This guide explains how to profile cache hits and misses for the simd-lookup benchmarks. Cache profiling is crucial for understanding why dual table lookups are slower and how bit-packing helps.

## Why Cache Profiling?

Your benchmarks show that:
- Single 3MB table: 1439 Melem/s (fits in L3 cache)
- Single 15MB table: 462 Melem/s (doesn't fit in L3 cache)
- **3.1x difference** - cache is critical!

Cache profiling helps you:
1. Verify cache miss rates for different table sizes
2. Understand why dual table (30MB) is slow
3. Confirm that bit-packing (7MB) reduces cache misses
4. Optimize memory access patterns

## Option 1: Linux Perf (Recommended)

Linux `perf` provides detailed cache statistics and is the most reliable option.

### Quick Start: Docker

The easiest way is to use Docker with the provided setup. **Run this from your macOS host** - Docker is handled automatically:

```bash
# Build the Docker image (one time, happens automatically)
# docker build -f Dockerfile.perf -t simd-lookup-perf .

# Run cache profiling (from macOS host - no need to enter container)
./scripts/docker_perf.sh single_table_lookup 10
```

**Note**: You run `docker_perf.sh` from your host machine (macOS). It automatically:
1. Builds the Docker image if needed
2. Runs the container with proper permissions
3. Executes perf inside the container
4. Shows you the results

If you want to manually use the other scripts inside the container:
```bash
./scripts/docker_perf.sh interactive
# Now you're inside the container and can run:
# ./scripts/perf_cache_profile.sh lookup_kernel_bench single_table_lookup 10
```

### Quick Start: Native Linux

If you're on Linux natively (no Docker needed):

```bash
# Install perf (if needed)
sudo apt-get install linux-perf  # Debian/Ubuntu
# or
sudo yum install perf            # Fedora/RHEL/CentOS
# or
sudo pacman -S perf              # Arch Linux

# Ensure perf permissions (may need to run once)
sudo sysctl kernel.perf_event_paranoid=-1

# Run cache profiling directly
./scripts/perf_cache_profile.sh lookup_kernel_bench single_table_lookup 10

# Or for detailed function-level analysis
./scripts/perf_record_cache.sh lookup_kernel_bench single_table_lookup 10
```

**Note**: On native Linux, you can run these scripts directly - no Docker needed!

### Understanding the Output

Key metrics to watch:

```
cache-references          # Total cache references
cache-misses              # Total cache misses
L1-dcache-load-misses    # L1 data cache misses
LLC-load-misses          # L3 cache misses (most important for 15MB tables)

Cache miss rate = cache-misses / cache-references
L3 miss rate = LLC-load-misses / LLC-loads
```

**Expected results:**
- **Single 3MB table**: Low L3 miss rate (< 5%)
- **Single 15MB table**: High L3 miss rate (> 20%)
- **Dual 30MB tables**: Very high L3 miss rate (> 40%)
- **Dual 7MB bit-packed**: Low L3 miss rate (< 10%)

### Detailed Profiling with perf record

For function-level cache analysis:

```bash
./scripts/perf_record_cache.sh lookup_kernel_bench single_table_lookup 10

# Then analyze
perf report --stdio --sort=symbol,cache-misses
```

This shows which functions have the most cache misses.

## Option 2: macOS Instruments (M1_CPUCounters_Profiler)

The `M1_CPUCounters_Profiler` template provides PMC (Performance Monitor Counter) access for cache profiling, but requires special configuration.

### The Problem

You're seeing this error:
```
Failed binding 'kdebug-counters-with-kdebug-sample' table: samplers differ:
requested = KPERF_SAMPLER_TINFO|KPERF_SAMPLER_PMC_CPU
configured = KPERF_SAMPLER_TINFO|KPERF_SAMPLER_USTACK
```

This means Instruments wants PMC counters but the system is configured for user stack sampling.

### Potential Solutions

#### Solution 1: Run with sudo (may help)

```bash
sudo cargo instruments -t M1_CPUCounters_Profiler --bench lookup_kernel_bench -- --bench single_table_lookup --profile-time 200
```

#### Solution 2: Use System Trace instead

System Trace may provide cache information without PMC issues:

```bash
cargo instruments -t "System Trace" --bench lookup_kernel_bench -- --bench single_table_lookup --profile-time 200
```

#### Solution 3: Check Instruments Template

The template file is at:
```
target/instruments/M1_CPUCounters_Profiler.tracetemplate
```

You may be able to modify it, but it's a binary plist format.

#### Solution 4: Use the trace anyway

The error says "trace is still ready to be viewed" - the trace file was created:
```
target/instruments/lookup_kernel_bench-13df22e6e72a6b8e_M1_CPUCounters_Profiler_2025-11-23_095130-677.trace
```

Open it in Instruments - it may have useful data despite the error.

### Opening the Trace

```bash
open target/instruments/lookup_kernel_bench-*.trace
```

In Instruments, look for:
- **Counters** instrument (shows PMC data if available)
- **CPU Counters** track (cache-related metrics)

## Comparison: Perf vs Instruments

| Feature | Linux Perf | macOS Instruments |
|---------|------------|-------------------|
| Cache metrics | ✅ Excellent | ⚠️ Requires PMC config |
| Ease of use | ✅ Simple | ⚠️ Complex setup |
| Function-level | ✅ Yes | ✅ Yes |
| Docker support | ✅ Yes | ❌ No |
| Cross-platform | ✅ Linux only | ✅ macOS only |

**Recommendation**: Use Linux Perf in Docker for the most reliable cache profiling.

## Example Workflow

1. **Profile single 3MB table** (baseline - should fit in cache):
   ```bash
   # Modify benchmark to use 3MB table, then:
   ./scripts/perf_cache_profile.sh lookup_kernel_bench single_table_lookup 10
   ```

2. **Profile single 15MB table** (should have cache misses):
   ```bash
   ./scripts/perf_cache_profile.sh lookup_kernel_bench single_table_lookup 10
   ```

3. **Profile dual 30MB tables** (should have many cache misses):
   ```bash
   ./scripts/perf_cache_profile.sh lookup_kernel_bench dual_table_lookup_v2 10
   ```

4. **Profile dual 7MB bit-packed** (should have fewer cache misses):
   ```bash
   ./scripts/perf_cache_profile.sh bitpacked_bench dual_table_comparison 10
   ```

5. **Compare results**:
   - L3 miss rate should decrease from 30MB → 7MB
   - This explains the performance improvement

## Troubleshooting

### perf: Permission denied

Run with sudo or ensure your user has perf permissions:
```bash
sudo sysctl kernel.perf_event_paranoid=-1
```

### perf: No events found

Some events may not be available on all CPUs. The scripts handle this gracefully.

### Docker: perf doesn't work

Ensure you're using `--privileged` flag:
```bash
docker run --privileged ...
```

### Instruments: Still getting PMC errors

Try using System Trace or Time Profiler - they won't show cache counters but will show CPU hotspots that correlate with cache misses.

## Further Reading

- [Linux perf wiki](https://perf.wiki.kernel.org/)
- [perf cache events](https://perf.wiki.kernel.org/index.php/Tutorial#Counting_with_perf_stat)
- [macOS Instruments documentation](https://developer.apple.com/documentation/instruments)

