# Benchmark Analysis: Cache Thrashing and Output Overhead

## Executive Summary

Two major findings from comprehensive benchmarking:

1. **Bit-packing provides 27% speedup** by reducing dual table from 30MB → 7MB
2. **Vec::push() overhead is massive** - costs 35% of total performance!

---

## Results Summary

### 1. Bit-Packing Effectiveness (dual_table_comparison)

| Method | Memory | Throughput | vs Baseline |
|--------|--------|-----------|-------------|
| Regular u8 dual table | 30 MB | 263 Melem/s | Baseline |
| 2-bit packed dual table | 7 MB | 334 Melem/s | **+27% faster** |

**Key Insight**: Moving from 30MB (doesn't fit in L3) to 7MB (fits in L3) provides significant speedup despite bit extraction overhead.

### 2. Output Processing Overhead (output_overhead_test)

| Output Method | Throughput | Overhead |
|---------------|-----------|----------|
| No output (just count) | 402 Melem/s | 0% (baseline) |
| Vec::push (no prealloc) | 261 Melem/s | **-35%** |
| Vec::push (preallocated) | 261 Melem/s | **-35%** (no help!) |
| Minimal (just touch) | ~420 Melem/s | Best case |

**Key Insight**: `Vec::push()` in hot loop costs 35% of performance. Pre-allocation doesn't help because the push itself is expensive (bounds check, length increment, etc).

### 3. Cache Size Impact (from diagnostic tests)

| Table Size | Single Table | Notes |
|------------|-------------|-------|
| 3 MB | 1439 Melem/s | Fits in L3 |
| 15 MB | 462 Melem/s | Doesn't fit in L3 |
| **Ratio** | **3.1x faster** | Cache is critical! |

---

## Detailed Analysis

### Why Only 27% from Bit-Packing? (Expected 2-3x)

We expected bigger gains because:
- Single 3MB (1439) vs Single 15MB (462) = 3.1x difference
- Dual 30MB → 7MB should show similar gains

But we only got 27%. Why?

**Reasons:**
1. **Dual lookup overhead exists regardless** - The interleaved access pattern penalty (~25% from diagnostics) applies to both
2. **Bit extraction cost** - Each lookup now costs 5 operations instead of 1
3. **Not cache-bound enough** - With conditional lookup (V2), second table is only accessed 20% of time, reducing cache pressure
4. **Output overhead dominates** - 35% of time is spent in Vec::push, which bit-packing doesn't improve

### The Vec::push Problem

Your benchmarks do this:
```rust
for &val in combined_array.iter() {
    if val != 0 {
        result_vec.push(val);  // ← This is VERY expensive!
    }
}
```

Each `push()` involves:
- Bounds check (is capacity sufficient?)
- Length increment
- Write to potentially non-sequential memory
- Prevents auto-vectorization
- Cache-unfriendly if vec grows

**Even with pre-allocation**, push() still does bounds checks and prevents optimization.

### Why V2 Kernel Behavior Varies by Size

From diagnostic tests:

| Table Size | V2 Throughput | Naive Dual | Winner |
|------------|--------------|-----------|--------|
| 15 MB | 266 Melem/s | 171 Melem/s | V2 better |
| 3 MB | 417 Melem/s | 640 Melem/s | Naive better |

**For large tables (15MB)**: V2's sequential processing helps cache (266 vs 171)
**For small tables (3MB)**: V2's overhead hurts, naive is simpler (417 vs 640)

---

## Recommendations

### 1. Use Bit-Packing for Large Dual Tables ✅

**When to use:**
- Dual table lookups
- Tables > 10MB each
- Value range fits in 2-3 bits
- Cache thrashing is suspected

**Expected gain:** 20-30% speedup

**Trade-offs:**
- Must accept limited value range (0-3 for 2-bit, 0-7 for 3-bit)
- Slightly more complex code
- Extraction overhead (mitigated by cache gains)

### 2. Eliminate Vec::push from Hot Loops ⚠️

**Current code** (261 Melem/s):
```rust
let mut result_vec = Vec::new();
for &val in array {
    if val != 0 {
        result_vec.push(val);  // 35% overhead!
    }
}
```

**Better alternatives:**

**Option A: Bulk write all values** (test after fixing):
```rust
let mut results = vec![0u8; total_capacity];
let mut write_idx = 0;
// Write full u8x16 chunks at once
results[write_idx..write_idx+16].copy_from_slice(arr);
write_idx += 16;
// Post-process to remove zeros if needed
```

**Option B: SIMD filtering** (future optimization):
```rust
// Use SIMD to create mask of non-zero positions
// Pack non-zero values using SIMD compress
// Single bulk write
```

**Option C: Accept all output** (simplest):
```rust
// Just write all values, let caller filter zeros
// Often faster than filtering during lookup
```

### 3. Choose Kernel by Table Size

| Table Size (per table) | Recommended Kernel | Why |
|------------------------|-------------------|------|
| < 5 MB | Naive dual lookup | Simplest, no V2 overhead |
| 5-10 MB | Test both | Depends on access pattern |
| > 10 MB | V2 kernel | Conditional lookup helps cache |
| > 15 MB (dual) | Bit-packed V2 | Fits in L3, big win |

### 4. Consider Sequential Access Pattern

If your use case allows:
```rust
// Pass 1: All table1 lookups
lookup1.lookup_batch(&keys1, &mut results1);

// Pass 2: All table2 lookups
lookup2.lookup_batch(&keys2, &mut results2);

// Pass 3: Combine results
for i in 0..len {
    combined[i] = results1[i] & results2[i];
}
```

From diagnostics: Sequential (206 Melem/s) vs Interleaved (147 Melem/s) = **40% faster**

---

## Predicted Performance with All Optimizations

Current dual 15MB V2 with Vec::push:
- Measured: 263 Melem/s

With bit-packing (2-bit, 7.5MB total):
- Measured: 334 Melem/s (+27%)

If we also eliminate Vec::push overhead:
- No output version: 402 Melem/s
- **Predicted total gain with both optimizations: +53%**

With sequential access pattern + bit-packing + no push:
- **Theoretical maximum: ~500-600 Melem/s** (2x-2.3x improvement)

---

## Benchmark Commands

```bash
# Diagnostic tests
cargo bench --bench dual_table_diagnostic

# Bit-packing comparison
cargo bench --bench bitpacked_bench -- dual_table_comparison

# Output overhead analysis
cargo bench --bench output_overhead_test

# Original benchmarks
cargo bench --bench lookup_kernel_bench
```

---

## Next Steps

1. **Fix output processing** in your production code
   - Replace Vec::push with bulk writes or accept all output
   - Expected: +35% speedup

2. **Deploy bit-packing** for large dual tables
   - Expected: +27% additional speedup
   - Combined: ~70% total improvement

3. **Consider SIMD bit extraction** (future)
   - Extract 8 values from 8 u64s in parallel
   - Could further reduce extraction overhead

4. **Profile real workload**
   - These are microbenchmarks
   - Real performance depends on your actual access patterns
   - May want to test with your production data

---

## Lessons Learned

1. **Cache is king** - 3MB vs 15MB = 3x performance difference
2. **Micro-optimizations matter** - Vec::push costs 35%
3. **Benchmark realistically** - Include all real work (output processing)
4. **Trade-offs are complex** - Bit-packing helps cache but costs extraction
5. **Size matters** - Different kernels optimal for different table sizes

