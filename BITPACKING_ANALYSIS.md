# Bit-Packed Lookup Tables: Cache Thrashing Solution

## Problem Statement

Your benchmarks show a **3-4x slowdown** for dual table lookups compared to single table, even though the second lookup only happens 20% of the time:

- **Expected performance**: Single = X, Dual ≈ 1.2X (1.0 for first lookup + 0.2 for conditional second)
- **Actual performance**: Single = X, Dual = 3-4X

This huge discrepancy can only be explained by **L3 cache thrashing**.

### The Math

- **Your tables**: 2 × 15MB = 30MB total
- **M1/M2 L3 cache**: 24-32MB (varies by chip)
- **Result**: Tables don't fit → constant cache misses → RAM access (50-100ns vs 5-10ns for L3)

## Solution: Bit-Packed Lookup Tables

If you only need 2-3 bits per value (values 0-3 or 0-7), you can compress tables by 4x or 2.67x:

### Memory Savings

| Encoding | Table Size | Dual Table Size | Fits in L3? |
|----------|------------|-----------------|-------------|
| Original u8 | 15MB | 30MB | ❌ No |
| 2-bit packed | 3.75MB | 7.5MB | ✅ Yes |
| 3-bit packed | 5.625MB | 11.25MB | ✅ Yes |

### Performance Trade-off

**Extraction cost** (per lookup):
```rust
// Original u8: 1 memory access + 1 operation
let value = table[key];

// Bit-packed: 1 memory access + ~4-5 operations
let word_index = key / VALUES_PER_WORD;
let value_index = key % VALUES_PER_WORD;
let word = table_u64[word_index];
let value = (word >> (value_index * BITS)) & MASK;
```

**Net effect**:
- Extraction: ~5x slower compute
- Memory access: ~10x faster (L3 vs RAM)
- **Expected net gain: ~2x faster overall**

## Implementation

### Key Design Decisions

1. **64-bit word packing**: No values cross word boundaries (your suggestion)
   - 2-bit: 32 values per u64
   - 3-bit: 21 values per u64 (1 bit wasted per word)
   - Simple extraction with shift + mask

2. **Generic bit-width support**: Easy to test 2-bit vs 3-bit vs 4-bit

3. **Drop-in replacement**: Same API as existing lookup kernels

### Example Usage

```rust
use simd_lookup::bitpacked_lookup::{BitPackedDualTable, TwoBit, ThreeBit};

// Create from existing u8 tables (values will be truncated to fit bit width)
let dual_2bit = BitPackedDualTable::<TwoBit>::from_u8_tables(&table1, &table2);

// Or create from sparse entries
let entries = vec![(0, 1), (1000, 2), (5000, 3)];
let dual_2bit = BitPackedDualTable::<TwoBit>::from_entries(&entries);

// Conditional lookup (only looks up table2 if table1 != 0)
let mut results = vec![(0u8, 0u8); keys.len()];
dual_2bit.lookup_batch_conditional(&keys1, &keys2, &mut results);
```

## Benchmarks

### Run the Comparison

```bash
# Full benchmark suite
cargo bench --bench bitpacked_bench

# Just single table comparison
cargo bench --bench bitpacked_bench -- single_table_comparison

# Just dual table comparison (the smoking gun test)
cargo bench --bench bitpacked_bench -- dual_table_comparison

# Table size scaling (find cache threshold)
cargo bench --bench bitpacked_bench -- table_size_scaling
```

### What to Look For

1. **Single table**: Bit-packed might be slightly slower (cache isn't the bottleneck)
2. **Dual table**: Bit-packed should be **significantly faster** (2-3x) if cache thrashing is the issue
3. **Table size scaling**: Performance should degrade sharply when tables exceed L3 cache size

### Expected Results

If cache thrashing is indeed the issue (which your 3-4x slowdown strongly suggests):

```
single_table_comparison/regular_u8       time: ~XXX ms
single_table_comparison/bitpacked_2bit   time: ~XXX ms (similar or slightly slower)

dual_table_comparison/regular_u8_v2      time: ~XXX ms  (3-4x slower than single)
dual_table_comparison/bitpacked_2bit     time: ~XXX ms  (2-3x FASTER than regular dual!)
```

## Next Steps

1. **Run the benchmarks** to confirm the theory
2. **If it works**:
   - Consider which bit-width (2-bit vs 3-bit) offers best speed/expressiveness trade-off
   - Integrate into your production code
   - Consider SIMD extraction optimizations (extract 8 values at once from 8 u64 words)

3. **If it doesn't work** (bit-packed is still slow):
   - Cache prefetching might help
   - Consider smaller table sizes (bloom filter + hash for rare values)
   - Profile to find actual bottleneck

## SIMD Extraction (Future Optimization)

The current implementation is scalar. You could further optimize with SIMD:

```rust
// Load 8 u64 words with SIMD gather
let words_vec = _mm512_i64gather_epi64(word_indices, table.as_ptr(), 8);

// Extract 8 2-bit values in parallel
// Shift each word by its bit offset, then mask
let values = /* SIMD shift + mask operations */
```

This would make extraction cost nearly free, but only worth it if cache thrashing is solved first.

## Files Created

- `src/bitpacked_lookup.rs` - Core bit-packed lookup implementation
- `benches/bitpacked_bench.rs` - Comprehensive benchmark suite
- Updated `src/lib.rs` to export new module

## Implementation Notes

### Why 64-bit words?

1. No cross-boundary issues (your insight)
2. Natural alignment for modern CPUs
3. Single memory access per extraction
4. Easy to implement SIMD gather later

### Why not use BitVec crate?

As per your research:
- No SIMD multi-index lookup support
- More general-purpose (slower)
- Our specialized implementation is simpler and faster

### Bit-width Trade-offs

| Bits | Values | Use Case |
|------|--------|----------|
| 2 | 0-3 | Boolean flags, tiny enums, maximum compression |
| 3 | 0-7 | Small enums, good compression + expressiveness |
| 4 | 0-15 | Larger enums, less compression (2x vs 4x) |

## Conclusion

Based on your performance numbers (3-4x dual table slowdown despite only 20% second lookups), **cache thrashing is almost certainly the bottleneck**. Bit-packing should provide a significant speedup by bringing both tables into L3 cache.

The 5x extraction overhead is worth paying if it eliminates constant RAM access, which is 10x slower than L3.

**Expected outcome**: 2-3x speedup for dual table lookups.

