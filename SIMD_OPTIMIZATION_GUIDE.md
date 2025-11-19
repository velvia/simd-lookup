# SIMD Optimization Guide for Bit-Packed Lookups

Based on your observation that conditional lookups (20% table2 access) have similar performance to unconditional (100% access), and your experience with `packed_simd_2`.

## Current Performance Bottlenecks

1. **Scalar bit extraction** - Extract one value at a time
2. **Conditional branches** - 16 branches per chunk in V2 kernel
3. **Sequential processing** - No vectorization of extraction

## Optimization Opportunities

### 1. Masked SIMD Gather (Eliminate Branches)

Using `packed_simd_2` for cleaner code:

```rust
use packed_simd_2::*;

// Instead of:
for j in 0..16 {
    if vocab1_array[j] != 0 {  // ← 16 branches!
        result[j] = table2[keys2[j]];
    }
}

// Do this:
let keys_vec = u32x8::from_slice_unaligned(keys2);
let vocab1_vec = u8x8::from_slice_unaligned(vocab1_array);

// Create mask from vocab1 (non-zero = true)
let mask = vocab1_vec.ne(u8x8::splat(0));

// Masked gather - only loads where mask is true
let gathered = unsafe {
    // packed_simd_2 doesn't have gather, but you can:
    // 1. Use target-specific intrinsics
    // 2. Or: unconditional gather + masked select
    let unconditional = gather_u8(table2, keys_vec);
    mask.select(unconditional, u8x8::splat(0))
};
```

### 2. SIMD Bit Extraction (Parallelize Extraction)

Extract 8 values from 8 u64 words simultaneously:

```rust
// For 2-bit values (32 per u64)
unsafe fn extract_8_values_2bit(
    words: &[u64; 8],        // 8 u64 words
    value_indices: &[u32; 8] // Which 2-bit value in each word (0-31)
) -> [u8; 8] {
    use std::arch::x86_64::*;

    // Load 8 u64 words
    let words_low = _mm256_loadu_si256(words[0..4].as_ptr() as *const __m256i);
    let words_high = _mm256_loadu_si256(words[4..8].as_ptr() as *const __m256i);

    // Calculate bit offsets (value_index * 2)
    let indices = _mm256_loadu_si256(value_indices.as_ptr() as *const __m256i);
    let bit_offsets = _mm256_slli_epi32::<1>(indices);

    // Variable shift each u64 by its bit offset
    // (Need to split into u32s because AVX2 doesn't have i64 variable shift)
    // ... more complex bit manipulation ...

    // Mask to extract 2 bits
    let mask = _mm256_set1_epi32(0x03);
    let values = _mm256_and_si256(shifted, mask);

    // Pack from i32 to u8
    // ... packing instructions ...

    let mut result = [0u8; 8];
    // Extract to result
    result
}
```

**Note**: This is complex! AVX2 doesn't have all the operations we need. AVX-512 would be better with `_mm512_srlv_epi64`.

### 3. Hybrid Approach (Recommended)

Based on your findings, here's what I'd recommend:

#### Option A: Remove Conditionals (Simplest)

Since you found 100% lookup ≈ 20% conditional lookup:

```rust
// Just always lookup both tables
pub fn lookup_batch_unconditional(&self, keys1: &[u32], keys2: &[u32]) -> Vec<(u8, u8)> {
    let mut results = vec![(0, 0); keys1.len()];
    for i in 0..keys1.len() {
        results[i] = (
            self.lookup1(keys1[i]),
            self.lookup2(keys2[i])  // Always lookup, no branch
        );
    }
    results
}
```

**Pros**: No branches, simpler code, enables auto-vectorization
**Cons**: Wastes 80% of table2 lookups (but you found this doesn't matter!)

#### Option B: Masked Operations with packed_simd_2

Add to Cargo.toml:
```toml
[dependencies]
packed_simd_2 = "0.3"
```

Then:
```rust
#[cfg(target_arch = "x86_64")]
unsafe fn lookup_batch_masked_simd(&self, keys1: &[u32], keys2: &[u32]) {
    use std::arch::x86_64::*;

    for chunk in keys1.chunks(8).zip(keys2.chunks(8)) {
        // Load keys
        let k1 = /* load 8 keys */;
        let k2 = /* load 8 keys */;

        // Lookup table1 (scalar for now, optimize later)
        let mut v1 = [0u8; 8];
        for i in 0..8 {
            v1[i] = self.lookup1(chunk.0[i]);
        }

        // Create mask
        let v1_vec = _mm_loadu_si64(v1.as_ptr() as *const u8);
        let zero = _mm_setzero_si128();
        let mask = _mm_cmpgt_epi8(v1_vec, zero);

        // Masked lookup table2
        // Use AVX2 _mm256_mask_i32gather_epi32 for byte table
        let default = _mm256_setzero_si256();
        let gathered = _mm256_mask_i32gather_epi32(
            default,
            self.table2.as_ptr() as *const i32,
            k2_vec,
            mask_expanded,  // Expand from i8 to i32 mask
            1
        );

        // Extract bytes and store
        // ...
    }
}
```

#### Option C: Use Existing SimdLookup with Masked Write

Your existing `SimdLookup` with VGATHER, but use masked store for results:

```rust
// Always gather from both tables
let v1 = simd_gather(table1, keys1);
let v2 = simd_gather(table2, keys2);

// Create mask from v1
let mask = v1.ne(u8x16::splat(0));

// Masked AND or masked write
let result = mask.select(v1 & v2, u8x16::splat(0));
```

### 4. AVX-512 Optimization (If Available)

AVX-512 has perfect instructions for this:

```rust
#[cfg(target_feature = "avx512f")]
unsafe fn lookup_avx512(words: &[u64; 8], bit_indices: [u32; 8]) -> [u8; 8] {
    use std::arch::x86_64::*;

    // Load 8 u64 words
    let words_vec = _mm512_loadu_si512(words.as_ptr() as *const i32);

    // Variable shift right
    let shifts = _mm512_loadu_si512(bit_indices.as_ptr() as *const i32);
    let shifted = _mm512_srlv_epi64(words_vec, shifts);

    // Mask 2 bits
    let masked = _mm512_and_si512(shifted, _mm512_set1_epi64(0x03));

    // Pack to bytes
    // ...
}
```

## Recommended Implementation Strategy

Based on your observations:

### Phase 1: Remove Conditionals (Quick Win)

1. Change V2 to always lookup both tables
2. Test performance - you said it's similar
3. Benefits: No branches, simpler code, enables compiler optimizations

### Phase 2: Bit-Packing with Unconditional Lookup

1. Use bit-packed tables (4x memory reduction)
2. Unconditional dual lookup
3. Expected: Cache benefits + no branch penalties = big win

### Phase 3: SIMD Batch Extraction (If Needed)

Only if profiling shows extraction is bottleneck:
1. Implement batch extraction for 8-16 keys at once
2. Use AVX-512 if available (perfect for this)
3. Fallback to AVX2 with more complex code

## Code Template for Unconditional Bit-Packed Dual Vocab

```rust
impl<S: BitPackStrategy> BitPackedDualVocab<S> {
    /// Unconditional dual lookup - always reads both tables
    /// Fastest for cases where both values usually needed
    #[inline]
    pub fn lookup_batch_unconditional(
        &self,
        keys1: &[u32],
        keys2: &[u32],
        results: &mut [(u8, u8)]
    ) {
        assert_eq!(keys1.len(), keys2.len());
        assert_eq!(keys1.len(), results.len());

        // No branches, clean loop
        // Compiler can auto-vectorize this!
        for i in 0..keys1.len() {
            results[i] = (
                self.lookup1.lookup(keys1[i]),
                self.lookup2.lookup(keys2[i])
            );
        }
    }

    /// Process in chunks for better cache behavior
    #[inline]
    pub fn lookup_batch_chunked(
        &self,
        keys1: &[u32],
        keys2: &[u32],
        chunk_size: usize
    ) -> Vec<(u8, u8)> {
        let mut results = Vec::with_capacity(keys1.len());

        for (chunk1, chunk2) in keys1.chunks(chunk_size).zip(keys2.chunks(chunk_size)) {
            let chunk_results = vec![(0, 0); chunk1.len()];
            // Process chunk...
            results.extend(chunk_results);
        }

        results
    }
}
```

## Performance Predictions

Based on your benchmarks:

| Method | Memory | Throughput | Notes |
|--------|--------|-----------|-------|
| Current V2 conditional | 30MB | 263 Melem/s | Baseline |
| Bit-packed conditional | 7.5MB | 334 Melem/s | +27% (measured) |
| Bit-packed unconditional | 7.5MB | ~380 Melem/s | +45% (predicted) |
| + Remove Vec::push | 7.5MB | ~500 Melem/s | +90% (predicted) |
| + SIMD extraction | 7.5MB | ~600 Melem/s | +128% (optimistic) |

## Next Steps

1. **Test unconditional lookup** with current (non-bit-packed) code
   - Verify it's really similar to conditional
   - Measure exact difference

2. **Combine bit-packing + unconditional**
   - Should be faster than bit-packed conditional
   - Simpler code

3. **Profile to find bottleneck**
   - Is it still cache? extraction? output?
   - Only add SIMD complexity if extraction is bottleneck

4. **Consider AVX-512**
   - If on newer Intel or future chips
   - Perfect instructions for bit extraction

Would you like me to implement the unconditional bit-packed version? It should be simple and fast!

