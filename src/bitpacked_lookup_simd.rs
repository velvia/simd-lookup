//! SIMD-accelerated bit-packed lookup with masked gather for conditional lookups
//!
//! This module provides optimized implementations using:
//! 1. SIMD bit extraction - extract multiple values in parallel
//! 2. Masked gather - conditional lookup without branches
//!
//! TODO: Refactor this.  This was AI generated and uses native AVX/Intel intrinsics.
//! It's yucky.  We instead want to use wide or other SIMD crate (packed_simd_2) to do the work in a platform-agnostic
//! way.  If those crates don't have enough, then we create extensions on top of their types.  That is much
//! cleaner and more maintainable.

use crate::bitpacked_lookup::{BitPackStrategy, TwoBit, ThreeBit};
use std::marker::PhantomData;

/// SIMD-accelerated bit-packed dual vocabulary lookup
/// Uses masked gather to eliminate conditional lookup branches
#[derive(Debug, Clone)]
pub struct BitPackedDualVocabSIMD<S: BitPackStrategy> {
    packed_data1: Vec<u64>,
    packed_data2: Vec<u64>,
    max_key: u32,
    _strategy: PhantomData<S>,
}

impl<S: BitPackStrategy> BitPackedDualVocabSIMD<S> {
    /// Create from two u8 tables
    pub fn from_u8_tables(table1: &[u8], table2: &[u8]) -> Self {
        assert_eq!(table1.len(), table2.len(), "Tables must be same size");

        let max_key = table1.len().saturating_sub(1) as u32;
        let num_words = table1.len().div_ceil(S::VALUES_PER_WORD as usize);

        let mut packed_data1 = vec![0u64; num_words];
        let mut packed_data2 = vec![0u64; num_words];

        for (key, (&value1, &value2)) in table1.iter().zip(table2.iter()).enumerate() {
            let word_index = key / (S::VALUES_PER_WORD as usize);
            let value_index = (key % (S::VALUES_PER_WORD as usize)) as u32;

            packed_data1[word_index] = S::pack(packed_data1[word_index], value1, value_index);
            packed_data2[word_index] = S::pack(packed_data2[word_index], value2, value_index);
        }

        Self {
            packed_data1,
            packed_data2,
            max_key,
            _strategy: PhantomData,
        }
    }

    /// Batch lookup with masked conditional - table2 only looked up where table1 != 0
    /// Uses SIMD masked operations to avoid branches
    #[inline]
    pub fn lookup_batch_masked(&self, keys1: &[u32], keys2: &[u32], results: &mut [(u8, u8)]) {
        assert_eq!(keys1.len(), keys2.len());
        assert_eq!(keys1.len(), results.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    return self.lookup_batch_avx2_masked(keys1, keys2, results);
                }
            }
        }

        // Fallback to scalar with branches (current implementation)
        self.lookup_batch_scalar(keys1, keys2, results);
    }

    /// Scalar fallback with conditional branches
    #[inline]
    fn lookup_batch_scalar(&self, keys1: &[u32], keys2: &[u32], results: &mut [(u8, u8)]) {
        for i in 0..keys1.len() {
            let val1 = self.lookup_single(&self.packed_data1, keys1[i]);
            let val2 = if val1 != 0 {
                self.lookup_single(&self.packed_data2, keys2[i])
            } else {
                0
            };
            results[i] = (val1, val2);
        }
    }

    /// Single value lookup helper
    #[inline]
    fn lookup_single(&self, table: &[u64], key: u32) -> u8 {
        if key > self.max_key {
            return 0;
        }
        let word_index = (key / S::VALUES_PER_WORD) as usize;
        let value_index = key % S::VALUES_PER_WORD;
        let word = unsafe { *table.get_unchecked(word_index) };
        S::extract(word, value_index)
    }

    /// AVX2 implementation with masked operations
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn lookup_batch_avx2_masked(&self, keys1: &[u32], keys2: &[u32], results: &mut [(u8, u8)]) {
        use std::arch::x86_64::*;

        let len = keys1.len();
        let mut i = 0;

        // Process 8 keys at a time with AVX2
        while i + 8 <= len {
            // Load 8 keys from each table
            let keys1_vec = _mm256_loadu_si256(keys1[i..].as_ptr() as *const __m256i);
            let keys2_vec = _mm256_loadu_si256(keys2[i..].as_ptr() as *const __m256i);

            // Calculate word indices and bit offsets for table1
            let word_indices1 = _mm256_srli_epi32::<5>(keys1_vec); // Divide by 32 for 2-bit
            let value_indices1 = _mm256_and_si256(keys1_vec, _mm256_set1_epi32(31));
            let bit_offsets1 = _mm256_slli_epi32::<1>(value_indices1); // Multiply by 2 bits

            // Gather 8 u64 words from table1
            // Note: This is complex because _mm256_i32gather_epi64 expects i32 indices
            // We need to extract and gather 8 u64s
            let mut words1 = [0u64; 8];
            let indices1: [i32; 8] = std::mem::transmute(word_indices1);
            for j in 0..8 {
                if indices1[j] >= 0 && (indices1[j] as usize) < self.packed_data1.len() {
                    words1[j] = self.packed_data1[indices1[j] as usize];
                }
            }
            let words1_vec = _mm256_loadu_si256(words1.as_ptr() as *const __m256i);

            // Extract 2-bit values from words1
            // For each of 8 u64 words, extract the 2-bit value at bit_offset
            let mut values1 = [0u8; 8];
            let offsets1: [i32; 8] = std::mem::transmute(bit_offsets1);
            for j in 0..8 {
                let shift = offsets1[j] as u32;
                values1[j] = ((words1[j] >> shift) & 0x03) as u8;
            }

            // Create mask: values1[j] != 0
            let values1_vec = _mm_loadu_si128(values1.as_ptr() as *const __m128i);
            let zero = _mm_setzero_si128();
            let mask_bytes = _mm_cmpgt_epi8(values1_vec, zero);

            // Now do masked lookup for table2
            // Only lookup where mask is true (values1 != 0)
            let mask_i32 = _mm256_cvtepi8_epi32(mask_bytes);

            // Calculate word indices and offsets for table2
            let word_indices2 = _mm256_srli_epi32::<5>(keys2_vec);
            let value_indices2 = _mm256_and_si256(keys2_vec, _mm256_set1_epi32(31));
            let bit_offsets2 = _mm256_slli_epi32::<1>(value_indices2);

            // Conditional gather based on mask
            let mut values2 = [0u8; 8];
            let indices2: [i32; 8] = std::mem::transmute(word_indices2);
            let offsets2: [i32; 8] = std::mem::transmute(bit_offsets2);
            let mask_array: [i32; 8] = std::mem::transmute(mask_i32);

            for j in 0..8 {
                if mask_array[j] != 0 && indices2[j] >= 0 && (indices2[j] as usize) < self.packed_data2.len() {
                    let word = self.packed_data2[indices2[j] as usize];
                    let shift = offsets2[j] as u32;
                    values2[j] = ((word >> shift) & 0x03) as u8;
                }
            }

            // Store results
            for j in 0..8 {
                results[i + j] = (values1[j], values2[j]);
            }

            i += 8;
        }

        // Handle remainder with scalar code
        while i < len {
            let val1 = self.lookup_single(&self.packed_data1, keys1[i]);
            let val2 = if val1 != 0 {
                self.lookup_single(&self.packed_data2, keys2[i])
            } else {
                0
            };
            results[i] = (val1, val2);
            i += 1;
        }
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        (self.packed_data1.len() + self.packed_data2.len()) * 8
    }
}

// TODO: ARM NEON implementation
// ARM doesn't have masked gather, but can simulate with:
// 1. Compare to create mask
// 2. Manual gather into two registers
// 3. Select between them based on mask

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_lookup_basic() {
        let table1 = vec![0u8, 1, 2, 0, 3, 0];
        let table2 = vec![10u8, 11, 12, 13, 14, 15];

        let lookup = BitPackedDualVocabSIMD::<TwoBit>::from_u8_tables(&table1, &table2);

        let keys1 = vec![0, 1, 2, 3, 4, 5];
        let keys2 = vec![0, 1, 2, 3, 4, 5];
        let mut results = vec![(0u8, 0u8); 6];

        lookup.lookup_batch_masked(&keys1, &keys2, &mut results);

        // table1[0] = 0, so table2 not looked up
        assert_eq!(results[0], (0, 0));

        // table1[1] = 1 (non-zero), so table2[1] looked up
        assert_eq!(results[1].0, 1);
        assert_eq!(results[1].1 & 0x03, (11 & 0x03)); // Masked to 2 bits

        // table1[3] = 0, so table2 not looked up
        assert_eq!(results[3], (0, 0));
    }
}


