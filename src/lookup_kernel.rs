//! Arrow-style "lookup kernel" similar to arrow=select::take::take kernel.
//! It does not do SIMD gather, rather relying on the speed on scalar lookups which are "in a row".
//! However, it does allow for SIMD processing of gathered u8x16 words.
//!
//! Anecdotally, this is not really faster than SIMD gather.
//!
//! ----------------- SIMD based lookup kernels - "Arrow" style ------------------------------------
//! These operate by leveraging constant-sized array reads from a slice with SIMD operations on looked up values,
//! especially fast for multiple vocabulary lookups and combinations thereof.
//! It turns out these don't significantly improve on a series of scalar lookups.  Also, I forgot that we need to
//! respect the SeriesIndex on input, as that may be used for series/time filtering.  So these don't really work.

use simd_aligned::{arch::u8x16, traits::Simd};

/// Single vocabulary lookup kernel - u32 to u8 lookup table kernel
/// The user is responsible for generating the lookup table - so this can be used for different use cases, including
/// CASE..WHEN and bitmasking/filtering.
/// Note: for general purpose vocab_expr where the lookup can be any type, instead just use `arrow::compute::take()`
/// to do a very efficient lookup where the lookup table can be any type, but then you pay the cost of write memory
/// I/O.  These kernels here allow user to operate on each looked up u8x16 and do something.
///
#[derive(Debug, Clone)]
pub struct SimdSingleVocabU32U8Lookup<'a> {
    lookup_table: &'a [u8],
}

impl<'a> SimdSingleVocabU32U8Lookup<'a> {
    #[inline]
    pub fn new(lookup_table: &'a [u8]) -> Self {
        Self { lookup_table }
    }

    /// Given a slice of u32 values, looks up each one and calls the user given function on an assembled u8x16 (16
    /// looked up values) at a time.
    ///
    /// The user function is passed (lookedup_values: u8x16, start_idx: usize), where start_idx is 0 for the first chunk
    /// call, 16 for the next one, etc.
    ///
    /// If the slice does not divide evenly into 16-item chunks, the rest is handled by filling missing values in the
    /// u8x16 with zeroes.  Thus, the lookup assumes the zero is basically a NOP.
    #[inline]
    pub fn lookup_func<F>(&self, values: &[u32], f: &mut F)
    where F: FnMut(u8x16, usize) {
        let (chunks, rest) = values.as_chunks::<16>();
        let mut idx = 0;
        for chunk in chunks {
            // Get looked up values - LLVM should be able to auto-vectorize this
            let mut values = [0u8; 16];
            values[0] = self.lookup_table[chunk[0] as usize];
            values[1] = self.lookup_table[chunk[1] as usize];
            values[2] = self.lookup_table[chunk[2] as usize];
            values[3] = self.lookup_table[chunk[3] as usize];
            values[4] = self.lookup_table[chunk[4] as usize];
            values[5] = self.lookup_table[chunk[5] as usize];
            values[6] = self.lookup_table[chunk[6] as usize];
            values[7] = self.lookup_table[chunk[7] as usize];
            values[8] = self.lookup_table[chunk[8] as usize];
            values[9] = self.lookup_table[chunk[9] as usize];
            values[10] = self.lookup_table[chunk[10] as usize];
            values[11] = self.lookup_table[chunk[11] as usize];
            values[12] = self.lookup_table[chunk[12] as usize];
            values[13] = self.lookup_table[chunk[13] as usize];
            values[14] = self.lookup_table[chunk[14] as usize];
            values[15] = self.lookup_table[chunk[15] as usize];

            // Call user function
            (f)(u8x16::from(values), idx);
            idx += 16;
        }

        // Handle the rest... just loop and do a lookup, feed to user function with 0's for items not in the slice.
        if !rest.is_empty() {
            let mut values = [0u8; 16];
            for i in 0..rest.len() {
                values[i] = self.lookup_table[rest[i] as usize];
            }
            (f)(u8x16::from(values), idx);
        }
    }

    /// Convenience function which does lookup and writes the results into a Vec of the same length as the input slice.
    /// Does not transform the looked up values.
    #[inline]
    pub fn lookup_into_vec(&self, values: &[u32]) -> Vec<u8> {
        // Allocate a vector with the same length as the input slice - setting the length so contents are uninitialized.
        // Safety: This is OK as this function explicitly overwrites every value, and there is no reading beforehand.
        let mut result = Vec::with_capacity(values.len());
        unsafe { result.set_len(values.len()); }

        // Call lookup_func with a closure that writes to the result vector
        // NOTE: we do as_chunks_mut as that allows for bulk writes - much more efficient than individual writes.
        let (write_slices, rest) = result[..].as_chunks_mut::<16>();
        self.lookup_func(values, &mut |lookedup_values, start_idx| {
            let slice_num = start_idx / 16;
            if slice_num < write_slices.len() {
                // write_slices[slice_num].copy_from_slice(&lookedup_values.as_array());

                // Safety: we have already validated slice_num is within range, and that also means the ensure slice
                //  is writeable.  Thus, skip bounds checks and do single instructon write.
                unsafe {
                    let ptr = write_slices[slice_num].as_mut_ptr() as *mut u8x16;
                    ptr.write_unaligned(lookedup_values);
                }
            } else {
                // Handle remainder - write only the needed bytes
                rest.copy_from_slice(&lookedup_values.as_array()[..rest.len()]);
            }
        });
        result
    }

    /// Version of lookup_into_vec which writes into a mutable u8x16 buffer, for cascaded lookups
    #[inline]
    pub fn lookup_into_u8x16_buffer(&self, values: &[u32], buffer: &mut [u8x16]) {
        assert!((buffer.len() * 16) >= values.len(), "Buffer must be at least as long as the input values");
        self.lookup_func(values, &mut |lookedup_values, start_idx| {
            buffer[start_idx / 16] = lookedup_values;
        });
    }

    /// Prepares a Vec of u8x16 for lookup_into_u8x16_buffer by setting the length and preparing.
    /// The Vec is extended by the amount necessary to hold the results.
    ///
    /// ## Safety
    /// - We unsafe set the length because we know we will overwrite every element.
    ///
    #[inline]
    pub fn lookup_extend_u8x16_vec(&self, values: &[u32], vec: &mut Vec<u8x16>) {
        let needed = values.len().div_ceil(16);
        let cur_len = vec.len();
        // Only reserve if we don't have enough capacity
        if vec.capacity() < cur_len + needed {
            vec.reserve(needed);
        }
        // Safety: we know we will overwrite every element, and we have already validated the length.
        unsafe { vec.set_len(cur_len + needed); }
        self.lookup_into_u8x16_buffer(values, &mut vec[cur_len..]);
    }
}

/// SIMD gather-based single vocabulary lookup kernel - u32 to u8 lookup table kernel
/// Uses VGATHER instructions for faster lookups on large tables.
///
/// This version uses SIMD gather to load u32 words containing target bytes,
/// then extracts the specific bytes. Much faster than scalar lookups for large tables.
/// Uses the same approach as SimdLookup::lookup_simd_8_impl() but processes 16 values at once.
#[derive(Debug, Clone)]
pub struct SimdSingleVocabU32U8LookupGather<'a> {
    lookup_table: &'a [u8],
    table_u32: Vec<u32>, // Same data viewed as u32 words for SIMD gather
    max_key: u32,
}

impl<'a> SimdSingleVocabU32U8LookupGather<'a> {
    #[inline]
    pub fn new(lookup_table: &'a [u8]) -> Self {
        let max_key = if lookup_table.is_empty() {
            0
        } else {
            (lookup_table.len() - 1) as u32
        };

        // Create u32 view of the table, padding to u32 boundary if needed
        let u32_len = (lookup_table.len() + 3) / 4; // Round up to nearest u32 boundary
        let mut padded_table = lookup_table.to_vec();
        padded_table.resize(u32_len * 4, 0); // Pad with zeros

        // Convert to u32 view - safe because we padded to u32 boundary
        let table_u32 = padded_table
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Self {
            lookup_table,
            table_u32,
            max_key,
        }
    }

    /// SIMD gather-based lookup function using VGATHER instructions.
    /// Given a slice of u32 values, looks up each one using SIMD gather and calls the user given function
    /// on an assembled u8x16 (16 looked up values) at a time.
    ///
    /// The user function is passed (lookedup_values: u8x16, start_idx: usize), where start_idx is 0 for the first chunk
    /// call, 16 for the next one, etc.
    ///
    /// If the slice does not divide evenly into 16-item chunks, the rest is handled by filling missing values in the
    /// u8x16 with zeroes.  Thus, the lookup assumes the zero is basically a NOP.
    #[inline]
    pub fn lookup_func<F>(&self, values: &[u32], f: &mut F)
    where F: FnMut(u8x16, usize) {
        let (chunks, rest) = values.as_chunks::<16>();
        let mut idx = 0;
        for chunk in chunks {
            let lookedup_values = self.lookup_chunk_16_gather(chunk);
            (f)(lookedup_values, idx);
            idx += 16;
        }

        // Handle the rest... just loop and do a lookup, feed to user function with 0's for items not in the slice.
        if !rest.is_empty() {
            let mut values = [0u8; 16];
            for i in 0..rest.len() {
                if rest[i] <= self.max_key {
                    values[i] = self.lookup_table[rest[i] as usize];
                }
            }
            (f)(u8x16::from(values), idx);
        }
    }

    /// Prepares a Vec of u8x16 for lookup by setting the length and preparing.
    /// The Vec is extended by the amount necessary to hold the results.
    /// Uses SIMD gather for faster lookups.
    ///
    /// ## Safety
    /// - We unsafe set the length because we know we will overwrite every element.
    ///
    #[inline]
    pub fn lookup_extend_u8x16_vec(&self, values: &[u32], vec: &mut Vec<u8x16>) {
        let needed = values.len().div_ceil(16);
        let cur_len = vec.len();
        // Only reserve if we don't have enough capacity
        if vec.capacity() < cur_len + needed {
            vec.reserve(needed);
        }
        // Safety: we know we will overwrite every element, and we have already validated the length.
        unsafe { vec.set_len(cur_len + needed); }
        self.lookup_func(values, &mut |lookedup_values, start_idx| {
            vec[cur_len + start_idx / 16] = lookedup_values;
        });
    }

    /// Lookup 16 u32 keys at once using SIMD gather, returning a u8x16
    #[inline]
    fn lookup_chunk_16_gather(&self, keys: &[u32; 16]) -> u8x16 {
        #[cfg(target_arch = "x86_64")]
        {
            self.lookup_chunk_16_avx2(keys)
        }

        #[cfg(target_arch = "aarch64")]
        {
            self.lookup_chunk_16_neon(keys)
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback to scalar
            let mut results = [0u8; 16];
            for (i, &key) in keys.iter().enumerate() {
                if key <= self.max_key {
                    results[i] = self.lookup_table[key as usize];
                }
            }
            u8x16::from(results)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn lookup_chunk_16_avx2(&self, keys: &[u32; 16]) -> u8x16 {
        unsafe {
            use std::arch::x86_64::*;

            if is_x86_feature_detected!("avx2") {
                // Process first 8 keys
                let keys1_vec = _mm256_loadu_si256(keys.as_ptr() as *const __m256i);

                // Check bounds for first 8
                let max_key_vec = _mm256_set1_epi32(self.max_key as i32);
                let bounds_check1 = _mm256_cmpeq_epi32(
                    _mm256_min_epu32(keys1_vec, max_key_vec),
                    keys1_vec
                );

                // Step 1: Divide keys by 4 to get u32 word indices
                let word_indices1 = _mm256_srli_epi32::<2>(keys1_vec);

                // Step 2: Calculate remainders (keys % 4)
                let mask_3 = _mm256_set1_epi32(3);
                let remainders1 = _mm256_and_si256(keys1_vec, mask_3);

                // Step 3: SIMD gather u32 words
                let gathered_words1 = _mm256_mask_i32gather_epi32(
                    _mm256_setzero_si256(),
                    self.table_u32.as_ptr() as *const i32,
                    word_indices1,
                    bounds_check1,
                    4,
                );

                // Process second 8 keys
                let keys2_vec = _mm256_loadu_si256(keys.as_ptr().add(8) as *const __m256i);

                // Check bounds for second 8
                let bounds_check2 = _mm256_cmpeq_epi32(
                    _mm256_min_epu32(keys2_vec, max_key_vec),
                    keys2_vec
                );

                let word_indices2 = _mm256_srli_epi32::<2>(keys2_vec);
                let remainders2 = _mm256_and_si256(keys2_vec, mask_3);

                let gathered_words2 = _mm256_mask_i32gather_epi32(
                    _mm256_setzero_si256(),
                    self.table_u32.as_ptr() as *const i32,
                    word_indices2,
                    bounds_check2,
                    4,
                );

                // Step 4: Extract bytes from gathered u32 words
                let mut results = [0u8; 16];
                let gathered_array1: [i32; 8] = std::mem::transmute(gathered_words1);
                let gathered_array2: [i32; 8] = std::mem::transmute(gathered_words2);
                let remainder_array1: [i32; 8] = std::mem::transmute(remainders1);
                let remainder_array2: [i32; 8] = std::mem::transmute(remainders2);
                let bounds_array1: [i32; 8] = std::mem::transmute(bounds_check1);
                let bounds_array2: [i32; 8] = std::mem::transmute(bounds_check2);

                for i in 0..8 {
                    if bounds_array1[i] != 0 {
                        let word = gathered_array1[i] as u32;
                        let byte_pos = remainder_array1[i] as usize;
                        results[i] = ((word >> (byte_pos * 8)) & 0xFF) as u8;
                    }
                }

                for i in 0..8 {
                    if bounds_array2[i] != 0 {
                        let word = gathered_array2[i] as u32;
                        let byte_pos = remainder_array2[i] as usize;
                        results[i + 8] = ((word >> (byte_pos * 8)) & 0xFF) as u8;
                    }
                }

                u8x16::from(results)
            } else {
                // Fallback to scalar if AVX2 not available
                let mut results = [0u8; 16];
                for (i, &key) in keys.iter().enumerate() {
                    if key <= self.max_key {
                        results[i] = self.lookup_table[key as usize];
                    }
                }
                u8x16::from(results)
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn lookup_chunk_16_neon(&self, keys: &[u32; 16]) -> u8x16 {
        unsafe {
            use std::arch::aarch64::*;

            // Load 16 keys (4 NEON vectors of 4 u32 each)
            let keys1 = vld1q_u32(keys.as_ptr());
            let keys2 = vld1q_u32(keys.as_ptr().add(4));
            let keys3 = vld1q_u32(keys.as_ptr().add(8));
            let keys4 = vld1q_u32(keys.as_ptr().add(12));

            // Check bounds
            let max_key_vec = vdupq_n_u32(self.max_key);
            let valid1 = vcleq_u32(keys1, max_key_vec);
            let valid2 = vcleq_u32(keys2, max_key_vec);
            let valid3 = vcleq_u32(keys3, max_key_vec);
            let valid4 = vcleq_u32(keys4, max_key_vec);

            // Divide by 4 to get u32 word indices
            let word_indices1 = vshrq_n_u32::<2>(keys1);
            let word_indices2 = vshrq_n_u32::<2>(keys2);
            let word_indices3 = vshrq_n_u32::<2>(keys3);
            let word_indices4 = vshrq_n_u32::<2>(keys4);

            // Calculate remainders
            let mask_3 = vdupq_n_u32(3);
            let remainders1 = vandq_u32(keys1, mask_3);
            let remainders2 = vandq_u32(keys2, mask_3);
            let remainders3 = vandq_u32(keys3, mask_3);
            let remainders4 = vandq_u32(keys4, mask_3);

            // Manual gather (ARM doesn't have gather instructions)
            let mut results = [0u8; 16];

            let process_chunk = |results: &mut [u8; 16], word_indices: uint32x4_t, remainders: uint32x4_t, valid: uint32x4_t, offset: usize| {
                let word_indices_array: [u32; 4] = std::mem::transmute(word_indices);
                let remainders_array: [u32; 4] = std::mem::transmute(remainders);
                let valid_array: [u32; 4] = std::mem::transmute(valid);

                for i in 0..4 {
                    if valid_array[i] != 0 {
                        let word_idx = word_indices_array[i] as usize;
                        if word_idx < self.table_u32.len() {
                            let word = self.table_u32[word_idx];
                            let byte_pos = remainders_array[i] as usize;
                            results[offset + i] = ((word >> (byte_pos * 8)) & 0xFF) as u8;
                        }
                    }
                }
            };

            process_chunk(&mut results, word_indices1, remainders1, valid1, 0);
            process_chunk(&mut results, word_indices2, remainders2, valid2, 4);
            process_chunk(&mut results, word_indices3, remainders3, valid3, 8);
            process_chunk(&mut results, word_indices4, remainders4, valid4, 12);

            u8x16::from(results)
        }
    }
}

/// Dual vocabulary lookup kernel - u32 to u8 lookup table kernel with custom SIMD function for combining the results.
/// This is perfect for event_value + page_screen combined lookup functions.
/// It is faster than combining multiple single vocabulary lookups due to SIMD combining function.
///
/// The user is responsible for generating the lookup tables - so this can be used for different use cases, including
/// CASE..WHEN and bitmasking/filtering.
#[derive(Debug, Clone)]
pub struct SimdDualVocabU32U8Lookup<'a> {
    lookup_table1: &'a [u8],
    lookup_table2: &'a [u8],
}

impl<'a> SimdDualVocabU32U8Lookup<'a> {
    /// Creates a new dual vocabulary lookup kernel with the given lookup tables.
    #[inline]
    pub fn new(lookup_table1: &'a [u8], lookup_table2: &'a [u8]) -> Self {
        Self { lookup_table1, lookup_table2 }
    }

    /// Given two slices of equal length &[u32] indices, looks up each one and calls the user given function
    /// on assembled u8x16 results.
    /// - lookup_table1 is used for the first slice, lookup_table2 is used for the second slice.
    /// - The user function is passed (lookedup_values1: u8x16, lookedup_values2: u8x16, start_idx: usize), where
    ///   start_idx is 0 for the first chunk call, 16 for the next one, etc.
    /// - If the slices do not divide evenly into 16-item chunks, the rest is handled by filling missing values in the
    ///   u8x16 with zeroes.  Thus, the lookup assumes the zero is basically a NOP.
    #[inline]
    pub fn lookup_func<F>(&self, values1: &[u32], values2: &[u32], f: &mut F)
    where F: FnMut(u8x16, u8x16, usize) {
        assert!(values1.len() == values2.len(), "Values1 and values2 must have the same length");
        let (chunks1, rest1) = values1.as_chunks::<16>();
        let (chunks2, rest2) = values2.as_chunks::<16>();
        let mut idx = 0;
        for (chunk1, chunk2) in chunks1.iter().zip(chunks2.iter()) {
            // Get looked up values - LLVM should be able to auto-vectorize this
            let mut values1 = [0u8; 16];
            values1[0] = self.lookup_table1[chunk1[0] as usize];
            values1[1] = self.lookup_table1[chunk1[1] as usize];
            values1[2] = self.lookup_table1[chunk1[2] as usize];
            values1[3] = self.lookup_table1[chunk1[3] as usize];
            values1[4] = self.lookup_table1[chunk1[4] as usize];
            values1[5] = self.lookup_table1[chunk1[5] as usize];
            values1[6] = self.lookup_table1[chunk1[6] as usize];
            values1[7] = self.lookup_table1[chunk1[7] as usize];
            values1[8] = self.lookup_table1[chunk1[8] as usize];
            values1[9] = self.lookup_table1[chunk1[9] as usize];
            values1[10] = self.lookup_table1[chunk1[10] as usize];
            values1[11] = self.lookup_table1[chunk1[11] as usize];
            values1[12] = self.lookup_table1[chunk1[12] as usize];
            values1[13] = self.lookup_table1[chunk1[13] as usize];
            values1[14] = self.lookup_table1[chunk1[14] as usize];
            values1[15] = self.lookup_table1[chunk1[15] as usize];

            let mut values2 = [0u8; 16];
            values2[0] = self.lookup_table2[chunk2[0] as usize];
            values2[1] = self.lookup_table2[chunk2[1] as usize];
            values2[2] = self.lookup_table2[chunk2[2] as usize];
            values2[3] = self.lookup_table2[chunk2[3] as usize];
            values2[4] = self.lookup_table2[chunk2[4] as usize];
            values2[5] = self.lookup_table2[chunk2[5] as usize];
            values2[6] = self.lookup_table2[chunk2[6] as usize];
            values2[7] = self.lookup_table2[chunk2[7] as usize];
            values2[8] = self.lookup_table2[chunk2[8] as usize];
            values2[9] = self.lookup_table2[chunk2[9] as usize];
            values2[10] = self.lookup_table2[chunk2[10] as usize];
            values2[11] = self.lookup_table2[chunk2[11] as usize];
            values2[12] = self.lookup_table2[chunk2[12] as usize];
            values2[13] = self.lookup_table2[chunk2[13] as usize];
            values2[14] = self.lookup_table2[chunk2[14] as usize];
            values2[15] = self.lookup_table2[chunk2[15] as usize];

            (f)(u8x16::from(values1), u8x16::from(values2), idx);
            idx += 16;
        }

        // Handle the rest... just loop and do a lookup, feed to user function with 0's for items not in the slice.
        if !rest1.is_empty() {
            let mut values1 = [0u8; 16];
            let mut values2 = [0u8; 16];
            for i in 0..rest1.len() {
                values1[i] = self.lookup_table1[rest1[i] as usize];
                values2[i] = self.lookup_table2[rest2[i] as usize];
            }
            (f)(u8x16::from(values1), u8x16::from(values2), idx);
        }
    }

    /// Convenience function which does dual lookup, combines the results using a user-defined combiner function,
    /// and writes the combined results into a Vec of the same length as the input slices.
    ///
    /// The combiner function `f` takes two u8x16 values (looked up from table1 and table2) and returns a combined u8x16.
    /// Unlike the single vocabulary version, this dual vocabulary version requires a combiner function.
    #[inline]
    pub fn lookup_into_vec<F>(&self, values1: &[u32], values2: &[u32], f: &mut F) -> Vec<u8>
    where F: FnMut(u8x16, u8x16) -> u8x16 {
        assert!(values1.len() == values2.len(), "Values1 and values2 must have the same length");

        // Allocate a vector with the same length as the input slices - setting the length so contents are uninitialized.
        // Safety: This is OK as this function explicitly overwrites every value, and there is no reading beforehand.
        let mut result = Vec::with_capacity(values1.len());
        unsafe { result.set_len(values1.len()); }

        // Call lookup_func with a closure that writes to the result vector
        // NOTE: we do as_chunks_mut as that allows for bulk writes - much more efficient than individual writes.
        let (write_slices, rest) = result[..].as_chunks_mut::<16>();
        self.lookup_func(values1, values2, &mut |lookedup_values1, lookedup_values2, start_idx| {
            let combined = (f)(lookedup_values1, lookedup_values2);
            let slice_num = start_idx / 16;
            if slice_num < write_slices.len() {
                // Safety: we have already validated slice_num is within range, and that also means the ensure slice
                //  is writeable.  Thus, skip bounds checks and do single instruction write.
                unsafe {
                    let ptr = write_slices[slice_num].as_mut_ptr() as *mut u8x16;
                    ptr.write_unaligned(combined);
                }
            } else {
                // Handle remainder - write only the needed bytes
                rest.copy_from_slice(&combined.as_array()[..rest.len()]);
            }
        });
        result
    }
}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_single_vocab_lookup_into_vec() {
        // Create a simple lookup table
        let lookup_table = vec![0u8, 10, 20, 30, 40];
        let lookup = SimdSingleVocabU32U8Lookup::new(&lookup_table);

        // Test with values that are less than lookup table size
        let values = vec![0u32, 1, 2, 3, 4, 1, 2, 3];
        let result = lookup.lookup_into_vec(&values);

        assert_eq!(result.len(), values.len());
        assert_eq!(result[0], 0);
        assert_eq!(result[1], 10);
        assert_eq!(result[2], 20);
        assert_eq!(result[3], 30);
        assert_eq!(result[4], 40);
        assert_eq!(result[5], 10);
        assert_eq!(result[6], 20);
        assert_eq!(result[7], 30);
    }

    #[test]
    fn test_dual_vocab_lookup_into_vec() {
        // Create two simple lookup tables
        let lookup_table1 = vec![0u8, 1, 2, 3, 4];
        let lookup_table2 = vec![0u8, 10, 20, 30, 40];
        let lookup = SimdDualVocabU32U8Lookup::new(&lookup_table1, &lookup_table2);

        // Test with values that are less than lookup table size
        let values1 = vec![0u32, 1, 2, 3, 4, 1, 2, 3];
        let values2 = vec![0u32, 1, 2, 3, 4, 1, 2, 3];

        // Use bitwise OR as the combiner function
        let result = lookup.lookup_into_vec(&values1, &values2, &mut |v1, v2| v1 | v2);

        assert_eq!(result.len(), values1.len());
        assert_eq!(result[0], 0);   // 0
        assert_eq!(result[1], 1 | 10);  // 11
        assert_eq!(result[2], 2 | 20);  // 22
        assert_eq!(result[3], 3 | 30);  // 31
        assert_eq!(result[4], 4 | 40);  // 44
        assert_eq!(result[5], 1 | 10);  // 11
        assert_eq!(result[6], 2 | 20);  // 22
        assert_eq!(result[7], 3 | 30);  // 31
    }

    #[test]
    fn test_dual_vocab_lookup_into_vec_large() {
        // Test with a larger dataset that spans multiple u8x16 chunks
        let lookup_table1 = vec![1u8; 100];
        let lookup_table2 = vec![2u8; 100];
        let lookup = SimdDualVocabU32U8Lookup::new(&lookup_table1, &lookup_table2);

        // Create 50 values (more than 16, so it tests multiple chunks)
        let values1 = vec![0u32; 50];
        let values2 = vec![0u32; 50];

        // Use addition as the combiner function
        let result = lookup.lookup_into_vec(&values1, &values2, &mut |v1, v2| v1 + v2);

        assert_eq!(result.len(), 50);
        // All results should be 1 + 2 = 3
        for &val in &result {
            assert_eq!(val, 3);
        }
    }
}
