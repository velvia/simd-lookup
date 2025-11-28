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

use wide::{u8x16, u32x8, u64x8};

use crate::{
    bulk_vec_extender::{BulkVecExtender, SliceU8SIMDExtender},
    wide_utils::WideUtilsExt,
};

/// Calculate real memory addresses from base pointer and u32 offsets using SIMD
/// Returns two u64x8 vectors containing the actual memory addresses
/// This avoids repeating address calculations in both lookup and prefetch operations
#[inline]
fn calc_real_addresses<T>(base: &T, offsets: &[u32; 16]) -> (u64x8, u64x8) {
    let base_ptr = base as *const T;
    let base_addr = base_ptr as u64;
    let base_simd = u64x8::splat(base_addr);

    // Convert slices directly to SIMD and multiply by element size
    let first_offsets_u32 = u32x8::from(<[u32; 8]>::try_from(&offsets[..8]).unwrap()) * (std::mem::size_of::<T>() as u32);
    let second_offsets_u32 = u32x8::from(<[u32; 8]>::try_from(&offsets[8..]).unwrap()) * (std::mem::size_of::<T>() as u32);

    // Widen to u64 and add to base address
    let first_offsets_u64 = first_offsets_u32.widen_to_u64x8();
    let second_offsets_u64 = second_offsets_u32.widen_to_u64x8();

    let first_addresses = base_simd + first_offsets_u64;
    let second_addresses = base_simd + second_offsets_u64;

    (first_addresses, second_addresses)
}

/// Lookup values directly from computed addresses (two u64x8 vectors)
/// Returns the looked up u8 values as a u8x16
#[inline]
fn lookup_from_addresses(addresses: (u64x8, u64x8)) -> u8x16 {
    let (first_addrs, second_addrs) = addresses;
    let first_array = first_addrs.as_array();
    let second_array = second_addrs.as_array();

    // Perform lookups directly from the computed addresses
    let values = [
        unsafe { *(first_array[0] as *const u8) },
        unsafe { *(first_array[1] as *const u8) },
        unsafe { *(first_array[2] as *const u8) },
        unsafe { *(first_array[3] as *const u8) },
        unsafe { *(first_array[4] as *const u8) },
        unsafe { *(first_array[5] as *const u8) },
        unsafe { *(first_array[6] as *const u8) },
        unsafe { *(first_array[7] as *const u8) },
        unsafe { *(second_array[0] as *const u8) },
        unsafe { *(second_array[1] as *const u8) },
        unsafe { *(second_array[2] as *const u8) },
        unsafe { *(second_array[3] as *const u8) },
        unsafe { *(second_array[4] as *const u8) },
        unsafe { *(second_array[5] as *const u8) },
        unsafe { *(second_array[6] as *const u8) },
        unsafe { *(second_array[7] as *const u8) },
    ];

    u8x16::from(values)
}

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
    /// The user function is passed (lookedup_values: u8x16, num_bytes: usize),
    /// where num_bytes is 16 other than the last/remainder chunk, where it may be less than that.
    ///
    /// If the slice does not divide evenly into 16-item chunks, the rest is handled by filling missing values in the
    /// u8x16 with zeroes.  Thus, the lookup assumes the zero is basically a NOP.
    #[inline]
    pub fn lookup_func<F>(&self, values: &[u32], f: &mut F)
    where
        F: FnMut(u8x16, usize),
    {
        let (chunks, rest) = values.as_chunks::<16>();
        for chunk in chunks {
            // Calculate addresses using SIMD and prefetch them
            let addresses = calc_real_addresses(&self.lookup_table[0], chunk);
            use crate::prefetch::{L3, prefetch_sixteen_addresses};
            prefetch_sixteen_addresses::<L3>(addresses);

            // Lookup values using SIMD address calculation
            let values = lookup_from_addresses(addresses);

            // Call user function
            (f)(values, 16);
        }

        // Handle the rest... just loop and do a lookup, feed to user function with 0's for items not in the slice.
        if !rest.is_empty() {
            let mut values = [0u8; 16];
            for i in 0..rest.len() {
                values[i] = self.lookup_table[rest[i] as usize];
            }
            (f)(u8x16::from(values), rest.len());
        }
    }

    /// Convenience function which does lookup and writes the results into a Vec of the same length as the input slice.
    /// Does not transform the looked up values.  Actually, extends a mutable Vec of u8.
    #[inline]
    pub fn lookup_into_vec(&self, values: &[u32], buffer: &mut Vec<u8>) {
        let mut write_guard = buffer.bulk_extend_guard(values.len());
        let mut write_slice = write_guard.as_mut_slice();
        let mut num_written = 0;
        self.lookup_func(values, &mut |lookedup_values, num_bytes| {
            write_slice.write_u8x16(num_written, lookedup_values, num_bytes);
            num_written += num_bytes;
        });
    }

    /// Version of lookup_into_vec which writes into a mutable u8x16 buffer, for cascaded lookups
    #[inline]
    pub fn lookup_into_u8x16_buffer(&self, values: &[u32], buffer: &mut [u8x16]) {
        assert!(
            (buffer.len() * 16) >= values.len(),
            "Buffer must be at least as long as the input values"
        );
        let mut idx = 0;
        self.lookup_func(values, &mut |lookedup_values, _num_bytes| {
            buffer[idx] = lookedup_values;
            idx += 1;
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
        let mut guard = vec.bulk_extend_guard(needed);
        self.lookup_into_u8x16_buffer(values, guard.as_mut_slice());
        // guard drops here, automatically finalizes to correct length
    }
}

/// Pipelined single vocabulary lookup kernel - u32 to u8 lookup table kernel with prefetch pipelining
///
/// This version pipelines prefetch operations with the actual lookup work to hide memory latency.
/// The algorithm works as follows:
/// 1. Read values from current chunk addresses
/// 2. Prefetch next chunk addresses while processing current values
/// 3. Call SIMD lookup function on current values
/// 4. Loop to next chunk
///
/// This pipelining allows memory prefetch latency to be hidden behind computation work.
#[derive(Debug, Clone)]
pub struct PipelinedSingleVocabU32U8Lookup<'a> {
    lookup_table: &'a [u8],
}

impl<'a> PipelinedSingleVocabU32U8Lookup<'a> {
    #[inline]
    pub fn new(lookup_table: &'a [u8]) -> Self {
        Self { lookup_table }
    }

    /// Pipelined lookup function that prefetches the next chunk while processing the current one
    ///
    /// The pipelining strategy:
    /// - Process chunks of 16 u32 values at a time
    /// - For each chunk: prefetch next chunk addresses, then process current chunk
    /// - This hides prefetch latency behind the lookup computation work
    #[inline]
    pub fn lookup_func<F>(&self, values: &[u32], f: &mut F)
    where
        F: FnMut(u8x16, usize),
    {
        let (chunks, rest) = values.as_chunks::<16>();

        if chunks.is_empty() {
            // Handle case where we have fewer than 16 values total
            if !rest.is_empty() {
                self.process_remainder(rest, f);
            }
            return;
        }

        // Calculate addresses for the first chunk
        let current_chunk = chunks[0];
        let mut current_addresses = calc_real_addresses(&self.lookup_table[0], &current_chunk);

        // For single chunk case, just process it directly
        if chunks.len() == 1 {
            let values = lookup_from_addresses(current_addresses);
            (f)(values, 16);
            if !rest.is_empty() {
                self.process_remainder(rest, f);
            }
            return;
        }

        // Pipeline: for each chunk, read current values, prefetch next, then process
        for (_, &next_chunk) in chunks.iter().enumerate().skip(1) {
            // Step 1: Read current values using pre-calculated addresses
            let current_values = lookup_from_addresses(current_addresses);

            // Step 2: Calculate and prefetch next chunk addresses while we have current values
            let next_addresses = calc_real_addresses(&self.lookup_table[0], &next_chunk);
            use crate::prefetch::{L3, prefetch_sixteen_addresses};
            prefetch_sixteen_addresses::<L3>(next_addresses);

            // Step 3: Call lookup function with current values (while prefetches are in flight)
            (f)(current_values, 16);

            // Move to next chunk - reuse the calculated addresses
            current_addresses = next_addresses;
        }

        // Process the last chunk using pre-calculated addresses
        let last_values = lookup_from_addresses(current_addresses);
        (f)(last_values, 16);

        // Handle remainder
        if !rest.is_empty() {
            self.process_remainder(rest, f);
        }
    }

    /// Process remainder elements (< 16 elements)
    #[inline]
    fn process_remainder<F>(&self, rest: &[u32], f: &mut F)
    where
        F: FnMut(u8x16, usize),
    {
        let mut values = [0u8; 16];
        for i in 0..rest.len() {
            values[i] = self.lookup_table[rest[i] as usize];
        }
        (f)(u8x16::from(values), rest.len());
    }

    /// Convenience function which does lookup and writes the results into a Vec
    #[inline]
    pub fn lookup_into_vec(&self, values: &[u32], buffer: &mut Vec<u8>) {
        let mut write_guard = buffer.bulk_extend_guard(values.len());
        let write_slice = write_guard.as_mut_slice();
        let mut num_written = 0;
        self.lookup_func(values, &mut |lookedup_values, num_bytes| {
            let target_slice = &mut write_slice[num_written..num_written + num_bytes];
            target_slice.copy_from_slice(&lookedup_values.to_array()[..num_bytes]);
            num_written += num_bytes;
        });
        // guard drops here, automatically finalizes to correct length
    }
}

/// Dual vocabulary lookup kernel - u32 to u8 lookup table kernel with custom SIMD function for combining the results.
/// It always does a lookup of the second table.
/// This is perfect for event_value + page_screen combined lookup functions.
/// It is faster than combining multiple single vocabulary lookups due to SIMD combining function.
/// Second lookup table is only looked up if the first lookup table returns a non-zero value.
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
        Self {
            lookup_table1,
            lookup_table2,
        }
    }

    /// Given two slices of equal length &[u32] indices, looks up each one and calls the user given function
    /// on assembled u8x16 results.
    /// - lookup_table1 is used for the first slice, lookup_table2 is used for the second slice.
    /// - The user function is passed (lookedup_values1: u8x16, lookedup_values2: u8x16, num_bytes), where
    ///   num_bytes is 16 other than the last/remainder chunk, where it may be less than that.
    /// - If the slices do not divide evenly into 16-item chunks, the rest is handled by filling missing values in the
    ///   u8x16 with zeroes.  Thus, the lookup assumes the zero is basically a NOP.
    #[inline]
    pub fn lookup_func<F>(&self, values1: &[u32], values2: &[u32], f: &mut F)
    where
        F: FnMut(u8x16, u8x16, usize),
    {
        assert!(
            values1.len() == values2.len(),
            "Values1 and values2 must have the same length"
        );
        let (chunks1, rest1) = values1.as_chunks::<16>();
        let (chunks2, rest2) = values2.as_chunks::<16>();
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
            for i in 0..16 {
                if values1[i] != 0 {
                    values2[i] = self.lookup_table2[chunk2[i] as usize];
                }
            }

            (f)(u8x16::from(values1), u8x16::from(values2), 16);
        }

        // Handle the rest... just loop and do a lookup, feed to user function with 0's for items not in the slice.
        if !rest1.is_empty() {
            let mut values1 = [0u8; 16];
            let mut values2 = [0u8; 16];
            for i in 0..rest1.len() {
                values1[i] = self.lookup_table1[rest1[i] as usize];
                values2[i] = self.lookup_table2[rest2[i] as usize];
            }
            (f)(u8x16::from(values1), u8x16::from(values2), rest1.len());
        }
    }

    /// Convenience function which does dual lookup, combines the results using a user-defined combiner function,
    /// and extends the combined results into a Vec (pushing all combined results)
    ///
    /// The combiner function `f` takes two u8x16 values (looked up from table1 and table2) and returns a combined u8x16.
    /// Unlike the single vocabulary version, this dual vocabulary version requires a combiner function.
    #[inline]
    pub fn lookup_into_vec<F>(
        &self,
        values1: &[u32],
        values2: &[u32],
        output: &mut Vec<u8>,
        f: &mut F,
    ) where
        F: FnMut(u8x16, u8x16) -> u8x16,
    {
        assert!(
            values1.len() == values2.len(),
            "Values1 and values2 must have the same length"
        );

        let mut write_guard = output.bulk_extend_guard(values1.len());
        let mut write_slice = write_guard.as_mut_slice();
        let mut num_written = 0;
        self.lookup_func(
            values1,
            values2,
            &mut |lookedup_values1, lookedup_values2, num_bytes| {
                let combined = (f)(lookedup_values1, lookedup_values2);
                write_slice.write_u8x16(num_written, combined, num_bytes);
                num_written += num_bytes;
            },
        );
        // write_guard drops here, automatically finalizes to correct length.  We don't have to set final
        // number of bytes since it is the same as the input, which is the default
    }
}

/// Dual vocabulary lookup kernel - u32 to u8 lookup table kernel with custom SIMD function for combining the results.
/// It tries to eliminate thrashing by using internally the single vocab kernel to write results out first to
/// a local temporary buffer, which is saved, then it looks up the second vocabulary, only if the first vocab returns
/// nonzero results - thus minimizing the number of reads from the second vocabulary.
/// By sequencing in this order, we hope to minimize the cache thrashing.
///
/// The user is responsible for generating the lookup tables - so this can be used for different use cases, including
/// CASE..WHEN and bitmasking/filtering.
#[derive(Debug, Clone)]
pub struct SimdDualVocabU32U8LookupV2<'a> {
    lookup1: SimdSingleVocabU32U8Lookup<'a>,
    lookup2: &'a [u8],
    temp_buffer: Vec<u8x16>,
}

impl<'a> SimdDualVocabU32U8LookupV2<'a> {
    #[inline]
    pub fn new(lookup_table1: &'a [u8], lookup_table2: &'a [u8]) -> Self {
        Self {
            lookup1: SimdSingleVocabU32U8Lookup::new(lookup_table1),
            lookup2: lookup_table2,
            temp_buffer: Vec::with_capacity(128),
        }
    }

    /// Given two slices of equal length &[u32] indices, looks up each one and calls the user given function
    /// on assembled u8x16 results.
    /// - lookup_table1 is used for the first slice, lookup_table2 is used for the second slice.
    /// - Only if the u8 from the first lookup table is nonzero, will the second lookup table be read.
    /// - The user function is passed (lookedup_values1: u8x16, lookedup_values2: u8x16, start_idx: usize), where
    ///   start_idx is 0 for the first chunk call, 16 for the next one, etc.
    /// - If the slices do not divide evenly into 16-item chunks, the rest is handled by filling missing values in the
    ///   u8x16 with zeroes.  Thus, the lookup assumes the zero is basically a NOP.
    ///
    /// The lookup function is passed these arguments: (lookedup_values1: u8x16, lookedup_values2: u8x16, num_bytes)
    /// - num_bytes: usually 16, but may be less for the last/remainder chunk.
    #[inline]
    pub fn lookup_func<F>(&mut self, values1: &[u32], values2: &[u32], f: &mut F)
    where
        F: FnMut(u8x16, u8x16, usize),
    {
        assert!(
            values1.len() == values2.len(),
            "Values1 and values2 must have the same length"
        );

        // Clear temp_buffer for reuse
        self.temp_buffer.clear();

        // First read the first vocabulary into the temporary buffer
        self.lookup1
            .lookup_extend_u8x16_vec(values1, &mut self.temp_buffer);

        let (chunks2, rest2) = values2.as_chunks::<16>();

        // Process full chunks
        for (i, chunk2) in chunks2.iter().enumerate() {
            let vocab1_result = self.temp_buffer[i];
            let vocab1_array = vocab1_result.as_array();

            // Only do lookup2 for positions where vocab1_result is nonzero
            // Use two u64 loops, somehow it's faster than writing to [u8; 16] directly.
            let local_chunk = *chunk2;

            // Process high 8 bytes (8-15) into first u64
            let mut result_high = 0u64;
            for j in (8..16).rev() {
                result_high <<= 8;
                if vocab1_array[j] != 0 {
                    result_high += self.lookup2[local_chunk[j] as usize] as u64;
                }
            }

            // Process low 8 bytes (0-7) into second u64
            let mut result_low = 0u64;
            for j in (0..8).rev() {
                result_low <<= 8;
                if vocab1_array[j] != 0 {
                    result_low += self.lookup2[local_chunk[j] as usize] as u64;
                }
            }

            // Combine into u128 for conversion to u8x16
            let result = ((result_high as u128) << 64) | (result_low as u128);

            // Call user function with both u8x16 results
            (f)(vocab1_result, u8x16::from(result.to_le_bytes()), 16);
        }

        // Handle the remainder
        if !rest2.is_empty() {
            // The remainder for vocab1 is already in temp_buffer (lookup_extend_u8x16_vec handles it)
            let vocab1_result = self.temp_buffer[chunks2.len()];
            let vocab1_array = vocab1_result.as_array();
            let mut vocab2_result = [0u8; 16];

            for i in 0..rest2.len() {
                if vocab1_array[i] != 0 {
                    // Only lookup if the first vocab returned nonzero
                    vocab2_result[i] = self.lookup2[rest2[i] as usize];
                }
            }
            (f)(vocab1_result, u8x16::from(vocab2_result), rest2.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipelined_single_vocab_lookup_basic() {
        // Create a simple lookup table: index -> index as u8
        let lookup_table: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let pipelined_lookup = PipelinedSingleVocabU32U8Lookup::new(&lookup_table);

        // Test with exactly 16 values (one chunk)
        let values = vec![
            10u32, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let mut results = Vec::new();
        pipelined_lookup.lookup_func(&values, &mut |lookedup_values, num_bytes| {
            let array = lookedup_values.to_array();
            results.extend_from_slice(&array[..num_bytes]);
        });

        let expected: Vec<u8> = values.iter().map(|&v| v as u8).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_pipelined_single_vocab_lookup_multiple_chunks() {
        // Create a simple lookup table
        let lookup_table: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let pipelined_lookup = PipelinedSingleVocabU32U8Lookup::new(&lookup_table);

        // Test with 35 values (2 full chunks + 3 remainder)
        let values: Vec<u32> = (1..36).collect();
        let mut results = Vec::new();
        pipelined_lookup.lookup_func(&values, &mut |lookedup_values, num_bytes| {
            let array = lookedup_values.to_array();
            results.extend_from_slice(&array[..num_bytes]);
        });

        let expected: Vec<u8> = values.iter().map(|&v| v as u8).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_pipelined_single_vocab_lookup_into_vec() {
        // Create a lookup table where each index maps to its double (mod 256)
        let lookup_table: Vec<u8> = (0..256).map(|i| ((i * 2) % 256) as u8).collect();
        let pipelined_lookup = PipelinedSingleVocabU32U8Lookup::new(&lookup_table);

        let values = vec![1u32, 2, 3, 4, 5, 100, 150, 200];
        let mut buffer = Vec::new();
        pipelined_lookup.lookup_into_vec(&values, &mut buffer);

        let expected: Vec<u8> = values.iter().map(|&v| ((v * 2) % 256) as u8).collect();
        assert_eq!(buffer, expected);
    }

    #[test]
    fn test_pipelined_vs_original_consistency() {
        // Test that pipelined version produces same results as original
        let lookup_table: Vec<u8> = (0..256).map(|i| (i ^ 0xAA) as u8).collect();

        let original_lookup = SimdSingleVocabU32U8Lookup::new(&lookup_table);
        let pipelined_lookup = PipelinedSingleVocabU32U8Lookup::new(&lookup_table);

        // Test with various sizes
        for size in [5, 16, 17, 32, 33, 100] {
            let values: Vec<u32> = (0..size).map(|i| (i * 7) % 256).collect();

            let mut original_results = Vec::new();
            original_lookup.lookup_into_vec(&values, &mut original_results);

            let mut pipelined_results = Vec::new();
            pipelined_lookup.lookup_into_vec(&values, &mut pipelined_results);

            assert_eq!(
                original_results, pipelined_results,
                "Results differ for size {}",
                size
            );
        }
    }

    #[test]
    fn test_single_vocab_lookup_into_vec() {
        // Create a simple lookup table
        let lookup_table = vec![0u8, 10, 20, 30, 40];
        let lookup = SimdSingleVocabU32U8Lookup::new(&lookup_table);

        // Test with values that are less than lookup table size
        let values = vec![0u32, 1, 2, 3, 4, 1, 2, 3];
        let mut result = Vec::new();
        lookup.lookup_into_vec(&values, &mut result);

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
        let mut result = Vec::new();
        lookup.lookup_into_vec(&values1, &values2, &mut result, &mut |v1, v2| v1 | v2);

        assert_eq!(result.len(), values1.len());
        assert_eq!(result[0], 0); // 0
        assert_eq!(result[1], 1 | 10); // 11
        assert_eq!(result[2], 2 | 20); // 22
        assert_eq!(result[3], 3 | 30); // 31
        assert_eq!(result[4], 4 | 40); // 44
        assert_eq!(result[5], 1 | 10); // 11
        assert_eq!(result[6], 2 | 20); // 22
        assert_eq!(result[7], 3 | 30); // 31
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
        let mut result = Vec::new();
        lookup.lookup_into_vec(&values1, &values2, &mut result, &mut |v1, v2| v1 + v2);

        assert_eq!(result.len(), 50);
        // All results should be 1 + 2 = 3
        for &val in &result {
            assert_eq!(val, 3);
        }
    }

    #[test]
    fn test_dual_vocab_v2_lookup_func_basic() {
        // Create two simple lookup tables
        let lookup_table1 = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let lookup_table2 = vec![0u8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let mut lookup = SimdDualVocabU32U8LookupV2::new(&lookup_table1, &lookup_table2);

        // Test with values that are less than lookup table size
        let values1 = vec![0u32, 1, 2, 3, 4];
        let values2 = vec![1u32, 2, 3, 4, 5];

        let mut vocab1_results = Vec::new();
        let mut vocab2_results = Vec::new();
        let mut num_bytes_list = Vec::new();

        lookup.lookup_func(&values1, &values2, &mut |v1, v2, num_bytes| {
            vocab1_results.push(v1);
            vocab2_results.push(v2);
            num_bytes_list.push(num_bytes);
        });

        // 5 values = 0 full chunks + 1 remainder of 5
        assert_eq!(num_bytes_list.len(), 1);
        assert_eq!(num_bytes_list[0], 5);

        // Check vocab1 results
        let v1_array = vocab1_results[0].as_array();
        assert_eq!(v1_array[0], 0);
        assert_eq!(v1_array[1], 1);
        assert_eq!(v1_array[2], 2);
        assert_eq!(v1_array[3], 3);
        assert_eq!(v1_array[4], 4);

        // Check vocab2 results - should only be looked up where vocab1 is nonzero
        let v2_array = vocab2_results[0].as_array();
        assert_eq!(v2_array[0], 0); // vocab1[0] == 0, so vocab2[0] should be 0 (not looked up)
        assert_eq!(v2_array[1], 20); // vocab1[1] == 1 (nonzero), so vocab2[1] == lookup_table2[values2[1]] == lookup_table2[2] == 20
        assert_eq!(v2_array[2], 30); // vocab1[2] == 2 (nonzero), so vocab2[2] == lookup_table2[values2[2]] == lookup_table2[3] == 30
        assert_eq!(v2_array[3], 40); // vocab1[3] == 3 (nonzero), so vocab2[3] == lookup_table2[values2[3]] == lookup_table2[4] == 40
        assert_eq!(v2_array[4], 50); // vocab1[4] == 4 (nonzero), so vocab2[4] == lookup_table2[values2[4]] == lookup_table2[5] == 50
    }

    #[test]
    fn test_dual_vocab_v2_lookup_func_remainder() {
        // Test remainder handling - values that don't divide evenly into 16
        let lookup_table1 = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let lookup_table2 = vec![0u8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let mut lookup = SimdDualVocabU32U8LookupV2::new(&lookup_table1, &lookup_table2);

        // Create 25 values (16 + 9 remainder)
        let values1: Vec<u32> = (0..25).map(|i| (i % 5) as u32).collect();
        let values2: Vec<u32> = (0..25).map(|i| ((i % 5) + 1) as u32).collect();

        let mut vocab1_results = Vec::new();
        let mut vocab2_results = Vec::new();
        let mut num_bytes_list = Vec::new();

        lookup.lookup_func(&values1, &values2, &mut |v1, v2, num_bytes| {
            vocab1_results.push(v1);
            vocab2_results.push(v2);
            num_bytes_list.push(num_bytes);
        });

        // Should have 2 chunks: one full (16) and one remainder (9)
        assert_eq!(num_bytes_list.len(), 2);
        assert_eq!(num_bytes_list[0], 16);
        assert_eq!(num_bytes_list[1], 9);

        // Check first chunk (full 16)
        let v1_chunk0 = vocab1_results[0].as_array();
        let v2_chunk0 = vocab2_results[0].as_array();
        for i in 0..16 {
            let expected_v1 = lookup_table1[values1[i] as usize];
            assert_eq!(v1_chunk0[i], expected_v1);
            if expected_v1 != 0 {
                assert_eq!(v2_chunk0[i], lookup_table2[values2[i] as usize]);
            } else {
                assert_eq!(v2_chunk0[i], 0);
            }
        }

        // Check remainder chunk (9 elements)
        let v1_remainder = vocab1_results[1].as_array();
        let v2_remainder = vocab2_results[1].as_array();
        for i in 0..9 {
            let expected_v1 = lookup_table1[values1[16 + i] as usize];
            assert_eq!(v1_remainder[i], expected_v1);
            if expected_v1 != 0 {
                assert_eq!(v2_remainder[i], lookup_table2[values2[16 + i] as usize]);
            } else {
                assert_eq!(v2_remainder[i], 0);
            }
        }
        // Remaining positions in the u8x16 should be zero
        for i in 9..16 {
            assert_eq!(v1_remainder[i], 0);
            assert_eq!(v2_remainder[i], 0);
        }
    }

    #[test]
    fn test_dual_vocab_v2_lookup_func_zero_filtering() {
        // Test that lookup2 is only performed when vocab1 is nonzero
        let lookup_table1 = vec![0u8, 0, 0, 5, 0, 0, 0, 10, 0, 0, 0];
        let lookup_table2 = vec![0u8, 100, 200, 50, 150, 250, 60, 70, 80, 90, 100];
        let mut lookup = SimdDualVocabU32U8LookupV2::new(&lookup_table1, &lookup_table2);

        // values1 will map to indices that are mostly zero in lookup_table1
        let values1 = vec![0u32, 1, 2, 3, 4, 5, 6, 7];
        let values2 = vec![1u32, 2, 3, 4, 5, 6, 7, 8]; // These would normally map to 100, 200, etc.

        let mut vocab1_results = Vec::new();
        let mut vocab2_results = Vec::new();

        lookup.lookup_func(&values1, &values2, &mut |v1, v2, _num_bytes| {
            vocab1_results.push(v1);
            vocab2_results.push(v2);
        });

        let v1_array = vocab1_results[0].as_array();
        let v2_array = vocab2_results[0].as_array();

        // Check that vocab2 is only looked up where vocab1 is nonzero
        assert_eq!(v1_array[0], 0);
        assert_eq!(v2_array[0], 0); // vocab1[0] == 0, so vocab2 not looked up

        assert_eq!(v1_array[1], 0);
        assert_eq!(v2_array[1], 0); // vocab1[1] == 0, so vocab2 not looked up

        assert_eq!(v1_array[2], 0);
        assert_eq!(v2_array[2], 0); // vocab1[2] == 0, so vocab2 not looked up

        assert_eq!(v1_array[3], 5);
        assert_eq!(v2_array[3], 150); // vocab1[3] == 5 (nonzero), so vocab2[3] == lookup_table2[values2[3]] == lookup_table2[4] == 150

        assert_eq!(v1_array[4], 0);
        assert_eq!(v2_array[4], 0); // vocab1[4] == 0, so vocab2 not looked up

        assert_eq!(v1_array[5], 0);
        assert_eq!(v2_array[5], 0); // vocab1[5] == 0, so vocab2 not looked up

        assert_eq!(v1_array[6], 0);
        assert_eq!(v2_array[6], 0); // vocab1[6] == 0, so vocab2 not looked up

        assert_eq!(v1_array[7], 10);
        assert_eq!(v2_array[7], 80); // vocab1[7] == 10 (nonzero), so vocab2[7] == lookup_table2[values2[7]] == lookup_table2[8] == 80
    }

    #[test]
    fn test_dual_vocab_v2_lookup_func_multiple_chunks() {
        // Test with multiple full chunks
        let lookup_table1 = vec![1u8; 100];
        let lookup_table2 = vec![2u8; 100];
        let mut lookup = SimdDualVocabU32U8LookupV2::new(&lookup_table1, &lookup_table2);

        // Create 50 values (3 full chunks: 16 + 16 + 16 + 2 remainder)
        let values1: Vec<u32> = (0..50).map(|i| (i % 10) as u32).collect();
        let values2: Vec<u32> = (0..50).map(|i| ((i % 10) + 1) as u32).collect();

        let mut vocab1_results = Vec::new();
        let mut vocab2_results = Vec::new();
        let mut num_bytes_list = Vec::new();

        lookup.lookup_func(&values1, &values2, &mut |v1, v2, num_bytes| {
            vocab1_results.push(v1);
            vocab2_results.push(v2);
            num_bytes_list.push(num_bytes);
        });

        // Should have 4 chunks: 3 full (16 each) + 1 remainder (2)
        assert_eq!(num_bytes_list.len(), 4);
        assert_eq!(num_bytes_list[0], 16);
        assert_eq!(num_bytes_list[1], 16);
        assert_eq!(num_bytes_list[2], 16);
        assert_eq!(num_bytes_list[3], 2);

        // Verify all chunks
        let mut global_idx = 0;
        for chunk_idx in 0..4 {
            let v1_chunk = vocab1_results[chunk_idx].as_array();
            let v2_chunk = vocab2_results[chunk_idx].as_array();
            let chunk_len = num_bytes_list[chunk_idx];

            for i in 0..chunk_len {
                let expected_v1 = lookup_table1[values1[global_idx] as usize];
                assert_eq!(v1_chunk[i], expected_v1);
                // Since vocab1 is always nonzero (lookup_table1 is all 1s), vocab2 should always be looked up
                assert_eq!(v2_chunk[i], lookup_table2[values2[global_idx] as usize]);
                global_idx += 1;
            }
        }
    }

    #[test]
    fn test_dual_vocab_v2_lookup_func_exact_multiple_of_16() {
        // Test with exactly 32 values (exactly 2 chunks, no remainder)
        let lookup_table1 = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let lookup_table2 = vec![0u8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let mut lookup = SimdDualVocabU32U8LookupV2::new(&lookup_table1, &lookup_table2);

        let values1: Vec<u32> = (0..32).map(|i| (i % 5) as u32).collect();
        let values2: Vec<u32> = (0..32).map(|i| ((i % 5) + 1) as u32).collect();

        let mut vocab1_results = Vec::new();
        let mut vocab2_results = Vec::new();
        let mut num_bytes_list = Vec::new();

        lookup.lookup_func(&values1, &values2, &mut |v1, v2, num_bytes| {
            vocab1_results.push(v1);
            vocab2_results.push(v2);
            num_bytes_list.push(num_bytes);
        });

        // Should have exactly 2 chunks, no remainder
        assert_eq!(num_bytes_list.len(), 2);
        assert_eq!(num_bytes_list[0], 16);
        assert_eq!(num_bytes_list[1], 16);

        // Verify both chunks
        let mut global_idx = 0;
        for chunk_idx in 0..2 {
            let v1_chunk = vocab1_results[chunk_idx].as_array();
            let v2_chunk = vocab2_results[chunk_idx].as_array();

            for i in 0..16 {
                let expected_v1 = lookup_table1[values1[global_idx] as usize];
                assert_eq!(v1_chunk[i], expected_v1);
                if expected_v1 != 0 {
                    assert_eq!(v2_chunk[i], lookup_table2[values2[global_idx] as usize]);
                } else {
                    assert_eq!(v2_chunk[i], 0);
                }
                global_idx += 1;
            }
        }
    }
}
