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

use wide::{u8x16, u32x16};

use rustc_hash::FxHashMap;

use crate::{
    bulk_vec_extender::{BulkVecExtender, SliceU8SIMDExtender}, compress_store_u8x16, compress_store_u32x16, gather_u32index_u32, gather_u32index_u8, prefetch::{L3, prefetch_eight_offsets}
};

/// Lookup values from lookup table using u32 offsets
/// Returns the looked up u8 values as a [u8; 16] array
/// This centralizes the lookup logic and avoids code duplication
#[inline]
fn lookup_from_offsets(lookup_table: &[u8], offsets: &[u32; 16]) -> [u8; 16] {
    [
        lookup_table[offsets[0] as usize],
        lookup_table[offsets[1] as usize],
        lookup_table[offsets[2] as usize],
        lookup_table[offsets[3] as usize],
        lookup_table[offsets[4] as usize],
        lookup_table[offsets[5] as usize],
        lookup_table[offsets[6] as usize],
        lookup_table[offsets[7] as usize],
        lookup_table[offsets[8] as usize],
        lookup_table[offsets[9] as usize],
        lookup_table[offsets[10] as usize],
        lookup_table[offsets[11] as usize],
        lookup_table[offsets[12] as usize],
        lookup_table[offsets[13] as usize],
        lookup_table[offsets[14] as usize],
        lookup_table[offsets[15] as usize],
    ]
}

/// Single vocabulary lookup kernel with SIMD function - u32 to u8 lookup table kernel
/// The user is responsible for generating the lookup table - so this can be used for different use cases, including
/// CASE..WHEN and bitmasking/filtering.
///
/// It allows for SIMD operations on looked up values, but SIMD isn't actually used in the lookups themselves as
/// there aren't major advantages for SIMD in terms of lookup for huge tables with random indices.
/// However, we look up 16 values at a time for efficiency.  This kernel makes sense to call on hundreds or thousands
/// of values at a time, columnar style.
///
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
            // Try prefetching lookup table entries with L3 cache level
            // NOTE: the below compiles down to nothing in release mode, as Rust can prove the below is statically
            //       safe with no need for bounds checking.
            // let first_half: &[u32; 8] = chunk[..8].try_into().unwrap();
            // let second_half: &[u32; 8] = chunk[8..].try_into().unwrap();

            // prefetch_eight_offsets::<_, L3>(&self.lookup_table[0], first_half);
            // prefetch_eight_offsets::<_, L3>(&self.lookup_table[0], second_half);

            // Get looked up values using centralized function
            let values = lookup_from_offsets(&self.lookup_table, chunk);
            (f)(u8x16::from(values), 16);
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

    /// Convenience function which compresses and extends two Vecs:
    /// - `nonzero_results` - Vec<u8> of nonzero looked up u8 results
    /// - `indices` - Vec<u32> of indices of the nonzero results
    ///
    /// This method is intended to be used with the cascading SIMD kernels which extend lookup into two or more
    /// vocabularies by leveraging the nonzero output to do packed lookups into the second vocabulary.
    ///
    /// ## Arguments
    /// - `values` - &[u32] of indices to lookup
    /// - `nonzero_results` - &mut Vec<u8> to store the nonzero looked up u8 results
    /// - `indices` - &mut Vec<u32> to store the indices of the nonzero results
    /// - `base_index` - base index value for the indices output.
    ///
    /// For example, if you wanted to extend empty Vecs (reusing them as temporary buffers), then
    /// pass `base_index = 0` and the indices will be 0, 16, 32, etc.  Also pass empty Vecs, and clear them
    /// every time before calling.
    ///
    /// ## Performance and Architecture
    ///
    /// The lookup function is heavily optimized for Intel AVX512, using VCOMPRESS kernel (simd_compress.rs).
    /// Using VCOMPRESS this is nearly as fast as lookup_into_vec() which does nothing but copy the results!
    /// On other platforms, it falls back to a scalar approach which will be potentially much slower.
    ///
    #[inline]
    pub fn lookup_compress_into_nonzeroes(&self, values: &[u32], nonzero_results: &mut Vec<u8>, indices: &mut Vec<u32>, base_index: u32) {
        // First, set up a SIMD vector of indices starting at the base index, representing indices of elements
        let mut indices_simd = u32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        indices_simd = indices_simd + u32x16::splat(base_index);

        let sixteen = u32x16::splat(16);
        let zeroes = u8x16::splat(0);

        // Use BulkVecExtender for bulk mutable slices - enables VCOMPRESS and efficient writes
        let mut result_guard = nonzero_results.bulk_extend_guard(values.len());
        let result_slice = result_guard.as_mut_slice();
        let mut indices_guard = indices.bulk_extend_guard(values.len());
        let indices_slice = indices_guard.as_mut_slice();
        let mut num_written = 0;

        self.lookup_func(values, &mut |lookedup_values, num_bytes| {
            // Check which values are nonzero and convert to bitmask
            // simd_eq returns 0xFF where equal to zero, so invert to get nonzero mask
            let eq_mask = lookedup_values.simd_eq(zeroes).to_bitmask();
            let nonzero_mask = !eq_mask as u16;

            // Compress nonzero values into result_slice
            let written = compress_store_u8x16(lookedup_values, nonzero_mask, &mut result_slice[num_written..]);
            let _ = compress_store_u32x16(indices_simd, nonzero_mask, &mut indices_slice[num_written..]);
            num_written += written;

            // Update indices based on num_bytes
            if num_bytes < 16 {
                indices_simd = indices_simd + u32x16::splat(num_bytes as u32);
            } else {
                indices_simd = indices_simd + sixteen;
            }
        });
        result_guard.set_written(num_written);
        indices_guard.set_written(num_written);
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

        // Deep prefetch pipeline: prefetch PREFETCH_DISTANCE chunks ahead (64 values)
        // This hides DRAM latency (~200-400 cycles) by having multiple memory requests in flight
        const PREFETCH_DISTANCE: usize = 4; // 4 chunks = 64 values ahead

        // Helper to prefetch a chunk
        let prefetch_chunk = |chunk: &[u32; 16]| {
            let first_half: &[u32; 8] = chunk[..8].try_into().unwrap();
            let second_half: &[u32; 8] = chunk[8..].try_into().unwrap();
            prefetch_eight_offsets::<_, L3>(&self.lookup_table[0], first_half);
            prefetch_eight_offsets::<_, L3>(&self.lookup_table[0], second_half);
        };

        // For small number of chunks, fall back to simple processing
        if chunks.len() <= PREFETCH_DISTANCE {
            for chunk in chunks {
                let values = lookup_from_offsets(&self.lookup_table, chunk);
                (f)(u8x16::from(values), 16);
            }
            if !rest.is_empty() {
                self.process_remainder(rest, f);
            }
            return;
        }

        // Prime the prefetch pipeline: prefetch first PREFETCH_DISTANCE chunks
        for i in 0..PREFETCH_DISTANCE {
            prefetch_chunk(&chunks[i]);
        }

        // Main loop: process chunk i while prefetching chunk i+PREFETCH_DISTANCE
        for i in 0..chunks.len() {
            // Prefetch chunk i+PREFETCH_DISTANCE (if it exists)
            if i + PREFETCH_DISTANCE < chunks.len() {
                prefetch_chunk(&chunks[i + PREFETCH_DISTANCE]);
            }

            // Process chunk i (which was prefetched PREFETCH_DISTANCE iterations ago)
            let values = lookup_from_offsets(&self.lookup_table, &chunks[i]);
            (f)(u8x16::from(values), 16);
        }

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
            let values1 = lookup_from_offsets(self.lookup_table1, chunk1);

            // Conditional lookup for table2 - only where table1 is nonzero
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
                if values1[i] != 0 {
                    values2[i] = self.lookup_table2[rest2[i] as usize];
                }
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

/// Dual vocabulary lookup kernel with JOINED/colocated lookup tables.
///
/// This kernel tests the hypothesis that TLB swaps or non-colocated tables could be a performance factor.
/// Instead of two separate lookup tables, it uses a single concatenated table:
/// - `joined_table` = table1 ++ table2 (concatenated)
/// - `table2_offset` = where table2 starts in the joined table (= table1.len())
///
/// To look up table2, we use: `joined_table[table2_offset + index2]`
///
/// The lookup behavior is the same as `SimdDualVocabU32U8Lookup`:
/// - Table2 is only looked up if table1 returns a non-zero value
/// - Results are passed to the user function as (u8x16, u8x16, num_bytes)
#[derive(Debug, Clone)]
pub struct SimdJoinedDualVocabU32U8Lookup<'a> {
    joined_table: &'a [u8],
    table2_offset: u32,
}

impl<'a> SimdJoinedDualVocabU32U8Lookup<'a> {
    /// Creates a new joined dual vocabulary lookup kernel.
    /// - `joined_table`: The concatenated table (table1 ++ table2)
    /// - `table2_offset`: The offset where table2 starts (typically table1.len())
    #[inline]
    pub fn new(joined_table: &'a [u8], table2_offset: usize) -> Self {
        Self {
            joined_table,
            table2_offset: table2_offset as u32,
        }
    }

    /// Given two slices of equal length &[u32] indices, looks up each one and calls the user given function
    /// on assembled u8x16 results.
    /// - Table1 indices are used directly: joined_table[index1]
    /// - Table2 indices are offset: joined_table[table2_offset + index2]
    /// - Table2 is only looked up if table1 returns a non-zero value
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
            // Lookup table1 directly
            let values1 = lookup_from_offsets(self.joined_table, chunk1);

            // Conditional lookup for table2 with offset - only where table1 is nonzero
            let mut values2 = [0u8; 16];
            for i in 0..16 {
                if values1[i] != 0 {
                    values2[i] = self.joined_table[(self.table2_offset + chunk2[i]) as usize];
                }
            }

            (f)(u8x16::from(values1), u8x16::from(values2), 16);
        }

        // Handle the rest
        if !rest1.is_empty() {
            let mut values1 = [0u8; 16];
            let mut values2 = [0u8; 16];
            for i in 0..rest1.len() {
                values1[i] = self.joined_table[rest1[i] as usize];
                if values1[i] != 0 {
                    values2[i] = self.joined_table[(self.table2_offset + rest2[i]) as usize];
                }
            }
            (f)(u8x16::from(values1), u8x16::from(values2), rest1.len());
        }
    }

    /// Convenience function which does dual lookup, combines the results using a user-defined combiner function,
    /// and extends the combined results into a Vec (pushing all combined results)
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
            let vocab1_array = vocab1_result.to_array();

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
            let vocab1_array = vocab1_result.to_array();
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

/// Dual vocabulary lookup kernel using FxHashMap for table2.
///
/// This kernel is optimized for the case where table2 has a very small number of entries
/// (sparse), making a hash map lookup more memory-efficient than a full lookup table.
/// NOTE: We also tried writing a kernel with PHF-based EntropyMap, but it is slower than the HashMap version.
///
/// - Table1: Standard `&[u8]` lookup table (can be large)
/// - Table2: `FxHashMap<u32, u8>` (optimized for sparse data with fast hashing)
///
/// The lookup behavior is the same as `SimdDualVocabU32U8Lookup`:
/// - Table2 is only looked up if table1 returns a non-zero value
/// - Results are passed to the user function as (u8x16, u8x16, num_bytes)
pub struct SimdDualVocabWithHashLookup<'a> {
    lookup_table1: &'a [u8],
    lookup_table2: &'a FxHashMap<u32, u8>,
}

impl<'a> SimdDualVocabWithHashLookup<'a> {
    /// Creates a new dual vocabulary lookup kernel with table1 as a slice and table2 as FxHashMap.
    #[inline]
    pub fn new(lookup_table1: &'a [u8], lookup_table2: &'a FxHashMap<u32, u8>) -> Self {
        Self {
            lookup_table1,
            lookup_table2,
        }
    }

    /// Given two slices of equal length &[u32] indices, looks up each one and calls the user given function
    /// on assembled u8x16 results.
    /// - lookup_table1 (slice) is used for the first slice
    /// - lookup_table2 (FxHashMap) is used for the second slice
    /// - Table2 is only looked up if table1 returns a non-zero value
    /// - The user function is passed (lookedup_values1: u8x16, lookedup_values2: u8x16, num_bytes)
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
            let values1 = lookup_from_offsets(self.lookup_table1, chunk1);

            // Conditional hash lookup for table2 - only where table1 is nonzero
            let mut values2 = [0u8; 16];
            for i in 0..16 {
                if values1[i] != 0 {
                    values2[i] = self.lookup_table2.get(&chunk2[i]).copied().unwrap_or(0);
                }
            }

            (f)(u8x16::from(values1), u8x16::from(values2), 16);
        }

        // Handle the rest
        if !rest1.is_empty() {
            let mut values1 = [0u8; 16];
            let mut values2 = [0u8; 16];
            for i in 0..rest1.len() {
                values1[i] = self.lookup_table1[rest1[i] as usize];
                if values1[i] != 0 {
                    values2[i] = self.lookup_table2.get(&rest2[i]).copied().unwrap_or(0);
                }
            }
            (f)(u8x16::from(values1), u8x16::from(values2), rest1.len());
        }
    }

    /// Convenience function which does dual lookup, combines the results using a user-defined combiner function,
    /// and extends the combined results into a Vec (pushing all combined results)
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
    }
}

/// SIMD "Cascading" 2nd/3rd Vocab Lookup Kernel
///
/// This kernel is designed to "cascade" and build on top of the primary SingleVocab kernel to efficiently look up
/// secondary or nonprimary vocabularies.  How does this work?
/// - First call [SimdSingleVocabU32U8Lookup] to look up the primary vocabulary, using the
///   `lookup_compress_into_nonzeroes()` method.  This returns compressed results and indices of the nonzero results.
/// - Now feed these Vecs into this kernel, which uses compressed output to do a packed lookup into the second
///   vocabulary.  This is faster than having to filter all the results from the first kernel.
/// - The lookup function is called for nonzero Vocab1 results and looked up second vocab lookups, and should
///   return results for all 16 values in the u8x16.
/// - Then, this kernel will COMPRESS the results and again output nonzero results and indices, filtered from the
///   input.
///
/// Basically, this kernel can be cascaded for additional vocabularies.
///
/// The theory is that this cascading and packed lookup approach allows us to come closest to kernels where
/// even with multiple vocabularies, the runtime is roughly O(num_nonzero_lookups).
#[derive(Debug, Clone)]
pub struct SimdCascadingVocabU32U8Lookup<'a> {
    lookup_table: &'a [u8],
}

impl<'a> SimdCascadingVocabU32U8Lookup<'a> {
    #[inline]
    pub fn new(lookup_table: &'a [u8]) -> Self {
        Self { lookup_table }
    }

    /// Given a slice of u32 values, looks up each one.
    /// Designed to work in cascading mode.  One needs to pass in the nonzero_results and indices output from
    /// [SimdSingleVocabU32U8Lookup]::lookup_compress_into_nonzeroes(), along with the values (which are the word IDs
    /// for the vocabulary/lookup table in this struct).
    ///
    /// For this to be efficient, the length of values probably should be at least hundreds or thousands of values.
    ///
    /// ## Arguments
    /// - `values` - &[u32] of indices to lookup.  NOTE: these are ORIGINAL values, NOT filtered, thus
    ///   its length should be the same length as the values fed into [SimdSingleVocabU32U8Lookup] kernel.
    ///   In other words, the length of values will probably be larger than in_nonzero_results.
    /// - `in_nonzero_results` - &[u8] of nonzero results from [SimdSingleVocabU32U8Lookup]::lookup_compress_into_nonzeroes()
    /// - `in_indices` - &[u32] of indices from [SimdSingleVocabU32U8Lookup]::lookup_compress_into_nonzeroes()
    ///   These indices should be indices into the values array.
    /// - `f` - function to mix the results from nonzero_results and the looked up values from this lookup table.
    ///         The results (u8x16) returned from this function, will be zero-compressed along with indices to
    ///         generate more nonzero output.
    /// - `out_results` - &mut Vec<u8> to store the nonzero results from the lookup function f
    /// - `out_indices` - &mut Vec<u32>, basically same as input indices but with nonzeroes compressed out
    #[inline]
    pub fn cascading_lookup<F>(&self,
        values: &[u32],
        in_nonzero_results: &[u8],
        in_indices: &[u32],
        f: F,
        out_results: &mut Vec<u8>,
        out_indices: &mut Vec<u32>)
    where
        F: Fn(u8x16, u8x16) -> u8x16,
    {
        // Use BulkVecExtender for bulk mutable slices - enables VCOMPRESS and efficient writes
        let mut result_guard = out_results.bulk_extend_guard(in_nonzero_results.len());
        let result_slice = result_guard.as_mut_slice();
        let mut indices_guard = out_indices.bulk_extend_guard(in_indices.len());
        let indices_slice = indices_guard.as_mut_slice();
        let mut num_written = 0;

        let zeroes = u8x16::splat(0);

        let (in_nonzero_chunks, in_nonzero_rest) = in_nonzero_results.as_chunks::<16>();
        let (in_indices_chunks, in_indices_rest) = in_indices.as_chunks::<16>();

        for (nonzero_chunk, indices_chunk) in in_nonzero_chunks.iter().zip(in_indices_chunks.iter()) {
            let in_results = u8x16::from(*nonzero_chunk);
            let in_indices_simd = u32x16::from(*indices_chunk);

            // Gather the lookup keys from the values array
            // NOTE: This is a great use for SIMD GATHER, since the indices should be very very close together
            //       in memory, so it should benefit from caching
            // Scale is 4 bytes per u32 element
            let lookup_keys = gather_u32index_u32(in_indices_simd, values, 4);

            // Lookup the values from the lookup table.  We hope this is much faster than a branch-based lookup.
            // TODO: switch this back to scalar lookup if this turns out to be slow
            let lookedup_values = gather_u32index_u8(lookup_keys, self.lookup_table, 1);

            // Mix the results and compress the output
            let mixed_results = f(in_results, lookedup_values);

            // Check which values are nonzero and convert to bitmask
            // simd_eq returns 0xFF where equal to zero, so invert to get nonzero mask
            let eq_mask = mixed_results.simd_eq(zeroes).to_bitmask();
            let nonzero_mask = !eq_mask as u16;

            let num_nonzeroes = compress_store_u8x16(mixed_results, nonzero_mask, &mut result_slice[num_written..]);
            let _ = compress_store_u32x16(in_indices_simd, nonzero_mask, &mut indices_slice[num_written..]);
            num_written += num_nonzeroes;
        }

        // Handle the "rest" of the inputs using scalar loop
        // Build up arrays of remaining elements to pass through the mix function
        if !in_nonzero_rest.is_empty() {
            let mut in_results_arr = [0u8; 16];
            let mut lookedup_arr = [0u8; 16];
            let mut indices_arr = [0u32; 16];

            for (i, (&in_result, &in_idx)) in in_nonzero_rest.iter().zip(in_indices_rest.iter()).enumerate() {
                // Look up the key from the original values array
                let lookup_key = values[in_idx as usize];
                // Look up the value from the lookup table
                let lookedup_value = self.lookup_table[lookup_key as usize];

                in_results_arr[i] = in_result;
                lookedup_arr[i] = lookedup_value;
                indices_arr[i] = in_idx;
            }

            // Call the mix function on the padded arrays
            let mixed = f(u8x16::from(in_results_arr), u8x16::from(lookedup_arr));
            let mixed_arr = mixed.to_array();

            // Write nonzero results only for the valid elements
            for i in 0..in_nonzero_rest.len() {
                if mixed_arr[i] != 0 {
                    result_slice[num_written] = mixed_arr[i];
                    indices_slice[num_written] = indices_arr[i];
                    num_written += 1;
                }
            }
        }

        result_guard.set_written(num_written);
        indices_guard.set_written(num_written);
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

    #[test]
    fn test_dual_vocab_with_hash_lookup_basic() {
        // Create table1 as a regular lookup table
        let lookup_table1: Vec<u8> =
            (0..256).map(|i| if i % 3 == 0 { 0 } else { i as u8 }).collect();

        // Create table2 as a FxHashMap with sparse entries
        let mut hash_table2: FxHashMap<u32, u8> = FxHashMap::default();
        hash_table2.insert(0, 100);
        hash_table2.insert(5, 105);
        hash_table2.insert(10, 110);
        hash_table2.insert(15, 115);
        hash_table2.insert(20, 120);
        hash_table2.insert(100, 200);

        let lookup = SimdDualVocabWithHashLookup::new(&lookup_table1, &hash_table2);

        // Test values
        let values1: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let values2: Vec<u32> = vec![0, 5, 10, 15, 20, 100, 0, 5, 10, 15, 20, 100, 0, 5, 10, 15];

        let mut vocab1_results = Vec::new();
        let mut vocab2_results = Vec::new();

        lookup.lookup_func(&values1, &values2, &mut |v1, v2, _num_bytes| {
            vocab1_results.push(v1);
            vocab2_results.push(v2);
        });

        assert_eq!(vocab1_results.len(), 1); // One chunk

        let v1_array = vocab1_results[0].as_array();
        let v2_array = vocab2_results[0].as_array();

        // Check that table2 is only looked up where table1 is nonzero
        for i in 0..16 {
            let expected_v1 = lookup_table1[values1[i] as usize];
            assert_eq!(v1_array[i], expected_v1, "v1 mismatch at index {}", i);

            if expected_v1 != 0 {
                let expected_v2 = hash_table2.get(&values2[i]).copied().unwrap_or(0);
                assert_eq!(v2_array[i], expected_v2, "v2 mismatch at index {}", i);
            } else {
                assert_eq!(v2_array[i], 0, "v2 should be 0 where v1 is 0 at index {}", i);
            }
        }
    }

    #[test]
    fn test_dual_vocab_with_hash_lookup_into_vec() {
        // Create table1 as a regular lookup table (all nonzero)
        let lookup_table1: Vec<u8> = (0..256).map(|i| (i + 1) as u8).collect();

        // Create table2 as a FxHashMap
        let mut hash_table2: FxHashMap<u32, u8> = FxHashMap::default();
        hash_table2.insert(0, 10);
        hash_table2.insert(1, 20);
        hash_table2.insert(2, 30);
        hash_table2.insert(3, 40);
        hash_table2.insert(4, 50);

        let lookup = SimdDualVocabWithHashLookup::new(&lookup_table1, &hash_table2);

        let values1: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let values2: Vec<u32> = vec![0, 1, 2, 3, 4, 0, 1, 2];

        let mut result = Vec::new();
        lookup.lookup_into_vec(&values1, &values2, &mut result, &mut |v1, v2| v1 & v2);

        // Result should be v1 & v2
        assert_eq!(result.len(), 8);
        // v1[0] = 1, v2[0] = 10, 1 & 10 = 0
        assert_eq!(result[0], 1 & 10);
        // v1[1] = 2, v2[1] = 20, 2 & 20 = 0
        assert_eq!(result[1], 2 & 20);
    }

    #[test]
    fn test_cascading_lookup_basic() {
        // Create lookup tables
        // Table 1: returns the index value (modulo 256) for indices 0-255
        let lookup_table1: Vec<u8> = (0..256).map(|i| i as u8).collect();
        // Table 2: returns 2 * index (modulo 256) for indices 0-127
        let lookup_table2: Vec<u8> = (0..128).map(|i| ((i * 2) % 256) as u8).collect();

        // Create kernels
        let single_vocab = SimdSingleVocabU32U8Lookup::new(&lookup_table1);
        let cascading_vocab = SimdCascadingVocabU32U8Lookup::new(&lookup_table2);

        // Create test values for vocab1 (indices into lookup_table1)
        // Some will be 0, some nonzero
        let values1: Vec<u32> = vec![0, 1, 2, 3, 0, 5, 0, 7, 8, 9, 0, 11, 12, 0, 14, 15];

        // Create test values for vocab2 (indices into lookup_table2)
        // These should be looked up only where vocab1 is nonzero
        let values2: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 1, 2, 3, 4];

        // Step 1: Call SingleVocab::lookup_compress_into_nonzeroes
        let mut nonzero_results: Vec<u8> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        single_vocab.lookup_compress_into_nonzeroes(&values1, &mut nonzero_results, &mut indices, 0);

        // Verify the nonzero results from step 1
        // Values 0, 4, 6, 10, 13 are zero in lookup_table1
        // Nonzero: 1, 2, 3, 5, 7, 8, 9, 11, 12, 14, 15 (11 values)
        assert_eq!(nonzero_results.len(), 11, "Expected 11 nonzero results from vocab1");
        assert_eq!(indices.len(), 11, "Expected 11 indices");

        // Verify the indices are correct (positions of nonzero values)
        let expected_indices: Vec<u32> = vec![1, 2, 3, 5, 7, 8, 9, 11, 12, 14, 15];
        assert_eq!(indices, expected_indices);

        // Verify the nonzero results
        let expected_nonzero: Vec<u8> = vec![1, 2, 3, 5, 7, 8, 9, 11, 12, 14, 15];
        assert_eq!(nonzero_results, expected_nonzero);

        // Step 2: Call CascadingVocab::cascading_lookup
        let mut out_results: Vec<u8> = Vec::new();
        let mut out_indices: Vec<u32> = Vec::new();

        // Mix function: bitwise AND
        cascading_vocab.cascading_lookup(
            &values2,
            &nonzero_results,
            &indices,
            |v1, v2| v1 & v2,
            &mut out_results,
            &mut out_indices,
        );

        // Verify cascading results
        // For each nonzero index i:
        //   - in_result = lookup_table1[values1[i]] = values1[i] (since lookup_table1[x] = x)
        //   - lookup_key = values2[i]
        //   - lookedup_value = lookup_table2[lookup_key] = 2 * lookup_key
        //   - mixed = in_result & lookedup_value
        // Expected (for indices 1,2,3,5,7,8,9,11,12,14,15):
        //   index 1: v1=1, lookup_key=20, v2=40, mixed=1&40=0
        //   index 2: v1=2, lookup_key=30, v2=60, mixed=2&60=0
        //   index 3: v1=3, lookup_key=40, v2=80, mixed=3&80=0
        //   index 5: v1=5, lookup_key=60, v2=120, mixed=5&120=0
        //   index 7: v1=7, lookup_key=80, v2=160%256=160, mixed=7&160=0
        //   index 8: v1=8, lookup_key=90, v2=180%256=180, mixed=8&180=0
        //   index 9: v1=9, lookup_key=100, v2=200%256=200, mixed=9&200=8
        //   index 11: v1=11, lookup_key=120, v2=240%256=240, mixed=11&240=0
        //   index 12: v1=12, lookup_key=1, v2=2, mixed=12&2=0
        //   index 14: v1=14, lookup_key=3, v2=6, mixed=14&6=6
        //   index 15: v1=15, lookup_key=4, v2=8, mixed=15&8=8

        // Only indices 9, 14, 15 have nonzero mixed results
        assert_eq!(out_results.len(), 3, "Expected 3 nonzero cascading results, got {:?}", out_results);
        assert_eq!(out_indices.len(), 3, "Expected 3 output indices");

        // Check values
        assert_eq!(out_results[0], 8);  // index 9: 9 & 200 = 8
        assert_eq!(out_results[1], 6);  // index 14: 14 & 6 = 6
        assert_eq!(out_results[2], 8);  // index 15: 15 & 8 = 8

        assert_eq!(out_indices[0], 9);
        assert_eq!(out_indices[1], 14);
        assert_eq!(out_indices[2], 15);
    }

    #[test]
    fn test_cascading_lookup_remainder() {
        // Test the remainder handling (< 16 elements)
        let lookup_table1: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let lookup_table2: Vec<u8> = (0..256).map(|i| i as u8).collect();

        let single_vocab = SimdSingleVocabU32U8Lookup::new(&lookup_table1);
        let cascading_vocab = SimdCascadingVocabU32U8Lookup::new(&lookup_table2);

        // Create 5 values (less than 16, so remainder only)
        let values1: Vec<u32> = vec![1, 2, 3, 4, 5];
        let values2: Vec<u32> = vec![10, 20, 30, 40, 50];

        let mut nonzero_results: Vec<u8> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        single_vocab.lookup_compress_into_nonzeroes(&values1, &mut nonzero_results, &mut indices, 0);

        // All 5 values are nonzero
        assert_eq!(nonzero_results.len(), 5);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);

        // Cascading lookup
        let mut out_results: Vec<u8> = Vec::new();
        let mut out_indices: Vec<u32> = Vec::new();

        cascading_vocab.cascading_lookup(
            &values2,
            &nonzero_results,
            &indices,
            |v1, v2| v1 & v2,
            &mut out_results,
            &mut out_indices,
        );

        // Expected mixed results:
        //   v1=1, v2=10, mixed=1&10=0
        //   v1=2, v2=20, mixed=2&20=0
        //   v1=3, v2=30, mixed=3&30=2
        //   v1=4, v2=40, mixed=4&40=0
        //   v1=5, v2=50, mixed=5&50=0
        // Only index 2 has nonzero result
        assert_eq!(out_results.len(), 1);
        assert_eq!(out_results[0], 2);  // 3 & 30 = 2
        assert_eq!(out_indices[0], 2);
    }

    #[test]
    fn test_cascading_lookup_multiple_chunks() {
        // Test with multiple chunks (> 16 nonzero results)
        let lookup_table1: Vec<u8> = (0..256).map(|i| ((i + 1) % 256) as u8).collect(); // All nonzero except 255
        let lookup_table2: Vec<u8> = (0..256).map(|i| i as u8).collect();

        let single_vocab = SimdSingleVocabU32U8Lookup::new(&lookup_table1);
        let cascading_vocab = SimdCascadingVocabU32U8Lookup::new(&lookup_table2);

        // Create 35 values (2 full chunks of 16 + 3 remainder)
        let values1: Vec<u32> = (0..35).collect();
        let values2: Vec<u32> = (0..35).collect();

        let mut nonzero_results: Vec<u8> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        single_vocab.lookup_compress_into_nonzeroes(&values1, &mut nonzero_results, &mut indices, 0);

        // All 35 values should be nonzero (lookup_table1[i] = (i+1) % 256)
        assert_eq!(nonzero_results.len(), 35);

        // Cascading lookup with identity mix function (just return v2)
        let mut out_results: Vec<u8> = Vec::new();
        let mut out_indices: Vec<u32> = Vec::new();

        cascading_vocab.cascading_lookup(
            &values2,
            &nonzero_results,
            &indices,
            |_v1, v2| v2,
            &mut out_results,
            &mut out_indices,
        );

        // v2 = lookup_table2[values2[i]] = values2[i] = i
        // Nonzero for i > 0
        assert_eq!(out_results.len(), 34, "Expected 34 nonzero results (all except index 0)");

        // Verify all results
        for (i, &result) in out_results.iter().enumerate() {
            assert_eq!(result, (i + 1) as u8, "Result at position {} should be {}", i, i + 1);
        }
    }
}
