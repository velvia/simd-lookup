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
