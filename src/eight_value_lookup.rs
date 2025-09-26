//! SIMD-accelerated lookup for finding positions in small u32 tables
//!
//! This module provides efficient position lookup for checking if a u32 value
//! exists in a table of up to 8 u32 values, returning the position (0-7) if found
//! or -1 if not found.

use simd_aligned::arch::u32x8;

/// SIMD-accelerated position lookup for finding a u32 value in a table of up to 8 values
///
/// This is optimized for the common pattern of finding the index/position of a value
/// in a small lookup table, which is more useful than simple membership testing.
pub struct EightValueLookup {
    table: u32x8,
    count: usize, // Number of actual values (≤ 8)
}

impl EightValueLookup {
    /// Create a new position lookup table from a slice of u32 values
    ///
    /// # Panics
    /// Panics if more than 8 values are provided
    pub fn new(values: &[u32]) -> Self {
        assert!(values.len() <= 8, "EightValueLookup supports at most 8 values");

        let mut array = [0u32; 8];
        for (i, &val) in values.iter().enumerate() {
            array[i] = val;
        }

        Self {
            table: u32x8::from(array),
            count: values.len(),
        }
    }

    /// Find the position of a u32 value in the lookup table
    /// Returns the position (0-7) if found, or -1 if not found
    #[inline]
    pub fn find_position(&self, value: u32) -> i32 {
        self.find_position_simd_impl(value)
    }

    /// Find positions for multiple values at once using SIMD
    /// Returns an array of positions where each element is the position (0-7) or -1
    #[inline]
    pub fn find_positions_batch(&self, values: u32x8) -> [i32; 8] {
        self.find_positions_batch_simd_impl(values)
    }

    /// Get the number of values in the lookup table
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the lookup table is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the underlying table as an array (includes padding zeros)
    pub fn as_array(&self) -> [u32; 8] {
        self.table.to_array()
    }

    /// Internal SIMD implementation for single value position lookup
    #[inline]
    fn find_position_simd_impl(&self, value: u32) -> i32 {
        if self.count == 0 {
            return -1;
        }

        #[cfg(target_arch = "x86_64")]
        {
            self.find_position_simd_avx2(value)
        }

        #[cfg(target_arch = "aarch64")]
        {
            self.find_position_simd_neon(value)
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback to scalar
            let table_array = self.table.to_array();
            for i in 0..self.count {
                if table_array[i] == value {
                    return i as i32;
                }
            }
            -1
        }
    }

    /// Internal SIMD implementation for batch position lookup
    #[inline]
    fn find_positions_batch_simd_impl(&self, values: u32x8) -> [i32; 8] {
        if self.count == 0 {
            return [-1; 8];
        }

        #[cfg(target_arch = "x86_64")]
        {
            self.find_positions_batch_simd_avx2(values)
        }

        #[cfg(target_arch = "aarch64")]
        {
            self.find_positions_batch_simd_neon(values)
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback to scalar
            let values_array = values.to_array();
            let table_array = self.table.to_array();
            let mut result = [-1i32; 8];

            for i in 0..8 {
                for j in 0..self.count {
                    if values_array[i] == table_array[j] {
                        result[i] = j as i32;
                        break;
                    }
                }
            }

            result
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn find_position_simd_avx2(&self, value: u32) -> i32 {
        unsafe {
            use std::arch::x86_64::*;

            if is_x86_feature_detected!("avx2") {
                // Broadcast the input value to all lanes
                let input_vec = _mm256_set1_epi32(value as i32);

                // Load our table values
                let table_values = self.table.to_array();
                let table_vec = _mm256_loadu_si256(table_values.as_ptr() as *const __m256i);

                // Compare all lanes at once
                let cmp_result = _mm256_cmpeq_epi32(input_vec, table_vec);

                // Extract the comparison mask
                let mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp_result));

                // Create a mask for only the valid lanes (based on self.count)
                let valid_mask = (1u32 << self.count) - 1;
                let masked_result = (mask as u32) & valid_mask;

                if masked_result == 0 {
                    -1
                } else {
                    // Find the position of the first set bit (trailing zeros)
                    masked_result.trailing_zeros() as i32
                }
            } else {
                // Fallback to scalar if AVX2 not available
                let table_array = self.table.to_array();
                for i in 0..self.count {
                    if table_array[i] == value {
                        return i as i32;
                    }
                }
                -1
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn find_positions_batch_simd_avx2(&self, values: u32x8) -> [i32; 8] {
        unsafe {
            use std::arch::x86_64::*;

            if is_x86_feature_detected!("avx2") {
                let values_array = values.to_array();
                let input_vec = _mm256_loadu_si256(values_array.as_ptr() as *const __m256i);

                let table_values = self.table.to_array();
                let table_vec = _mm256_loadu_si256(table_values.as_ptr() as *const __m256i);

                let mut result = [-1i32; 8];

                // For each table position, check which input values match
                for table_pos in 0..self.count {
                    // Broadcast current table value to all lanes
                    let table_broadcast = _mm256_set1_epi32(table_values[table_pos] as i32);

                    // Compare with all input values
                    let cmp_result = _mm256_cmpeq_epi32(input_vec, table_broadcast);

                    // Extract mask and update results
                    let mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp_result));

                    for i in 0..8 {
                        if (mask & (1 << i)) != 0 && result[i] == -1 {
                            // First match for this input value
                            result[i] = table_pos as i32;
                        }
                    }
                }

                result
            } else {
                // Fallback to scalar
                let values_array = values.to_array();
                let table_array = self.table.to_array();
                let mut result = [-1i32; 8];

                for i in 0..8 {
                    for j in 0..self.count {
                        if values_array[i] == table_array[j] {
                            result[i] = j as i32;
                            break;
                        }
                    }
                }

                result
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn find_position_simd_neon(&self, value: u32) -> i32 {
        unsafe {
            use std::arch::aarch64::*;

            // Load our table values (2 NEON vectors of 4 u32 each)
            let table_values = self.table.to_array();
            let table_vec1 = vld1q_u32(table_values.as_ptr());
            let table_vec2 = vld1q_u32(table_values.as_ptr().add(4));

            // Broadcast the input value
            let input_vec = vdupq_n_u32(value);

            // Compare with both halves
            let cmp1 = vceqq_u32(input_vec, table_vec1);
            let cmp2 = vceqq_u32(input_vec, table_vec2);

            // Convert to arrays to find position
            let cmp1_array: [u32; 4] = std::mem::transmute(cmp1);
            let cmp2_array: [u32; 4] = std::mem::transmute(cmp2);

            // Check first half
            for i in 0..4.min(self.count) {
                if cmp1_array[i] != 0 {
                    return i as i32;
                }
            }

            // Check second half if needed
            if self.count > 4 {
                for i in 0..(self.count - 4) {
                    if cmp2_array[i] != 0 {
                        return (i + 4) as i32;
                    }
                }
            }

            -1
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn find_positions_batch_simd_neon(&self, values: u32x8) -> [i32; 8] {
        unsafe {
            use std::arch::aarch64::*;

            let values_array = values.to_array();
            let input_vec1 = vld1q_u32(values_array.as_ptr());
            let input_vec2 = vld1q_u32(values_array.as_ptr().add(4));

            let table_values = self.table.to_array();

            let mut result = [-1i32; 8];

            // Check each table position against all input values
            for table_pos in 0..self.count {
                let table_broadcast = vdupq_n_u32(table_values[table_pos]);

                let cmp1 = vceqq_u32(input_vec1, table_broadcast);
                let cmp2 = vceqq_u32(input_vec2, table_broadcast);

                let cmp1_array: [u32; 4] = std::mem::transmute(cmp1);
                let cmp2_array: [u32; 4] = std::mem::transmute(cmp2);

                // Update results for first half
                for i in 0..4 {
                    if cmp1_array[i] != 0 && result[i] == -1 {
                        result[i] = table_pos as i32;
                    }
                }

                // Update results for second half
                for i in 0..4 {
                    if cmp2_array[i] != 0 && result[i + 4] == -1 {
                        result[i + 4] = table_pos as i32;
                    }
                }
            }

            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_position_lookup() {
        let lookup = EightValueLookup::new(&[10, 20, 30, 40]);

        assert_eq!(lookup.find_position(10), 0);
        assert_eq!(lookup.find_position(20), 1);
        assert_eq!(lookup.find_position(30), 2);
        assert_eq!(lookup.find_position(40), 3);
        assert_eq!(lookup.find_position(5), -1);
        assert_eq!(lookup.find_position(50), -1);
    }

    #[test]
    fn test_full_table() {
        let lookup = EightValueLookup::new(&[1, 2, 3, 4, 5, 6, 7, 8]);

        for i in 1..=8 {
            assert_eq!(lookup.find_position(i), (i - 1) as i32);
        }

        assert_eq!(lookup.find_position(0), -1);
        assert_eq!(lookup.find_position(9), -1);
    }

    #[test]
    fn test_empty_table() {
        let lookup = EightValueLookup::new(&[]);

        assert_eq!(lookup.find_position(0), -1);
        assert_eq!(lookup.find_position(1), -1);
        assert!(lookup.is_empty());
        assert_eq!(lookup.len(), 0);
    }

    #[test]
    fn test_single_value() {
        let lookup = EightValueLookup::new(&[42]);

        assert_eq!(lookup.find_position(42), 0);
        assert_eq!(lookup.find_position(41), -1);
        assert_eq!(lookup.find_position(43), -1);
        assert_eq!(lookup.len(), 1);
    }

    #[test]
    fn test_batch_position_lookup() {
        let lookup = EightValueLookup::new(&[10, 20, 30, 40, 50]);

        let test_values = u32x8::from([10, 15, 20, 25, 30, 35, 40, 45]);
        let results = lookup.find_positions_batch(test_values);

        let expected = [0, -1, 1, -1, 2, -1, 3, -1];
        assert_eq!(results, expected);
    }

    #[test]
    fn test_duplicates_return_first_position() {
        let lookup = EightValueLookup::new(&[10, 20, 10, 30, 20]);

        // Should return the first occurrence
        assert_eq!(lookup.find_position(10), 0);
        assert_eq!(lookup.find_position(20), 1);
        assert_eq!(lookup.find_position(30), 3);
    }

    #[test]
    fn test_large_values() {
        let lookup = EightValueLookup::new(&[
            u32::MAX - 7,
            u32::MAX - 5,
            u32::MAX - 3,
            u32::MAX - 1,
            u32::MAX,
        ]);

        assert_eq!(lookup.find_position(u32::MAX), 4);
        assert_eq!(lookup.find_position(u32::MAX - 1), 3);
        assert_eq!(lookup.find_position(u32::MAX - 3), 2);
        assert_eq!(lookup.find_position(u32::MAX - 5), 1);
        assert_eq!(lookup.find_position(u32::MAX - 7), 0);

        assert_eq!(lookup.find_position(u32::MAX - 2), -1);
        assert_eq!(lookup.find_position(u32::MAX - 4), -1);
    }

    #[test]
    fn test_batch_vs_single_consistency() {
        let lookup = EightValueLookup::new(&[5, 15, 25, 35, 45, 55, 65, 75]);

        let test_values = u32x8::from([5, 10, 15, 20, 25, 30, 35, 40]);
        let batch_results = lookup.find_positions_batch(test_values);

        let test_array = test_values.to_array();
        for (i, &test_val) in test_array.iter().enumerate() {
            let single_result = lookup.find_position(test_val);
            assert_eq!(batch_results[i], single_result,
                "Mismatch for value {} at index {}: batch={}, single={}",
                test_val, i, batch_results[i], single_result);
        }
    }

    #[test]
    #[should_panic(expected = "EightValueLookup supports at most 8 values")]
    fn test_too_many_values() {
        EightValueLookup::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_as_array() {
        let lookup = EightValueLookup::new(&[10, 20, 30]);
        let array = lookup.as_array();

        assert_eq!(array[0], 10);
        assert_eq!(array[1], 20);
        assert_eq!(array[2], 30);
        // Remaining elements should be 0 (padding)
        for i in 3..8 {
            assert_eq!(array[i], 0);
        }
    }
}
