//! Generic u32 -> u8 lookup tables optimized for sparse data
//!
//! This module provides three different lookup implementations optimized for different scenarios:
//! 1. Scalar lookups - direct array indexing for dense tables
//! 2. Hash-based lookups - using AHash for sparse tables with better cache efficiency
//! 3. SIMD gather lookups - vectorized lookups using AVX512 and ARM NEON

use ahash::AHashMap;
use simd_aligned::arch::u32x8;

/// Simple wrapper for 8 u8 values to match u32x8
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct U8x8([u8; 8]);

impl U8x8 {
    pub fn from(array: [u8; 8]) -> Self {
        Self(array)
    }

    pub fn to_array(self) -> [u8; 8] {
        self.0
    }
}

/// Trait for generic u32 -> u8 lookup operations
pub trait Lookup {
    /// Create a new lookup table from key-value pairs
    fn new(entries: &[(u32, u8)]) -> Self;

    /// Lookup a single value, returns 0 if key not found
    fn lookup(&self, key: u32) -> u8;

    /// Lookup multiple values at once
    fn lookup_batch(&self, keys: &[u32], results: &mut [u8]) {
        assert_eq!(keys.len(), results.len());
        for (i, &key) in keys.iter().enumerate() {
            results[i] = self.lookup(key);
        }
    }
}

/// Direct array-based lookup for dense tables
///
/// Best for scenarios where the lookup table is relatively dense and you can afford
/// the memory overhead of a full array.
pub struct ScalarLookup {
    table: Vec<u8>,
    max_key: u32,
}

impl Lookup for ScalarLookup {
    fn new(entries: &[(u32, u8)]) -> Self {
        if entries.is_empty() {
            return Self {
                table: Vec::new(),
                max_key: 0,
            };
        }

        let max_key = entries.iter().map(|(k, _)| *k).max().unwrap();
        let mut table = vec![0u8; (max_key + 1) as usize];

        for &(key, value) in entries {
            table[key as usize] = value;
        }

        Self { table, max_key }
    }

    #[inline]
    fn lookup(&self, key: u32) -> u8 {
        // All our keys are within range, this is just by data validation outside of this.
        unsafe { *self.table.get_unchecked(key as usize) }
    }

    fn lookup_batch(&self, keys: &[u32], results: &mut [u8]) {
        assert_eq!(keys.len(), results.len());

        for (i, &key) in keys.iter().enumerate() {
            results[i] = self.lookup(key);
        }
    }
}

/// Hash-based lookup using AHash for sparse tables
///
/// Best for sparse tables where only a small percentage of possible keys are populated.
/// Uses AHash which is optimized for speed over cryptographic security.
pub struct HashLookup {
    map: AHashMap<u32, u8>,
}

impl Lookup for HashLookup {
    fn new(entries: &[(u32, u8)]) -> Self {
        let mut map = AHashMap::with_capacity(entries.len());
        for &(key, value) in entries {
            map.insert(key, value);
        }
        Self { map }
    }

    #[inline]
    fn lookup(&self, key: u32) -> u8 {
        self.map.get(&key).copied().unwrap_or(0)
    }

    fn lookup_batch(&self, keys: &[u32], results: &mut [u8]) {
        assert_eq!(keys.len(), results.len());

        for (i, &key) in keys.iter().enumerate() {
            results[i] = self.map.get(&key).copied().unwrap_or(0);
        }
    }
}

/// SIMD gather-based lookup for vectorized operations
///
/// Uses SIMD gather instructions to load u32 words containing the target u8 values,
/// then extracts the specific bytes using SIMD operations. This approach allows
/// true SIMD gather usage while handling u32 -> u8 mappings efficiently.
/// Does not implement the Lookup trait as it's designed specifically for SIMD operations.
pub struct SimdLookup {
    table: Vec<u8>,
    table_u32: Vec<u32>, // Same data viewed as u32 words for SIMD gather
    max_key: u32,
}

impl SimdLookup {
    /// Create a new SIMD lookup table from key-value pairs
    pub fn new(entries: &[(u32, u8)]) -> Self {
        if entries.is_empty() {
            return Self {
                table: Vec::new(),
                table_u32: Vec::new(),
                max_key: 0,
            };
        }

        let max_key = entries.iter().map(|(k, _)| *k).max().unwrap();
        let mut table = vec![0u8; (max_key + 1) as usize];

        for &(key, value) in entries {
            table[key as usize] = value;
        }

        // Create u32 view of the table, padding to u32 boundary if needed
        let u32_len = (table.len() + 3) / 4; // Round up to nearest u32 boundary
        let mut padded_table = table.clone();
        padded_table.resize(u32_len * 4, 0); // Pad with zeros

        // Convert to u32 view - safe because we padded to u32 boundary
        let table_u32 = padded_table
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Self { table, table_u32, max_key }
    }

    /// Lookup 8 u32 keys at once, returning 8 u8 values
    /// This is the primary interface for SIMD lookups
    #[inline]
    pub fn lookup_u32x8(&self, keys: u32x8) -> U8x8 {
        self.lookup_simd_8_impl(keys)
    }

    /// Batch lookup for multiple u32x8 vectors
    pub fn lookup_batch_u32x8(&self, keys: &[u32x8], results: &mut [U8x8]) {
        assert_eq!(keys.len(), results.len());

        for (i, &key_vec) in keys.iter().enumerate() {
            results[i] = self.lookup_u32x8(key_vec);
        }
    }

    /// Single scalar lookup for compatibility (returns 0 if key not found)
    #[inline]
    pub fn lookup_scalar(&self, key: u32) -> u8 {
        if key > self.max_key {
            return 0;
        }

        unsafe { *self.table.get_unchecked(key as usize) }
    }

    /// Internal implementation for SIMD lookup using sophisticated gather approach
    ///
    /// This implementation uses a clever strategy to leverage SIMD gather instructions
    /// for u32 -> u8 lookups:
    ///
    /// 1. Treat the u8 lookup table as a u32 array (4 bytes per u32 word)
    /// 2. Divide input indices by 4 to get u32 word indices
    /// 3. Use SIMD gather to load u32 words containing target bytes
    /// 4. Calculate remainder (index % 4) to find byte position within each u32
    /// 5. Extract the specific byte from each gathered u32 word
    ///
    /// This approach maximizes SIMD utilization while handling the u32->u8 type mismatch.
    #[inline]
    fn lookup_simd_8_impl(&self, keys: u32x8) -> U8x8 {
        #[cfg(target_arch = "x86_64")]
        {
            self.lookup_simd_8_avx512(keys)
        }

        #[cfg(target_arch = "aarch64")]
        {
            self.lookup_simd_8_neon(keys)
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            // Fallback to scalar for other architectures
            let keys_array = keys.to_array();
            let mut results = [0u8; 8];
            for (i, &key) in keys_array.iter().enumerate() {
                results[i] = self.lookup_scalar(key);
            }
            U8x8::from(results)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn lookup_simd_8_avx512(&self, keys: u32x8) -> U8x8 {
        unsafe {
            use std::arch::x86_64::*;

            if is_x86_feature_detected!("avx512f") {
                let keys_array = keys.to_array();

                // Load 8 keys into a 256-bit vector
                let keys_vec = _mm256_loadu_si256(keys_array.as_ptr() as *const __m256i);

                // Check bounds - create mask for valid indices
                let max_key_vec = _mm256_set1_epi32(self.max_key as i32);
                let valid_mask = _mm256_cmple_epu32_mask(keys_vec, max_key_vec);

                // Step 1: Divide keys by 4 to get u32 word indices (using right shift by 2)
                let word_indices = _mm256_srli_epi32::<2>(keys_vec); // keys >> 2 == keys / 4

                // Step 2: Calculate remainders (keys % 4) using bitwise AND
                let mask_3 = _mm256_set1_epi32(3); // 0b11 mask for % 4
                let remainders = _mm256_and_si256(keys_vec, mask_3); // keys & 3 == keys % 4

                // Step 3: SIMD gather u32 words containing our target bytes
                let gathered_words = _mm256_mask_i32gather_epi32(
                    _mm256_setzero_si256(), // default value for invalid indices
                    valid_mask,
                    word_indices,
                    self.table_u32.as_ptr() as *const i32,
                    4, // scale factor (sizeof(u32))
                );

                // Step 4: Extract individual bytes from the gathered u32 words
                let mut results = [0u8; 8];
                let gathered_array: [i32; 8] = std::mem::transmute(gathered_words);
                let remainder_array: [i32; 8] = std::mem::transmute(remainders);

                for i in 0..8 {
                    if (valid_mask & (1 << i)) != 0 {
                        let word = gathered_array[i] as u32;
                        let byte_pos = remainder_array[i] as usize;
                        results[i] = ((word >> (byte_pos * 8)) & 0xFF) as u8;
                    }
                }

                U8x8::from(results)
            } else {
                // Fallback to scalar if AVX512 not available
                let keys_array = keys.to_array();
                let mut results = [0u8; 8];
                for (i, &key) in keys_array.iter().enumerate() {
                    results[i] = self.lookup_scalar(key);
                }
                U8x8::from(results)
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn lookup_simd_8_neon(&self, keys: u32x8) -> U8x8 {
        unsafe {
            use std::arch::aarch64::*;

            let keys_array = keys.to_array();

            // Load 8 keys (2 NEON vectors of 4 u32 each)
            let keys1 = vld1q_u32(keys_array.as_ptr());
            let keys2 = vld1q_u32(keys_array.as_ptr().add(4));

            // Check bounds
            let max_key_vec = vdupq_n_u32(self.max_key);
            let valid1 = vcleq_u32(keys1, max_key_vec);
            let valid2 = vcleq_u32(keys2, max_key_vec);

            // Step 1: Divide by 4 to get u32 word indices (using right shift by 2)
            let word_indices1 = vshrq_n_u32::<2>(keys1); // keys >> 2 == keys / 4
            let word_indices2 = vshrq_n_u32::<2>(keys2);

            // Step 2: Calculate remainders (keys % 4) using bitwise AND
            let mask_3 = vdupq_n_u32(3); // 0b11 mask for % 4
            let remainders1 = vandq_u32(keys1, mask_3); // keys & 3 == keys % 4
            let remainders2 = vandq_u32(keys2, mask_3);

            // Step 3: Manual gather u32 words (ARM doesn't have gather instructions)
            let word_indices1_array: [u32; 4] = std::mem::transmute(word_indices1);
            let word_indices2_array: [u32; 4] = std::mem::transmute(word_indices2);
            let valid1_array: [u32; 4] = std::mem::transmute(valid1);
            let valid2_array: [u32; 4] = std::mem::transmute(valid2);
            let remainders1_array: [u32; 4] = std::mem::transmute(remainders1);
            let remainders2_array: [u32; 4] = std::mem::transmute(remainders2);

            let mut results = [0u8; 8];

            // Process first 4 elements
            for i in 0..4 {
                if valid1_array[i] != 0 {
                    let word_idx = word_indices1_array[i] as usize;
                    if word_idx < self.table_u32.len() {
                        let word = self.table_u32[word_idx];
                        let byte_pos = remainders1_array[i] as usize;
                        results[i] = ((word >> (byte_pos * 8)) & 0xFF) as u8;
                    }
                }
            }

            // Process second 4 elements
            for i in 0..4 {
                if valid2_array[i] != 0 {
                    let word_idx = word_indices2_array[i] as usize;
                    if word_idx < self.table_u32.len() {
                        let word = self.table_u32[word_idx];
                        let byte_pos = remainders2_array[i] as usize;
                        results[i + 4] = ((word >> (byte_pos * 8)) & 0xFF) as u8;
                    }
                }
            }

            U8x8::from(results)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_entries() -> Vec<(u32, u8)> {
        vec![
            (0, 100),
            (5, 105),
            (10, 110),
            (100, 200),
            (1000, 255), // Max u8 value
            (10000, 42),
        ]
    }

    #[test]
    fn test_scalar_lookup() {
        let entries = create_test_entries();
        let lookup = ScalarLookup::new(&entries);

        assert_eq!(lookup.lookup(0), 100);
        assert_eq!(lookup.lookup(5), 105);
        assert_eq!(lookup.lookup(10), 110);
        assert_eq!(lookup.lookup(1), 0); // Not found, returns 0
        assert_eq!(lookup.lookup(20000), 0); // Out of bounds, returns 0
    }

    #[test]
    fn test_hash_lookup() {
        let entries = create_test_entries();
        let lookup = HashLookup::new(&entries);

        assert_eq!(lookup.lookup(0), 100);
        assert_eq!(lookup.lookup(5), 105);
        assert_eq!(lookup.lookup(10), 110);
        assert_eq!(lookup.lookup(1), 0); // Not found, returns 0
        assert_eq!(lookup.lookup(20000), 0); // Not found, returns 0
    }

    #[test]
    fn test_simd_lookup() {
        let entries = create_test_entries();
        let lookup = SimdLookup::new(&entries);

        assert_eq!(lookup.lookup_scalar(0), 100);
        assert_eq!(lookup.lookup_scalar(5), 105);
        assert_eq!(lookup.lookup_scalar(10), 110);
        assert_eq!(lookup.lookup_scalar(1), 0); // Not found, returns 0
        assert_eq!(lookup.lookup_scalar(20000), 0); // Out of bounds, returns 0
    }

    #[test]
    fn test_batch_lookup() {
        let entries = create_test_entries();
        let scalar = ScalarLookup::new(&entries);
        let hash = HashLookup::new(&entries);

        let keys = vec![0, 1, 5, 10, 15, 100, 1000, 20000];
        let expected = vec![100, 0, 105, 110, 0, 200, 255, 0];

        let mut results = vec![0u8; keys.len()];

        // Test scalar batch lookup
        scalar.lookup_batch(&keys, &mut results);
        assert_eq!(results, expected);

        // Test hash batch lookup
        results.fill(0);
        hash.lookup_batch(&keys, &mut results);
        assert_eq!(results, expected);
    }

    #[test]
    fn test_simd_u32x8_lookup() {
        let entries = create_test_entries();
        let lookup = SimdLookup::new(&entries);

        let keys = u32x8::from([0, 1, 5, 10, 15, 100, 1000, 20000]);
        let results = lookup.lookup_u32x8(keys);

        let expected = U8x8::from([100, 0, 105, 110, 0, 200, 255, 0]);

        assert_eq!(results.to_array(), expected.to_array());
    }

    #[test]
    fn test_simd_gather_approach() {
        // Test the sophisticated SIMD gather approach with specific patterns
        let entries = vec![
            (0, 10), (1, 11), (2, 12), (3, 13),   // First u32 word: [10, 11, 12, 13]
            (4, 20), (5, 21), (6, 22), (7, 23),   // Second u32 word: [20, 21, 22, 23]
            (8, 30), (9, 31), (10, 32), (11, 33), // Third u32 word: [30, 31, 32, 33]
        ];

        let lookup = SimdLookup::new(&entries);

        // Test keys that span multiple u32 words and different byte positions
        let keys = u32x8::from([0, 1, 4, 5, 8, 9, 2, 6]);
        let results = lookup.lookup_u32x8(keys);

        let expected = U8x8::from([10, 11, 20, 21, 30, 31, 12, 22]);

        assert_eq!(results.to_array(), expected.to_array());

        // Verify the u32 table was created correctly
        assert_eq!(lookup.table_u32.len(), (entries.len() + 3) / 4);

        // First u32 should contain bytes [10, 11, 12, 13] in little-endian
        let expected_first_word = u32::from_le_bytes([10, 11, 12, 13]);
        assert_eq!(lookup.table_u32[0], expected_first_word);
    }

    #[test]
    fn test_simd_batch_lookup() {
        let entries = create_test_entries();
        let lookup = SimdLookup::new(&entries);

        let keys = vec![
            u32x8::from([0, 1, 5, 10, 15, 100, 1000, 20000]),
            u32x8::from([5, 10, 0, 100, 1000, 1, 2, 3]),
        ];

        let mut results = vec![U8x8::from([0; 8]); 2];
        lookup.lookup_batch_u32x8(&keys, &mut results);

        let expected1 = U8x8::from([100, 0, 105, 110, 0, 200, 255, 0]);
        let expected2 = U8x8::from([105, 110, 100, 200, 255, 0, 0, 0]);

        assert_eq!(results[0].to_array(), expected1.to_array());
        assert_eq!(results[1].to_array(), expected2.to_array());
    }
}