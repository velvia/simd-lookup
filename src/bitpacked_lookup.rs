//! Bit-packed lookup tables for memory-efficient vocabulary lookups
//!
//! Uses 64-bit word packing to avoid cross-boundary issues while dramatically
//! reducing memory footprint for small value spaces (2-3 bits).
//!
//! Key insight: For large dual-vocabulary lookups, the memory savings (4x for 2-bit)
//! can move tables from RAM-bound to L3-bound, providing massive speedup despite
//! the bit extraction overhead.
//!
//! TODO: investigate using both SIMD and also memory pre-fetch instructions to avoid cache thrashing
//! TODO: Change the API so we don't have to decide on number of bits at compile time

use std::marker::PhantomData;

/// Trait for bit-packed value encoding strategies
pub trait BitPackStrategy: Copy + Clone {
    /// Number of bits per value
    const BITS_PER_VALUE: u32;

    /// Values per u64 word
    const VALUES_PER_WORD: u32;

    /// Bit mask for extracting a value
    const VALUE_MASK: u64;

    /// Extract a value from a u64 word at the given value index (0..VALUES_PER_WORD-1)
    #[inline]
    fn extract(word: u64, value_index: u32) -> u8 {
        let bit_offset = value_index * Self::BITS_PER_VALUE;
        ((word >> bit_offset) & Self::VALUE_MASK) as u8
    }

    /// Pack a value into a u64 word at the given value index
    #[inline]
    fn pack(word: u64, value: u8, value_index: u32) -> u64 {
        let bit_offset = value_index * Self::BITS_PER_VALUE;
        let mask = Self::VALUE_MASK << bit_offset;
        let cleared = word & !mask;
        let packed = ((value as u64) & Self::VALUE_MASK) << bit_offset;
        cleared | packed
    }
}

/// 2-bit packing: 32 values per u64, supports values 0-3
#[derive(Debug, Copy, Clone)]
pub struct TwoBit;

impl BitPackStrategy for TwoBit {
    const BITS_PER_VALUE: u32 = 2;
    const VALUES_PER_WORD: u32 = 32;
    const VALUE_MASK: u64 = 0b11;
}

/// 3-bit packing: 21 values per u64 (1 bit wasted), supports values 0-7
#[derive(Debug, Copy, Clone)]
pub struct ThreeBit;

impl BitPackStrategy for ThreeBit {
    const BITS_PER_VALUE: u32 = 3;
    const VALUES_PER_WORD: u32 = 21; // 21 * 3 = 63 bits (1 bit wasted per word)
    const VALUE_MASK: u64 = 0b111;
}

/// 4-bit packing: 16 values per u64, supports values 0-15
#[derive(Debug, Copy, Clone)]
pub struct FourBit;

impl BitPackStrategy for FourBit {
    const BITS_PER_VALUE: u32 = 4;
    const VALUES_PER_WORD: u32 = 16;
    const VALUE_MASK: u64 = 0b1111;
}

/// Bit-packed lookup table that stores values in u64 words
#[derive(Debug, Clone)]
pub struct BitPackedLookup<S: BitPackStrategy> {
    /// Packed data: each u64 contains S::VALUES_PER_WORD values
    packed_data: Vec<u64>,
    /// Maximum valid key
    max_key: u32,
    /// Phantom data for strategy
    _strategy: PhantomData<S>,
}

impl<S: BitPackStrategy> BitPackedLookup<S> {
    /// Create a new bit-packed lookup table from a slice of u8 values
    /// Values are truncated to fit the bit width (e.g., 2 bits = 0-3)
    #[inline]
    pub fn from_u8_table(table: &[u8]) -> Self {
        let num_words = table.len().div_ceil(S::VALUES_PER_WORD as usize);
        let mut packed_data = vec![0u64; num_words];

        for (key, &value) in table.iter().enumerate() {
            let word_index = key / (S::VALUES_PER_WORD as usize);
            let value_index = (key % (S::VALUES_PER_WORD as usize)) as u32;
            packed_data[word_index] = S::pack(packed_data[word_index], value, value_index);
        }

        Self {
            packed_data,
            max_key: table.len().saturating_sub(1) as u32,
            _strategy: PhantomData,
        }
    }

    /// Create from sparse entries (key, value pairs)
    #[inline]
    pub fn from_entries(entries: &[(u32, u8)]) -> Self {
        if entries.is_empty() {
            return Self {
                packed_data: vec![0u64; 1],
                max_key: 0,
                _strategy: PhantomData,
            };
        }

        let max_key = entries.iter().map(|(k, _)| *k).max().unwrap();
        let table_size = (max_key as usize) + 1;
        let num_words = table_size.div_ceil(S::VALUES_PER_WORD as usize);
        let mut packed_data = vec![0u64; num_words];

        for &(key, value) in entries {
            let word_index = (key / S::VALUES_PER_WORD) as usize;
            let value_index = key % S::VALUES_PER_WORD;
            packed_data[word_index] = S::pack(packed_data[word_index], value, value_index);
        }

        Self {
            packed_data,
            max_key,
            _strategy: PhantomData,
        }
    }

    /// Lookup a single value
    #[inline]
    pub fn lookup(&self, key: u32) -> u8 {
        if key > self.max_key {
            return 0;
        }

        let word_index = (key / S::VALUES_PER_WORD) as usize;
        let value_index = key % S::VALUES_PER_WORD;
        let word = unsafe { *self.packed_data.get_unchecked(word_index) };
        S::extract(word, value_index)
    }

    /// Batch lookup into a pre-allocated result slice
    #[inline]
    pub fn lookup_batch(&self, keys: &[u32], results: &mut [u8]) {
        assert_eq!(
            keys.len(),
            results.len(),
            "Keys and results must have same length"
        );

        for (i, &key) in keys.iter().enumerate() {
            results[i] = self.lookup(key);
        }
    }

    /// Get memory usage in bytes
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.packed_data.len() * 8
    }

    /// Get the maximum valid key
    #[inline]
    pub fn max_key(&self) -> u32 {
        self.max_key
    }
}

/// Single vocabulary bit-packed lookup kernel
#[derive(Debug, Clone)]
pub struct BitPackedSingleVocab<S: BitPackStrategy> {
    lookup: BitPackedLookup<S>,
}

impl<S: BitPackStrategy> BitPackedSingleVocab<S> {
    #[inline]
    pub fn new(lookup: BitPackedLookup<S>) -> Self {
        Self { lookup }
    }

    #[inline]
    pub fn from_u8_table(table: &[u8]) -> Self {
        Self::new(BitPackedLookup::from_u8_table(table))
    }

    #[inline]
    pub fn from_entries(entries: &[(u32, u8)]) -> Self {
        Self::new(BitPackedLookup::from_entries(entries))
    }

    /// Batch lookup
    #[inline]
    pub fn lookup_batch(&self, keys: &[u32], results: &mut [u8]) {
        self.lookup.lookup_batch(keys, results);
    }

    /// Memory usage in bytes
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.lookup.memory_bytes()
    }
}

/// Dual vocabulary bit-packed lookup kernel
/// This is where bit-packing really shines: two 15MB tables → two 3.75MB tables
#[derive(Debug, Clone)]
pub struct BitPackedDualVocab<S: BitPackStrategy> {
    lookup1: BitPackedLookup<S>,
    lookup2: BitPackedLookup<S>,
}

impl<S: BitPackStrategy> BitPackedDualVocab<S> {
    #[inline]
    pub fn new(lookup1: BitPackedLookup<S>, lookup2: BitPackedLookup<S>) -> Self {
        Self { lookup1, lookup2 }
    }

    #[inline]
    pub fn from_u8_tables(table1: &[u8], table2: &[u8]) -> Self {
        Self::new(
            BitPackedLookup::from_u8_table(table1),
            BitPackedLookup::from_u8_table(table2),
        )
    }

    /// Dual lookup with conditional second lookup (like V2)
    /// Only looks up table2 if table1 result is non-zero
    #[inline]
    pub fn lookup_batch_conditional(&self, keys1: &[u32], keys2: &[u32], results: &mut [(u8, u8)]) {
        assert_eq!(keys1.len(), keys2.len(), "Key slices must have same length");
        assert_eq!(
            keys1.len(),
            results.len(),
            "Keys and results must have same length"
        );

        for i in 0..keys1.len() {
            let val1 = self.lookup1.lookup(keys1[i]);
            let val2 = if val1 != 0 {
                self.lookup2.lookup(keys2[i])
            } else {
                0
            };
            results[i] = (val1, val2);
        }
    }

    /// Dual lookup with UNCONDITIONAL second lookup
    /// Always looks up both tables - no branches!
    /// User reports this is similar speed to conditional despite doing more work,
    /// because it eliminates branch misprediction overhead.
    #[inline]
    pub fn lookup_batch_unconditional(
        &self,
        keys1: &[u32],
        keys2: &[u32],
        results: &mut [(u8, u8)],
    ) {
        assert_eq!(keys1.len(), keys2.len(), "Key slices must have same length");
        assert_eq!(
            keys1.len(),
            results.len(),
            "Keys and results must have same length"
        );

        // Clean branch-free loop - compiler can auto-vectorize!
        for i in 0..keys1.len() {
            results[i] = (self.lookup1.lookup(keys1[i]), self.lookup2.lookup(keys2[i]));
        }
    }

    /// Dual lookup with combining function
    #[inline]
    pub fn lookup_batch_combined<F>(
        &self,
        keys1: &[u32],
        keys2: &[u32],
        results: &mut [u8],
        combine: F,
    ) where
        F: Fn(u8, u8) -> u8,
    {
        assert_eq!(keys1.len(), keys2.len(), "Key slices must have same length");
        assert_eq!(
            keys1.len(),
            results.len(),
            "Keys and results must have same length"
        );

        for i in 0..keys1.len() {
            let val1 = self.lookup1.lookup(keys1[i]);
            let val2 = self.lookup2.lookup(keys2[i]);
            results[i] = combine(val1, val2);
        }
    }

    /// Total memory usage in bytes
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.lookup1.memory_bytes() + self.lookup2.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_bit_pack_extract() {
        // Test packing all 4 possible 2-bit values
        let mut word = 0u64;
        word = TwoBit::pack(word, 0, 0); // 0b00 at position 0
        word = TwoBit::pack(word, 1, 1); // 0b01 at position 1
        word = TwoBit::pack(word, 2, 2); // 0b10 at position 2
        word = TwoBit::pack(word, 3, 3); // 0b11 at position 3

        assert_eq!(TwoBit::extract(word, 0), 0);
        assert_eq!(TwoBit::extract(word, 1), 1);
        assert_eq!(TwoBit::extract(word, 2), 2);
        assert_eq!(TwoBit::extract(word, 3), 3);
    }

    #[test]
    fn test_three_bit_pack_extract() {
        let mut word = 0u64;
        word = ThreeBit::pack(word, 0, 0); // 0b000
        word = ThreeBit::pack(word, 7, 1); // 0b111
        word = ThreeBit::pack(word, 5, 2); // 0b101

        assert_eq!(ThreeBit::extract(word, 0), 0);
        assert_eq!(ThreeBit::extract(word, 1), 7);
        assert_eq!(ThreeBit::extract(word, 2), 5);
    }

    #[test]
    fn test_bitpacked_lookup_basic() {
        let table = vec![0u8, 1, 2, 3, 0, 1, 2, 3];
        let lookup = BitPackedLookup::<TwoBit>::from_u8_table(&table);

        assert_eq!(lookup.lookup(0), 0);
        assert_eq!(lookup.lookup(1), 1);
        assert_eq!(lookup.lookup(2), 2);
        assert_eq!(lookup.lookup(3), 3);
        assert_eq!(lookup.lookup(4), 0);
        assert_eq!(lookup.lookup(5), 1);
    }

    #[test]
    fn test_bitpacked_lookup_batch() {
        let table = vec![0u8, 1, 2, 3, 0, 1, 2, 3];
        let lookup = BitPackedLookup::<TwoBit>::from_u8_table(&table);

        let keys = vec![0, 1, 2, 3, 4, 5];
        let mut results = vec![0u8; keys.len()];
        lookup.lookup_batch(&keys, &mut results);

        assert_eq!(results, vec![0, 1, 2, 3, 0, 1]);
    }

    #[test]
    fn test_memory_savings() {
        let size = 15_000_000;
        let table = vec![1u8; size];

        // Original u8 table
        let original_bytes = size;

        // 2-bit packed
        let lookup_2bit = BitPackedLookup::<TwoBit>::from_u8_table(&table);
        let packed_2bit_bytes = lookup_2bit.memory_bytes();

        // Should be ~4x smaller
        assert!(packed_2bit_bytes < original_bytes / 3);
        assert!(packed_2bit_bytes > original_bytes / 5);

        println!("Original: {} MB", original_bytes / 1_000_000);
        println!("2-bit packed: {} MB", packed_2bit_bytes / 1_000_000);
        println!(
            "Compression ratio: {:.2}x",
            original_bytes as f64 / packed_2bit_bytes as f64
        );
    }

    #[test]
    fn test_dual_vocab_conditional() {
        let table1 = vec![0u8, 1, 2, 0, 3, 0];
        let table2 = vec![0u8, 10, 20, 30, 40, 50];

        let dual = BitPackedDualVocab::<TwoBit>::from_u8_tables(&table1, &table2);

        let keys1 = vec![0, 1, 2, 3, 4];
        let keys2 = vec![0, 1, 2, 3, 4];
        let mut results = vec![(0u8, 0u8); keys1.len()];

        dual.lookup_batch_conditional(&keys1, &keys2, &mut results);

        // table1[0] = 0, so table2 not looked up
        assert_eq!(results[0], (0, 0));

        // table1[1] = 1 (non-zero), so table2[1] = 10, but truncated to 2 bits = 2
        assert_eq!(results[1].0, 1);

        // table1[2] = 2 (non-zero), so table2[2] = 20, but truncated to 2 bits = 0
        assert_eq!(results[2].0, 2);

        // table1[3] = 0, so table2 not looked up
        assert_eq!(results[3], (0, 0));

        // table1[4] = 3 (non-zero), so table2[4] is looked up
        assert_eq!(results[4].0, 3);
    }
}
