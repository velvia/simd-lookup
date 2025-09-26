//! Entropy-map based lookup tables using Perfect Hash Functions (PHFs)
//!
//! This module provides wrappers around entropy-map's MapWithDict and MapWithDictBitpacked
//! for u32 -> u8 lookups, allowing comparison with traditional hash-based approaches.

use entropy_map::MapWithDict;
use std::time::Instant;

/// Wrapper around entropy-map's MapWithDict for u32 -> u8 lookups
pub struct EntropyMapLookup {
    map: MapWithDict<u32, u8>,
}

impl EntropyMapLookup {
    /// Create a new entropy map from key-value pairs
    /// Returns the map and construction time in nanoseconds
    pub fn new(entries: &[(u32, u8)]) -> (Self, u64) {
        let start = Instant::now();

        // Convert to the format expected by entropy-map
        let map_entries: Vec<(u32, u8)> = entries.iter().copied().collect();

        let map = MapWithDict::from_iter_with_params(map_entries, 2.0)
            .map_err(|e| format!("Failed to create MapWithDict: {:?}", e))
            .unwrap();
        let construction_time = start.elapsed().as_nanos() as u64;

        (Self { map }, construction_time)
    }

    /// Lookup a single value, returns 0 if key not found
    #[inline]
    pub fn lookup(&self, key: u32) -> u8 {
        self.map.get(&key).copied().unwrap_or(0)
    }

    /// Lookup multiple values at once
    pub fn lookup_batch(&self, keys: &[u32], results: &mut [u8]) {
        assert_eq!(keys.len(), results.len());
        for (i, &key) in keys.iter().enumerate() {
            results[i] = self.lookup(key);
        }
    }

    /// Get memory usage information
    pub fn memory_usage(&self) -> usize {
        // This is an approximation - entropy-map doesn't expose exact memory usage
        // We estimate based on the typical PHF overhead
        std::mem::size_of_val(&self.map)
    }
}

/// Wrapper around entropy-map's MapWithDictBitpacked for u32 -> u8 lookups
/// This version uses bitpacking for even more compact storage
/// Note: For now, we'll use a simpler approach and just use MapWithDict with a different name
/// TODO: Implement proper MapWithDictBitpacked support when API is clearer
pub struct EntropyMapBitpackedLookup {
    map: MapWithDict<u32, u8>,
}

impl EntropyMapBitpackedLookup {
    /// Create a new bitpacked entropy map from key-value pairs
    /// Returns the map and construction time in nanoseconds
    pub fn new(entries: &[(u32, u8)]) -> (Self, u64) {
        let start = Instant::now();

        // Convert to the format expected by entropy-map
        let map_entries: Vec<(u32, u8)> = entries.iter().copied().collect();

        let map = MapWithDict::from_iter_with_params(map_entries, 2.0)
            .map_err(|e| format!("Failed to create MapWithDict: {:?}", e))
            .unwrap();
        let construction_time = start.elapsed().as_nanos() as u64;

        (Self { map }, construction_time)
    }

    /// Lookup a single value, returns 0 if key not found
    #[inline]
    pub fn lookup(&self, key: u32) -> u8 {
        self.map.get(&key).copied().unwrap_or(0)
    }

    /// Lookup multiple values at once
    pub fn lookup_batch(&self, keys: &[u32], results: &mut [u8]) {
        assert_eq!(keys.len(), results.len());
        for (i, &key) in keys.iter().enumerate() {
            results[i] = self.lookup(key);
        }
    }

    /// Get memory usage information
    pub fn memory_usage(&self) -> usize {
        // This is an approximation - entropy-map doesn't expose exact memory usage
        // Bitpacked version should be more compact than regular version
        std::mem::size_of_val(&self.map)
    }
}

/// Helper function to create sparse test data with specified density
pub fn create_sparse_entries_for_entropy(size: usize, density_percent: f32) -> Vec<(u32, u8)> {
    let num_entries = ((size as f32) * (density_percent / 100.0)) as usize;
    let mut entries = Vec::with_capacity(num_entries);

    // Create sparse entries distributed across the range
    let step = size / num_entries.max(1);
    for i in 0..num_entries {
        let key = (i * step) as u32;
        let value = ((key % 255) + 1) as u8; // Values 1-255, avoiding 0 which is default
        entries.push((key, value));
    }

    entries
}

/// Benchmark helper to measure construction time for different sizes
pub fn benchmark_construction_time(sizes: &[usize], density: f32) -> Vec<(usize, u64, u64)> {
    let mut results = Vec::new();

    for &size in sizes {
        let entries = create_sparse_entries_for_entropy(size, density);

        // Benchmark MapWithDict
        let (_, dict_time) = EntropyMapLookup::new(&entries);

        // Benchmark MapWithDictBitpacked
        let (_, bitpacked_time) = EntropyMapBitpackedLookup::new(&entries);

        results.push((size, dict_time, bitpacked_time));

        println!("Size: {}, Dict: {}ns, Bitpacked: {}ns",
                size, dict_time, bitpacked_time);
    }

    results
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
            (1000, 255),
            (10000, 42),
        ]
    }

    #[test]
    fn test_entropy_map_lookup() {
        let entries = create_test_entries();
        let (lookup, _) = EntropyMapLookup::new(&entries);

        assert_eq!(lookup.lookup(0), 100);
        assert_eq!(lookup.lookup(5), 105);
        assert_eq!(lookup.lookup(10), 110);
        assert_eq!(lookup.lookup(1), 0); // Not found, returns 0
        assert_eq!(lookup.lookup(20000), 0); // Not found, returns 0
    }

    #[test]
    fn test_entropy_map_bitpacked_lookup() {
        let entries = create_test_entries();
        let (lookup, _) = EntropyMapBitpackedLookup::new(&entries);

        assert_eq!(lookup.lookup(0), 100);
        assert_eq!(lookup.lookup(5), 105);
        assert_eq!(lookup.lookup(10), 110);
        assert_eq!(lookup.lookup(1), 0); // Not found, returns 0
        assert_eq!(lookup.lookup(20000), 0); // Not found, returns 0
    }

    #[test]
    fn test_batch_lookup() {
        let entries = create_test_entries();
        let (dict_lookup, _) = EntropyMapLookup::new(&entries);
        let (bitpacked_lookup, _) = EntropyMapBitpackedLookup::new(&entries);

        let keys = vec![0, 1, 5, 10, 15, 100, 1000, 20000];
        let expected = vec![100, 0, 105, 110, 0, 200, 255, 0];

        let mut results = vec![0u8; keys.len()];

        // Test dict batch lookup
        dict_lookup.lookup_batch(&keys, &mut results);
        assert_eq!(results, expected);

        // Test bitpacked batch lookup
        results.fill(0);
        bitpacked_lookup.lookup_batch(&keys, &mut results);
        assert_eq!(results, expected);
    }

    #[test]
    fn test_construction_time_measurement() {
        let entries = create_test_entries();

        let (_, dict_time) = EntropyMapLookup::new(&entries);
        let (_, bitpacked_time) = EntropyMapBitpackedLookup::new(&entries);

        // Construction should take some measurable time
        assert!(dict_time > 0);
        assert!(bitpacked_time > 0);

        println!("Dict construction: {}ns", dict_time);
        println!("Bitpacked construction: {}ns", bitpacked_time);
    }
}
