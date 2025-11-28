pub mod bitpacked_lookup;
pub mod bulk_vec_extender;
pub mod eight_value_lookup;
pub mod entropy_map_lookup;
pub mod lookup;
pub mod lookup_kernel;
pub mod prefetch;
pub mod simd_compress;
pub mod table64;
pub mod wide_utils;

// Re-export the main types for convenience
pub use eight_value_lookup::EightValueLookup;
pub use entropy_map_lookup::{EntropyMapBitpackedLookup, EntropyMapLookup};
pub use lookup::{HashLookup, Lookup, ScalarLookup, SimdLookup, U8x8};
pub use lookup_kernel::{
    PipelinedSingleVocabU32U8Lookup, SimdDualVocabWithHashLookup, SimdJoinedDualVocabU32U8Lookup,
};
pub use simd_compress::{
    compress_store_u32x8, compress_store_u32x16, compress_store_u8x16,
    compress_u32x8, compress_u32x16, compress_u8x16,
};
pub use wide_utils::{
    FromBitmask, SimdSplit, WideUtilsExt,
};

#[cfg(test)]
mod tests {
    #[test]
    fn test_cpu_features() {
        // Check if AVX512 features are enabled at compile time
        #[cfg(target_feature = "avx512f")]
        {
            println!("✓ AVX-512 Foundation (AVX512F): ENABLED");
        }
        #[cfg(not(target_feature = "avx512f"))]
        {
            println!("✗ AVX-512 Foundation (AVX512F): DISABLED");
        }

        #[cfg(target_feature = "avx512bw")]
        {
            println!("✓ AVX-512 Byte and Word (AVX512BW): ENABLED");
        }
        #[cfg(not(target_feature = "avx512bw"))]
        {
            println!("✗ AVX-512 Byte and Word (AVX512BW): DISABLED");
        }

        #[cfg(target_feature = "avx512vl")]
        {
            println!("✓ AVX-512 Vector Length (AVX512VL): ENABLED");
        }
        #[cfg(not(target_feature = "avx512vl"))]
        {
            println!("✗ AVX-512 Vector Length (AVX512VL): DISABLED");
        }

        #[cfg(target_feature = "avx2")]
        {
            println!("✓ AVX2: ENABLED");
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            println!("✗ AVX2: DISABLED");
        }

        #[cfg(target_feature = "avx")]
        {
            println!("✓ AVX: ENABLED");
        }
        #[cfg(not(target_feature = "avx"))]
        {
            println!("✗ AVX: DISABLED");
        }
    }
}
