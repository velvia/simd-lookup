pub mod table64;
pub mod lookup;
pub mod eight_value_lookup;
pub mod entropy_map_lookup;
pub mod lookup_kernel;
pub mod bitpacked_lookup;
pub mod bulk_vec_extender;

// Re-export the main types for convenience
pub use lookup::{SimdLookup, ScalarLookup, HashLookup, Lookup, U8x8};
pub use eight_value_lookup::EightValueLookup;
pub use entropy_map_lookup::{EntropyMapLookup, EntropyMapBitpackedLookup};

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
