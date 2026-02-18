//! # simd-lookup
//!
//! High-performance SIMD utilities for fast table lookups, compression, and data processing.
//!
//! ## Features
//!
//! - **Cross-platform SIMD**: Automatic dispatch to optimal implementation (AVX-512, AVX2, NEON)
//! - **Zero-cost abstractions**: Thin wrappers over platform intrinsics via the [`wide`] crate
//! - **ARM NEON optimized**: Compress operations achieve up to 12 Gelem/s on Apple Silicon
//!
//! ## Core Modules
//!
//! - [`simd_compress`] — Stream compaction (VCOMPRESS): pack selected elements by bitmask
//! - [`simd_gather`] — Parallel memory gather with SIMD indices
//! - [`small_table`] — 64-byte lookup tables using ARM TBL4 / AVX-512 VPERMB
//! - [`wide_utils`] — Shuffle, widen, split, and bitmask utilities for `wide` types
//! - [`prefetch`] — Cross-platform memory prefetch (L1/L2/L3 hints)
//!
//!
//! ## Quick Example
//!
//! ```rust
//! use simd_lookup::{compress_store_u32x8};
//! use wide::u32x8;
//!
//! // Compress: select elements where mask bits are set
//! let data = u32x8::from([10, 20, 30, 40, 50, 60, 70, 80]);
//! let mask = 0b10110010u8; // Select positions 1, 4, 5, 7
//! let mut output = [0u32; 8];
//!
//! let count = compress_store_u32x8(data, mask, &mut output);
//! assert_eq!(count, 4);
//! assert_eq!(&output[..count], &[20, 50, 60, 80]);
//! ```
//!
//! ## Platform Support
//!
//! | Platform | Optimization Level |
//! |----------|-------------------|
//! | ARM aarch64 (Apple Silicon) | Full NEON optimization |
//! | x86-64 AVX-512 (Ice Lake+) | Native compress/gather |
//! | x86-64 AVX2 | Shuffle-based fallbacks |
//! | Other | Scalar fallbacks |

pub mod bulk_vec_extender;
pub mod eight_value_lookup;
pub mod entropy_map_lookup;
pub mod lookup_kernel;
pub mod prefetch;
pub mod simd_compress;
pub mod simd_gather;
pub mod small_table;
pub mod wide_utils;

// Re-export the main types for convenience
pub use eight_value_lookup::EightValueLookup;
pub use entropy_map_lookup::{EntropyMapBitpackedLookup, EntropyMapLookup};
pub use lookup_kernel::{
    PipelinedSingleTableU32U8Lookup, SimdDualTableWithHashLookup,
};
pub use simd_compress::{
    compress_store_u32x8, compress_store_u32x16, compress_store_u8x16,
    compress_u32x8, compress_u32x16, compress_u8x16,
};
pub use wide_utils::{
    FromBitmask, SimdSplit, WideUtilsExt,
};
pub use simd_gather::{
    gather_u32index_u8, gather_masked_u32index_u8,
    gather_u32index_u32, gather_masked_u32index_u32,
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
