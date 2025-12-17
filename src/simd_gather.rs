//! SIMD gather operations for efficient indexed memory access
//!
//! This module provides vectorized gather functions that load multiple values from
//! memory using SIMD indices. On AVX-512, these use hardware gather instructions
//! (`_mm512_i32gather_epi32`, `_mm512_mask_i32gather_epi32`); on other platforms,
//! they fall back to scalar loops.
//!
//! # CPU Feature Requirements (Intel x86_64)
//!
//! ## Optimal Performance (AVX-512)
//!
//! - **`gather_u32index_u8` / `gather_masked_u32index_u8`**: Requires **AVX512F** + **AVX512BW**
//!   - Uses `VGATHERDPS` (`_mm512_i32gather_epi32`) + `VPMOVDB` (`_mm512_cvtepi32_epi8`)
//!   - Available on: Intel Skylake-X (Xeon), Ice Lake, Tiger Lake, and later
//!   - Fallback: Scalar loop (works on all architectures)
//!
//! - **`gather_u32index_u32` / `gather_masked_u32index_u32`**: Requires **AVX512F**
//!   - Uses `VGATHERDPS` (`_mm512_i32gather_epi32`)
//!   - Available on: Intel Skylake-X (Xeon), Ice Lake, Tiger Lake, and later
//!   - Fallback: Scalar loop (works on all architectures)
//!
//! ## Fallback Behavior
//!
//! All functions automatically fall back to scalar implementations when AVX-512
//! features are not available. The fallback implementations work on:
//! - x86_64 without AVX-512 (uses AVX2 gather if available, or scalar)
//! - aarch64 (ARM NEON) - scalar fallback
//! - All other architectures (scalar fallback)
//!
//! # Functions
//!
//! - [`gather_u32index_u8`] - Gather 16 bytes using u32 indices
//! - [`gather_masked_u32index_u8`] - Masked gather of bytes with fallback values
//! - [`gather_u32index_u32`] - Gather 16 u32 values using u32 indices
//! - [`gather_masked_u32index_u32`] - Masked gather of u32 values with fallback
//!
//! # Important: Masked Gather Behavior on Intel
//!
//! When using masked gather functions, be aware of two distinct behaviors:
//!
//! ## 1. Architectural Fault Suppression (AVX-512)
//!
//! AVX-512 masked gathers are *architecturally* designed to **suppress page faults**
//! for masked-off elements. If a masked element (mask bit = 0) points to an invalid
//! address, it will NOT cause a page fault. This is documented in the Intel® 64 and
//! IA-32 Architectures SDM, Vol. 1, Section 15.6.4.
//!
//! This means masked gathers are safe to use when some indices may be invalid, as long
//! as those lanes are masked off.
//!
//! ## 2. Speculative Memory Access (Performance Reality)
//!
//! Despite the mask, the hardware may still **speculatively access all memory locations**
//! regardless of mask state. This was the root cause of the Gather Data Sampling (GDS)
//! vulnerability (CVE-2022-40982).
//!
//! From Intel's GDS documentation:
//! > "When a gather instruction performs loads from memory, different data elements are
//! > merged into the destination vector register according to the mask specified. In some
//! > situations, due to hardware optimizations specific to gather instructions, stale data
//! > from previous usage of architectural or internal vector registers may get transiently
//! > forwarded to dependent instructions without being updated by the gather loads."
//!
//! **Practical implications:**
//! - The mask **does NOT reduce memory bandwidth** - all lanes likely issue loads
//! - The mask **does NOT skip cache misses** on masked lanes
//! - Post-GDS microcode updates add latency but fix the speculation issue
//!
//! ## Architecture Comparison
//!
//! | Feature                         | AVX2 Gather    | AVX-512 Gather           |
//! |---------------------------------|----------------|--------------------------|
//! | Masked fault suppression        | Limited/None   | Architecturally guaranteed |
//! | Speculative access (pre-GDS)    | Yes            | Yes                      |
//! | Post-GDS microcode              | N/A            | Adds latency, fixes spec |
//!
//! ## When to Use Masked Gathers
//!
//! **Good use cases:**
//! - Conditional semantics (keeping fallback values for some lanes)
//! - Fault suppression (safe to have invalid pointers in masked lanes on AVX-512)
//! - Avoiding branching in vectorized code
//!
//! **NOT useful for:**
//! - Reducing memory bandwidth (all locations still accessed)
//! - Skipping expensive cache misses on masked lanes
//! - Performance gains from partial masking
//!
//! # References
//!
//! - Intel® 64 and IA-32 Architectures SDM, Vol. 1, Section 15.6.4 (AVX-512 Masking)
//! - [Intel Gather Data Sampling (GDS) Documentation](https://www.intel.com/content/www/us/en/developer/articles/technical/software-security-guidance/technical-documentation/gather-data-sampling.html)
//!
//! # Example
//!
//! ```rust
//! use wide::u32x16;
//! use simd_lookup::simd_gather::gather_u32index_u8;
//!
//! let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
//! let indices = u32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
//! let result = gather_u32index_u8(indices, &data, 1);
//! assert_eq!(result.to_array(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
//! ```

use wide::{u8x16, u32x16};

// =============================================================================
// Public API
// =============================================================================

/// Gather 16 bytes from memory using u32 indices.
///
/// Computes: `result[i] = base[indices[i] * scale]` for each lane.
///
/// # Arguments
/// * `indices` - Vector of 16 u32 indices
/// * `base` - Base slice to gather from
/// * `scale` - Scale factor applied to each index (1, 2, 4, or 8)
///
/// # Safety
/// The caller must ensure that `indices[i] * scale < base.len()` for all lanes.
/// Out-of-bounds access is undefined behavior.
#[inline]
pub fn gather_u32index_u8(indices: u32x16, base: &[u8], scale: u8) -> u8x16 {
    #[cfg(target_arch = "x86_64")]
    {
        // Requires AVX512BW for _mm512_cvtepi32_epi8
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return unsafe { gather_u32index_u8_avx512(indices, base, scale) };
        }
    }

    gather_u32index_u8_scalar(indices, base, scale)
}

/// Masked gather of 16 bytes from memory using u32 indices.
///
/// For lanes where mask bit is 1: `result[i] = base[indices[i] * scale]`
/// For lanes where mask bit is 0: `result[i] = fallback[i] as u8`
///
/// # Arguments
/// * `indices` - Vector of 16 u32 indices
/// * `base` - Base slice to gather from
/// * `scale` - Scale factor applied to each index (1, 2, 4, or 8)
/// * `mask` - 16-bit mask indicating which lanes to gather (1 = gather, 0 = use fallback)
/// * `fallback` - Fallback values (low byte used) for masked-off lanes
///
/// # Safety
/// For lanes where the mask bit is 1, the caller must ensure that
/// `indices[i] * scale < base.len()`. Out-of-bounds access is undefined behavior.
#[inline]
pub fn gather_masked_u32index_u8(
    indices: u32x16,
    base: &[u8],
    scale: u8,
    mask: u16,
    fallback: u32x16,
) -> u8x16 {
    #[cfg(target_arch = "x86_64")]
    {
        // Requires AVX512BW for _mm512_cvtepi32_epi8
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            return unsafe { gather_masked_u32index_u8_avx512(indices, base, scale, mask, fallback) };
        }
    }

    gather_masked_u32index_u8_scalar(indices, base, scale, mask, fallback)
}

/// Gather 16 u32 values from memory using u32 indices.
///
/// Computes: `result[i] = base[indices[i] * scale / 4]` for each lane.
///
/// # Arguments
/// * `indices` - Vector of 16 u32 indices
/// * `base` - Base slice to gather from
/// * `scale` - Scale factor applied to each index (1, 2, 4, or 8)
///
/// # Safety
/// The caller must ensure that the computed byte offset is valid.
/// Out-of-bounds access is undefined behavior.
#[inline]
pub fn gather_u32index_u32(indices: u32x16, base: &[u32], scale: u8) -> u32x16 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { gather_u32index_u32_avx512(indices, base, scale) };
        }
    }

    gather_u32index_u32_scalar(indices, base, scale)
}

/// Masked gather of 16 u32 values from memory using u32 indices.
///
/// For lanes where mask bit is 1: `result[i] = base[indices[i] * scale / 4]`
/// For lanes where mask bit is 0: `result[i] = fallback[i]`
///
/// # Arguments
/// * `indices` - Vector of 16 u32 indices
/// * `base` - Base slice to gather from
/// * `scale` - Scale factor applied to each index (1, 2, 4, or 8)
/// * `mask` - 16-bit mask indicating which lanes to gather (1 = gather, 0 = use fallback)
/// * `fallback` - Fallback values for masked-off lanes
///
/// # Safety
/// For lanes where the mask bit is 1, the caller must ensure that
/// the computed byte offset is valid. Out-of-bounds access is undefined behavior.
#[inline]
pub fn gather_masked_u32index_u32(
    indices: u32x16,
    base: &[u32],
    scale: u8,
    mask: u16,
    fallback: u32x16,
) -> u32x16 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { gather_masked_u32index_u32_avx512(indices, base, scale, mask, fallback) };
        }
    }

    gather_masked_u32index_u32_scalar(indices, base, scale, mask, fallback)
}

// =============================================================================
// x86_64 AVX512 Implementations
// =============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

/// AVX512 implementation of gather_u32index_u8
///
/// Uses `_mm512_i32gather_epi32` to gather 32-bit words, then extracts low bytes
/// using `_mm512_cvtepi32_epi8`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn gather_u32index_u8_avx512(indices: u32x16, base: &[u8], scale: u8) -> u8x16 {
    unsafe {
        let idx = std::mem::transmute::<u32x16, __m512i>(indices);

        // Gather 32-bit values (we only care about the low byte of each)
        let gathered = match scale {
            1 => _mm512_i32gather_epi32::<1>(idx, base.as_ptr() as *const i32),
            2 => _mm512_i32gather_epi32::<2>(idx, base.as_ptr() as *const i32),
            4 => _mm512_i32gather_epi32::<4>(idx, base.as_ptr() as *const i32),
            8 => _mm512_i32gather_epi32::<8>(idx, base.as_ptr() as *const i32),
            _ => _mm512_i32gather_epi32::<1>(idx, base.as_ptr() as *const i32),
        };

        // Extract low byte from each 32-bit lane
        extract_low_bytes_avx512(gathered)
    }
}

/// AVX512 implementation of gather_masked_u32index_u8
///
/// Uses `_mm512_cvtepi32_epi8` to extract low bytes from gathered 32-bit values.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn gather_masked_u32index_u8_avx512(
    indices: u32x16,
    base: &[u8],
    scale: u8,
    mask: u16,
    fallback: u32x16,
) -> u8x16 {
    unsafe {
        let idx = std::mem::transmute::<u32x16, __m512i>(indices);
        let src = std::mem::transmute::<u32x16, __m512i>(fallback);

        // Masked gather: where mask bit is 1, gather from memory; where 0, use src
        let gathered = match scale {
            1 => _mm512_mask_i32gather_epi32::<1>(src, mask, idx, base.as_ptr() as *const i32),
            2 => _mm512_mask_i32gather_epi32::<2>(src, mask, idx, base.as_ptr() as *const i32),
            4 => _mm512_mask_i32gather_epi32::<4>(src, mask, idx, base.as_ptr() as *const i32),
            8 => _mm512_mask_i32gather_epi32::<8>(src, mask, idx, base.as_ptr() as *const i32),
            _ => _mm512_mask_i32gather_epi32::<1>(src, mask, idx, base.as_ptr() as *const i32),
        };

        // Extract low byte from each 32-bit lane
        extract_low_bytes_avx512(gathered)
    }
}

/// AVX512 implementation of gather_u32index_u32
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn gather_u32index_u32_avx512(indices: u32x16, base: &[u32], scale: u8) -> u32x16 {
    unsafe {
        let idx = std::mem::transmute::<u32x16, __m512i>(indices);

        let gathered = match scale {
            1 => _mm512_i32gather_epi32::<1>(idx, base.as_ptr() as *const i32),
            2 => _mm512_i32gather_epi32::<2>(idx, base.as_ptr() as *const i32),
            4 => _mm512_i32gather_epi32::<4>(idx, base.as_ptr() as *const i32),
            8 => _mm512_i32gather_epi32::<8>(idx, base.as_ptr() as *const i32),
            _ => _mm512_i32gather_epi32::<4>(idx, base.as_ptr() as *const i32),
        };

        std::mem::transmute::<__m512i, u32x16>(gathered)
    }
}

/// AVX512 implementation of gather_masked_u32index_u32
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn gather_masked_u32index_u32_avx512(
    indices: u32x16,
    base: &[u32],
    scale: u8,
    mask: u16,
    fallback: u32x16,
) -> u32x16 {
    unsafe {
        let idx = std::mem::transmute::<u32x16, __m512i>(indices);
        let src = std::mem::transmute::<u32x16, __m512i>(fallback);

        let gathered = match scale {
            1 => _mm512_mask_i32gather_epi32::<1>(src, mask, idx, base.as_ptr() as *const i32),
            2 => _mm512_mask_i32gather_epi32::<2>(src, mask, idx, base.as_ptr() as *const i32),
            4 => _mm512_mask_i32gather_epi32::<4>(src, mask, idx, base.as_ptr() as *const i32),
            8 => _mm512_mask_i32gather_epi32::<8>(src, mask, idx, base.as_ptr() as *const i32),
            _ => _mm512_mask_i32gather_epi32::<4>(src, mask, idx, base.as_ptr() as *const i32),
        };

        std::mem::transmute::<__m512i, u32x16>(gathered)
    }
}

/// Extract the low byte from each 32-bit lane of a 512-bit vector.
///
/// Uses `_mm512_cvtepi32_epi8` which truncates each 32-bit element to 8 bits
/// and packs all 16 results into a 128-bit vector (u8x16).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn extract_low_bytes_avx512(gathered: __m512i) -> u8x16 {
    unsafe {
        // Truncate each 32-bit lane to 8 bits, pack into __m128i
        let packed = _mm512_cvtepi32_epi8(gathered);
        std::mem::transmute::<__m128i, u8x16>(packed)
    }
}

// =============================================================================
// Scalar Fallback Implementations
// =============================================================================

/// Scalar fallback for gather_u32index_u8
#[inline]
fn gather_u32index_u8_scalar(indices: u32x16, base: &[u8], scale: u8) -> u8x16 {
    let idx_arr = indices.to_array();
    let scale = scale as usize;
    let mut result = [0u8; 16];

    for i in 0..16 {
        let offset = idx_arr[i] as usize * scale;
        result[i] = base[offset];
    }

    u8x16::from(result)
}

/// Scalar fallback for gather_masked_u32index_u8
#[inline]
fn gather_masked_u32index_u8_scalar(
    indices: u32x16,
    base: &[u8],
    scale: u8,
    mask: u16,
    fallback: u32x16,
) -> u8x16 {
    let idx_arr = indices.to_array();
    let fallback_arr = fallback.to_array();
    let scale = scale as usize;
    let mut result = [0u8; 16];

    for i in 0..16 {
        if (mask >> i) & 1 != 0 {
            let offset = idx_arr[i] as usize * scale;
            result[i] = base[offset];
        } else {
            result[i] = fallback_arr[i] as u8;
        }
    }

    u8x16::from(result)
}

/// Scalar fallback for gather_u32index_u32
#[inline]
fn gather_u32index_u32_scalar(indices: u32x16, base: &[u32], scale: u8) -> u32x16 {
    let idx_arr = indices.to_array();
    let scale = scale as usize;
    let mut result = [0u32; 16];

    for i in 0..16 {
        // Scale is in bytes, so divide by 4 for u32 indexing
        let offset = (idx_arr[i] as usize * scale) / 4;
        result[i] = base[offset];
    }

    u32x16::from(result)
}

/// Scalar fallback for gather_masked_u32index_u32
#[inline]
fn gather_masked_u32index_u32_scalar(
    indices: u32x16,
    base: &[u32],
    scale: u8,
    mask: u16,
    fallback: u32x16,
) -> u32x16 {
    let idx_arr = indices.to_array();
    let fallback_arr = fallback.to_array();
    let scale = scale as usize;
    let mut result = [0u32; 16];

    for i in 0..16 {
        if (mask >> i) & 1 != 0 {
            // Scale is in bytes, so divide by 4 for u32 indexing
            let offset = (idx_arr[i] as usize * scale) / 4;
            result[i] = base[offset];
        } else {
            result[i] = fallback_arr[i];
        }
    }

    u32x16::from(result)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gather_u32index_u8_basic() {
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let indices = u32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        let result = gather_u32index_u8(indices, &data, 1);
        assert_eq!(
            result.to_array(),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );
    }

    #[test]
    fn test_gather_u32index_u8_scaled() {
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        // With scale=2, indices are multiplied by 2
        let indices = u32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        let result = gather_u32index_u8(indices, &data, 2);
        assert_eq!(
            result.to_array(),
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        );
    }

    #[test]
    fn test_gather_u32index_u8_non_sequential() {
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let indices = u32x16::from([100, 50, 200, 25, 150, 75, 225, 10, 0, 255, 128, 64, 192, 32, 96, 160]);

        let result = gather_u32index_u8(indices, &data, 1);
        assert_eq!(
            result.to_array(),
            [100, 50, 200, 25, 150, 75, 225, 10, 0, 255, 128, 64, 192, 32, 96, 160]
        );
    }

    #[test]
    fn test_gather_masked_u32index_u8() {
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let indices = u32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let fallback = u32x16::from([255; 16]);
        // Mask: only gather even lanes (bits 0, 2, 4, 6, 8, 10, 12, 14)
        let mask = 0b0101010101010101u16;

        let result = gather_masked_u32index_u8(indices, &data, 1, mask, fallback);
        assert_eq!(
            result.to_array(),
            [0, 255, 2, 255, 4, 255, 6, 255, 8, 255, 10, 255, 12, 255, 14, 255]
        );
    }

    #[test]
    fn test_gather_masked_u32index_u8_all_masked() {
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let indices = u32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let fallback = u32x16::from([42; 16]);
        let mask = 0u16; // No lanes active

        let result = gather_masked_u32index_u8(indices, &data, 1, mask, fallback);
        assert_eq!(result.to_array(), [42; 16]);
    }

    #[test]
    fn test_gather_u32index_u32_basic() {
        let data: Vec<u32> = (0..256).map(|i| i as u32 * 1000).collect();
        let indices = u32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        // Scale 4 means each index addresses a u32 directly
        let result = gather_u32index_u32(indices, &data, 4);
        assert_eq!(
            result.to_array(),
            [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000]
        );
    }

    #[test]
    fn test_gather_masked_u32index_u32() {
        let data: Vec<u32> = (0..256).map(|i| i as u32 * 100).collect();
        let indices = u32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let fallback = u32x16::from([999; 16]);
        // Mask: only odd lanes (bits 1, 3, 5, 7, 9, 11, 13, 15)
        let mask = 0b1010101010101010u16;

        let result = gather_masked_u32index_u32(indices, &data, 4, mask, fallback);
        assert_eq!(
            result.to_array(),
            [999, 100, 999, 300, 999, 500, 999, 700, 999, 900, 999, 1100, 999, 1300, 999, 1500]
        );
    }

    #[test]
    fn test_gather_u32index_u32_non_sequential() {
        let data: Vec<u32> = (0..256).map(|i| i as u32).collect();
        let indices = u32x16::from([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);

        let result = gather_u32index_u32(indices, &data, 4);
        assert_eq!(
            result.to_array(),
            [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        );
    }
}

