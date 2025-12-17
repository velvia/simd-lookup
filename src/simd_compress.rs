//! SIMD compress operations
//!
//! This module provides compress/compact operations similar to AVX-512's VCOMPRESS instruction.
//! Elements where the corresponding mask bit is set are packed contiguously to the front of
//! the destination buffer.
//!
//! # CPU Feature Requirements (Intel x86_64)
//!
//! ## Optimal Performance (AVX-512)
//!
//! - **`compress_store_u32x8` / `compress_u32x8`**: Requires **AVX512F** + **AVX512VL**
//!   - Uses `VPCOMPRESSD` instruction (`_mm256_mask_compressstoreu_epi32`)
//!   - Available on: Intel Skylake-X (Xeon), Ice Lake, Tiger Lake, and later
//!   - Fallback: Shuffle-based table lookup (works on all architectures)
//!
//! - **`compress_store_u32x16` / `compress_u32x16`**: Requires **AVX512F**
//!   - Uses `VPCOMPRESSD` instruction (`_mm512_mask_compressstoreu_epi32`)
//!   - Available on: Intel Skylake-X (Xeon), Ice Lake, Tiger Lake, and later
//!   - Fallback: Two `compress_store_u32x8` operations (works on all architectures)
//!
//! - **`compress_store_u8x16` / `compress_u8x16`**: Requires **AVX512VBMI2** + **AVX512VL**
//!   - Uses `VPCOMPRESSB` instruction (`_mm256_mask_compressstoreu_epi8`)
//!   - Available on: Intel Ice Lake, Tiger Lake, and later (not available on Skylake-X)
//!   - Fallback: Gather-style direct writes (works on all architectures)
//!
//! ## Fallback Behavior
//!
//! All functions automatically fall back to scalar/shuffle implementations when AVX-512
//! features are not available. The fallback implementations work on:
//! - x86_64 without AVX-512 (uses AVX2/SSE if available, or scalar)
//! - aarch64 (ARM NEON)
//! - All other architectures (scalar fallback)
//!
//! ## Performance Impact
//!
//! AVX-512 compress instructions are **3-5× faster** than shuffle-based fallbacks for
//! typical mask densities (10-50% of elements selected).
//!
//! # Example
//! ```ignore
//! use wide::u32x8;
//! use simd_lookup::simd_compress::compress_store_u32x8;
//!
//! let data = u32x8::from([10, 20, 30, 40, 50, 60, 70, 80]);
//! let mask = 0b10110010u8; // Select elements at positions 1, 4, 5, 7
//! let mut output = [0u32; 8];
//!
//! let count = compress_store_u32x8(data, mask, &mut output);
//! // count == 4
//! // output[0..4] == [20, 50, 60, 80]
//! ```

use wide::{u8x16, u32x8, u32x16};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

use crate::wide_utils::{
    SimdSplit, WideUtilsExt, get_compress_indices_u32x8,
    SHUFFLE_COMPRESS_IDX_U8_HI, SHUFFLE_COMPRESS_IDX_U8_LO,
};

// =============================================================================
// u32x8 Compress Operations
// =============================================================================

/// Compress and store u32x8 elements where mask bits are set.
///
/// # Arguments
/// * `data` - Source vector of 8 u32 values
/// * `mask` - 8-bit mask where bit i selects element i
/// * `dest` - Destination slice (must have at least `mask.count_ones()` elements)
///
/// # Returns
/// Number of elements written (equal to `mask.count_ones()`)
///
/// # Panics
/// Panics if `dest` is smaller than the number of set bits in mask.
#[inline]
pub fn compress_store_u32x8(data: u32x8, mask: u8, dest: &mut [u32]) -> usize {
    let count = mask.count_ones() as usize;
    assert!(dest.len() >= count, "destination buffer too small");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            unsafe { compress_store_u32x8_avx512(data, mask, dest) };
            return count;
        }
    }

    // Fallback: gather-style direct write (faster than shuffle on ARM)
    compress_store_u32x8_gather(data, mask, dest);
    count
}

/// Compress u32x8 and return both the compressed vector and element count.
/// Unwritten lanes contain undefined values.
#[inline]
pub fn compress_u32x8(data: u32x8, mask: u8) -> (u32x8, usize) {
    let count = mask.count_ones() as usize;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            let result = unsafe { compress_u32x8_avx512(data, mask) };
            return (result, count);
        }
    }

    // Fallback: use shuffle with SIMD indices (zero-cost table lookup via transmute)
    let indices = get_compress_indices_u32x8(mask);
    let result = data.shuffle(indices);
    (result, count)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f", enable = "avx512vl")]
unsafe fn compress_store_u32x8_avx512(data: u32x8, mask: u8, dest: &mut [u32]) {
    unsafe {
        let raw = std::mem::transmute::<u32x8, __m256i>(data);
        _mm256_mask_compressstoreu_epi32(dest.as_mut_ptr() as *mut i32, mask, raw);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f", enable = "avx512vl")]
unsafe fn compress_u32x8_avx512(data: u32x8, mask: u8) -> u32x8 {
    unsafe {
        let raw = std::mem::transmute::<u32x8, __m256i>(data);
        let compressed = _mm256_maskz_compress_epi32(mask, raw);
        std::mem::transmute::<__m256i, u32x8>(compressed)
    }
}

/// Gather-style compress for u32x8 - direct indexed writes to destination.
#[inline]
fn compress_store_u32x8_gather(data: u32x8, mask: u8, dest: &mut [u32]) {
    let arr = data.to_array();
    let mut idx = 0;
    if mask & (1 << 0) != 0 { dest[idx] = arr[0]; idx += 1; }
    if mask & (1 << 1) != 0 { dest[idx] = arr[1]; idx += 1; }
    if mask & (1 << 2) != 0 { dest[idx] = arr[2]; idx += 1; }
    if mask & (1 << 3) != 0 { dest[idx] = arr[3]; idx += 1; }
    if mask & (1 << 4) != 0 { dest[idx] = arr[4]; idx += 1; }
    if mask & (1 << 5) != 0 { dest[idx] = arr[5]; idx += 1; }
    if mask & (1 << 6) != 0 { dest[idx] = arr[6]; idx += 1; }
    if mask & (1 << 7) != 0 { dest[idx] = arr[7]; }
}

// =============================================================================
// u32x16 Compress Operations (512-bit)
// =============================================================================

/// Compress and store u32x16 elements where mask bits are set.
///
/// # Arguments
/// * `data` - Source vector of 16 u32 values
/// * `mask` - 16-bit mask where bit i selects element i
/// * `dest` - Destination slice (must have at least `mask.count_ones()` elements)
///
/// # Returns
/// Number of elements written (equal to `mask.count_ones()`)
///
/// # Panics
/// Panics if `dest` is smaller than the number of set bits in mask.
#[inline]
pub fn compress_store_u32x16(data: u32x16, mask: u16, dest: &mut [u32]) -> usize {
    let count = mask.count_ones() as usize;
    assert!(dest.len() >= count, "destination buffer too small");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { compress_store_u32x16_avx512(data, mask, dest) };
            return count;
        }
    }

    // Fallback: split into two u32x8 halves and compress each
    compress_store_u32x16_fallback(data, mask, dest);
    count
}

/// Compress u32x16 and return both the compressed vector and element count.
/// Unwritten lanes contain undefined values.
#[inline]
pub fn compress_u32x16(data: u32x16, mask: u16) -> (u32x16, usize) {
    let count = mask.count_ones() as usize;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            let result = unsafe { compress_u32x16_avx512(data, mask) };
            return (result, count);
        }
    }

    // Fallback: use two u32x8 compress operations
    let result = compress_u32x16_fallback_to_vec(data, mask);
    (result, count)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn compress_store_u32x16_avx512(data: u32x16, mask: u16, dest: &mut [u32]) {
    unsafe {
        let raw = std::mem::transmute::<u32x16, __m512i>(data);
        _mm512_mask_compressstoreu_epi32(dest.as_mut_ptr() as *mut i32, mask, raw);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn compress_u32x16_avx512(data: u32x16, mask: u16) -> u32x16 {
    unsafe {
        let raw = std::mem::transmute::<u32x16, __m512i>(data);
        let compressed = _mm512_maskz_compress_epi32(mask, raw);
        std::mem::transmute::<__m512i, u32x16>(compressed)
    }
}

/// Fallback: compress u32x16 by splitting into two u32x8 halves using SimdSplit
#[inline]
fn compress_store_u32x16_fallback(data: u32x16, mask: u16, dest: &mut [u32]) {
    // Use efficient SimdSplit to extract halves
    let (lo, hi) = data.split_low_high();

    let lo_mask = (mask & 0xFF) as u8;
    let hi_mask = ((mask >> 8) & 0xFF) as u8;

    // Compress low half
    let lo_count = compress_store_u32x8(lo, lo_mask, dest);

    // Compress high half, writing after the low results
    let _ = compress_store_u32x8(hi, hi_mask, &mut dest[lo_count..]);
}

/// Fallback: compress u32x16 to vector using two u32x8 operations
#[inline]
fn compress_u32x16_fallback_to_vec(data: u32x16, mask: u16) -> u32x16 {
    // Use efficient SimdSplit to extract halves
    let (lo, hi) = data.split_low_high();

    let lo_mask = (mask & 0xFF) as u8;
    let hi_mask = ((mask >> 8) & 0xFF) as u8;

    // Compress each half
    let (lo_compressed, lo_count) = compress_u32x8(lo, lo_mask);
    let (hi_compressed, hi_count) = compress_u32x8(hi, hi_mask);

    // Combine results using slice operations
    let lo_arr = lo_compressed.to_array();
    let hi_arr = hi_compressed.to_array();

    let mut result = [0u32; 16];

    // Copy compressed low elements
    result[..lo_count].copy_from_slice(&lo_arr[..lo_count]);

    // Copy compressed high elements after low
    let hi_copy_count = hi_count.min(16 - lo_count);
    result[lo_count..lo_count + hi_copy_count].copy_from_slice(&hi_arr[..hi_copy_count]);

    u32x16::from(result)
}

// =============================================================================
// u8x16 Compress Operations
// =============================================================================

/// Compress and store u8x16 elements where mask bits are set.
///
/// # Arguments
/// * `data` - Source vector of 16 u8 values
/// * `mask` - 16-bit mask where bit i selects element i
/// * `dest` - Destination slice (must have at least `mask.count_ones()` elements)
///
/// # Returns
/// Number of elements written (equal to `mask.count_ones()`)
///
/// # Panics
/// Panics if `dest` is smaller than the number of set bits in mask.
#[inline]
pub fn compress_store_u8x16(data: u8x16, mask: u16, dest: &mut [u8]) -> usize {
    let count = mask.count_ones() as usize;
    assert!(dest.len() >= count, "destination buffer too small");

    #[cfg(target_arch = "x86_64")]
    {
        // AVX512VBMI2 has native u8 compress
        if is_x86_feature_detected!("avx512vbmi2") && is_x86_feature_detected!("avx512vl") {
            unsafe { compress_store_u8x16_avx512(data, mask, dest) };
            return count;
        }
    }

    // Fallback: gather-style direct write (faster than shuffle on ARM)
    // Avoids shuffle index building, shuffle operation, and intermediate copies
    compress_store_u8x16_gather(data, mask, dest);
    count
}

/// Compress u8x16 and return both the compressed vector and element count.
/// Unwritten lanes contain undefined values.
#[inline]
pub fn compress_u8x16(data: u8x16, mask: u16) -> (u8x16, usize) {
    let count = mask.count_ones() as usize;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vbmi2") && is_x86_feature_detected!("avx512vl") {
            let result = unsafe { compress_u8x16_avx512(data, mask) };
            return (result, count);
        }
    }

    // Fallback: two-pass shuffle approach
    let result = compress_u8x16_shuffle(data, mask);
    (result, count)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512vbmi2", enable = "avx512vl")]
unsafe fn compress_store_u8x16_avx512(data: u8x16, mask: u16, dest: &mut [u8]) {
    unsafe {
        let raw = std::mem::transmute::<u8x16, __m128i>(data);
        _mm_mask_compressstoreu_epi8(dest.as_mut_ptr() as *mut i8, mask, raw);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512vbmi2", enable = "avx512vl")]
unsafe fn compress_u8x16_avx512(data: u8x16, mask: u16) -> u8x16 {
    unsafe {
        let raw = std::mem::transmute::<u8x16, __m128i>(data);
        let compressed = _mm_maskz_compress_epi8(mask, raw);
        std::mem::transmute::<__m128i, u8x16>(compressed)
    }
}

/// Gather-style compress for u8x16 - direct indexed writes to destination.
/// Faster than shuffle-based approach on ARM because it avoids:
/// - Building shuffle indices from lookup tables
/// - The shuffle operation itself
/// - Intermediate array copies
#[inline]
fn compress_store_u8x16_gather(data: u8x16, mask: u16, dest: &mut [u8]) {
    let arr = data.to_array();
    let mut idx = 0;
    // Unrolled gather: each element is conditionally written
    // Compiler can optimize this to efficient indexed stores
    if mask & (1 << 0) != 0 { dest[idx] = arr[0]; idx += 1; }
    if mask & (1 << 1) != 0 { dest[idx] = arr[1]; idx += 1; }
    if mask & (1 << 2) != 0 { dest[idx] = arr[2]; idx += 1; }
    if mask & (1 << 3) != 0 { dest[idx] = arr[3]; idx += 1; }
    if mask & (1 << 4) != 0 { dest[idx] = arr[4]; idx += 1; }
    if mask & (1 << 5) != 0 { dest[idx] = arr[5]; idx += 1; }
    if mask & (1 << 6) != 0 { dest[idx] = arr[6]; idx += 1; }
    if mask & (1 << 7) != 0 { dest[idx] = arr[7]; idx += 1; }
    if mask & (1 << 8) != 0 { dest[idx] = arr[8]; idx += 1; }
    if mask & (1 << 9) != 0 { dest[idx] = arr[9]; idx += 1; }
    if mask & (1 << 10) != 0 { dest[idx] = arr[10]; idx += 1; }
    if mask & (1 << 11) != 0 { dest[idx] = arr[11]; idx += 1; }
    if mask & (1 << 12) != 0 { dest[idx] = arr[12]; idx += 1; }
    if mask & (1 << 13) != 0 { dest[idx] = arr[13]; idx += 1; }
    if mask & (1 << 14) != 0 { dest[idx] = arr[14]; idx += 1; }
    if mask & (1 << 15) != 0 { dest[idx] = arr[15]; }
}

/// Compress u8x16 using shuffle tables (used by compress_u8x16 which returns a vector).
/// This is a two-pass approach that handles each 8-byte half separately.
#[inline]
fn compress_u8x16_shuffle(data: u8x16, mask: u16) -> u8x16 {
    let lo_mask = (mask & 0xFF) as u8;
    let hi_mask = ((mask >> 8) & 0xFF) as u8;

    let lo_count = lo_mask.count_ones() as usize;
    let hi_count = hi_mask.count_ones() as usize;

    // Get shuffle indices for each half
    let lo_indices = &SHUFFLE_COMPRESS_IDX_U8_LO[lo_mask as usize];
    let hi_indices = &SHUFFLE_COMPRESS_IDX_U8_HI[hi_mask as usize];

    // Build the full 16-byte shuffle index using slice operations
    let mut indices = [0u8; 16];

    // Copy low indices (variable count based on mask popcount)
    indices[..lo_count].copy_from_slice(&lo_indices[..lo_count]);

    // Copy high indices after low results
    let hi_copy_count = hi_count.min(16 - lo_count);
    indices[lo_count..lo_count + hi_copy_count].copy_from_slice(&hi_indices[..hi_copy_count]);

    // Remaining positions stay 0 (safe filler)
    data.shuffle(u8x16::from(indices))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_u32x8_basic() {
        let data = u32x8::from([10, 20, 30, 40, 50, 60, 70, 80]);
        let mask = 0b10110010u8;
        let mut output = [0u32; 8];

        let count = compress_store_u32x8(data, mask, &mut output);

        assert_eq!(count, 4);
        assert_eq!(output[0], 20);
        assert_eq!(output[1], 50);
        assert_eq!(output[2], 60);
        assert_eq!(output[3], 80);
    }

    #[test]
    fn test_compress_u32x8_all() {
        let data = u32x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
        let mask = 0xFFu8;
        let mut output = [0u32; 8];

        let count = compress_store_u32x8(data, mask, &mut output);

        assert_eq!(count, 8);
        assert_eq!(output, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_compress_u32x8_none() {
        let data = u32x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
        let mask = 0x00u8;
        let mut output = [0u32; 8];

        let count = compress_store_u32x8(data, mask, &mut output);

        assert_eq!(count, 0);
    }

    #[test]
    fn test_compress_u32x8_first_only() {
        let data = u32x8::from([42, 2, 3, 4, 5, 6, 7, 8]);
        let mask = 0b00000001u8;
        let mut output = [0u32; 8];

        let count = compress_store_u32x8(data, mask, &mut output);

        assert_eq!(count, 1);
        assert_eq!(output[0], 42);
    }

    #[test]
    fn test_compress_u32x8_last_only() {
        let data = u32x8::from([1, 2, 3, 4, 5, 6, 7, 99]);
        let mask = 0b10000000u8;
        let mut output = [0u32; 8];

        let count = compress_store_u32x8(data, mask, &mut output);

        assert_eq!(count, 1);
        assert_eq!(output[0], 99);
    }

    #[test]
    fn test_compress_u8x16_basic() {
        let data = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let mask = 0b1000000100000101u16;
        let mut output = [0u8; 16];

        let count = compress_store_u8x16(data, mask, &mut output);

        assert_eq!(count, 4);
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 2);
        assert_eq!(output[2], 8);
        assert_eq!(output[3], 15);
    }

    #[test]
    fn test_compress_u8x16_all() {
        let data = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let mask = 0xFFFFu16;
        let mut output = [0u8; 16];

        let count = compress_store_u8x16(data, mask, &mut output);

        assert_eq!(count, 16);
        assert_eq!(output, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    }

    #[test]
    fn test_compress_u8x16_none() {
        let data = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let mask = 0x0000u16;
        let mut output = [0u8; 16];

        let count = compress_store_u8x16(data, mask, &mut output);

        assert_eq!(count, 0);
    }

    #[test]
    fn test_compress_u8x16_low_half_only() {
        let data = u8x16::from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]);
        let mask = 0b0000000010101010u16;
        let mut output = [0u8; 16];

        let count = compress_store_u8x16(data, mask, &mut output);

        assert_eq!(count, 4);
        assert_eq!(output[0], 20);
        assert_eq!(output[1], 40);
        assert_eq!(output[2], 60);
        assert_eq!(output[3], 80);
    }

    #[test]
    fn test_compress_u8x16_high_half_only() {
        let data = u8x16::from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]);
        let mask = 0b0101010100000000u16;
        let mut output = [0u8; 16];

        let count = compress_store_u8x16(data, mask, &mut output);

        assert_eq!(count, 4);
        assert_eq!(output[0], 90);
        assert_eq!(output[1], 110);
        assert_eq!(output[2], 130);
        assert_eq!(output[3], 150);
    }

    #[test]
    fn test_compress_u32x8_return_vector() {
        let data = u32x8::from([10, 20, 30, 40, 50, 60, 70, 80]);
        let mask = 0b10110010u8;

        let (result, count) = compress_u32x8(data, mask);
        let arr = result.to_array();

        assert_eq!(count, 4);
        assert_eq!(arr[0], 20);
        assert_eq!(arr[1], 50);
        assert_eq!(arr[2], 60);
        assert_eq!(arr[3], 80);
    }

    #[test]
    fn test_compress_u8x16_return_vector() {
        let data = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let mask = 0b1000000100000101u16;

        let (result, count) = compress_u8x16(data, mask);
        let arr = result.to_array();

        assert_eq!(count, 4);
        assert_eq!(arr[0], 0);
        assert_eq!(arr[1], 2);
        assert_eq!(arr[2], 8);
        assert_eq!(arr[3], 15);
    }

    // =========================================================================
    // u32x16 Tests
    // =========================================================================

    #[test]
    fn test_compress_u32x16_basic() {
        let data = u32x16::from([
            10, 20, 30, 40, 50, 60, 70, 80,
            90, 100, 110, 120, 130, 140, 150, 160
        ]);
        let mask = 0b1000000110110010u16;
        let mut output = [0u32; 16];

        let count = compress_store_u32x16(data, mask, &mut output);

        assert_eq!(count, 6);
        assert_eq!(output[0], 20);
        assert_eq!(output[1], 50);
        assert_eq!(output[2], 60);
        assert_eq!(output[3], 80);
        assert_eq!(output[4], 90);
        assert_eq!(output[5], 160);
    }

    #[test]
    fn test_compress_u32x16_all() {
        let data = u32x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let mask = 0xFFFFu16;
        let mut output = [0u32; 16];

        let count = compress_store_u32x16(data, mask, &mut output);

        assert_eq!(count, 16);
        assert_eq!(output, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    }

    #[test]
    fn test_compress_u32x16_none() {
        let data = u32x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let mask = 0x0000u16;
        let mut output = [0u32; 16];

        let count = compress_store_u32x16(data, mask, &mut output);

        assert_eq!(count, 0);
    }

    #[test]
    fn test_compress_u32x16_low_half_only() {
        let data = u32x16::from([
            10, 20, 30, 40, 50, 60, 70, 80,
            90, 100, 110, 120, 130, 140, 150, 160
        ]);
        let mask = 0b0000000001010101u16;
        let mut output = [0u32; 16];

        let count = compress_store_u32x16(data, mask, &mut output);

        assert_eq!(count, 4);
        assert_eq!(output[0], 10);
        assert_eq!(output[1], 30);
        assert_eq!(output[2], 50);
        assert_eq!(output[3], 70);
    }

    #[test]
    fn test_compress_u32x16_high_half_only() {
        let data = u32x16::from([
            10, 20, 30, 40, 50, 60, 70, 80,
            90, 100, 110, 120, 130, 140, 150, 160
        ]);
        let mask = 0b0101010100000000u16;
        let mut output = [0u32; 16];

        let count = compress_store_u32x16(data, mask, &mut output);

        assert_eq!(count, 4);
        assert_eq!(output[0], 90);
        assert_eq!(output[1], 110);
        assert_eq!(output[2], 130);
        assert_eq!(output[3], 150);
    }

    #[test]
    fn test_compress_u32x16_return_vector() {
        let data = u32x16::from([
            10, 20, 30, 40, 50, 60, 70, 80,
            90, 100, 110, 120, 130, 140, 150, 160
        ]);
        let mask = 0b1000000110110010u16;

        let (result, count) = compress_u32x16(data, mask);
        let arr = result.to_array();

        assert_eq!(count, 6);
        assert_eq!(arr[0], 20);
        assert_eq!(arr[1], 50);
        assert_eq!(arr[2], 60);
        assert_eq!(arr[3], 80);
        assert_eq!(arr[4], 90);
        assert_eq!(arr[5], 160);
    }

    #[test]
    fn test_compress_u32x16_first_and_last() {
        let data = u32x16::from([
            100, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 200
        ]);
        let mask = 0b1000000000000001u16;
        let mut output = [0u32; 16];

        let count = compress_store_u32x16(data, mask, &mut output);

        assert_eq!(count, 2);
        assert_eq!(output[0], 100);
        assert_eq!(output[1], 200);
    }
}
