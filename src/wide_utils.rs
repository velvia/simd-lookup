//! SIMD utilities and trait extensions for the `wide` crate
//!
//! This module provides optimized platform-specific implementations for common SIMD operations
//! that are not directly available in the `wide` crate, including:
//! - Widening operations (u32x8 → u64x8)
//! - Bitmask to vector conversion
//! - Cross-platform optimizations for x86_64 and aarch64
//!
//! # Examples
//!
//! ```rust
//! use simd_lookup::wide_utils::{WideUtilsExt, FromBitmask};
//! use wide::{u32x8, u64x8};
//!
//! let input = u32x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
//! let widened: u64x8 = input.widen_to_u64x8();
//!
//! let mask = 0b10101010u8;
//! let mask_vec: u64x8 = u64x8::from_bitmask(mask);
//! ```

use wide::{u32x4, u32x8, u64x4, u64x8};

/// Trait extension for `wide` SIMD types providing additional utility operations
pub trait WideUtilsExt<T> {
    /// The output type for widening operations
    type Widened;

    /// Widen the vector elements to a larger type
    fn widen_to_u64x8(self) -> Self::Widened;
}

/// Trait for creating SIMD vectors from bitmasks
pub trait FromBitmask<T> {
    /// Create a SIMD vector from a bitmask where each bit becomes 0 or T::MAX
    fn from_bitmask(mask: u8) -> Self;
}

// Implementation for u32x8 → u64x8 widening
impl WideUtilsExt<u32> for u32x8 {
    type Widened = u64x8;

    #[inline(always)]
    fn widen_to_u64x8(self) -> u64x8 {
        #[cfg(target_arch = "x86_64")]
        {
            // Use AVX-512 if available, fallback to AVX2
            if is_x86_feature_detected!("avx512f") {
                unsafe { widen_u32x8_to_u64x8_avx512(self) }
            } else if is_x86_feature_detected!("avx2") {
                unsafe { widen_u32x8_to_u64x8_avx2(self) }
            } else {
                // Fallback to scalar
                widen_u32x8_to_u64x8_scalar(self)
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { widen_u32x8_to_u64x8_neon(self) }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            widen_u32x8_to_u64x8_scalar(self)
        }
    }
}

// Implementation for u32x4 → u64x4 widening
impl WideUtilsExt<u32> for u32x4 {
    type Widened = u64x4;

    #[inline(always)]
    fn widen_to_u64x8(self) -> u64x4 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { widen_u32x4_to_u64x4_avx2(self) }
            } else {
                widen_u32x4_to_u64x4_scalar(self)
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { widen_u32x4_to_u64x4_neon(self) }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            widen_u32x4_to_u64x4_scalar(self)
        }
    }
}

// Implementation for creating u64x8 from bitmask
impl FromBitmask<u64> for u64x8 {
    #[inline(always)]
    fn from_bitmask(mask: u8) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe { u64x8_from_bitmask_avx512(mask) }
            } else if is_x86_feature_detected!("avx2") {
                unsafe { u64x8_from_bitmask_avx2(mask) }
            } else {
                u64x8_from_bitmask_scalar(mask)
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { u64x8_from_bitmask_neon(mask) }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            u64x8_from_bitmask_scalar(mask)
        }
    }
}

// Implementation for creating u32x8 from bitmask
impl FromBitmask<u32> for u32x8 {
    #[inline(always)]
    fn from_bitmask(mask: u8) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                unsafe { u32x8_from_bitmask_avx512(mask) }
            } else if is_x86_feature_detected!("avx2") {
                unsafe { u32x8_from_bitmask_avx2(mask) }
            } else {
                u32x8_from_bitmask_scalar(mask)
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { u32x8_from_bitmask_neon(mask) }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            u32x8_from_bitmask_scalar(mask)
        }
    }
}

// =============================================================================
// x86_64 Implementations
// =============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn widen_u32x8_to_u64x8_avx512(input: u32x8) -> u64x8 {
    let raw = std::mem::transmute::<u32x8, __m256i>(input);
    let widened = _mm512_cvtepu32_epi64(raw);
    std::mem::transmute::<__m512i, u64x8>(widened)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn widen_u32x8_to_u64x8_avx2(input: u32x8) -> u64x8 {
    let raw = std::mem::transmute::<u32x8, __m256i>(input);

    // Split into two 128-bit halves
    let low = _mm256_extracti128_si256(raw, 0);
    let high = _mm256_extracti128_si256(raw, 1);

    // Widen each half
    let low_wide = _mm256_cvtepu32_epi64(low);
    let high_wide = _mm256_cvtepu32_epi64(high);

    // Combine into 512-bit result (we'll use two 256-bit operations)
    // For now, let's create the result array manually
    let low_array: [u64; 4] = std::mem::transmute(low_wide);
    let high_array: [u64; 4] = std::mem::transmute(high_wide);

    u64x8::from([
        low_array[0],
        low_array[1],
        low_array[2],
        low_array[3],
        high_array[0],
        high_array[1],
        high_array[2],
        high_array[3],
    ])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn widen_u32x4_to_u64x4_avx2(input: u32x4) -> u64x4 {
    let raw = std::mem::transmute::<u32x4, __m128i>(input);
    let widened = _mm256_cvtepu32_epi64(raw);
    std::mem::transmute::<__m256i, u64x4>(widened)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn u64x8_from_bitmask_avx512(mask: u8) -> u64x8 {
    let kmask = mask;
    let vec = _mm512_maskz_set1_epi64(kmask, -1i64);
    std::mem::transmute::<__m512i, u64x8>(vec)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn u64x8_from_bitmask_avx2(mask: u8) -> u64x8 {
    // Use lookup table approach for AVX2 - fastest method
    let mut values = [0u64; 8];
    for i in 0..8 {
        values[i] = if (mask >> i) & 1 != 0 { u64::MAX } else { 0 };
    }
    u64x8::from(values)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn u32x8_from_bitmask_avx512(mask: u8) -> u32x8 {
    let kmask = mask;
    let vec = _mm256_maskz_set1_epi32(kmask, -1i32);
    std::mem::transmute::<__m256i, u32x8>(vec)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn u32x8_from_bitmask_avx2(mask: u8) -> u32x8 {
    // Use lookup table approach for AVX2
    let mut values = [0u32; 8];
    for i in 0..8 {
        values[i] = if (mask >> i) & 1 != 0 { u32::MAX } else { 0 };
    }
    u32x8::from(values)
}

// =============================================================================
// ARM NEON Implementations
// =============================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
unsafe fn widen_u32x8_to_u64x8_neon(input: u32x8) -> u64x8 {
    let array = input.to_array();

    unsafe {
        // Load as two uint32x4_t vectors
        let low_input = vld1q_u32(array.as_ptr());
        let high_input = vld1q_u32(array.as_ptr().add(4));

        // Widen each half
        let (low_0, low_1) = widen_u32x4_to_u64x4_neon_raw(low_input);
        let (high_0, high_1) = widen_u32x4_to_u64x4_neon_raw(high_input);

        // Store results
        let mut result = [0u64; 8];
        vst1q_u64(result.as_mut_ptr(), low_0);
        vst1q_u64(result.as_mut_ptr().add(2), low_1);
        vst1q_u64(result.as_mut_ptr().add(4), high_0);
        vst1q_u64(result.as_mut_ptr().add(6), high_1);

        u64x8::from(result)
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn widen_u32x4_to_u64x4_neon(input: u32x4) -> u64x4 {
    let array = input.to_array();
    unsafe {
        let neon_input = vld1q_u32(array.as_ptr());
        let (low, high) = widen_u32x4_to_u64x4_neon_raw(neon_input);

        let mut result = [0u64; 4];
        vst1q_u64(result.as_mut_ptr(), low);
        vst1q_u64(result.as_mut_ptr().add(2), high);

        u64x4::from(result)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn widen_u32x4_to_u64x4_neon_raw(input: uint32x4_t) -> (uint64x2_t, uint64x2_t) {
    let low = vmovl_u32(vget_low_u32(input)); // u64[0,1]
    let high = vmovl_u32(vget_high_u32(input)); // u64[2,3]
    (low, high)
}

#[cfg(target_arch = "aarch64")]
unsafe fn u64x8_from_bitmask_neon(mask: u8) -> u64x8 {
    // NEON doesn't have native mask registers, use manual approach
    let mut values = [0u64; 8];
    for (i, value) in values.iter_mut().enumerate() {
        *value = if (mask >> i) & 1 != 0 { u64::MAX } else { 0 };
    }
    u64x8::from(values)
}

#[cfg(target_arch = "aarch64")]
unsafe fn u32x8_from_bitmask_neon(mask: u8) -> u32x8 {
    // Manual approach for NEON
    let mut values = [0u32; 8];
    for (i, value) in values.iter_mut().enumerate() {
        *value = if (mask >> i) & 1 != 0 { u32::MAX } else { 0 };
    }
    u32x8::from(values)
}

// =============================================================================
// Scalar Fallback Implementations
// =============================================================================

#[allow(dead_code)]
fn widen_u32x8_to_u64x8_scalar(input: u32x8) -> u64x8 {
    let array = input.to_array();
    u64x8::from(array.map(|x| x as u64))
}

#[allow(dead_code)]
fn widen_u32x4_to_u64x4_scalar(input: u32x4) -> u64x4 {
    let array = input.to_array();
    u64x4::from(array.map(|x| x as u64))
}

#[allow(dead_code)]
fn u64x8_from_bitmask_scalar(mask: u8) -> u64x8 {
    let mut values = [0u64; 8];
    for (i, value) in values.iter_mut().enumerate() {
        *value = if (mask >> i) & 1 != 0 { u64::MAX } else { 0 };
    }
    u64x8::from(values)
}

#[allow(dead_code)]
fn u32x8_from_bitmask_scalar(mask: u8) -> u32x8 {
    let mut values = [0u32; 8];
    for (i, value) in values.iter_mut().enumerate() {
        *value = if (mask >> i) & 1 != 0 { u32::MAX } else { 0 };
    }
    u32x8::from(values)
}

// =============================================================================
// Feature Detection Helper
// =============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::is_x86_feature_detected;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u32x8_widening() {
        let input = u32x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
        let widened: u64x8 = input.widen_to_u64x8();
        let result = widened.to_array();

        assert_eq!(result, [1u64, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_u32x4_widening() {
        let input = u32x4::from([1, 2, 3, 4]);
        let widened: u64x4 = input.widen_to_u64x8();
        let result = widened.to_array();

        assert_eq!(result, [1u64, 2, 3, 4]);
    }

    #[test]
    fn test_u64x8_from_bitmask() {
        let mask = 0b10101010u8;
        let mask_vec: u64x8 = u64x8::from_bitmask(mask);
        let result = mask_vec.to_array();

        let expected = [
            0u64,     // bit 0 = 0
            u64::MAX, // bit 1 = 1
            0u64,     // bit 2 = 0
            u64::MAX, // bit 3 = 1
            0u64,     // bit 4 = 0
            u64::MAX, // bit 5 = 1
            0u64,     // bit 6 = 0
            u64::MAX, // bit 7 = 1
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_u32x8_from_bitmask() {
        let mask = 0b11000011u8;
        let mask_vec: u32x8 = u32x8::from_bitmask(mask);
        let result = mask_vec.to_array();

        let expected = [
            u32::MAX, // bit 0 = 1
            u32::MAX, // bit 1 = 1
            0u32,     // bit 2 = 0
            0u32,     // bit 3 = 0
            0u32,     // bit 4 = 0
            0u32,     // bit 5 = 0
            u32::MAX, // bit 6 = 1
            u32::MAX, // bit 7 = 1
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_edge_cases() {
        // Test all zeros
        let mask_zero = 0b00000000u8;
        let vec_zero: u64x8 = u64x8::from_bitmask(mask_zero);
        assert_eq!(vec_zero.to_array(), [0u64; 8]);

        // Test all ones
        let mask_all = 0b11111111u8;
        let vec_all: u64x8 = u64x8::from_bitmask(mask_all);
        assert_eq!(vec_all.to_array(), [u64::MAX; 8]);
    }
}
