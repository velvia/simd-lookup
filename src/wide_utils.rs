//! SIMD utilities and trait extensions for the `wide` crate
//!
//! This module provides optimized platform-specific implementations for common SIMD operations
//! that are not directly available in the `wide` crate, including:
//! - Widening operations (u32x8 → u64x8)
//! - Bitmask to vector conversion
//! - Shuffle/permute operations
//! - Vector splitting (high/low extraction)
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

use wide::{u8x16, u32x4, u32x8, u32x16, u64x4, u64x8};

// =============================================================================
// Shuffle Lookup Tables for Compress Operations
// =============================================================================

/// Shuffle indices for compressing u32x8 based on an 8-bit mask.
/// Each entry contains the indices to shuffle selected elements to the front.
/// Stored as arrays; use `get_compress_indices_u32x8()` for zero-cost SIMD access.
pub static SHUFFLE_COMPRESS_IDX_U32X8: [[u32; 8]; 256] = {
    let mut table = [[7u32; 8]; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut dest_pos = 0usize;
        let mut src_pos = 0usize;
        while src_pos < 8 {
            if (mask >> src_pos) & 1 != 0 {
                table[mask][dest_pos] = src_pos as u32;
                dest_pos += 1;
            }
            src_pos += 1;
        }
        mask += 1;
    }
    table
};

/// Get compress indices as u32x8 (zero-cost transmute from array)
#[inline(always)]
pub fn get_compress_indices_u32x8(mask: u8) -> u32x8 {
    // Safety: [u32; 8] has same size and alignment as u32x8
    unsafe { std::mem::transmute(SHUFFLE_COMPRESS_IDX_U32X8[mask as usize]) }
}

/// Legacy alias for backwards compatibility
pub static SHUFFLE_COMPRESS_IDX_U32: &[[u32; 8]; 256] = &SHUFFLE_COMPRESS_IDX_U32X8;

/// Shuffle indices for compressing u8x16 (low half).
pub static SHUFFLE_COMPRESS_IDX_U8_LO: [[u8; 8]; 256] = {
    let mut table = [[0u8; 8]; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut dest_pos = 0usize;
        let mut src_pos = 0usize;
        while src_pos < 8 {
            if (mask >> src_pos) & 1 != 0 {
                table[mask][dest_pos] = src_pos as u8;
                dest_pos += 1;
            }
            src_pos += 1;
        }
        while dest_pos < 8 {
            table[mask][dest_pos] = 0;
            dest_pos += 1;
        }
        mask += 1;
    }
    table
};

/// High byte shuffle indices (with +8 offset baked in)
pub static SHUFFLE_COMPRESS_IDX_U8_HI: [[u8; 8]; 256] = {
    let mut table = [[0u8; 8]; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut dest_pos = 0usize;
        let mut src_pos = 0usize;
        while src_pos < 8 {
            if (mask >> src_pos) & 1 != 0 {
                table[mask][dest_pos] = (src_pos + 8) as u8;
                dest_pos += 1;
            }
            src_pos += 1;
        }
        while dest_pos < 8 {
            table[mask][dest_pos] = 8;
            dest_pos += 1;
        }
        mask += 1;
    }
    table
};

// =============================================================================
// Main Trait: WideUtilsExt - All SIMD utility operations in one place
// =============================================================================

/// Trait extension for `wide` SIMD types providing additional utility operations
pub trait WideUtilsExt: Sized {
    /// The output type for widening operations
    type Widened;

    /// Widen the vector elements to a larger type
    fn widen_to_u64x8(self) -> Self::Widened;

    /// Shuffle elements according to the given indices (indices are same SIMD type).
    /// `result[i] = self[indices[i]]`
    fn shuffle(self, indices: Self) -> Self;

    /// Double each element (self + self). Wraps on overflow.
    ///
    /// This is the most efficient way to multiply by 2 since addition is
    /// well-supported on all SIMD architectures (NEON `vaddq`, SSE `paddq`).
    ///
    /// For multiply by powers of 2, chain calls:
    /// - `x.double()` = x * 2
    /// - `x.double().double()` = x * 4
    /// - `x.double().double().double()` = x * 8
    #[inline(always)]
    fn double(self) -> Self
    where
        Self: std::ops::Add<Output = Self> + Copy,
    {
        self + self
    }
}

/// Trait for creating SIMD vectors from bitmasks
pub trait FromBitmask<T> {
    /// Create a SIMD vector from a bitmask where each bit becomes 0 or T::MAX
    fn from_bitmask(mask: u8) -> Self;
}

/// Trait for splitting SIMD vectors into high/low halves efficiently
pub trait SimdSplit: Sized {
    /// The half-width type (e.g., u32x8 for u32x16)
    type Half;

    /// Split into (low, high) halves using efficient intrinsics
    fn split_low_high(self) -> (Self::Half, Self::Half);

    /// Extract the low half
    #[inline(always)]
    fn low_half(self) -> Self::Half {
        self.split_low_high().0
    }

    /// Extract the high half
    #[inline(always)]
    fn high_half(self) -> Self::Half {
        self.split_low_high().1
    }
}


// =============================================================================
// SimdSplit Implementations
// =============================================================================

impl SimdSplit for u32x16 {
    type Half = u32x8;

    #[inline(always)]
    fn split_low_high(self) -> (u32x8, u32x8) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return unsafe { split_u32x16_avx512(self) };
            }
        }

        // Fallback: use pointer casting (zero-copy)
        split_u32x16_cast(self)
    }
}

impl SimdSplit for u64x8 {
    type Half = u64x4;

    #[inline(always)]
    fn split_low_high(self) -> (u64x4, u64x4) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return unsafe { split_u64x8_avx512(self) };
            }
        }

        // Fallback: use pointer casting
        split_u64x8_cast(self)
    }
}

impl SimdSplit for u8x16 {
    type Half = [u8; 8];

    #[inline(always)]
    fn split_low_high(self) -> ([u8; 8], [u8; 8]) {
        let arr = self.to_array();
        let mut lo = [0u8; 8];
        let mut hi = [0u8; 8];
        lo.copy_from_slice(&arr[0..8]);
        hi.copy_from_slice(&arr[8..16]);
        (lo, hi)
    }
}

// =============================================================================
// WideUtilsExt Implementations
// =============================================================================

impl WideUtilsExt for u32x8 {
    type Widened = u64x8;

    #[inline(always)]
    fn widen_to_u64x8(self) -> u64x8 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return unsafe { widen_u32x8_to_u64x8_avx512(self) };
            } else if is_x86_feature_detected!("avx2") {
                return unsafe { widen_u32x8_to_u64x8_avx2(self) };
            }
            return widen_u32x8_to_u64x8_scalar(self);
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { widen_u32x8_to_u64x8_neon(self) };
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            widen_u32x8_to_u64x8_scalar(self)
        }
    }

    #[inline(always)]
    fn shuffle(self, indices: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { shuffle_u32x8_avx2(self, indices) };
            }
            return shuffle_u32x8_scalar(self, indices);
        }

        // On ARM, try SVE first (has native permute), fall back to scalar
        // (NEON TBL requires byte-level index conversion which adds overhead)
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                return unsafe { shuffle_u32x8_sve(self, indices) };
            }
            return shuffle_u32x8_scalar(self, indices);
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            shuffle_u32x8_scalar(self, indices)
        }
    }
}

impl WideUtilsExt for u32x4 {
    type Widened = u64x4;

    #[inline(always)]
    fn widen_to_u64x8(self) -> u64x4 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { widen_u32x4_to_u64x4_avx2(self) };
            }
            return widen_u32x4_to_u64x4_scalar(self);
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { widen_u32x4_to_u64x4_neon(self) };
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            widen_u32x4_to_u64x4_scalar(self)
        }
    }

    #[inline(always)]
    fn shuffle(self, indices: Self) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                return unsafe { shuffle_u32x4_sve(self, indices) };
            }
        }
        // Scalar fallback - faster than NEON TBL for u32 (no byte-index conversion)
        shuffle_u32x4_scalar(self, indices)
    }
}

impl WideUtilsExt for u8x16 {
    type Widened = (); // u8 doesn't widen in the same way

    #[inline(always)]
    fn widen_to_u64x8(self) -> () {
        // Not applicable for u8x16
    }

    #[inline(always)]
    fn shuffle(self, indices: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("ssse3") {
                return unsafe { shuffle_u8x16_ssse3(self, indices) };
            }
            return shuffle_u8x16_scalar(self, indices);
        }

        #[cfg(target_arch = "aarch64")]
        {
            // SVE has native tbl, but NEON TBL is already optimal for u8x16
            return unsafe { shuffle_u8x16_neon(self, indices) };
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            shuffle_u8x16_scalar(self, indices)
        }
    }
}

// =============================================================================
// FromBitmask Implementations
// =============================================================================

impl FromBitmask<u64> for u64x8 {
    #[inline(always)]
    fn from_bitmask(mask: u8) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return unsafe { u64x8_from_bitmask_avx512(mask) };
            } else if is_x86_feature_detected!("avx2") {
                return unsafe { u64x8_from_bitmask_avx2(mask) };
            }
            return u64x8_from_bitmask_scalar(mask);
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { u64x8_from_bitmask_neon(mask) };
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            u64x8_from_bitmask_scalar(mask)
        }
    }
}

impl FromBitmask<u32> for u32x8 {
    #[inline(always)]
    fn from_bitmask(mask: u8) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return unsafe { u32x8_from_bitmask_avx512(mask) };
            } else if is_x86_feature_detected!("avx2") {
                return unsafe { u32x8_from_bitmask_avx2(mask) };
            }
            return u32x8_from_bitmask_scalar(mask);
        }

        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { u32x8_from_bitmask_neon(mask) };
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
use std::arch::is_x86_feature_detected;

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn split_u32x16_avx512(input: u32x16) -> (u32x8, u32x8) {
    unsafe {
        let raw = std::mem::transmute::<u32x16, __m512i>(input);
        let lo = _mm512_castsi512_si256(raw);
        let hi = _mm512_extracti64x4_epi64(raw, 1);
        (
            std::mem::transmute::<__m256i, u32x8>(lo),
            std::mem::transmute::<__m256i, u32x8>(hi),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn split_u64x8_avx512(input: u64x8) -> (u64x4, u64x4) {
    unsafe {
        let raw = std::mem::transmute::<u64x8, __m512i>(input);
        let lo = _mm512_castsi512_si256(raw);
        let hi = _mm512_extracti64x4_epi64(raw, 1);
        (
            std::mem::transmute::<__m256i, u64x4>(lo),
            std::mem::transmute::<__m256i, u64x4>(hi),
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn widen_u32x8_to_u64x8_avx512(input: u32x8) -> u64x8 {
    unsafe {
        let raw = std::mem::transmute::<u32x8, __m256i>(input);
        let widened = _mm512_cvtepu32_epi64(raw);
        std::mem::transmute::<__m512i, u64x8>(widened)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn widen_u32x8_to_u64x8_avx2(input: u32x8) -> u64x8 {
    unsafe {
        let raw = std::mem::transmute::<u32x8, __m256i>(input);
        let low = _mm256_extracti128_si256(raw, 0);
        let high = _mm256_extracti128_si256(raw, 1);
        let low_wide = _mm256_cvtepu32_epi64(low);
        let high_wide = _mm256_cvtepu32_epi64(high);
        let low_array: [u64; 4] = std::mem::transmute(low_wide);
        let high_array: [u64; 4] = std::mem::transmute(high_wide);
        u64x8::from([
            low_array[0], low_array[1], low_array[2], low_array[3],
            high_array[0], high_array[1], high_array[2], high_array[3],
        ])
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn widen_u32x4_to_u64x4_avx2(input: u32x4) -> u64x4 {
    unsafe {
        let raw = std::mem::transmute::<u32x4, __m128i>(input);
        let widened = _mm256_cvtepu32_epi64(raw);
        std::mem::transmute::<__m256i, u64x4>(widened)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn u64x8_from_bitmask_avx512(mask: u8) -> u64x8 {
    unsafe {
        let vec = _mm512_maskz_set1_epi64(mask, -1i64);
        std::mem::transmute::<__m512i, u64x8>(vec)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn u64x8_from_bitmask_avx2(mask: u8) -> u64x8 {
    let mut values = [0u64; 8];
    for i in 0..8 {
        values[i] = if (mask >> i) & 1 != 0 { u64::MAX } else { 0 };
    }
    u64x8::from(values)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn u32x8_from_bitmask_avx512(mask: u8) -> u32x8 {
    unsafe {
        let vec = _mm256_maskz_set1_epi32(mask, -1i32);
        std::mem::transmute::<__m256i, u32x8>(vec)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn u32x8_from_bitmask_avx2(mask: u8) -> u32x8 {
    let mut values = [0u32; 8];
    for i in 0..8 {
        values[i] = if (mask >> i) & 1 != 0 { u32::MAX } else { 0 };
    }
    u32x8::from(values)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn shuffle_u32x8_avx2(input: u32x8, indices: u32x8) -> u32x8 {
    unsafe {
        let raw = std::mem::transmute::<u32x8, __m256i>(input);
        let idx = std::mem::transmute::<u32x8, __m256i>(indices);
        let shuffled = _mm256_permutevar8x32_epi32(raw, idx);
        std::mem::transmute::<__m256i, u32x8>(shuffled)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "ssse3")]
unsafe fn shuffle_u8x16_ssse3(input: u8x16, indices: u8x16) -> u8x16 {
    unsafe {
        let raw = std::mem::transmute::<u8x16, __m128i>(input);
        let idx = std::mem::transmute::<u8x16, __m128i>(indices);
        let shuffled = _mm_shuffle_epi8(raw, idx);
        std::mem::transmute::<__m128i, u8x16>(shuffled)
    }
}

// =============================================================================
// ARM NEON Implementations
// =============================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn widen_u32x8_to_u64x8_neon(input: u32x8) -> u64x8 {
    let array = input.to_array();
    unsafe {
        let low_input = vld1q_u32(array.as_ptr());
        let high_input = vld1q_u32(array.as_ptr().add(4));
        let (low_0, low_1) = widen_u32x4_to_u64x4_neon_raw(low_input);
        let (high_0, high_1) = widen_u32x4_to_u64x4_neon_raw(high_input);
        let mut result = [0u64; 8];
        vst1q_u64(result.as_mut_ptr(), low_0);
        vst1q_u64(result.as_mut_ptr().add(2), low_1);
        vst1q_u64(result.as_mut_ptr().add(4), high_0);
        vst1q_u64(result.as_mut_ptr().add(6), high_1);
        u64x8::from(result)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
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
#[inline]
#[target_feature(enable = "neon")]
unsafe fn widen_u32x4_to_u64x4_neon_raw(input: uint32x4_t) -> (uint64x2_t, uint64x2_t) {
    let low = vmovl_u32(vget_low_u32(input));
    let high = vmovl_u32(vget_high_u32(input));
    (low, high)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn u64x8_from_bitmask_neon(mask: u8) -> u64x8 {
    let mut values = [0u64; 8];
    for (i, value) in values.iter_mut().enumerate() {
        *value = if (mask >> i) & 1 != 0 { u64::MAX } else { 0 };
    }
    u64x8::from(values)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn u32x8_from_bitmask_neon(mask: u8) -> u32x8 {
    let mut values = [0u32; 8];
    for (i, value) in values.iter_mut().enumerate() {
        *value = if (mask >> i) & 1 != 0 { u32::MAX } else { 0 };
    }
    u32x8::from(values)
}

/// NEON u8x16 shuffle using TBL instruction
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn shuffle_u8x16_neon(input: u8x16, indices: u8x16) -> u8x16 {
    unsafe {
        let arr = input.to_array();
        let idx_arr = indices.to_array();
        let data = vld1q_u8(arr.as_ptr());
        let idx = vld1q_u8(idx_arr.as_ptr());
        let result = vqtbl1q_u8(data, idx);
        let mut out = [0u8; 16];
        vst1q_u8(out.as_mut_ptr(), result);
        u8x16::from(out)
    }
}

// NOTE: NEON TBL-based u32x4/u32x8 shuffle implementations were removed.
// While NEON TBL is fast for u8x16 shuffles, using it for u32 shuffles requires
// converting u32 indices to byte indices (4 bytes per element) via a loop,
// which adds significant overhead. Scalar shuffle is faster for u32 types on ARM
// unless SVE is available.

// =============================================================================
// SVE Implementations (ARM Scalable Vector Extension)
// =============================================================================

/// SVE u32x4 shuffle using TBL instruction via inline assembly
/// SVE TBL does element-wise table lookup with native u32 indices (no byte conversion needed)
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "sve")]
unsafe fn shuffle_u32x4_sve(input: u32x4, indices: u32x4) -> u32x4 {
    use std::arch::asm;
    let data_arr = input.to_array();
    let idx_arr = indices.to_array();
    let mut out = [0u32; 4];

    unsafe {
        asm!(
            "ptrue p0.s, vl4",           // Predicate for 4 elements
            "ld1w {{z0.s}}, p0/z, [{data}]",  // Load data
            "ld1w {{z1.s}}, p0/z, [{idx}]",   // Load indices
            "tbl z2.s, {{z0.s}}, z1.s",       // Table lookup permute
            "st1w {{z2.s}}, p0, [{out}]",     // Store result
            data = in(reg) data_arr.as_ptr(),
            idx = in(reg) idx_arr.as_ptr(),
            out = in(reg) out.as_mut_ptr(),
            options(nostack)
        );
    }
    u32x4::from(out)
}

/// SVE u32x8 shuffle using TBL instruction via inline assembly
/// Processes as two u32x4 halves
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "sve")]
unsafe fn shuffle_u32x8_sve(input: u32x8, indices: u32x8) -> u32x8 {
    use std::arch::asm;
    let data_arr = input.to_array();
    let idx_arr = indices.to_array();
    let mut out = [0u32; 8];

    unsafe {
        // Process low half (indices 0-3 reference data 0-3)
        asm!(
            "ptrue p0.s, vl4",
            "ld1w {{z0.s}}, p0/z, [{data}]",
            "ld1w {{z1.s}}, p0/z, [{idx}]",
            "tbl z2.s, {{z0.s}}, z1.s",
            "st1w {{z2.s}}, p0, [{out}]",
            data = in(reg) data_arr.as_ptr(),
            idx = in(reg) idx_arr.as_ptr(),
            out = in(reg) out.as_mut_ptr(),
            options(nostack)
        );

        // Process high half (indices 4-7 reference data 4-7, adjusted by -4)
        asm!(
            "ptrue p0.s, vl4",
            "ld1w {{z0.s}}, p0/z, [{data}]",  // Load high data
            "ld1w {{z1.s}}, p0/z, [{idx}]",   // Load high indices
            "mov z3.s, #4",                    // Load constant 4
            "sub z1.s, z1.s, z3.s",           // Subtract 4 from indices
            "tbl z2.s, {{z0.s}}, z1.s",       // Table lookup
            "st1w {{z2.s}}, p0, [{out}]",
            data = in(reg) data_arr.as_ptr().add(4),
            idx = in(reg) idx_arr.as_ptr().add(4),
            out = in(reg) out.as_mut_ptr().add(4),
            options(nostack)
        );
    }
    u32x8::from(out)
}

// =============================================================================
// Scalar/Portable Fallback Implementations
// =============================================================================

#[inline(always)]
fn split_u32x16_cast(input: u32x16) -> (u32x8, u32x8) {
    // True zero-copy: (u32x8, u32x8) has identical layout to u32x16
    // Safety: Both are 64 bytes of contiguous u32 values
    unsafe { std::mem::transmute(input) }
}

#[inline(always)]
fn split_u64x8_cast(input: u64x8) -> (u64x4, u64x4) {
    // True zero-copy: (u64x4, u64x4) has identical layout to u64x8
    // Safety: Both are 64 bytes of contiguous u64 values
    unsafe { std::mem::transmute(input) }
}

#[allow(dead_code)]
#[inline]
fn widen_u32x8_to_u64x8_scalar(input: u32x8) -> u64x8 {
    let array = input.to_array();
    u64x8::from(array.map(|x| x as u64))
}

#[allow(dead_code)]
#[inline]
fn widen_u32x4_to_u64x4_scalar(input: u32x4) -> u64x4 {
    let array = input.to_array();
    u64x4::from(array.map(|x| x as u64))
}

#[allow(dead_code)]
#[inline]
fn u64x8_from_bitmask_scalar(mask: u8) -> u64x8 {
    let mut values = [0u64; 8];
    for (i, value) in values.iter_mut().enumerate() {
        *value = if (mask >> i) & 1 != 0 { u64::MAX } else { 0 };
    }
    u64x8::from(values)
}

#[allow(dead_code)]
#[inline]
fn u32x8_from_bitmask_scalar(mask: u8) -> u32x8 {
    let mut values = [0u32; 8];
    for (i, value) in values.iter_mut().enumerate() {
        *value = if (mask >> i) & 1 != 0 { u32::MAX } else { 0 };
    }
    u32x8::from(values)
}

#[allow(dead_code)]
#[inline]
fn shuffle_u32x8_scalar(input: u32x8, indices: u32x8) -> u32x8 {
    let arr = input.to_array();
    let idx = indices.to_array();
    let mut result = [0u32; 8];
    for i in 0..8 {
        result[i] = arr[(idx[i] & 7) as usize];
    }
    u32x8::from(result)
}

#[allow(dead_code)]
#[inline]
fn shuffle_u32x4_scalar(input: u32x4, indices: u32x4) -> u32x4 {
    let arr = input.to_array();
    let idx = indices.to_array();
    let mut result = [0u32; 4];
    for i in 0..4 {
        result[i] = arr[(idx[i] & 3) as usize];
    }
    u32x4::from(result)
}

#[allow(dead_code)]
#[inline]
fn shuffle_u8x16_scalar(input: u8x16, indices: u8x16) -> u8x16 {
    let arr = input.to_array();
    let idx = indices.to_array();
    let mut result = [0u8; 16];
    for i in 0..16 {
        let index = idx[i] as usize;
        result[i] = if index < 16 { arr[index] } else { 0 };
    }
    u8x16::from(result)
}

// =============================================================================
// Scalar Mul/Div Implementations
// =============================================================================


// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u32x8_widening() {
        let input = u32x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
        let widened: u64x8 = input.widen_to_u64x8();
        assert_eq!(widened.to_array(), [1u64, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_u32x4_widening() {
        let input = u32x4::from([1, 2, 3, 4]);
        let widened: u64x4 = input.widen_to_u64x8();
        assert_eq!(widened.to_array(), [1u64, 2, 3, 4]);
    }

    #[test]
    fn test_u64x8_from_bitmask() {
        let mask = 0b10101010u8;
        let mask_vec: u64x8 = u64x8::from_bitmask(mask);
        let expected = [0u64, u64::MAX, 0u64, u64::MAX, 0u64, u64::MAX, 0u64, u64::MAX];
        assert_eq!(mask_vec.to_array(), expected);
    }

    #[test]
    fn test_u32x8_from_bitmask() {
        let mask = 0b11000011u8;
        let mask_vec: u32x8 = u32x8::from_bitmask(mask);
        let expected = [u32::MAX, u32::MAX, 0u32, 0u32, 0u32, 0u32, u32::MAX, u32::MAX];
        assert_eq!(mask_vec.to_array(), expected);
    }

    #[test]
    fn test_edge_cases() {
        let mask_zero = 0b00000000u8;
        let vec_zero: u64x8 = u64x8::from_bitmask(mask_zero);
        assert_eq!(vec_zero.to_array(), [0u64; 8]);

        let mask_all = 0b11111111u8;
        let vec_all: u64x8 = u64x8::from_bitmask(mask_all);
        assert_eq!(vec_all.to_array(), [u64::MAX; 8]);
    }

    #[test]
    fn test_shuffle_u32x8() {
        let input = u32x8::from([10, 20, 30, 40, 50, 60, 70, 80]);

        // Identity shuffle
        let indices = u32x8::from([0, 1, 2, 3, 4, 5, 6, 7]);
        let result = input.shuffle(indices);
        assert_eq!(result.to_array(), [10, 20, 30, 40, 50, 60, 70, 80]);

        // Reverse shuffle
        let indices = u32x8::from([7, 6, 5, 4, 3, 2, 1, 0]);
        let result = input.shuffle(indices);
        assert_eq!(result.to_array(), [80, 70, 60, 50, 40, 30, 20, 10]);

        // Compress-like shuffle
        let indices = u32x8::from([1, 4, 5, 7, 0, 0, 0, 0]);
        let result = input.shuffle(indices);
        assert_eq!(result.to_array()[0..4], [20, 50, 60, 80]);
    }

    #[test]
    fn test_shuffle_u8x16() {
        let input = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        // Identity
        let indices = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let result = input.shuffle(indices);
        assert_eq!(result.to_array(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        // Reverse
        let indices = u8x16::from([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
        let result = input.shuffle(indices);
        assert_eq!(result.to_array(), [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_simd_compress_indices_table() {
        // Mask 0b10110010 = bits 1,4,5,7 set
        let mask = 0b10110010u8;
        let indices = get_compress_indices_u32x8(mask);
        let arr = indices.to_array();
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 4);
        assert_eq!(arr[2], 5);
        assert_eq!(arr[3], 7);

        // All set = identity
        let indices = get_compress_indices_u32x8(0xFF);
        assert_eq!(indices.to_array(), [0, 1, 2, 3, 4, 5, 6, 7]);

        // Test raw array access
        let raw = SHUFFLE_COMPRESS_IDX_U32X8[0b10110010];
        assert_eq!(raw[0], 1);
        assert_eq!(raw[1], 4);
    }

    #[test]
    fn test_simd_split_u32x16() {
        let input = u32x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let (lo, hi) = input.split_low_high();

        assert_eq!(lo.to_array(), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(hi.to_array(), [9, 10, 11, 12, 13, 14, 15, 16]);
    }

    #[test]
    fn test_simd_split_u64x8() {
        let input = u64x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
        let (lo, hi) = input.split_low_high();

        assert_eq!(lo.to_array(), [1, 2, 3, 4]);
        assert_eq!(hi.to_array(), [5, 6, 7, 8]);
    }

    #[test]
    fn test_shuffle_u32x4() {
        let input = u32x4::from([10, 20, 30, 40]);

        // Identity
        let indices = u32x4::from([0, 1, 2, 3]);
        let result = input.shuffle(indices);
        assert_eq!(result.to_array(), [10, 20, 30, 40]);

        // Reverse
        let indices = u32x4::from([3, 2, 1, 0]);
        let result = input.shuffle(indices);
        assert_eq!(result.to_array(), [40, 30, 20, 10]);

        // Broadcast first element
        let indices = u32x4::from([0, 0, 0, 0]);
        let result = input.shuffle(indices);
        assert_eq!(result.to_array(), [10, 10, 10, 10]);
    }

    #[test]
    fn test_double_u8x16() {
        let a = u8x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let result = a.double();
        assert_eq!(
            result.to_array(),
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        );
    }

    #[test]
    fn test_double_triple_for_x8() {
        // x.double().double().double() = x * 8
        let a = u8x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let result = a.double().double().double();
        assert_eq!(
            result.to_array(),
            [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
        );
    }

    #[test]
    fn test_double_overflow() {
        // Test wrapping: 128 * 2 = 256 wraps to 0
        let a = u8x16::splat(128);
        let result = a.double();
        assert_eq!(result.to_array(), [0u8; 16]);

        // 200 * 2 = 400 wraps to 144
        let a = u8x16::splat(200);
        let result = a.double();
        assert_eq!(result.to_array(), [144u8; 16]); // 400 & 0xFF = 144
    }
}
