//! Cross-platform prefetch intrinsics for x86 and ARM architectures.
//!
//! This module provides a unified API for prefetching memory addresses across different
//! architectures and cache levels. It uses compile-time generics to eliminate runtime
//! branches and provide direct intrinsic calls.
//!
//! # Why u8/i8 pointers?
//!
//! Prefetch instructions operate at the cache line level (typically 64 bytes) and don't
//! care about the actual data type being prefetched. They only need a memory address.
//! Using u8/i8 pointers is the standard convention because:
//! - Prefetch works on cache lines, not individual data elements
//! - The CPU prefetches entire cache lines regardless of data type
//! - u8 provides byte-level addressing which is what the hardware expects
//!
//! # Examples
//!
//! ```rust
//! use simd_lookup::prefetch::{prefetch_eight_addresses, prefetch_eight_masked, L1, NTA};
//!
//! let data = vec![0u32; 1000];
//! let offsets = [10, 20, 30, 40, 50, 60, 70, 80];
//!
//! // Prefetch 8 addresses for L1 cache
//! prefetch_eight_addresses::<_, L1>(&data, &offsets);
//!
//! // Prefetch with mask - only prefetch where mask bit is 1
//! let mask = 0b10101010; // prefetch offsets[1], [3], [5], [7]
//! prefetch_eight_masked::<_, L1>(&data, offsets, mask);
//! ```

use crate::wide_utils::{FromBitmask, WideUtilsExt};
#[cfg(target_arch = "aarch64")]
use std::arch::asm;

/// Cache level marker traits for compile-time dispatch
pub trait CacheLevel {
    /// The prefetch hint value for this cache level
    const HINT: i32;
}

/// Non-temporal access - bypass cache hierarchy
pub struct NTA;
impl CacheLevel for NTA {
    #[cfg(target_arch = "x86_64")]
    const HINT: i32 = 0; // _MM_HINT_NTA

    #[cfg(target_arch = "aarch64")]
    const HINT: i32 = 0; // pldl1strm (streaming)
}

/// L1 cache prefetch
pub struct L1;
impl CacheLevel for L1 {
    #[cfg(target_arch = "x86_64")]
    const HINT: i32 = 3; // _MM_HINT_T0

    #[cfg(target_arch = "aarch64")]
    const HINT: i32 = 1; // pldl1keep
}

/// L2 cache prefetch
pub struct L2;
impl CacheLevel for L2 {
    #[cfg(target_arch = "x86_64")]
    const HINT: i32 = 2; // _MM_HINT_T1

    #[cfg(target_arch = "aarch64")]
    const HINT: i32 = 2; // pldl2keep
}

/// L3 cache prefetch
pub struct L3;
impl CacheLevel for L3 {
    #[cfg(target_arch = "x86_64")]
    const HINT: i32 = 1; // _MM_HINT_T2

    #[cfg(target_arch = "aarch64")]
    const HINT: i32 = 3; // pldl3keep
}

/// Prefetch a single memory address for the specified cache level
#[inline(always)]
pub fn prefetch_address<T, L: CacheLevel>(base: &T, offset: u32) {
    let ptr = unsafe { (base as *const T).add(offset as usize) as *const i8 };

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        unsafe {
            match L::HINT {
                0 => _mm_prefetch(ptr, _MM_HINT_NTA),
                1 => _mm_prefetch(ptr, _MM_HINT_T2),
                2 => _mm_prefetch(ptr, _MM_HINT_T1),
                3 => _mm_prefetch(ptr, _MM_HINT_T0),
                _ => _mm_prefetch(ptr, _MM_HINT_T0),
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            match L::HINT {
                0 => asm!("prfm pldl1strm, [{0}]", in(reg) ptr), // Non-temporal (streaming)
                1 => asm!("prfm pldl1keep, [{0}]", in(reg) ptr), // L1 keep
                2 => asm!("prfm pldl2keep, [{0}]", in(reg) ptr), // L2 keep
                3 => asm!("prfm pldl3keep, [{0}]", in(reg) ptr), // L3 keep
                _ => asm!("prfm pldl1keep, [{0}]", in(reg) ptr), // Default fallback
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // No-op for unsupported architectures
        let _ = ptr;
    }
}

/// Prefetch eight memory addresses at once for the specified cache level
///
/// This function takes a base pointer and an array of 8 offsets, then prefetches
/// each calculated address. The prefetch operations are unrolled for maximum
/// performance and use SIMD address calculations where beneficial.
///
/// # Arguments
/// * `base` - Base pointer to the data structure
/// * `offsets` - Array of exactly 8 u32 offsets from the base
///
/// # Type Parameters
/// * `T` - The type of the base data structure
/// * `L` - Cache level implementing the `CacheLevel` trait
#[inline(always)]
pub fn prefetch_eight_addresses<T, L: CacheLevel>(base: &T, offsets: &[u32; 8]) {
    let base_ptr = base as *const T;

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        unsafe {
            // Calculate all 8 addresses using typed pointer arithmetic (no multiplication)
            let ptrs = [
                (base_ptr.add(offsets[0] as usize) as *const i8),
                (base_ptr.add(offsets[1] as usize) as *const i8),
                (base_ptr.add(offsets[2] as usize) as *const i8),
                (base_ptr.add(offsets[3] as usize) as *const i8),
                (base_ptr.add(offsets[4] as usize) as *const i8),
                (base_ptr.add(offsets[5] as usize) as *const i8),
                (base_ptr.add(offsets[6] as usize) as *const i8),
                (base_ptr.add(offsets[7] as usize) as *const i8),
            ];

            // Unroll all 8 prefetch operations - no branches
            match L::HINT {
                0 => {
                    _mm_prefetch(ptrs[0], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[1], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[2], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[3], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[4], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[5], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[6], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[7], _MM_HINT_NTA);
                }
                1 => {
                    _mm_prefetch(ptrs[0], _MM_HINT_T2);
                    _mm_prefetch(ptrs[1], _MM_HINT_T2);
                    _mm_prefetch(ptrs[2], _MM_HINT_T2);
                    _mm_prefetch(ptrs[3], _MM_HINT_T2);
                    _mm_prefetch(ptrs[4], _MM_HINT_T2);
                    _mm_prefetch(ptrs[5], _MM_HINT_T2);
                    _mm_prefetch(ptrs[6], _MM_HINT_T2);
                    _mm_prefetch(ptrs[7], _MM_HINT_T2);
                }
                2 => {
                    _mm_prefetch(ptrs[0], _MM_HINT_T1);
                    _mm_prefetch(ptrs[1], _MM_HINT_T1);
                    _mm_prefetch(ptrs[2], _MM_HINT_T1);
                    _mm_prefetch(ptrs[3], _MM_HINT_T1);
                    _mm_prefetch(ptrs[4], _MM_HINT_T1);
                    _mm_prefetch(ptrs[5], _MM_HINT_T1);
                    _mm_prefetch(ptrs[6], _MM_HINT_T1);
                    _mm_prefetch(ptrs[7], _MM_HINT_T1);
                }
                3 => {
                    _mm_prefetch(ptrs[0], _MM_HINT_T0);
                    _mm_prefetch(ptrs[1], _MM_HINT_T0);
                    _mm_prefetch(ptrs[2], _MM_HINT_T0);
                    _mm_prefetch(ptrs[3], _MM_HINT_T0);
                    _mm_prefetch(ptrs[4], _MM_HINT_T0);
                    _mm_prefetch(ptrs[5], _MM_HINT_T0);
                    _mm_prefetch(ptrs[6], _MM_HINT_T0);
                    _mm_prefetch(ptrs[7], _MM_HINT_T0);
                }
                _ => {
                    _mm_prefetch(ptrs[0], _MM_HINT_T0);
                    _mm_prefetch(ptrs[1], _MM_HINT_T0);
                    _mm_prefetch(ptrs[2], _MM_HINT_T0);
                    _mm_prefetch(ptrs[3], _MM_HINT_T0);
                    _mm_prefetch(ptrs[4], _MM_HINT_T0);
                    _mm_prefetch(ptrs[5], _MM_HINT_T0);
                    _mm_prefetch(ptrs[6], _MM_HINT_T0);
                    _mm_prefetch(ptrs[7], _MM_HINT_T0);
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            // Calculate all 8 addresses using typed pointer arithmetic (no multiplication)
            let addrs = [
                base_ptr.add(offsets[0] as usize) as *const u8,
                base_ptr.add(offsets[1] as usize) as *const u8,
                base_ptr.add(offsets[2] as usize) as *const u8,
                base_ptr.add(offsets[3] as usize) as *const u8,
                base_ptr.add(offsets[4] as usize) as *const u8,
                base_ptr.add(offsets[5] as usize) as *const u8,
                base_ptr.add(offsets[6] as usize) as *const u8,
                base_ptr.add(offsets[7] as usize) as *const u8,
            ];

            // Unroll all 8 prefetch operations with correct cache level
            match L::HINT {
                0 => {
                    // Non-temporal (streaming)
                    asm!("prfm pldl1strm, [{0}]", in(reg) addrs[0]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) addrs[1]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) addrs[2]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) addrs[3]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) addrs[4]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) addrs[5]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) addrs[6]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) addrs[7]);
                }
                1 => {
                    // L1 keep
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[0]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[1]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[2]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[3]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[4]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[5]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[6]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[7]);
                }
                2 => {
                    // L2 keep
                    asm!("prfm pldl2keep, [{0}]", in(reg) addrs[0]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) addrs[1]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) addrs[2]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) addrs[3]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) addrs[4]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) addrs[5]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) addrs[6]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) addrs[7]);
                }
                3 => {
                    // L3 keep
                    asm!("prfm pldl3keep, [{0}]", in(reg) addrs[0]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) addrs[1]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) addrs[2]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) addrs[3]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) addrs[4]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) addrs[5]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) addrs[6]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) addrs[7]);
                }
                _ => {
                    // Default fallback
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[0]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[1]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[2]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[3]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[4]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[5]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[6]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) addrs[7]);
                }
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // No-op for unsupported architectures
        let _ = (base_ptr, offsets);
    }
}

/// Prefetch eight addresses with a bitmask to control which addresses to prefetch
///
/// This function uses **true SIMD operations** with the `wide` crate to ensure minimum CPU cycles.
///
/// ## SIMD Implementation Details:
/// 1. **Address Calculation**: Uses `wide::u64x8` SIMD vectors to calculate all 8 addresses simultaneously
/// 2. **Mask Conversion**: Converts 8-bit mask to SIMD mask vector (0 or u64::MAX per lane)
/// 3. **SIMD Blend**: Uses bitwise operations `(real_addrs & mask) | (dummy_addrs & !mask)` for branchless selection
/// 4. **Consistent Timing**: Always executes exactly 8 prefetch instructions regardless of mask pattern
///
/// For each bit in the mask:
/// - If bit is 1: prefetch from base + offset[i]
/// - If bit is 0: prefetch from dummy address (base address - offset 0)
///
/// This achieves true branchless execution using SIMD blend operations instead of scalar conditionals.
///
/// # Arguments
/// * `base` - Base pointer to the data structure
/// * `offsets` - Array of exactly 8 u32 offsets from the base
/// * `mask` - 8-bit mask where bit i controls whether offset[i] is prefetched
///
/// # Type Parameters
/// * `T` - The type of the base data structure
/// * `L` - Cache level implementing the `CacheLevel` trait
#[inline(always)]
pub fn prefetch_eight_masked<T, L: CacheLevel>(base: &T, offsets: [u32; 8], mask: u8) {
    let base_ptr = base as *const T;

    // Use SIMD for address calculation and masking
    let base_addr = base_ptr as u64;
    let base_simd = wide::u64x8::splat(base_addr);

    // Multiply offsets by size_of<T>, blend with dummy value of 0, and widen to u64x8
    let offsets_u32_simd = wide::u32x8::from(offsets) * (std::mem::size_of::<T>() as u32);
    let zero_offsets_simd = wide::u32x8::splat(0);
    let mask_simd = wide::u32x8::from_bitmask(mask);
    // Blend API: mask.blend(true_value, false_value)
    let blended_offsets_simd = mask_simd.blend(offsets_u32_simd, zero_offsets_simd);

    // Calculate final addresses
    let offsets_u64_simd = blended_offsets_simd.widen_to_u64x8();
    let selected_addrs_simd = base_simd + offsets_u64_simd;

    let selected_addrs = selected_addrs_simd.to_array();

    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        unsafe {
            // Convert u64 addresses to i8 pointers for prefetch
            let ptrs = [
                selected_addrs[0] as *const i8,
                selected_addrs[1] as *const i8,
                selected_addrs[2] as *const i8,
                selected_addrs[3] as *const i8,
                selected_addrs[4] as *const i8,
                selected_addrs[5] as *const i8,
                selected_addrs[6] as *const i8,
                selected_addrs[7] as *const i8,
            ];

            // Unroll all 8 prefetch operations - consistent timing
            match L::HINT {
                0 => {
                    _mm_prefetch(ptrs[0], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[1], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[2], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[3], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[4], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[5], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[6], _MM_HINT_NTA);
                    _mm_prefetch(ptrs[7], _MM_HINT_NTA);
                }
                1 => {
                    _mm_prefetch(ptrs[0], _MM_HINT_T2);
                    _mm_prefetch(ptrs[1], _MM_HINT_T2);
                    _mm_prefetch(ptrs[2], _MM_HINT_T2);
                    _mm_prefetch(ptrs[3], _MM_HINT_T2);
                    _mm_prefetch(ptrs[4], _MM_HINT_T2);
                    _mm_prefetch(ptrs[5], _MM_HINT_T2);
                    _mm_prefetch(ptrs[6], _MM_HINT_T2);
                    _mm_prefetch(ptrs[7], _MM_HINT_T2);
                }
                2 => {
                    _mm_prefetch(ptrs[0], _MM_HINT_T1);
                    _mm_prefetch(ptrs[1], _MM_HINT_T1);
                    _mm_prefetch(ptrs[2], _MM_HINT_T1);
                    _mm_prefetch(ptrs[3], _MM_HINT_T1);
                    _mm_prefetch(ptrs[4], _MM_HINT_T1);
                    _mm_prefetch(ptrs[5], _MM_HINT_T1);
                    _mm_prefetch(ptrs[6], _MM_HINT_T1);
                    _mm_prefetch(ptrs[7], _MM_HINT_T1);
                }
                3 => {
                    _mm_prefetch(ptrs[0], _MM_HINT_T0);
                    _mm_prefetch(ptrs[1], _MM_HINT_T0);
                    _mm_prefetch(ptrs[2], _MM_HINT_T0);
                    _mm_prefetch(ptrs[3], _MM_HINT_T0);
                    _mm_prefetch(ptrs[4], _MM_HINT_T0);
                    _mm_prefetch(ptrs[5], _MM_HINT_T0);
                    _mm_prefetch(ptrs[6], _MM_HINT_T0);
                    _mm_prefetch(ptrs[7], _MM_HINT_T0);
                }
                _ => {
                    _mm_prefetch(ptrs[0], _MM_HINT_T0);
                    _mm_prefetch(ptrs[1], _MM_HINT_T0);
                    _mm_prefetch(ptrs[2], _MM_HINT_T0);
                    _mm_prefetch(ptrs[3], _MM_HINT_T0);
                    _mm_prefetch(ptrs[4], _MM_HINT_T0);
                    _mm_prefetch(ptrs[5], _MM_HINT_T0);
                    _mm_prefetch(ptrs[6], _MM_HINT_T0);
                    _mm_prefetch(ptrs[7], _MM_HINT_T0);
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            // Convert u64 addresses to u8 pointers for prefetch
            let ptrs = [
                selected_addrs[0] as *const u8,
                selected_addrs[1] as *const u8,
                selected_addrs[2] as *const u8,
                selected_addrs[3] as *const u8,
                selected_addrs[4] as *const u8,
                selected_addrs[5] as *const u8,
                selected_addrs[6] as *const u8,
                selected_addrs[7] as *const u8,
            ];

            // Unroll all 8 prefetch operations with correct cache level
            match L::HINT {
                0 => {
                    // Non-temporal (streaming)
                    asm!("prfm pldl1strm, [{0}]", in(reg) ptrs[0]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) ptrs[1]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) ptrs[2]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) ptrs[3]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) ptrs[4]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) ptrs[5]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) ptrs[6]);
                    asm!("prfm pldl1strm, [{0}]", in(reg) ptrs[7]);
                }
                1 => {
                    // L1 keep
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[0]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[1]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[2]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[3]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[4]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[5]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[6]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[7]);
                }
                2 => {
                    // L2 keep
                    asm!("prfm pldl2keep, [{0}]", in(reg) ptrs[0]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) ptrs[1]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) ptrs[2]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) ptrs[3]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) ptrs[4]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) ptrs[5]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) ptrs[6]);
                    asm!("prfm pldl2keep, [{0}]", in(reg) ptrs[7]);
                }
                3 => {
                    // L3 keep
                    asm!("prfm pldl3keep, [{0}]", in(reg) ptrs[0]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) ptrs[1]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) ptrs[2]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) ptrs[3]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) ptrs[4]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) ptrs[5]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) ptrs[6]);
                    asm!("prfm pldl3keep, [{0}]", in(reg) ptrs[7]);
                }
                _ => {
                    // Default fallback
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[0]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[1]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[2]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[3]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[4]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[5]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[6]);
                    asm!("prfm pldl1keep, [{0}]", in(reg) ptrs[7]);
                }
            }
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        // No-op for unsupported architectures
        let _ = selected_addrs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_level_constants() {
        // Test that cache level constants are defined
        assert_eq!(NTA::HINT >= 0, true);
        assert_eq!(L1::HINT >= 0, true);
        assert_eq!(L2::HINT >= 0, true);
        assert_eq!(L3::HINT >= 0, true);
    }

    #[test]
    fn test_prefetch_single_address() {
        let data = vec![0u32; 100];

        // These should not panic or crash
        prefetch_address::<_, NTA>(&data, 10);
        prefetch_address::<_, L1>(&data, 20);
        prefetch_address::<_, L2>(&data, 30);
        prefetch_address::<_, L3>(&data, 40);
    }

    #[test]
    fn test_prefetch_eight_addresses() {
        let data = vec![0u32; 100];
        let offsets = [10, 20, 30, 40, 50, 60, 70, 80];

        // These should not panic or crash
        prefetch_eight_addresses::<_, NTA>(&data, &offsets);
        prefetch_eight_addresses::<_, L1>(&data, &offsets);
        prefetch_eight_addresses::<_, L2>(&data, &offsets);
        prefetch_eight_addresses::<_, L3>(&data, &offsets);
    }

    #[test]
    fn test_prefetch_eight_masked() {
        let data = vec![0u32; 100];
        let offsets = [10, 20, 30, 40, 50, 60, 70, 80];

        // Test different mask patterns
        prefetch_eight_masked::<_, L1>(&data, offsets, 0xFF); // All bits set
        prefetch_eight_masked::<_, L1>(&data, offsets, 0x00); // No bits set
        prefetch_eight_masked::<_, L1>(&data, offsets, 0xAA); // Alternating pattern
        prefetch_eight_masked::<_, L1>(&data, offsets, 0x55); // Alternating pattern
        prefetch_eight_masked::<_, L1>(&data, offsets, 0x0F); // First 4 bits
        prefetch_eight_masked::<_, L1>(&data, offsets, 0xF0); // Last 4 bits
    }

    #[test]
    fn test_different_data_types() {
        let u32_data = vec![0u32; 100];
        let u64_data = vec![0u64; 100];
        let f32_data = vec![0.0f32; 100];

        let offsets = [1, 2, 3, 4, 5, 6, 7, 8];

        prefetch_eight_addresses::<_, L1>(&u32_data, &offsets);
        prefetch_eight_addresses::<_, L1>(&u64_data, &offsets);
        prefetch_eight_addresses::<_, L1>(&f32_data, &offsets);

        prefetch_eight_masked::<_, L1>(&u32_data, offsets, 0xFF);
        prefetch_eight_masked::<_, L1>(&u64_data, offsets, 0xAA);
        prefetch_eight_masked::<_, L1>(&f32_data, offsets, 0x55);
    }
}
