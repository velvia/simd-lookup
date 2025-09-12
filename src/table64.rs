//! Fast SIMD enabled lookup table with <= 64 u8 values
//!

    use simd_aligned::arch::u8x16;

    //------------------- SIMD small table lookup functions (ARM NEON VTBL etc.) ---------------------------------------
    // The idea is optimized small table (say <64 entries) lookup, which can be done in only a few instructions.
    // We don't need these right now, they are intended for fast SIMD pattern detection algorithms, but not as high priority.
    // Just writing them down so we can preserve some AI output.

    // This is from the ChatGPt chat: https://chatgpt.com/share/e/68c20b23-4ba8-8008-9d52-cdbbaba08b43

    /// A SIMD-optimized 64-entry lookup table, able to do extremely efficient lookups in ARM NEON and Intel AVX-512VBMI
    pub struct Table64 {
        #[cfg(target_arch = "aarch64")]
        neon_tbl: core::arch::aarch64::uint8x16x4_t,

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        bytes: [u8; 64],

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        zmm: Option<core::arch::x86_64::__m512i>, // preloaded 64B table for AVX-512VBMI
    }

    impl Table64 {
        pub fn new(table: &[u8; 64]) -> Self {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                let zmm = if is_x86_avx512_vbmi() {
                    unsafe {
                        let z = core::arch::x86_64::_mm512_loadu_si512(table.as_ptr() as *const _);
                        Some(z)
                    }
                } else {
                    None
                };

                Self {
                    bytes: *table,
                    zmm,
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                Self {
                    neon_tbl: unsafe {
                        use core::arch::aarch64::*;
                        let t0 = vld1q_u8(table.as_ptr());
                        let t1 = vld1q_u8(table.as_ptr().add(16));
                        let t2 = vld1q_u8(table.as_ptr().add(32));
                        let t3 = vld1q_u8(table.as_ptr().add(48));
                        uint8x16x4_t(t0, t1, t2, t3)
                    },
                }
            }
        }

        /// Dynamic lookup: each byte of `idx[k]` (0..63) selects from this 64B table.
        /// - Requires: `idx.len() == out.len()`
        /// - No element tails (I/O is in whole `u8x16` blocks).
        #[inline]
        pub fn lookup(&self, idx: &[u8x16], out: &mut [u8x16]) {
            assert_eq!(idx.len(), out.len());

            #[cfg(target_arch = "aarch64")]
            unsafe {
                use core::arch::aarch64::*;
                // Treat &[u8x16] as a flat &[u8] for direct loads/stores.
                let idx_bytes = idx.as_ptr() as *const u8;
                let out_bytes = out.as_mut_ptr() as *mut u8;

                for b in 0..idx.len() {
                    let i_ptr = idx_bytes.add(b * 16);
                    let o_ptr = out_bytes.add(b * 16);

                    let i = vld1q_u8(i_ptr);
                    let r = vqtbl4q_u8(self.neon_tbl, i); // 64-entry dynamic table
                    vst1q_u8(o_ptr, r);
                }
            }

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            unsafe {
                use core::arch::x86_64::*;

                let mut i = 0usize;
                if let Some(tzmm) = self.zmm {
                    // Process 4×u8x16 at a time (64 bytes) with one vpermb.
                    let idx_bytes = idx.as_ptr() as *const u8;
                    let out_bytes = out.as_mut_ptr() as *mut u8;

                    while i + 4 <= idx.len() {
                        let off = i * 16;
                        let iv = _mm512_loadu_si512(idx_bytes.add(off) as *const __m512i);
                        let rv = _mm512_permutexvar_epi8(iv, tzmm);
                        _mm512_storeu_si512(out_bytes.add(off) as *mut __m512i, rv);
                        i += 4;
                    }
                }

                // Handle remainder blocks — scalar per 16B block; still no per-byte tails.
                for k in i..idx.len() {
                    out[k] = scalar_lookup_1x16(&self.bytes, idx[k]);
                }
            }

            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
            compile_error!("Table64::lookup is implemented for aarch64 (NEON) and x86/x86_64 (AVX-512VBMI).");
        }
    }

    // ------------------
    // Helpers
    // ------------------

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    fn is_x86_avx512_vbmi() -> bool {
        use std::arch::is_x86_feature_detected as det;
        det!("avx512bw") && det!("avx512vbmi")
    }

    /// Scalar per-vector fallback: takes/returns `u8x16`; no element tails.
    /// Preconditions: every lane < 64.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    fn scalar_lookup_1x16(table: &[u8; 64], idx: u8x16) -> u8x16 {
        let i = idx.to_array();
        debug_assert!(i.iter().all(|&x| x < 64));
        let mut out = [0u8; 16];
        // Manual unroll to encourage ILP.
        out[0]  = table[i[0]  as usize];
        out[1]  = table[i[1]  as usize];
        out[2]  = table[i[2]  as usize];
        out[3]  = table[i[3]  as usize];
        out[4]  = table[i[4]  as usize];
        out[5]  = table[i[5]  as usize];
        out[6]  = table[i[6]  as usize];
        out[7]  = table[i[7]  as usize];
        out[8]  = table[i[8]  as usize];
        out[9]  = table[i[9]  as usize];
        out[10] = table[i[10] as usize];
        out[11] = table[i[11] as usize];
        out[12] = table[i[12] as usize];
        out[13] = table[i[13] as usize];
        out[14] = table[i[14] as usize];
        out[15] = table[i[15] as usize];
        u8x16::from(out)
    }
