//! SIMD enabled efficient small table lookups - for 64 entries or 64K entries.
//! May be 2-D lookups as well.

use crate::wide_utils::WideUtilsExt;
use wide::u8x16;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{uint8x16x4_t, vld1q_u8, vqtbl4q_u8, vst1q_u8};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::{
    __m512i, _mm512_loadu_si512, _mm512_permutexvar_epi8, _mm512_storeu_si512,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::is_x86_feature_detected as det;

//------------------- SIMD small table lookup functions (ARM NEON VTBL etc.) ---------------------------------------
// The idea is optimized small table (say <=64 entries) lookup, which can be done in only a few instructions.
// Or, you can think of it as an 8x8 lookup table.

/// A SIMD-optimized 64-entry lookup table, able to do extremely efficient lookups in ARM NEON and Intel AVX-512VBMI.
///
/// # 2D Interpretation
///
/// `Table64` can also be viewed as an 8×8 two-dimensional table stored in row-major order:
///
/// ```text
///        col 0  col 1  col 2  col 3  col 4  col 5  col 6  col 7
/// row 0:   0      1      2      3      4      5      6      7
/// row 1:   8      9     10     11     12     13     14     15
/// row 2:  16     17     18     19     20     21     22     23
/// row 3:  24     25     26     27     28     29     30     31
/// row 4:  32     33     34     35     36     37     38     39
/// row 5:  40     41     42     43     44     45     46     47
/// row 6:  48     49     50     51     52     53     54     55
/// row 7:  56     57     58     59     60     61     62     63
/// ```
///
/// Use [`lookup_one_2d`](Self::lookup_one_2d) to perform lookups using (row, column) coordinates.
pub struct Table64 {
    #[cfg(target_arch = "aarch64")]
    neon_tbl: uint8x16x4_t,

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    bytes: [u8; 64],

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    zmm: Option<__m512i>, // preloaded 64B table for AVX-512VBMI
}

impl Table64 {
    #[inline]
    pub fn new(table: &[u8; 64]) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let zmm = if is_x86_avx512_vbmi() {
                unsafe {
                    let z = _mm512_loadu_si512(table.as_ptr() as *const _);
                    Some(z)
                }
            } else {
                None
            };

            Self { bytes: *table, zmm }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                neon_tbl: unsafe {
                    let t0 = vld1q_u8(table.as_ptr());
                    let t1 = vld1q_u8(table.as_ptr().add(16));
                    let t2 = vld1q_u8(table.as_ptr().add(32));
                    let t3 = vld1q_u8(table.as_ptr().add(48));
                    uint8x16x4_t(t0, t1, t2, t3)
                },
            }
        }
    }

    /// Single-vector lookup: each byte of `idx` (0..63) selects from this 64B table.
    /// Returns a `u8x16` with the looked-up values.
    #[inline]
    pub fn lookup_one(&self, idx: u8x16) -> u8x16 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let i = vld1q_u8(idx.as_array().as_ptr());
            let r = vqtbl4q_u8(self.neon_tbl, i);
            let mut out = [0u8; 16];
            vst1q_u8(out.as_mut_ptr(), r);
            u8x16::from(out)
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some(tzmm) = self.zmm {
                unsafe {
                    let iv = _mm512_loadu_si512(idx.as_array().as_ptr() as *const __m512i);
                    let rv = _mm512_permutexvar_epi8(iv, tzmm);
                    let mut out = [0u8; 64];
                    _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, rv);
                    // Only take the first 16 bytes
                    let mut result = [0u8; 16];
                    result.copy_from_slice(&out[0..16]);
                    u8x16::from(result)
                }
            } else {
                scalar_lookup_1x16(&self.bytes, idx)
            }
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
        compile_error!(
            "Table64::lookup_one is implemented for aarch64 (NEON) and x86/x86_64 (AVX-512VBMI)."
        );
    }

    /// 2D lookup: treats the 64-entry table as an 8×8 row-major matrix.
    ///
    /// Each lane computes `index = row * 8 + col` and looks up the corresponding value.
    ///
    /// # Arguments
    /// - `rows`: Row indices (0..7) for each of the 16 lanes
    /// - `cols`: Column indices (0..7) for each of the 16 lanes
    ///
    /// # Panics (debug only)
    /// Debug-asserts that all row and column values are in range 0..8.
    ///
    /// # Example
    /// ```ignore
    /// let table = Table64::new(&data);
    /// let rows = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
    /// let cols = u8x16::from([0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7]);
    /// let result = table.lookup_one_2d(rows, cols);
    /// // Looks up indices [0, 8, 16, 24, 32, 40, 48, 56, 7, 15, 23, 31, 39, 47, 55, 63]
    /// ```
    #[inline]
    pub fn lookup_one_2d(&self, rows: u8x16, cols: u8x16) -> u8x16 {
        debug_assert!(
            rows.to_array().iter().all(|&r| r < 8),
            "All row indices must be < 8"
        );
        debug_assert!(
            cols.to_array().iter().all(|&c| c < 8),
            "All column indices must be < 8"
        );

        // index = row * 8 + col
        // Use double().double().double() for efficient ×8 via SIMD addition
        // x86-64 does not have SIMD support for u8 multiply unfortunately
        let idx = rows.double().double().double() + cols;
        self.lookup_one(idx)
    }

    /// Dynamic lookup: each byte of `idx[k]` (0..63) selects from this 64B table.
    /// - Requires: `idx.len() == out.len()`
    /// - No element tails (I/O is in whole `u8x16` blocks).
    #[inline]
    pub fn lookup(&self, idx: &[u8x16], out: &mut [u8x16]) {
        assert_eq!(idx.len(), out.len());

        #[cfg(target_arch = "aarch64")]
        unsafe {
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
        compile_error!(
            "Table64::lookup is implemented for aarch64 (NEON) and x86/x86_64 (AVX-512VBMI)."
        );
    }
}

// ------------------
// Helpers
// ------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn is_x86_avx512_vbmi() -> bool {
    det!("avx512bw") && det!("avx512vbmi")
}

/// Scalar per-vector fallback: takes/returns `u8x16`; no element tails.
/// Preconditions: every lane < 64.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn scalar_lookup_1x16(table: &[u8; 64], idx: u8x16) -> u8x16 {
    let i = idx.to_array();
    debug_assert!(i.iter().all(|&x| x < 64));
    let out = [
        table[i[0] as usize],
        table[i[1] as usize],
        table[i[2] as usize],
        table[i[3] as usize],
        table[i[4] as usize],
        table[i[5] as usize],
        table[i[6] as usize],
        table[i[7] as usize],
        table[i[8] as usize],
        table[i[9] as usize],
        table[i[10] as usize],
        table[i[11] as usize],
        table[i[12] as usize],
        table[i[13] as usize],
        table[i[14] as usize],
        table[i[15] as usize],
    ];
    u8x16::from(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_table() -> [u8; 64] {
        let mut table = [0u8; 64];
        for i in 0..64 {
            table[i] = (i * 3 + 7) as u8; // Pattern: 7, 10, 13, 16, ...
        }
        table
    }

    #[test]
    fn test_table64_new() {
        let table_data = create_test_table();
        let _table = Table64::new(&table_data);
        // Just ensure construction doesn't panic
    }

    #[test]
    fn test_lookup_one_basic() {
        let table_data = create_test_table();
        let table = Table64::new(&table_data);

        // Lookup indices 0-15
        let idx = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let result = table.lookup_one(idx);
        let result_arr = result.to_array();

        // Verify each lookup
        for i in 0..16 {
            assert_eq!(
                result_arr[i], table_data[i],
                "Mismatch at index {}: expected {}, got {}",
                i, table_data[i], result_arr[i]
            );
        }
    }

    #[test]
    fn test_lookup_one_scattered_indices() {
        let table_data = create_test_table();
        let table = Table64::new(&table_data);

        // Scattered indices across the table
        let idx = u8x16::from([0, 63, 32, 16, 48, 1, 62, 31, 15, 47, 8, 56, 4, 60, 20, 40]);
        let result = table.lookup_one(idx);
        let result_arr = result.to_array();
        let idx_arr = idx.to_array();

        for i in 0..16 {
            assert_eq!(
                result_arr[i],
                table_data[idx_arr[i] as usize],
                "Mismatch at position {}: idx={}, expected {}, got {}",
                i,
                idx_arr[i],
                table_data[idx_arr[i] as usize],
                result_arr[i]
            );
        }
    }

    #[test]
    fn test_lookup_one_all_same_index() {
        let table_data = create_test_table();
        let table = Table64::new(&table_data);

        // All indices are the same
        let idx = u8x16::splat(42);
        let result = table.lookup_one(idx);
        let result_arr = result.to_array();

        let expected = table_data[42];
        for i in 0..16 {
            assert_eq!(
                result_arr[i], expected,
                "All lookups should return the same value"
            );
        }
    }

    #[test]
    fn test_lookup_batch() {
        let table_data = create_test_table();
        let table = Table64::new(&table_data);

        let indices = vec![
            u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            u8x16::from([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
            u8x16::from([32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]),
            u8x16::from([48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]),
        ];
        let mut output = vec![u8x16::splat(0); 4];

        table.lookup(&indices, &mut output);

        // Verify all lookups
        for (vec_idx, out_vec) in output.iter().enumerate() {
            let out_arr = out_vec.to_array();
            for lane in 0..16 {
                let table_idx = vec_idx * 16 + lane;
                assert_eq!(
                    out_arr[lane], table_data[table_idx],
                    "Mismatch at vec {}, lane {}: expected {}, got {}",
                    vec_idx, lane, table_data[table_idx], out_arr[lane]
                );
            }
        }
    }

    #[test]
    fn test_lookup_one_matches_lookup_batch() {
        let table_data = create_test_table();
        let table = Table64::new(&table_data);

        let idx = u8x16::from([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 0, 32, 63, 1]);

        // Single lookup
        let single_result = table.lookup_one(idx);

        // Batch lookup with single element
        let mut batch_output = vec![u8x16::splat(0); 1];
        table.lookup(&[idx], &mut batch_output);

        assert_eq!(
            single_result.to_array(),
            batch_output[0].to_array(),
            "lookup_one and lookup should produce the same result"
        );
    }

    #[test]
    fn test_identity_table() {
        // Create an identity table where table[i] = i
        let mut table_data = [0u8; 64];
        for i in 0..64 {
            table_data[i] = i as u8;
        }
        let table = Table64::new(&table_data);

        let idx = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let result = table.lookup_one(idx);

        assert_eq!(idx.to_array(), result.to_array(), "Identity table should return input indices");
    }

    // ==================== 2D Lookup Tests ====================

    /// Create an 8x8 table where table[row][col] = row * 10 + col
    /// This makes it easy to verify 2D lookups: result should be row*10 + col
    fn create_2d_test_table() -> [u8; 64] {
        let mut table = [0u8; 64];
        for row in 0..8 {
            for col in 0..8 {
                table[row * 8 + col] = (row * 10 + col) as u8;
            }
        }
        table
    }

    #[test]
    fn test_lookup_one_2d_basic() {
        let table_data = create_2d_test_table();
        let table = Table64::new(&table_data);

        // Lookup first row (row=0, cols=0..7) and second row (row=1, cols=0..7)
        let rows = u8x16::from([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]);
        let cols = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);

        let result = table.lookup_one_2d(rows, cols);
        let result_arr = result.to_array();

        // First 8: row 0, should be 0, 1, 2, 3, 4, 5, 6, 7
        for col in 0..8 {
            assert_eq!(result_arr[col], col as u8, "Row 0, col {}", col);
        }
        // Next 8: row 1, should be 10, 11, 12, 13, 14, 15, 16, 17
        for col in 0..8 {
            assert_eq!(result_arr[8 + col], (10 + col) as u8, "Row 1, col {}", col);
        }
    }

    #[test]
    fn test_lookup_one_2d_diagonal() {
        let table_data = create_2d_test_table();
        let table = Table64::new(&table_data);

        // Diagonal: (0,0), (1,1), (2,2), ..., (7,7), then reverse diagonal
        let rows = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0]);
        let cols = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);

        let result = table.lookup_one_2d(rows, cols);
        let result_arr = result.to_array();

        // Main diagonal: row*10 + col where row == col
        for i in 0..8 {
            let expected = (i * 10 + i) as u8; // 0, 11, 22, 33, 44, 55, 66, 77
            assert_eq!(result_arr[i], expected, "Main diagonal position {}", i);
        }

        // Anti-diagonal part: row=7-i, col=i
        let expected_anti = [70, 61, 52, 43, 34, 25, 16, 7u8];
        for i in 0..8 {
            assert_eq!(result_arr[8 + i], expected_anti[i], "Anti-diagonal position {}", i);
        }
    }

    #[test]
    fn test_lookup_one_2d_corners() {
        let table_data = create_2d_test_table();
        let table = Table64::new(&table_data);

        // Test all four corners repeated
        let rows = u8x16::from([0, 0, 7, 7, 0, 0, 7, 7, 0, 0, 7, 7, 0, 0, 7, 7]);
        let cols = u8x16::from([0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7, 0, 7]);

        let result = table.lookup_one_2d(rows, cols);
        let result_arr = result.to_array();

        // Expected: (0,0)=0, (0,7)=7, (7,0)=70, (7,7)=77
        let expected = [0u8, 7, 70, 77, 0, 7, 70, 77, 0, 7, 70, 77, 0, 7, 70, 77];
        assert_eq!(result_arr, expected, "Corner lookups");
    }

    #[test]
    fn test_lookup_one_2d_same_row() {
        let table_data = create_2d_test_table();
        let table = Table64::new(&table_data);

        // All from row 5
        let rows = u8x16::splat(5);
        let cols = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0]);

        let result = table.lookup_one_2d(rows, cols);
        let result_arr = result.to_array();
        let cols_arr = cols.to_array();

        for i in 0..16 {
            let expected = (50 + cols_arr[i]) as u8;
            assert_eq!(result_arr[i], expected, "Row 5, col {}", cols_arr[i]);
        }
    }

    #[test]
    fn test_lookup_one_2d_same_col() {
        let table_data = create_2d_test_table();
        let table = Table64::new(&table_data);

        // All from column 3
        let rows = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]);
        let cols = u8x16::splat(3);

        let result = table.lookup_one_2d(rows, cols);
        let result_arr = result.to_array();

        // Column 3: 3, 13, 23, 33, 43, 53, 63, 73
        for i in 0..8 {
            let expected = (i * 10 + 3) as u8;
            assert_eq!(result_arr[i], expected, "Row {}, col 3", i);
            assert_eq!(result_arr[8 + i], expected, "Row {}, col 3 (second half)", i);
        }
    }

    #[test]
    fn test_lookup_one_2d_matches_lookup_one() {
        let table_data = create_2d_test_table();
        let table = Table64::new(&table_data);

        // Random (row, col) pairs
        let rows = u8x16::from([0, 3, 7, 2, 5, 1, 6, 4, 7, 0, 3, 5, 2, 6, 1, 4]);
        let cols = u8x16::from([5, 2, 0, 7, 3, 6, 1, 4, 7, 0, 4, 2, 6, 3, 5, 1]);

        // Compute expected indices manually
        let rows_arr = rows.to_array();
        let cols_arr = cols.to_array();
        let mut expected_idx = [0u8; 16];
        for i in 0..16 {
            expected_idx[i] = rows_arr[i] * 8 + cols_arr[i];
        }

        let result_2d = table.lookup_one_2d(rows, cols);
        let result_1d = table.lookup_one(u8x16::from(expected_idx));

        assert_eq!(
            result_2d.to_array(),
            result_1d.to_array(),
            "lookup_one_2d should match lookup_one with computed indices"
        );
    }
}

