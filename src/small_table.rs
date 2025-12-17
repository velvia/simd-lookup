//! SIMD enabled efficient small table lookups - for 64 entries or 64K entries.
//! May be 2-D lookups as well.
//!
//! # CPU Feature Requirements
//!
//! ## Table64 (64-entry lookup table)
//!
//! **`Table64` is primarily optimized for ARM NEON** and provides excellent performance on Apple Silicon
//! and other ARMv8+ CPUs. It also supports Intel AVX-512 on newer CPUs.
//!
//! ### ARM aarch64 (Primary Optimization Target)
//! - **Optimal**: Uses ARM NEON `TBL4` instruction (`vqtbl4q_u8`)
//!   - Native hardware support on all ARMv8+ CPUs (including Apple M1/M2/M3)
//!   - Extremely efficient single-instruction 64-byte table lookup
//!   - No fallback needed - full SIMD acceleration on ARM
//!   - The `TBL4` instruction can perform 64-entry lookups in a single operation
//!
//! ### Intel x86_64
//! - **Optimal**: Requires **AVX512BW** + **AVX512VBMI**
//!   - Uses `VPERMB` instruction (`_mm512_permutexvar_epi8`) for 64-byte table lookups
//!   - Available on: Intel Ice Lake, Tiger Lake, and later (not available on Skylake-X)
//!   - Fallback: Scalar lookup (works on all x86_64 CPUs)
//!
//! ## Table2dU8xU8 (2D lookup table, up to 64K entries)
//!
//! ### Intel x86_64
//! - **Optimal**: Requires **AVX512F** + **AVX512BW** (via `simd_gather` module)
//!   - Uses `VGATHERDPS` + `VPMOVDB` for parallel lookups
//!   - Available on: Intel Skylake-X (Xeon), Ice Lake, Tiger Lake, and later
//!   - Fallback: Scalar lookup (works on all architectures)
//!
//! ### ARM aarch64
//! - Uses scalar fallback (NEON gather is not significantly faster than scalar for this use case)

use crate::simd_gather::gather_u32index_u8;
use crate::wide_utils::WideUtilsExt;
use std::fmt;
use wide::{u8x16, u16x16, u32x16};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{uint8x16x4_t, vld1q_u8, vqtbl4q_u8, vst1q_u8};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::{
    __m128i, __m512i, _mm_loadu_si128, _mm_storeu_si128, _mm512_castsi128_si512,
    _mm512_castsi512_si128, _mm512_loadu_si512, _mm512_permutexvar_epi8, _mm512_storeu_si512,
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
                    // Load only 16 bytes (safe) into XMM register
                    let iv_128 = _mm_loadu_si128(idx.as_array().as_ptr() as *const __m128i);
                    // Zero-cost cast to ZMM (upper bytes undefined, but we don't use them)
                    let iv = _mm512_castsi128_si512(iv_128);
                    // VPERMB: only first 16 result bytes are valid
                    let rv = _mm512_permutexvar_epi8(iv, tzmm);
                    // Extract low 128 bits (zero latency - register rename)
                    let rv_128 = _mm512_castsi512_si128(rv);
                    // Store only 16 bytes
                    let mut result = [0u8; 16];
                    _mm_storeu_si128(result.as_mut_ptr() as *mut __m128i, rv_128);
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

    /// Get the underlying bytes array (for debugging/display purposes).
    /// This extracts the data from platform-specific storage.
    #[inline]
    fn as_bytes(&self) -> [u8; 64] {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                let mut bytes = [0u8; 64];
                vst1q_u8(bytes.as_mut_ptr(), self.neon_tbl.0);
                vst1q_u8(bytes.as_mut_ptr().add(16), self.neon_tbl.1);
                vst1q_u8(bytes.as_mut_ptr().add(32), self.neon_tbl.2);
                vst1q_u8(bytes.as_mut_ptr().add(48), self.neon_tbl.3);
                bytes
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            self.bytes
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

impl Clone for Table64 {
    fn clone(&self) -> Self {
        let bytes = self.as_bytes();
        Self::new(&bytes)
    }
}

impl Default for Table64 {
    fn default() -> Self {
        Self::new(&[0u8; 64])
    }
}

impl fmt::Debug for Table64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bytes = self.as_bytes();
        writeln!(f, "Table64 {{")?;
        writeln!(f, "        col 0  col 1  col 2  col 3  col 4  col 5  col 6  col 7")?;
        for row in 0..8 {
            write!(f, "row {}: ", row)?;
            for col in 0..8 {
                let idx = row * 8 + col;
                write!(f, "{:5} ", bytes[idx])?;
            }
            writeln!(f)?;
        }
        write!(f, "}}")
    }
}

// =============================================================================
// Table2dU8xU8 - 2D lookup table with up to 64K entries (256×256)
// =============================================================================

/// A 2D SIMD lookup table for `u8 × u8` coordinates, supporting up to 64K entries.
///
/// This table stores data in row-major order and uses SIMD gather operations for
/// efficient parallel lookups. Each lookup takes a row index (0..num_rows) and
/// column index (0..num_cols), both as u8, and returns the corresponding value.
///
/// # Index Calculation
///
/// For row `r` and column `c`, the flat index is: `index = r * num_cols + c`
///
/// Since row and column are both u8 (max 255), and num_cols is at most 256,
/// the maximum index is 255 * 256 + 255 = 65535, which fits in u16.
///
/// # Example
///
/// ```ignore
/// // Create a 16x16 multiplication table
/// let mut data = vec![0u8; 256];
/// for r in 0..16u8 {
///     for c in 0..16u8 {
///         data[(r as usize) * 16 + (c as usize)] = r.wrapping_mul(c);
///     }
/// }
/// let table = Table2dU8xU8::from_flat(&data, 16);
///
/// // Look up multiple (row, col) pairs in parallel
/// let rows = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
/// let cols = u8x16::splat(5);  // All looking up column 5
/// let result = table.lookup_one(rows, cols);
/// // result[i] = i * 5
/// ```
#[derive(Clone, Default)]
pub struct Table2dU8xU8 {
    data: Vec<u8>,
    num_cols: u16,
}

impl Table2dU8xU8 {
    /// Create a 2D table from a flat slice with the given number of columns.
    ///
    /// The data is stored in row-major order: `data[row * num_cols + col]`.
    ///
    /// # Arguments
    /// - `data`: Flat slice of values, length must be `num_rows * num_cols`
    /// - `num_cols`: Number of columns per row (1..=256)
    ///
    /// # Panics
    /// - Panics if `num_cols` is 0 or greater than 256
    /// - Panics if `data.len()` is not a multiple of `num_cols`
    /// - Panics if `data.len() > 65536`
    #[inline]
    pub fn from_flat(data: &[u8], num_cols: usize) -> Self {
        assert!(num_cols > 0 && num_cols <= 256, "num_cols must be 1..=256");
        assert!(data.len() % num_cols == 0, "data length must be multiple of num_cols");
        assert!(data.len() <= 65536, "data length must be <= 65536 (64K entries)");

        Self {
            data: data.to_vec(),
            num_cols: num_cols as u16,
        }
    }

    /// Create a 2D table from a 2D matrix (Vec of rows).
    ///
    /// All rows must have the same length.
    ///
    /// # Panics
    /// - Panics if the matrix is empty
    /// - Panics if rows have different lengths
    /// - Panics if total size exceeds 65536
    #[inline]
    pub fn from_2d(matrix: &[&[u8]]) -> Self {
        assert!(!matrix.is_empty(), "matrix cannot be empty");
        let num_cols = matrix[0].len();
        assert!(num_cols > 0 && num_cols <= 256, "num_cols must be 1..=256");
        assert!(matrix.iter().all(|row| row.len() == num_cols), "all rows must have same length");
        assert!(matrix.len() * num_cols <= 65536, "total size must be <= 65536");

        let mut data = Vec::with_capacity(matrix.len() * num_cols);
        for row in matrix {
            data.extend_from_slice(row);
        }

        Self {
            data,
            num_cols: num_cols as u16,
        }
    }

    /// Returns the number of columns per row.
    #[inline]
    pub fn num_cols(&self) -> usize {
        self.num_cols as usize
    }

    /// Returns the number of rows in the table.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.data.len() / self.num_cols as usize
    }

    /// Returns the total number of entries in the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the table is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Look up 16 values in parallel using (row, col) coordinates.
    ///
    /// Computes `result[i] = table[rows[i]][cols[i]]` for all 16 lanes.
    ///
    /// # Arguments
    /// - `rows`: Row indices (0..num_rows) for each of the 16 lanes
    /// - `cols`: Column indices (0..num_cols) for each of the 16 lanes
    ///
    /// # Safety
    /// In debug mode, asserts that all indices are in bounds.
    /// In release mode, out-of-bounds access is undefined behavior.
    #[inline]
    pub fn lookup_one(&self, rows: u8x16, cols: u8x16) -> u8x16 {
        // Widen u8x16 → u16x16 for arithmetic
        let rows_u16: u16x16 = u16x16::from(rows);
        let cols_u16: u16x16 = u16x16::from(cols);
        let num_cols_u16 = u16x16::splat(self.num_cols);

        // index = row * num_cols + col (all in u16x16)
        let indices_u16 = rows_u16 * num_cols_u16 + cols_u16;

        // Widen u16x16 → u32x16 for gather
        let indices_u32: u32x16 = u32x16::from(indices_u16);

        // Debug bounds check
        #[cfg(debug_assertions)]
        {
            let idx_arr = indices_u32.to_array();
            for (i, &idx) in idx_arr.iter().enumerate() {
                debug_assert!(
                    (idx as usize) < self.data.len(),
                    "Index out of bounds at lane {}: {} >= {}",
                    i, idx, self.data.len()
                );
            }
        }

        // Use SIMD gather (AVX-512 on x86, scalar fallback elsewhere)
        gather_u32index_u8(indices_u32, &self.data, 1)
    }

    /// Scalar lookup for a single (row, col) coordinate.
    ///
    /// # Panics
    /// Panics if row or col is out of bounds.
    #[inline]
    pub fn get(&self, row: u8, col: u8) -> u8 {
        let index = (row as usize) * (self.num_cols as usize) + (col as usize);
        self.data[index]
    }
}

impl fmt::Debug for Table2dU8xU8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let num_rows = self.num_rows();
        let num_cols = self.num_cols as usize;

        writeln!(f, "Table2dU8xU8 {{")?;
        writeln!(f, "  dimensions: {} rows × {} cols", num_rows, num_cols)?;

        if self.data.is_empty() {
            return write!(f, "  (empty)}}");
        }

        // Limit display to reasonable size: max 20 rows and 20 cols
        const MAX_DISPLAY_ROWS: usize = 20;
        const MAX_DISPLAY_COLS: usize = 20;

        let display_rows = num_rows.min(MAX_DISPLAY_ROWS);
        let display_cols = num_cols.min(MAX_DISPLAY_COLS);
        let show_row_ellipsis = num_rows > MAX_DISPLAY_ROWS;
        let show_col_ellipsis = num_cols > MAX_DISPLAY_COLS;

        // Print column headers
        write!(f, "  ")?;
        for col in 0..display_cols {
            write!(f, " col{:3}", col)?;
        }
        if show_col_ellipsis {
            write!(f, " ...")?;
        }
        writeln!(f)?;

        // Print rows
        for row in 0..display_rows {
            write!(f, "  row{:3}:", row)?;
            for col in 0..display_cols {
                let idx = row * num_cols + col;
                write!(f, "{:5}", self.data[idx])?;
            }
            if show_col_ellipsis {
                write!(f, " ...")?;
            }
            writeln!(f)?;
        }

        if show_row_ellipsis {
            writeln!(f, "  ...")?;
        }

        write!(f, "}}")
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
        let table = Table64::new(&table_data);
        println!("\n{:?}", table);
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

    // ==================== Table2dU8xU8 Tests ====================

    /// Create a test table where value = row * 10 + col
    fn create_table2d_test_data(num_rows: usize, num_cols: usize) -> Vec<u8> {
        let mut data = Vec::with_capacity(num_rows * num_cols);
        for r in 0..num_rows {
            for c in 0..num_cols {
                data.push(((r * 10 + c) % 256) as u8);
            }
        }
        data
    }

    #[test]
    fn test_table2d_from_flat_basic() {
        let data = create_table2d_test_data(16, 16);
        let table = Table2dU8xU8::from_flat(&data, 16);

        println!("\n{:?}", table);
        assert_eq!(table.num_rows(), 16);
        assert_eq!(table.num_cols(), 16);
        assert_eq!(table.len(), 256);
    }

    #[test]
    fn test_table2d_from_2d() {
        let row0: &[u8] = &[0, 1, 2, 3];
        let row1: &[u8] = &[10, 11, 12, 13];
        let row2: &[u8] = &[20, 21, 22, 23];
        let matrix: &[&[u8]] = &[row0, row1, row2];

        let table = Table2dU8xU8::from_2d(matrix);

        assert_eq!(table.num_rows(), 3);
        assert_eq!(table.num_cols(), 4);
        assert_eq!(table.len(), 12);

        // Verify scalar lookup
        assert_eq!(table.get(0, 0), 0);
        assert_eq!(table.get(0, 3), 3);
        assert_eq!(table.get(1, 0), 10);
        assert_eq!(table.get(2, 3), 23);
    }

    #[test]
    fn test_table2d_lookup_one_basic() {
        let data = create_table2d_test_data(16, 16);
        let table = Table2dU8xU8::from_flat(&data, 16);

        // Look up row 0, cols 0..15
        let rows = u8x16::splat(0);
        let cols = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        let result = table.lookup_one(rows, cols);
        let result_arr = result.to_array();

        // Row 0: values are 0, 1, 2, ..., 15
        for i in 0..16 {
            assert_eq!(result_arr[i], i as u8, "Row 0, col {}", i);
        }
    }

    #[test]
    fn test_table2d_lookup_one_different_rows() {
        let data = create_table2d_test_data(16, 16);
        let table = Table2dU8xU8::from_flat(&data, 16);

        // Look up different rows, same column
        let rows = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let cols = u8x16::splat(5);

        let result = table.lookup_one(rows, cols);
        let result_arr = result.to_array();

        // Column 5: values are 5, 15, 25, 35, ... (row * 10 + 5)
        for i in 0..16 {
            let expected = ((i * 10 + 5) % 256) as u8;
            assert_eq!(result_arr[i], expected, "Row {}, col 5", i);
        }
    }

    #[test]
    fn test_table2d_lookup_one_scattered() {
        let data = create_table2d_test_data(16, 16);
        let table = Table2dU8xU8::from_flat(&data, 16);

        // Scattered lookups
        let rows = u8x16::from([0, 5, 10, 15, 3, 8, 12, 1, 7, 14, 2, 9, 4, 11, 6, 13]);
        let cols = u8x16::from([0, 15, 5, 10, 3, 8, 12, 1, 7, 14, 2, 9, 4, 11, 6, 13]);

        let result = table.lookup_one(rows, cols);
        let result_arr = result.to_array();
        let rows_arr = rows.to_array();
        let cols_arr = cols.to_array();

        for i in 0..16 {
            let expected = ((rows_arr[i] as usize * 10 + cols_arr[i] as usize) % 256) as u8;
            assert_eq!(
                result_arr[i], expected,
                "Mismatch at lane {}: row={}, col={}, expected={}, got={}",
                i, rows_arr[i], cols_arr[i], expected, result_arr[i]
            );
        }
    }

    #[test]
    fn test_table2d_lookup_matches_scalar() {
        let data = create_table2d_test_data(32, 20);
        let table = Table2dU8xU8::from_flat(&data, 20);

        let rows = u8x16::from([0, 5, 10, 15, 20, 25, 30, 31, 1, 6, 11, 16, 21, 26, 28, 29]);
        let cols = u8x16::from([0, 5, 10, 15, 19, 0, 5, 10, 1, 6, 11, 16, 18, 1, 6, 11]);

        let result = table.lookup_one(rows, cols);
        let result_arr = result.to_array();
        let rows_arr = rows.to_array();
        let cols_arr = cols.to_array();

        // Verify against scalar get()
        for i in 0..16 {
            let expected = table.get(rows_arr[i], cols_arr[i]);
            assert_eq!(
                result_arr[i], expected,
                "Mismatch at lane {}: SIMD={}, scalar={}",
                i, result_arr[i], expected
            );
        }
    }

    #[test]
    fn test_table2d_large_table() {
        // 256 × 256 = 64K entries (maximum size)
        let mut data = vec![0u8; 65536];
        for r in 0..256 {
            for c in 0..256 {
                data[r * 256 + c] = (r ^ c) as u8; // XOR pattern
            }
        }
        let table = Table2dU8xU8::from_flat(&data, 256);

        assert_eq!(table.num_rows(), 256);
        assert_eq!(table.num_cols(), 256);

        // Test some lookups
        let rows = u8x16::from([0, 255, 128, 64, 32, 16, 8, 4, 2, 1, 100, 200, 50, 150, 75, 175]);
        let cols = u8x16::from([255, 0, 128, 64, 32, 16, 8, 4, 2, 1, 50, 100, 200, 75, 175, 150]);

        let result = table.lookup_one(rows, cols);
        let result_arr = result.to_array();
        let rows_arr = rows.to_array();
        let cols_arr = cols.to_array();

        for i in 0..16 {
            let expected = rows_arr[i] ^ cols_arr[i];
            assert_eq!(result_arr[i], expected, "XOR mismatch at lane {}", i);
        }
    }

    #[test]
    fn test_table2d_non_power_of_two_cols() {
        // Test with 17 columns (not power of 2)
        let data = create_table2d_test_data(10, 17);
        let table = Table2dU8xU8::from_flat(&data, 17);

        assert_eq!(table.num_rows(), 10);
        assert_eq!(table.num_cols(), 17);

        let rows = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5]);
        let cols = u8x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 15, 14, 13, 12, 11]);

        let result = table.lookup_one(rows, cols);
        let result_arr = result.to_array();
        let rows_arr = rows.to_array();
        let cols_arr = cols.to_array();

        for i in 0..16 {
            let expected = table.get(rows_arr[i], cols_arr[i]);
            assert_eq!(result_arr[i], expected, "Mismatch at lane {}", i);
        }
    }
}

