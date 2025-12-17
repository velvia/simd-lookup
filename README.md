# simd-lookup

High-performance SIMD utilities for fast table lookups, compression and data processing in Rust.

## Features

- **Cross-platform SIMD**: Automatic dispatch to optimal implementation (AVX-512, AVX2, NEON)
- **Zero-cost abstractions**: Thin wrappers over platform intrinsics via the `wide` crate
- **Comprehensive utilities**: Compress, shuffle, widen, split, and bitmask operations

## CPU Feature Requirements

This crate automatically detects and uses the best available CPU features, with fallbacks for older CPUs.
The crate is optimized for both **ARM NEON** (aarch64) and **Intel AVX-512** (x86_64) architectures.

**Note**: `Table64` is **primarily optimized for ARM NEON** using the `TBL4` instruction, which provides
excellent performance on Apple Silicon and other ARMv8+ CPUs. On Intel x86_64, it requires newer AVX-512
features (Ice Lake+).

### Summary Table

| Module/Feature | Required CPU Features | Available CPUs | Fallback |
|----------------|----------------------|----------------|----------|
| **simd_compress** (`compress_store_u32x8`) | AVX512F + AVX512VL | Skylake-X+, Ice Lake+ | Shuffle table |
| **simd_compress** (`compress_store_u32x16`) | AVX512F | Skylake-X+, Ice Lake+ | Two u32x8 compresses |
| **simd_compress** (`compress_store_u8x16`) | AVX512VBMI2 + AVX512VL | Ice Lake+, Tiger Lake+ | Gather-style writes |
| **simd_gather** (`gather_u32index_u8`) | AVX512F + AVX512BW | Skylake-X+, Ice Lake+ | Scalar loop |
| **simd_gather** (`gather_u32index_u32`) | AVX512F | Skylake-X+, Ice Lake+ | Scalar loop |
| **Table64** | **ARM NEON TBL4** (aarch64) or AVX512BW + AVX512VBMI (x86_64) | All ARMv8+ (Apple Silicon), Ice Lake+ | Scalar lookup (x86_64 only) |
| **Table2dU8xU8** | AVX512F + AVX512BW | Skylake-X+, Ice Lake+ | Scalar lookup |
| **Cascading Lookup Kernel** | AVX512F + AVX512VL + AVX512BW + AVX512VBMI2 | Ice Lake+, Tiger Lake+ | Scalar lookup |

### Detailed Requirements

#### SIMD Compress Kernels (`simd_compress` module)

- **`compress_store_u32x8`**: Requires **AVX512F** + **AVX512VL**
  - Uses `VPCOMPRESSD` instruction
  - Available on: Intel Skylake-X (Xeon), Ice Lake, Tiger Lake, and later
  - Fallback: Shuffle-based table lookup (works on all architectures)

- **`compress_store_u32x16`**: Requires **AVX512F**
  - Uses `VPCOMPRESSD` instruction (512-bit variant)
  - Available on: Intel Skylake-X (Xeon), Ice Lake, Tiger Lake, and later
  - Fallback: Two `compress_store_u32x8` operations

- **`compress_store_u8x16`**: Requires **AVX512VBMI2** + **AVX512VL**
  - Uses `VPCOMPRESSB` instruction
  - Available on: Intel Ice Lake, Tiger Lake, and later (**not available on Skylake-X**)
  - Fallback: Gather-style direct writes

#### SIMD Gather Operations (`simd_gather` module)

- **`gather_u32index_u8`**: Requires **AVX512F** + **AVX512BW**
  - Uses `VGATHERDPS` + `VPMOVDB` instructions
  - Available on: Intel Skylake-X (Xeon), Ice Lake, Tiger Lake, and later
  - Fallback: Scalar loop

- **`gather_u32index_u32`**: Requires **AVX512F**
  - Uses `VGATHERDPS` instruction
  - Available on: Intel Skylake-X (Xeon), Ice Lake, Tiger Lake, and later
  - Fallback: Scalar loop

#### Small Table Lookups (`small_table` module)

- **`Table64`**: **Highly optimized for ARM NEON** (primary optimization target)
  - **ARM aarch64 (Apple Silicon, etc.)**: Uses ARM NEON `TBL4` instruction (`vqtbl4q_u8`)
    - Native hardware support on all ARMv8+ CPUs (including Apple M1/M2/M3)
    - Extremely efficient single-instruction 64-byte table lookup
    - No fallback needed - full SIMD acceleration on ARM
  - **Intel x86_64**: Requires **AVX512BW** + **AVX512VBMI**
    - Uses `VPERMB` instruction (`_mm512_permutexvar_epi8`) for 64-byte table lookups
    - Available on: Intel Ice Lake, Tiger Lake, and later (**not available on Skylake-X**)
    - Fallback: Scalar lookup (works on all x86_64 CPUs)

- **`Table2dU8xU8`**: Requires **AVX512F** + **AVX512BW** (via `simd_gather`)
  - Uses `VGATHERDPS` + `VPMOVDB` for parallel lookups
  - Available on: Intel Skylake-X (Xeon), Ice Lake, Tiger Lake, and later
  - Fallback: Scalar lookup

#### Cascading Lookup Kernels (`lookup_kernel` module)

- **`SimdCascadingTableU32U8Lookup`**: Requires **AVX512F** + **AVX512VL** + **AVX512BW** + **AVX512VBMI2**
  - Uses `compress_store_u8x16`, `compress_store_u32x16`, and `gather_u32index_u8`
  - Provides 40-50% speedup over scalar implementations on large tables
  - Available on: Intel Ice Lake, Tiger Lake, and later (**not available on Skylake-X**)
  - Fallback: Scalar lookup (works on all architectures)

### CPU Generation Reference

- **Skylake-X (2017)**: AVX512F, AVX512VL, AVX512BW ✅ | AVX512VBMI ❌ | AVX512VBMI2 ❌
- **Ice Lake (2019)**: AVX512F, AVX512VL, AVX512BW, AVX512VBMI, AVX512VBMI2 ✅
- **Tiger Lake (2020)**: AVX512F, AVX512VL, AVX512BW, AVX512VBMI, AVX512VBMI2 ✅
- **Apple Silicon (M1/M2/M3)**: ARM NEON (TBL4) ✅ - no AVX-512 equivalent needed

### Checking CPU Features

You can check which features your CPU supports:

```bash
# Linux
grep flags /proc/cpuinfo | head -1

# Or use Rust's feature detection
cargo run --example check_features
```

All functions automatically detect available CPU features at runtime and use the best available implementation.

## SIMD Utilities (`wide_utils` module)

This crate provides a rich set of SIMD utilities built on top of the `wide` crate, with optimized implementations for x86_64 (AVX-512/AVX2) and aarch64 (NEON).

### Compress Operations (`simd_compress` module)

Stream compaction similar to AVX-512's `VCOMPRESS` instruction — pack selected elements contiguously based on a bitmask.

```rust
use simd_lookup::{compress_store_u32x8, compress_store_u32x16, compress_store_u8x16};
use wide::{u32x8, u32x16, u8x16};

// Compress u32x8: select elements where mask bits are set
let data = u32x8::from([10, 20, 30, 40, 50, 60, 70, 80]);
let mask = 0b10110010u8; // Select positions 1, 4, 5, 7
let mut output = [0u32; 8];

let count = compress_store_u32x8(data, mask, &mut output);
// count == 4, output[0..4] == [20, 50, 60, 80]

// Also available for u32x16 (512-bit) and u8x16
```

| Function | AVX-512 | Fallback |
|----------|---------|----------|
| `compress_store_u32x8` | `VPCOMPRESSD` (AVX512VL) | Shuffle table |
| `compress_store_u32x16` | `VPCOMPRESSD` (AVX512F) | 2× u32x8 compress |
| `compress_store_u8x16` | `VPCOMPRESSB` (AVX512VBMI2) | Shuffle table |

### Shuffle/Permute Operations

Variable-index shuffle using the same SIMD type for indices (zero-copy from lookup tables):

```rust
use simd_lookup::WideUtilsExt;
use wide::u32x8;

let data = u32x8::from([10, 20, 30, 40, 50, 60, 70, 80]);
let indices = u32x8::from([7, 6, 5, 4, 3, 2, 1, 0]); // Reverse

let reversed = data.shuffle(indices);
// reversed == [80, 70, 60, 50, 40, 30, 20, 10]
```

| Type | AVX2 | NEON | Scalar |
|------|------|------|--------|
| `u32x8` | `VPERMD` | `TBL2` (byte-level) | Loop |
| `u32x4` | — | `TBL` (byte-level) | Loop |
| `u8x16` | `PSHUFB` | `TBL` | Loop |

### Vector Splitting (`SimdSplit` trait)

Efficiently extract high/low halves of wide vectors:

```rust
use simd_lookup::SimdSplit;
use wide::u32x16;

let data = u32x16::from([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
let (lo, hi) = data.split_low_high();
// lo: u32x8 = [1,2,3,4,5,6,7,8]
// hi: u32x8 = [9,10,11,12,13,14,15,16]

// Or extract just one half
let low_half = data.low_half();
let high_half = data.high_half();
```

| Type | AVX-512 | Fallback |
|------|---------|----------|
| `u32x16 → u32x8` | `_mm512_extracti64x4_epi64` | Array slicing |
| `u64x8 → u64x4` | `_mm512_extracti64x4_epi64` | Array slicing |

### Widening Operations

Zero-extend smaller types to larger types:

```rust
use simd_lookup::WideUtilsExt;
use wide::{u32x8, u64x8};

let input = u32x8::from([1, 2, 3, 4, 5, 6, 7, 8]);
let widened: u64x8 = input.widen_to_u64x8();
// widened == [1u64, 2, 3, 4, 5, 6, 7, 8]
```

| Type | AVX-512 | AVX2 | NEON |
|------|---------|------|------|
| `u32x8 → u64x8` | `VPMOVZXDQ` | 2× `VPMOVZXDQ` | `VMOVL` |
| `u32x4 → u64x4` | — | `VPMOVZXDQ` | `VMOVL` |

### Bitmask to Vector Conversion

Convert a scalar bitmask to a SIMD mask vector:

```rust
use simd_lookup::FromBitmask;
use wide::u64x8;

let mask = 0b10101010u8;
let mask_vec: u64x8 = u64x8::from_bitmask(mask);
// mask_vec == [0, MAX, 0, MAX, 0, MAX, 0, MAX]
```

| Type | AVX-512 | AVX2/NEON |
|------|---------|-----------|
| `u64x8` | `VPBROADCASTQ` + mask | Loop |
| `u32x8` | `VPBROADCASTD` + mask | Loop |

### Double (`double()` method on `WideUtilsExt`)

Efficiently double each element via `self + self`. Addition is well-supported on all architectures (NEON `vaddq`, SSE `paddb`), making this the most efficient way to multiply by powers of 2:

```rust
use simd_lookup::WideUtilsExt;
use wide::u8x16;

let a = u8x16::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

// x * 2
let doubled = a.double();
// doubled == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

// x * 8 (chain three doubles)
let times_8 = a.double().double().double();
// times_8 == [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
```

This is more efficient than scalar multiplication for types like `u8x16` where x86 lacks native byte multiply/shift instructions.

### Shuffle Index Tables

Pre-computed shuffle indices for compress operations (256 entries for 8-element masks):

```rust
use simd_lookup::{SHUFFLE_COMPRESS_IDX_U32X8, get_compress_indices_u32x8};

// Raw array access
let indices: [u32; 8] = SHUFFLE_COMPRESS_IDX_U32X8[0b10110010];
// indices == [1, 4, 5, 7, 7, 7, 7, 7] (unused positions filled with 7)

// Zero-cost SIMD access via transmute
let simd_indices = get_compress_indices_u32x8(0b10110010u8);
```

## Other Modules

### `small_table` — Small Table SIMD Lookup
64-entry lookup table **primarily optimized for ARM NEON `TBL4`** (excellent performance on Apple Silicon)
and also supports AVX-512 `VPERMB` on Intel Ice Lake+. Useful for fast pattern detection and small dictionary lookups.

### `prefetch` — SIMD Memory Prefetch
Cross-platform memory prefetch utilities including masked prefetch for 8 addresses at once. Supports L1/L2/L3 cache hints.

### `lookup_kernel` — High-Performance Lookup Kernels
Production-ready SIMD lookup kernels for dictionary/table lookups:
- `PipelinedSingleTableU32U8Lookup` — Pipelined single-table lookup with software prefetching
- `SimdDualTableU32U8Lookup` — Dual-table lookup for join-like operations
- `SimdCascadingTableU32U8Lookup` — Cascading multi-table lookup with VGATHER/VCOMPRESS
- `SimdDualTableWithHashLookup` — Dual table with hash fallback for unknown keys

### `bulk_vec_extender` — Efficient Vec Extension
Utilities for efficiently extending `Vec` with SIMD-produced results, minimizing bounds checks and reallocations.

### `entropy_map_lookup` — Entropy-Optimized Lookups
Lookup structures optimized for low-entropy (few unique values) data, using bitpacking and small lookup tables.

### `eight_value_lookup` — 8-Value Fast Path
Specialized lookup for tables with ≤8 unique values, using SIMD comparison and bitmask extraction.

## Performance Notes

- **AVX-512**: Native compress instructions are ~3-5× faster than shuffle-based fallback
- **NEON u32 shuffle**: Uses `TBL`/`TBL2` with byte-level indexing (converts u32 indices to byte offsets)
- **Lookup tables**: 256×8×4 = 8KB for u32x8 compress indices; fits in L1 cache
- **SimdSplit**: AVX-512 uses single extract instruction; fallback is zero-cost transmute

## TODO list

- Build proper SIMD extensions for memory prefetch, masked VGATHER, etc that are reusable in different places.
  For example, build traits on top of wide's SIMD types and implement them for different architectures.
- Refactor and get rid of all of the ugly AI generated intrinsic code
- Good looking SIMD bitvec core, no AI generated intrinsics
- As we build the SIMD intrinsics and other lookup utilities, add plenty of RustDoc detailing the WHY's, performance
  space/memory and other tradeoffs.