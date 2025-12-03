# simd-lookup

High-performance SIMD utilities for fast table lookups, compression and data processing in Rust.

## Features

- **Cross-platform SIMD**: Automatic dispatch to optimal implementation (AVX-512, AVX2, NEON)
- **Zero-cost abstractions**: Thin wrappers over platform intrinsics via the `wide` crate
- **Comprehensive utilities**: Compress, shuffle, widen, split, and bitmask operations

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

### `table64` — Small Table SIMD Lookup
64-entry lookup table optimized for NEON `TBL4` and AVX-512 `VPERMB`. Useful for fast pattern detection and small dictionary lookups.

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