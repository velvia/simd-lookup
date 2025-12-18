# Contributing to simd-lookup

Thank you for your interest in contributing to simd-lookup! This document provides guidelines and information for contributors.

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## Getting Started

### Prerequisites

- Rust 1.75 or later (for stable SIMD features)
- For ARM development: Apple Silicon Mac or ARMv8+ Linux system
- For AVX-512 development: Intel Ice Lake+ or Tiger Lake+ CPU (or emulation)

### Setting Up Your Development Environment

```bash
# Clone the repository
git clone https://github.com/evanchan/simd-lookup.git
cd simd-lookup

# Build the project
cargo build

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Cross-Compilation

To verify changes work on both ARM and x86-64:

```bash
# On Apple Silicon, cross-compile for x86-64
rustup target add x86_64-apple-darwin
cargo build --target x86_64-apple-darwin

# Check for Linux x86-64
rustup target add x86_64-unknown-linux-gnu
cargo check --target x86_64-unknown-linux-gnu
```

## How to Contribute

### Reporting Issues

- Search existing issues before creating a new one
- Include your Rust version, CPU architecture, and OS
- For performance issues, include benchmark results and CPU model
- Provide a minimal reproducible example when possible

### Pull Requests

1. **Fork the repository** and create a feature branch from `main`
2. **Write tests** for new functionality
3. **Run the full test suite**: `cargo test`
4. **Run clippy**: `cargo clippy -- -D warnings`
5. **Format code**: `cargo fmt`
6. **Update documentation** for any API changes
7. **Add a changelog entry** if the change is user-facing

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add AVX-512 optimized gather for u64 types
fix: correct bitmask expansion for negative values on ARM
perf: improve compress_store_u8x16 throughput by 2x
docs: add CPU feature requirements table
```

Prefixes: `feat`, `fix`, `perf`, `docs`, `test`, `refactor`, `chore`

## Architecture Guidelines

### SIMD Implementation Pattern

When adding new SIMD operations, follow this pattern:

```rust
/// Brief description of what this function does.
///
/// # Platform-Specific Implementation
///
/// | Platform | Instruction | Requirements |
/// |----------|------------|--------------|
/// | x86-64 | `VPXXX` | AVX512XX |
/// | ARM | `vxxx` | NEON |
/// | Fallback | scalar loop | — |
///
/// # Arguments
///
/// * `input` - Description
/// * `mask` - Description
///
/// # Returns
///
/// Description of return value
#[inline]
pub fn my_simd_op(input: SomeType, mask: u8) -> OutputType {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        my_simd_op_avx512(input, mask)
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        my_simd_op_neon(input, mask)
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx512f"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        my_simd_op_fallback(input, mask)
    }
}
```

### Performance Considerations

1. **Minimize branches** in hot paths - use branchless algorithms where possible
2. **Use lookup tables** for complex index computations (but document cache impact)
3. **Avoid variable-length copies** - prefer fixed-size SIMD stores when possible
4. **Document memory requirements** for any lookup tables

### Testing Requirements

- Unit tests for correctness on all supported platforms
- Edge cases: empty inputs, single elements, maximum capacity
- Property-based tests where applicable
- Benchmark any performance-critical changes

### Documentation Standards

- All public APIs must have rustdoc comments
- Include code examples for non-trivial functions
- Document platform-specific behavior
- Note any unsafe code and its invariants

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench simd_compress_bench

# Run with specific filter
cargo bench -- compress_store_u32x8
```

### Interpreting Results

- Report throughput in **elements per second** (not bytes)
- Note your CPU model and relevant features
- Run multiple times to account for variance
- Compare against baseline before/after changes

## Architecture-Specific Notes

### ARM NEON

- Use `vqtbl1q_u8` / `vqtbl2q_u8` for table lookups (TBL instruction)
- Prefer `vst1q_u8` for full-vector stores over slice copies
- Use signed widening (`vmovl_s8`) for bitmask expansion

### x86-64 AVX-512

- Check feature availability: `AVX512F`, `AVX512VL`, `AVX512BW`, `AVX512VBMI`, `AVX512VBMI2`
- Use `_mm512_mask_compressstoreu_*` for compress operations
- Be aware of frequency throttling on heavy AVX-512 workloads

## Questions?

- Open a GitHub issue for technical questions
- Check existing issues and discussions first
- Performance questions should include benchmark data

Thank you for contributing!

