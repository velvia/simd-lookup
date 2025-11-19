# simd-lookup

SIMD utilties for fast lookups.

## TODO list

- Build proper SIMD extensions for memory prefetch, masked VGATHER, etc that are reusable in different places.
  For example, build traits on top of wide's SIMD types and implement them for different architectures.
- Refactor and get rid of all of the ugly AI generated intrinsic code
- Good looking SIMD bitvec core, no AI generated intrinsics
- As we build the SIMD intrinsics and other lookup utilities, add plenty of RustDoc detailing the WHY's, performance
  space/memory and other tradeoffs.