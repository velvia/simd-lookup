use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::prelude::*;
use rustc_hash::FxHashMap;
use simd_lookup::bulk_vec_extender::{BulkVecExtender, SliceU8SIMDExtender};
// use simd_lookup::entropy_map_lookup::EntropyMapLookup;
use simd_lookup::lookup_kernel::{
    SimdDualVocabU32U8Lookup, SimdDualVocabU32U8LookupV2, SimdDualVocabWithHashLookup,
    SimdJoinedDualVocabU32U8Lookup, SimdSingleVocabU32U8Lookup,
};
use simd_lookup::{PipelinedSingleVocabU32U8Lookup, compress_store_u8x16, compress_store_u32x16};
use wide::{u8x16, u32x16};

/// Create sparse entries for kernel benchmarks (same as lookup_bench.rs)
/// Returns Vec<(u32, u8)> entries that can be converted to a lookup table
fn create_sparse_entries_for_kernel(size: usize, density_percent: f32) -> Vec<(u32, u8)> {
    let num_entries = ((size as f32) * (density_percent / 100.0)) as usize;
    let mut entries = Vec::with_capacity(num_entries);

    // Create sparse entries distributed across the range [0, size-1]
    // Ensure we cover the full range so vocab size is exactly 'size'
    let step = if num_entries > 1 {
        (size - 1) / (num_entries - 1).max(1)
    } else {
        0
    };

    for i in 0..num_entries {
        let key = if num_entries == 1 {
            0
        } else if i == num_entries - 1 {
            // Ensure last entry is at size-1 to make vocab size exactly 'size'
            (size - 1) as u32
        } else {
            (i * step) as u32
        };
        let value = ((key % 255) + 1) as u8; // Values 1-255, avoiding 0 which is default
        entries.push((key, value));
    }

    entries
}

/// Create test values (u32 indices) for lookup
fn create_test_values(num_values: usize, max_index: usize) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducible benchmarks
    let mut values = Vec::with_capacity(num_values);

    for _ in 0..num_values {
        let idx = rng.gen_range(0..max_index);
        values.push(idx as u32);
    }

    values.shuffle(&mut rng);
    values
}

/// Benchmark SimdSingleVocabU32U8Lookup with chunks of 500
/// Compares simple (direct write) vs complex (filter zeros and track indices) lookup functions
fn bench_single_vocab_lookup(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0; // 20% density

    println!(
        "Creating lookup table: {} entries, {}% density",
        table_size, density
    );
    // Use the same table creation method as batch_lookup benchmark
    let entries = create_sparse_entries_for_kernel(table_size, density);
    let lookup_table = simd_lookup::lookup::create_scalar_lookup_table(&entries);
    let lookup = SimdSingleVocabU32U8Lookup::new(&lookup_table);

    // Create 1 million test values (divisible by 500)
    let num_values = 1_000_000;
    let test_values = create_test_values(num_values, table_size);

    let mut group = c.benchmark_group("single_vocab_lookup");
    group.throughput(Throughput::Elements(num_values as u64));

    // Simple version: direct write to Vec
    group.bench_function("chunks_of_500_simple", |b| {
        // Pre-allocate to avoid repeated reserve() calls (2000 calls for 1M elements in chunks of 500)
        let mut result_vec = Vec::with_capacity(num_values);
        b.iter(|| {
            // Reset the vec for each iteration
            result_vec.clear();
            // Process in chunks of 500
            for chunk in test_values.chunks_exact(500) {
                lookup.lookup_into_vec(black_box(chunk), &mut result_vec);
            }
            black_box(&result_vec);
        })
    });

    // Complex version: filter zeros and track indices
    group.bench_function("chunks_of_500_complex", |b| {
        let mut result_vec = Vec::new();
        let mut indices_vec = Vec::new();
        b.iter(|| {
            result_vec.clear();
            indices_vec.clear();
            let mut indices_simd = u32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            let sixteen = u32x16::splat(16);
            let zeroes = u8x16::splat(0);
            // Process in chunks of 500
            for chunk in test_values.chunks_exact(500) {
                let mut result_guard = result_vec.bulk_extend_guard(chunk.len());
                let result_slice = result_guard.as_mut_slice();
                let mut indices_guard = indices_vec.bulk_extend_guard(chunk.len());
                let indices_slice = indices_guard.as_mut_slice();
                let mut num_written = 0;
                lookup.lookup_func(black_box(chunk), &mut |lookedup_values, num_bytes| {
                    // This is the old scalar code.  The SIMD Compress code on Intel AVX512 is super fast!
                    // let array = lookedup_values.to_array();
                    // for i in 0..num_bytes {
                    //     if array[i] != 0 {
                    //         result_slice[num_written] = array[i];
                    //         indices_slice[num_written] = (global_idx + i) as u32;
                    //         num_written += 1;
                    //     }
                    // }
                    // global_idx += num_bytes;

                    // Check which values are nonzero and convert to bitmask
                    // simd_eq returns 0xFF where equal to zero, so invert to get nonzero mask
                    let eq_mask = lookedup_values.simd_eq(zeroes).to_bitmask();
                    let nonzero_mask = !eq_mask as u16;

                    // Compress nonzero values into result_slice
                    let written = compress_store_u8x16(lookedup_values, nonzero_mask, &mut result_slice[num_written..]);
                    let _ = compress_store_u32x16(indices_simd, nonzero_mask, &mut indices_slice[num_written..]);
                    num_written += written;

                    // Update indices based on num_bytes
                    if num_bytes < 16 {
                        indices_simd = indices_simd + u32x16::splat(num_bytes as u32);
                    } else {
                        indices_simd = indices_simd + sixteen;
                    }
                });
                result_guard.set_written(num_written);
                indices_guard.set_written(num_written);
            }
            black_box(&result_vec);
            black_box(&indices_vec);
        })
    });

    group.finish();
}

/// Benchmark PipelinedSingleVocabU32U8Lookup with chunks of 500
/// Compares simple (direct write) vs complex (filter zeros and track indices) lookup functions
/// Uses the same parameters as single_vocab_lookup for direct comparison
fn bench_pipelined_single_vocab_lookup(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0; // 20% density

    println!(
        "Creating pipelined lookup table: {} entries, {}% density",
        table_size, density
    );
    // Use the same table creation method as batch_lookup benchmark
    let entries = create_sparse_entries_for_kernel(table_size, density);
    let lookup_table = simd_lookup::lookup::create_scalar_lookup_table(&entries);
    let lookup = PipelinedSingleVocabU32U8Lookup::new(&lookup_table);

    // Create 1 million test values (divisible by 500)
    let num_values = 1_000_000;
    let test_values = create_test_values(num_values, table_size);

    let mut group = c.benchmark_group("pipelined_single_vocab_lookup");
    group.throughput(Throughput::Elements(num_values as u64));

    // Simple version: direct write to Vec
    group.bench_function("chunks_of_500_simple", |b| {
        // Pre-allocate to avoid repeated reserve() calls (2000 calls for 1M elements in chunks of 500)
        let mut result_vec = Vec::with_capacity(num_values);
        b.iter(|| {
            // Reset the vec for each iteration
            result_vec.clear();
            // Process in chunks of 500
            for chunk in test_values.chunks_exact(500) {
                lookup.lookup_into_vec(black_box(chunk), &mut result_vec);
            }
            black_box(&result_vec);
        })
    });

    // Complex version: filter zeros and track indices (using SIMD compress + BulkVecExtender)
    group.bench_function("chunks_of_500_complex", |b| {
        let mut result_vec = Vec::new();
        let mut indices_vec = Vec::new();
        b.iter(|| {
            result_vec.clear();
            indices_vec.clear();
            let mut indices_simd = u32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            let sixteen = u32x16::splat(16);
            let zeroes = u8x16::splat(0);
            // Process in chunks of 500
            for chunk in test_values.chunks_exact(500) {
                let mut result_guard = result_vec.bulk_extend_guard(chunk.len());
                let result_slice = result_guard.as_mut_slice();
                let mut indices_guard = indices_vec.bulk_extend_guard(chunk.len());
                let indices_slice = indices_guard.as_mut_slice();
                let mut num_written = 0;
                lookup.lookup_func(black_box(chunk), &mut |lookedup_values, num_bytes| {
                    // SIMD compress: filter zeros and track indices without branching
                    let eq_mask = lookedup_values.simd_eq(zeroes).to_bitmask();
                    let nonzero_mask = !eq_mask as u16;

                    let written = compress_store_u8x16(lookedup_values, nonzero_mask, &mut result_slice[num_written..]);
                    let _ = compress_store_u32x16(indices_simd, nonzero_mask, &mut indices_slice[num_written..]);
                    num_written += written;

                    if num_bytes < 16 {
                        indices_simd = indices_simd + u32x16::splat(num_bytes as u32);
                    } else {
                        indices_simd = indices_simd + sixteen;
                    }
                });
                result_guard.set_written(num_written);
                indices_guard.set_written(num_written);
            }
            black_box(&result_vec);
            black_box(&indices_vec);
        })
    });

    group.finish();
}

/// Benchmark SimdDualVocabU32U8Lookup with chunks of 500
/// Takes bitwise AND of the two lookup results
fn bench_dual_vocab_lookup(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0; // 20% density

    println!(
        "Creating dual lookup tables: {} entries, {}% density",
        table_size, density
    );
    // Use the same table creation method as batch_lookup benchmark
    let entries1 = create_sparse_entries_for_kernel(table_size, density);
    let entries2 = create_sparse_entries_for_kernel(table_size, density);
    let lookup_table1 = simd_lookup::lookup::create_scalar_lookup_table(&entries1);
    let lookup_table2 = simd_lookup::lookup::create_scalar_lookup_table(&entries2);
    let lookup = SimdDualVocabU32U8Lookup::new(&lookup_table1, &lookup_table2);

    // Create 1 million test values (two sets, divisible by 500)
    let num_values = 1_000_000;
    let test_values1 = create_test_values(num_values, table_size);
    let test_values2 = create_test_values(num_values, table_size);

    let mut group = c.benchmark_group("dual_vocab_lookup");
    group.throughput(Throughput::Elements(num_values as u64));

    group.bench_function("chunks_of_500_bitwise_and", |b| {
        b.iter(|| {
            let mut all_results = Vec::new();
            // Process in chunks of 500
            for (chunk1, chunk2) in test_values1
                .chunks_exact(500)
                .zip(test_values2.chunks_exact(500))
            {
                // Use bitwise AND as the combiner function, but write results to a Vec
                lookup.lookup_into_vec(
                    black_box(chunk1),
                    black_box(chunk2),
                    &mut all_results,
                    &mut |v1, v2| v1 & v2,
                );
            }
            black_box(all_results);
        })
    });

    group.finish();
}

/// Benchmark SimdJoinedDualVocabU32U8Lookup - tests colocated/joined lookup tables
/// to see if TLB swaps or non-colocated tables could be a performance factor.
fn bench_joined_dual_vocab_lookup(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0; // 20% density

    println!(
        "Creating joined dual lookup table: {} entries each, {}% density",
        table_size, density
    );

    // Create separate entries for table1 and table2
    let entries1 = create_sparse_entries_for_kernel(table_size, density);
    let entries2 = create_sparse_entries_for_kernel(table_size, density);
    let lookup_table1 = simd_lookup::lookup::create_scalar_lookup_table(&entries1);
    let lookup_table2 = simd_lookup::lookup::create_scalar_lookup_table(&entries2);

    // Create joined table: table1 ++ table2
    let mut joined_table = lookup_table1.clone();
    let table2_offset = joined_table.len();
    joined_table.extend_from_slice(&lookup_table2);

    println!(
        "Joined table size: {} bytes, table2_offset: {}",
        joined_table.len(),
        table2_offset
    );

    let lookup = SimdJoinedDualVocabU32U8Lookup::new(&joined_table, table2_offset);

    // Create 1 million test values (two sets, divisible by 500)
    let num_values = 1_000_000;
    let test_values1 = create_test_values(num_values, table_size);
    let test_values2 = create_test_values(num_values, table_size);

    let mut group = c.benchmark_group("joined_dual_vocab_lookup");
    group.throughput(Throughput::Elements(num_values as u64));

    group.bench_function("chunks_of_500_bitwise_and", |b| {
        b.iter(|| {
            let mut all_results = Vec::new();
            // Process in chunks of 500
            for (chunk1, chunk2) in test_values1
                .chunks_exact(500)
                .zip(test_values2.chunks_exact(500))
            {
                // Use bitwise AND as the combiner function, but write results to a Vec
                lookup.lookup_into_vec(
                    black_box(chunk1),
                    black_box(chunk2),
                    &mut all_results,
                    &mut |v1, v2| v1 & v2,
                );
            }
            black_box(all_results);
        })
    });

    group.finish();
}

/// Benchmark SimdDualVocabU32U8LookupV2 with varying second lookup table sizes
/// Takes bitwise AND of the two lookup results and writes nonzero u8's into a Vec
/// This benchmark tests the effect of lookup table 2 size on throughput
fn bench_dual_vocab_lookup_v2(c: &mut Criterion) {
    let table1_size = 15_000_000;
    let density = 20.0; // 20% density
    let chunk_size = 1_000_000;

    // Create table 1 (fixed at 15M)
    println!(
        "Creating lookup table 1: {} entries, {}% density",
        table1_size, density
    );
    let entries1 = create_sparse_entries_for_kernel(table1_size, density);
    let lookup_table1 = simd_lookup::lookup::create_scalar_lookup_table(&entries1);

    // Create 1 million test values for table 1
    let num_values = 1_000_000;
    let test_values1 = create_test_values(num_values, table1_size);

    let mut group = c.benchmark_group("dual_vocab_v2_table2_size");
    group.throughput(Throughput::Elements(num_values as u64));

    // Vary second lookup table size: 100k, 500k, 4M, 15M
    for table2_size in [100_000, 500_000, 4_000_000, 15_000_000] {
        let size_label = match table2_size {
            100_000 => "100k",
            500_000 => "500k",
            4_000_000 => "4M",
            15_000_000 => "15M",
            _ => "unknown",
        };

        println!("Creating lookup table 2: {} entries", table2_size);
        let entries2 = create_sparse_entries_for_kernel(table2_size, density);
        let lookup_table2 = simd_lookup::lookup::create_scalar_lookup_table(&entries2);
        let mut lookup = SimdDualVocabU32U8LookupV2::new(&lookup_table1, &lookup_table2);

        // Create test values for table 2 (indices within table2_size)
        let test_values2 = create_test_values(num_values, table2_size);

        group.bench_function(BenchmarkId::new("nonzero_filter", size_label), |b| {
            b.iter(|| {
                let mut result_vec: Vec<u8> = Vec::new();
                let mut indices_vec: Vec<u32> = Vec::new();
                let mut indices_simd = u32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
                let sixteen = u32x16::splat(16);
                let zeroes = u8x16::splat(0);
                // Process in chunks
                for (chunk1, chunk2) in test_values1
                    .chunks_exact(chunk_size)
                    .zip(test_values2.chunks_exact(chunk_size))
                {
                    let mut result_guard = result_vec.bulk_extend_guard(chunk1.len());
                    let result_slice = result_guard.as_mut_slice();
                    let mut indices_guard = indices_vec.bulk_extend_guard(chunk1.len());
                    let indices_slice = indices_guard.as_mut_slice();
                    let mut num_written = 0;

                    lookup.lookup_func(
                        black_box(chunk1),
                        black_box(chunk2),
                        &mut |v1, v2, num_bytes| {
                            // AND the two u8x16 values, then SIMD compress nonzero results
                            let combined = v1 & v2;
                            let eq_mask = combined.simd_eq(zeroes).to_bitmask();
                            let nonzero_mask = !eq_mask as u16;

                            let written = compress_store_u8x16(combined, nonzero_mask, &mut result_slice[num_written..]);
                            let _ = compress_store_u32x16(indices_simd, nonzero_mask, &mut indices_slice[num_written..]);
                            num_written += written;

                            if num_bytes < 16 {
                                indices_simd = indices_simd + u32x16::splat(num_bytes as u32);
                            } else {
                                indices_simd = indices_simd + sixteen;
                            }
                        },
                    );

                    result_guard.set_written(num_written);
                    indices_guard.set_written(num_written);
                }
                black_box(&result_vec);
                black_box(&indices_vec);
            })
        });
    }

    group.finish();
}

/// Benchmark SimdDualVocabU32U8LookupV2 with simple direct output using BulkVecExtender
/// This version writes all u8x16 results directly without filtering, to measure raw throughput
fn bench_dual_vocab_lookup_v2_simple(c: &mut Criterion) {
    let table1_size = 15_000_000;
    let density = 20.0; // 20% density
    let chunk_size = 1_000_000;

    // Create table 1 (fixed at 15M)
    println!(
        "Creating lookup table 1 (simple): {} entries, {}% density",
        table1_size, density
    );
    let entries1 = create_sparse_entries_for_kernel(table1_size, density);
    let lookup_table1 = simd_lookup::lookup::create_scalar_lookup_table(&entries1);

    // Create 1 million test values for table 1
    let num_values = 1_000_000;
    let test_values1 = create_test_values(num_values, table1_size);

    let mut group = c.benchmark_group("dual_vocab_v2_simple_output");
    group.throughput(Throughput::Elements(num_values as u64));

    // Vary second lookup table size: 100k, 500k, 4M, 15M
    for table2_size in [100_000, 500_000, 4_000_000, 15_000_000] {
        let size_label = match table2_size {
            100_000 => "100k",
            500_000 => "500k",
            4_000_000 => "4M",
            15_000_000 => "15M",
            _ => "unknown",
        };

        println!("Creating lookup table 2 (simple): {} entries", table2_size);
        let entries2 = create_sparse_entries_for_kernel(table2_size, density);
        let lookup_table2 = simd_lookup::lookup::create_scalar_lookup_table(&entries2);
        let mut lookup = SimdDualVocabU32U8LookupV2::new(&lookup_table1, &lookup_table2);

        // Create test values for table 2 (indices within table2_size)
        let test_values2 = create_test_values(num_values, table2_size);

        group.bench_function(BenchmarkId::new("direct_write", size_label), |b| {
            b.iter(|| {
                let mut result_vec: Vec<u8> = Vec::with_capacity(num_values);
                // Process in chunks of 500
                for (chunk1, chunk2) in test_values1
                    .chunks_exact(chunk_size)
                    .zip(test_values2.chunks_exact(chunk_size))
                {
                    // Pre-extend the vec for this chunk
                    let mut guard = result_vec.bulk_extend_guard(chunk1.len());
                    let mut write_slice = guard.as_mut_slice();
                    let mut num_written = 0;

                    lookup.lookup_func(
                        black_box(chunk1),
                        black_box(chunk2),
                        &mut |v1, v2, num_bytes| {
                            // AND the two u8x16 values and write directly
                            let combined = v1 & v2;
                            write_slice.write_u8x16(num_written, combined, num_bytes);
                            num_written += num_bytes;
                        },
                    );
                    // guard drops here, finalizes to correct length
                }
                black_box(&result_vec);
            })
        });
    }

    group.finish();
}

/// Benchmark dual vocab lookup with FxHashMap for table2
/// Tests varying hash table sizes: 1k, 3k, 10k, 30k, 100k, 300k entries
/// Table1 is fixed at 15M entries with 20% density
fn bench_dual_vocab_with_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("dual_vocab_with_hash");

    // Fixed parameters for table1
    let table1_size = 15_000_000;
    let table1_density = 20.0;
    let num_values = 1_000_000;
    let chunk_size = 500;
    let max_word_id = 15_000_000usize; // Max word ID for table2 keys

    // Create table1 (same as other dual vocab benchmarks)
    println!(
        "Creating lookup table 1: {} entries, {}% density",
        table1_size, table1_density
    );
    let entries1 = create_sparse_entries_for_kernel(table1_size, table1_density);
    let lookup_table1 = simd_lookup::lookup::create_scalar_lookup_table(&entries1);

    // Create test values for table1
    let test_values1 = create_test_values(num_values, table1_size);

    group.throughput(Throughput::Elements(num_values as u64));

    // Hash table sizes to test
    let hash_sizes = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000];

    for &hash_size in &hash_sizes {
        // Create FxHashMap entries with keys distributed across max_word_id range
        println!(
            "Creating FxHashMap table2 with {} entries (max key = {})",
            hash_size, max_word_id
        );
        let mut rng = StdRng::seed_from_u64(123);
        let mut hash_table2: FxHashMap<u32, u8> = FxHashMap::default();
        let mut used_keys = std::collections::HashSet::new();

        while hash_table2.len() < hash_size {
            let key = rng.gen_range(0..max_word_id) as u32;
            if !used_keys.contains(&key) {
                used_keys.insert(key);
                let value = ((key % 255) + 1) as u8;
                hash_table2.insert(key, value);
            }
        }

        // Create test values for table2 (random keys within max_word_id range)
        let test_values2 = create_test_values(num_values, max_word_id);

        let lookup = SimdDualVocabWithHashLookup::new(&lookup_table1, &hash_table2);

        let size_label = match hash_size {
            1_000 => "1k",
            3_000 => "3k",
            10_000 => "10k",
            30_000 => "30k",
            100_000 => "100k",
            300_000 => "300k",
            _ => "unknown",
        };

        group.bench_function(BenchmarkId::new("nonzero_filter", size_label), |b| {
            b.iter(|| {
                let mut result_vec: Vec<u8> = Vec::new();
                let mut indices_vec: Vec<u32> = Vec::new();
                let mut indices_simd = u32x16::from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
                let sixteen = u32x16::splat(16);
                let zeroes = u8x16::splat(0);
                // Process in chunks
                for (chunk1, chunk2) in test_values1
                    .chunks_exact(chunk_size)
                    .zip(test_values2.chunks_exact(chunk_size))
                {
                    let mut result_guard = result_vec.bulk_extend_guard(chunk1.len());
                    let result_slice = result_guard.as_mut_slice();
                    let mut indices_guard = indices_vec.bulk_extend_guard(chunk1.len());
                    let indices_slice = indices_guard.as_mut_slice();
                    let mut num_written = 0;

                    lookup.lookup_func(
                        black_box(chunk1),
                        black_box(chunk2),
                        &mut |v1, v2, num_bytes| {
                            // AND the two u8x16 values, then SIMD compress nonzero results
                            let combined = v1 & v2;
                            let eq_mask = combined.simd_eq(zeroes).to_bitmask();
                            let nonzero_mask = !eq_mask as u16;

                            let written = compress_store_u8x16(combined, nonzero_mask, &mut result_slice[num_written..]);
                            let _ = compress_store_u32x16(indices_simd, nonzero_mask, &mut indices_slice[num_written..]);
                            num_written += written;

                            if num_bytes < 16 {
                                indices_simd = indices_simd + u32x16::splat(num_bytes as u32);
                            } else {
                                indices_simd = indices_simd + sixteen;
                            }
                        },
                    );

                    result_guard.set_written(num_written);
                    indices_guard.set_written(num_written);
                }
                black_box(&result_vec);
                black_box(&indices_vec);
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_vocab_lookup,
    bench_pipelined_single_vocab_lookup,
    bench_dual_vocab_lookup,
    bench_joined_dual_vocab_lookup,
    bench_dual_vocab_lookup_v2,
    bench_dual_vocab_lookup_v2_simple,
    bench_dual_vocab_with_hash
);
criterion_main!(benches);
