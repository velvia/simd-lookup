use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rand::prelude::*;
use simd_aligned::{arch::u8x16, traits::Simd};
use simd_lookup::lookup_kernel::{SimdSingleVocabU32U8Lookup, SimdSingleVocabU32U8LookupGather, ScalarSingleVocabU32U8Lookup, SimdDualVocabU32U8Lookup, SimdDualVocabU32U8LookupV2};

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
fn bench_single_vocab_lookup(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0; // 20% density

    println!("Creating lookup table: {} entries, {}% density", table_size, density);
    // Use the same table creation method as batch_lookup benchmark
    let entries = create_sparse_entries_for_kernel(table_size, density);
    let lookup_table = simd_lookup::lookup::create_scalar_lookup_table(&entries);
    let lookup = SimdSingleVocabU32U8Lookup::new(&lookup_table);

    // Create 1 million test values (divisible by 500)
    let num_values = 1_000_000;
    let test_values = create_test_values(num_values, table_size);

    let mut group = c.benchmark_group("single_vocab_lookup");
    group.throughput(Throughput::Elements(num_values as u64));

    group.bench_function("chunks_of_500", |b| {
        // Pre-allocate to avoid repeated reserve() calls (2000 calls for 1M elements in chunks of 500)
        let mut result_vec = Vec::<u8x16>::with_capacity(num_values.div_ceil(16));
        b.iter(|| {
            // Reset the vec for each iteration
            result_vec.clear();
            // Process in chunks of 500
            for chunk in test_values.chunks_exact(500) {
                lookup.lookup_extend_u8x16_vec(black_box(chunk), &mut result_vec);
            }
            black_box(&result_vec);
        })
    });

    group.finish();
}

/// Benchmark SimdSingleVocabU32U8LookupGather with chunks of 500 (SIMD gather version)
fn bench_single_vocab_lookup_gather(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0; // 20% density

    println!("Creating lookup table (gather): {} entries, {}% density", table_size, density);
    // Use the same table creation method as batch_lookup benchmark
    let entries = create_sparse_entries_for_kernel(table_size, density);
    let lookup_table = simd_lookup::lookup::create_scalar_lookup_table(&entries);
    let lookup = SimdSingleVocabU32U8LookupGather::new(&lookup_table);

    // Create 1 million test values (divisible by 500)
    let num_values = 1_000_000;
    let test_values = create_test_values(num_values, table_size);

    let mut group = c.benchmark_group("single_vocab_lookup_gather");
    group.throughput(Throughput::Elements(num_values as u64));

    group.bench_function("chunks_of_500", |b| {
        // Pre-allocate to avoid repeated reserve() calls (2000 calls for 1M elements in chunks of 500)
        let mut result_vec = Vec::<u8x16>::with_capacity(num_values.div_ceil(16));
        b.iter(|| {
            // Reset the vec for each iteration
            result_vec.clear();
            // Process in chunks of 500
            for chunk in test_values.chunks_exact(500) {
                lookup.lookup_extend_u8x16_vec(black_box(chunk), &mut result_vec);
            }
            black_box(&result_vec);
        })
    });

    group.finish();
}

/// Benchmark ScalarSingleVocabU32U8Lookup with chunks of 500 (scalar-only, no u8x16 SIMD)
fn bench_single_vocab_lookup_scalar(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0; // 20% density

    println!("Creating lookup table (scalar): {} entries, {}% density", table_size, density);
    // Use the same table creation method as batch_lookup benchmark
    let entries = create_sparse_entries_for_kernel(table_size, density);
    let lookup_table = simd_lookup::lookup::create_scalar_lookup_table(&entries);
    let lookup = ScalarSingleVocabU32U8Lookup::new(&lookup_table);

    // Create 1 million test values (divisible by 500)
    let num_values = 1_000_000;
    let test_values = create_test_values(num_values, table_size);

    let mut group = c.benchmark_group("single_vocab_lookup_scalar");
    group.throughput(Throughput::Elements(num_values as u64));

    group.bench_function("chunks_of_500", |b| {
        b.iter(|| {
            // Process in chunks of 500, reusing the same vec like SimdSingleVocabU32U8Lookup
            let mut result_vec = Vec::with_capacity(num_values);
            for chunk in test_values.chunks_exact(500) {
                lookup.lookup_extend_vec(black_box(chunk), &mut result_vec);
            }
            result_vec.clear(); // Reset for next iteration
            black_box(result_vec);
        })
    });

    group.finish();
}

/// Benchmark SimdDualVocabU32U8Lookup with chunks of 500
/// Takes bitwise AND of the two lookup results
fn bench_dual_vocab_lookup(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0; // 20% density

    println!("Creating dual lookup tables: {} entries, {}% density", table_size, density);
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
            for (chunk1, chunk2) in test_values1.chunks_exact(500).zip(test_values2.chunks_exact(500)) {
                // Use bitwise AND as the combiner function
                let result = lookup.lookup_into_vec(
                    black_box(chunk1),
                    black_box(chunk2),
                    &mut |v1, v2| v1 & v2
                );
                all_results.extend(result);
            }
            black_box(all_results);
        })
    });

    group.finish();
}

/// Benchmark SimdDualVocabU32U8LookupV2 with different chunk sizes
/// Takes bitwise AND of the two lookup results and writes nonzero u8's into a Vec
fn bench_dual_vocab_lookup_v2(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0; // 20% density

    println!("Creating dual lookup tables (V2): {} entries, {}% density", table_size, density);
    // Use the same table creation method as batch_lookup benchmark
    let entries1 = create_sparse_entries_for_kernel(table_size, density);
    let entries2 = create_sparse_entries_for_kernel(table_size, density);
    let lookup_table1 = simd_lookup::lookup::create_scalar_lookup_table(&entries1);
    let lookup_table2 = simd_lookup::lookup::create_scalar_lookup_table(&entries2);
    let mut lookup = SimdDualVocabU32U8LookupV2::new(&lookup_table1, &lookup_table2);

    // Create 1 million test values (two sets)
    let num_values = 1_000_000;
    let test_values1 = create_test_values(num_values, table_size);
    let test_values2 = create_test_values(num_values, table_size);

    let mut group = c.benchmark_group("dual_vocab_lookup_v2");
    group.throughput(Throughput::Elements(num_values as u64));

    for chunk_size in [100, 250, 500, 1000] {
        group.bench_function(format!("chunks_of_{}_bitwise_and", chunk_size), |b| {
            b.iter(|| {
                let mut result_vec = Vec::new();
                // Process in chunks
                for (chunk1, chunk2) in test_values1.chunks_exact(chunk_size).zip(test_values2.chunks_exact(chunk_size)) {
                    lookup.lookup_func(
                        black_box(chunk1),
                        black_box(chunk2),
                        &mut |v1, v2, _idx| {
                            // AND the two u8x16 values
                            let combined = v1 & v2;
                            let combined_array = combined.as_array();

                            // Write any nonzero u8's into the result Vec
                            for &val in combined_array.iter() {
                                if val != 0 {
                                    result_vec.push(val);
                                }
                            }
                        }
                    );
                }
                result_vec.clear(); // Reset for next iteration
                black_box(result_vec);
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_vocab_lookup,
    bench_single_vocab_lookup_gather,
    bench_single_vocab_lookup_scalar,
    bench_dual_vocab_lookup,
    bench_dual_vocab_lookup_v2
);
criterion_main!(benches);

