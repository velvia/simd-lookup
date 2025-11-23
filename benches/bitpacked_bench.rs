use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use rand::prelude::*;
use simd_aligned::traits::Simd;
use simd_lookup::bitpacked_lookup::{BitPackedSingleVocab, BitPackedDualVocab, TwoBit, ThreeBit};
use simd_lookup::lookup_kernel::{ScalarSingleVocabU32U8Lookup, SimdDualVocabU32U8LookupV2};

/// Create sparse entries for kernel benchmarks
fn create_sparse_entries(size: usize, density_percent: f32, max_value: u8) -> Vec<(u32, u8)> {
    let num_entries = ((size as f32) * (density_percent / 100.0)) as usize;
    let mut entries = Vec::with_capacity(num_entries);

    let step = if num_entries > 1 {
        (size - 1) / (num_entries - 1).max(1)
    } else {
        0
    };

    for i in 0..num_entries {
        let key = if num_entries == 1 {
            0
        } else if i == num_entries - 1 {
            (size - 1) as u32
        } else {
            (i * step) as u32
        };
        // Limit values to the allowed range for bit-packing
        let value = ((key % (max_value as u32)) + 1) as u8;
        entries.push((key, value));
    }

    entries
}

/// Create test values (u32 indices) for lookup
fn create_test_values(num_values: usize, max_index: usize) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut values = Vec::with_capacity(num_values);

    for _ in 0..num_values {
        let idx = rng.gen_range(0..max_index);
        values.push(idx as u32);
    }

    values.shuffle(&mut rng);
    values
}

/// Benchmark single vocab: regular u8 vs 2-bit packed
fn bench_single_vocab_comparison(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0;
    let num_values = 1_000_000;

    let mut group = c.benchmark_group("single_vocab_comparison");
    group.throughput(Throughput::Elements(num_values as u64));

    // Create entries with values 0-3 (2-bit range)
    let entries = create_sparse_entries(table_size, density, 3);
    let test_values = create_test_values(num_values, table_size);

    // Regular u8 table
    let lookup_table_u8 = simd_lookup::lookup::create_scalar_lookup_table(&entries);
    let lookup_u8 = ScalarSingleVocabU32U8Lookup::new(&lookup_table_u8);

    println!("Single vocab - Regular u8 table: {} MB", lookup_table_u8.len() / 1_000_000);

    group.bench_function("regular_u8", |b| {
        let mut results = vec![0u8; num_values];
        b.iter(|| {
            lookup_u8.lookup_into_slice(black_box(&test_values), black_box(&mut results));
            black_box(&results);
        })
    });

    // 2-bit packed
    let lookup_2bit = BitPackedSingleVocab::<TwoBit>::from_entries(&entries);

    println!("Single vocab - 2-bit packed: {} MB", lookup_2bit.memory_bytes() / 1_000_000);
    println!("Compression ratio: {:.2}x",
             lookup_table_u8.len() as f64 / lookup_2bit.memory_bytes() as f64);

    group.bench_function("bitpacked_2bit", |b| {
        let mut results = vec![0u8; num_values];
        b.iter(|| {
            lookup_2bit.lookup_batch(black_box(&test_values), black_box(&mut results));
            black_box(&results);
        })
    });

    group.finish();
}

/// Benchmark dual vocab: regular u8 vs 2-bit packed
/// This is where we expect bit-packing to REALLY shine due to cache thrashing
fn bench_dual_vocab_comparison(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0;
    let num_values = 1_000_000;

    let mut group = c.benchmark_group("dual_vocab_comparison");
    group.throughput(Throughput::Elements(num_values as u64));

    // Create entries with values 0-3 (2-bit range)
    let entries1 = create_sparse_entries(table_size, density, 3);
    let entries2 = create_sparse_entries(table_size, density, 3);
    let test_values1 = create_test_values(num_values, table_size);
    let test_values2 = create_test_values(num_values, table_size);

    // Regular u8 tables with V2 (conditional lookup)
    let lookup_table1_u8 = simd_lookup::lookup::create_scalar_lookup_table(&entries1);
    let lookup_table2_u8 = simd_lookup::lookup::create_scalar_lookup_table(&entries2);
    let mut lookup_u8_v2 = SimdDualVocabU32U8LookupV2::new(&lookup_table1_u8, &lookup_table2_u8);

    let total_mb_u8 = (lookup_table1_u8.len() + lookup_table2_u8.len()) / 1_000_000;
    println!("\nDual vocab - Regular u8 tables: {} MB total", total_mb_u8);

    group.bench_function("regular_u8_v2", |b| {
        b.iter(|| {
            let mut result_count = 0u64;
            // Process in chunks like the original benchmark
            for (chunk1, chunk2) in test_values1.chunks_exact(500).zip(test_values2.chunks_exact(500)) {
                lookup_u8_v2.lookup_func(
                    black_box(chunk1),
                    black_box(chunk2),
                    &mut |v1, v2, _idx| {
                        // AND the two values and count non-zeros
                        let combined = v1 & v2;
                        let combined_array = combined.as_array();
                        for &val in combined_array.iter() {
                            if val != 0 {
                                result_count += 1;
                            }
                        }
                    }
                );
            }
            black_box(result_count);
        })
    });

    // 2-bit packed
    let lookup_2bit_dual = BitPackedDualVocab::<TwoBit>::from_u8_tables(&lookup_table1_u8, &lookup_table2_u8);

    let total_mb_2bit = lookup_2bit_dual.memory_bytes() / 1_000_000;
    println!("Dual vocab - 2-bit packed: {} MB total", total_mb_2bit);
    println!("Compression ratio: {:.2}x",
             (lookup_table1_u8.len() + lookup_table2_u8.len()) as f64 / lookup_2bit_dual.memory_bytes() as f64);

    group.bench_function("bitpacked_2bit", |b| {
        b.iter(|| {
            let mut result_count = 0u64;
            // Process in chunks
            for (chunk1, chunk2) in test_values1.chunks_exact(500).zip(test_values2.chunks_exact(500)) {
                let mut results = vec![(0u8, 0u8); chunk1.len()];
                lookup_2bit_dual.lookup_batch_conditional(
                    black_box(chunk1),
                    black_box(chunk2),
                    black_box(&mut results)
                );
                // Count non-zero ANDs
                for (v1, v2) in results {
                    if v1 & v2 != 0 {
                        result_count += 1;
                    }
                }
            }
            black_box(result_count);
        })
    });

    group.finish();
}

/// Test different table sizes to find cache threshold
fn bench_table_size_scaling(c: &mut Criterion) {
    let density = 20.0;
    let num_values = 1_000_000;

    let mut group = c.benchmark_group("dual_vocab_table_size_scaling");

    // Test different table sizes from L3-fittable to beyond
    // M1/M2 chips typically have 24-32MB L3 cache
    for size_mb in [1, 3, 5, 10, 15, 20] {
        let table_size = size_mb * 1_000_000;

        let entries1 = create_sparse_entries(table_size, density, 3);
        let entries2 = create_sparse_entries(table_size, density, 3);
        let test_values1 = create_test_values(num_values, table_size);
        let test_values2 = create_test_values(num_values, table_size);

        // Regular u8
        let lookup_table1_u8 = simd_lookup::lookup::create_scalar_lookup_table(&entries1);
        let lookup_table2_u8 = simd_lookup::lookup::create_scalar_lookup_table(&entries2);
        let mut lookup_u8_v2 = SimdDualVocabU32U8LookupV2::new(&lookup_table1_u8, &lookup_table2_u8);

        let total_mb = (lookup_table1_u8.len() + lookup_table2_u8.len()) / 1_000_000;

        group.throughput(Throughput::Elements(num_values as u64));
        group.bench_with_input(
            BenchmarkId::new("regular_u8", format!("{}MB", total_mb)),
            &total_mb,
            |b, _| {
                b.iter(|| {
                    let mut result_count = 0u64;
                    for (chunk1, chunk2) in test_values1.chunks_exact(500).zip(test_values2.chunks_exact(500)) {
                        lookup_u8_v2.lookup_func(
                            black_box(chunk1),
                            black_box(chunk2),
                            &mut |v1, v2, _idx| {
                                let combined = v1 & v2;
                                for &val in combined.as_array().iter() {
                                    if val != 0 {
                                        result_count += 1;
                                    }
                                }
                            }
                        );
                    }
                    black_box(result_count);
                })
            }
        );

        // 2-bit packed
        let lookup_2bit_dual = BitPackedDualVocab::<TwoBit>::from_u8_tables(&lookup_table1_u8, &lookup_table2_u8);
        let packed_mb = lookup_2bit_dual.memory_bytes() / 1_000_000;

        group.bench_with_input(
            BenchmarkId::new("bitpacked_2bit", format!("{}MB", packed_mb)),
            &packed_mb,
            |b, _| {
                b.iter(|| {
                    let mut result_count = 0u64;
                    for (chunk1, chunk2) in test_values1.chunks_exact(500).zip(test_values2.chunks_exact(500)) {
                        let mut results = vec![(0u8, 0u8); chunk1.len()];
                        lookup_2bit_dual.lookup_batch_conditional(
                            black_box(chunk1),
                            black_box(chunk2),
                            black_box(&mut results)
                        );
                        for (v1, v2) in results {
                            if v1 & v2 != 0 {
                                result_count += 1;
                            }
                        }
                    }
                    black_box(result_count);
                })
            }
        );
    }

    group.finish();
}

/// Test 3-bit encoding (values 0-7) vs 2-bit
fn bench_bit_width_comparison(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0;
    let num_values = 1_000_000;

    let mut group = c.benchmark_group("dual_vocab_bit_width");
    group.throughput(Throughput::Elements(num_values as u64));

    // 2-bit (values 0-3)
    let entries1_2bit = create_sparse_entries(table_size, density, 3);
    let entries2_2bit = create_sparse_entries(table_size, density, 3);
    let test_values1 = create_test_values(num_values, table_size);
    let test_values2 = create_test_values(num_values, table_size);

    let lookup_table1_u8 = simd_lookup::lookup::create_scalar_lookup_table(&entries1_2bit);
    let lookup_table2_u8 = simd_lookup::lookup::create_scalar_lookup_table(&entries2_2bit);
    let lookup_2bit = BitPackedDualVocab::<TwoBit>::from_u8_tables(&lookup_table1_u8, &lookup_table2_u8);

    println!("\n2-bit dual vocab: {} MB", lookup_2bit.memory_bytes() / 1_000_000);

    group.bench_function("2bit", |b| {
        b.iter(|| {
            let mut result_count = 0u64;
            for (chunk1, chunk2) in test_values1.chunks_exact(500).zip(test_values2.chunks_exact(500)) {
                let mut results = vec![(0u8, 0u8); chunk1.len()];
                lookup_2bit.lookup_batch_conditional(
                    black_box(chunk1),
                    black_box(chunk2),
                    black_box(&mut results)
                );
                for (v1, v2) in results {
                    if v1 & v2 != 0 {
                        result_count += 1;
                    }
                }
            }
            black_box(result_count);
        })
    });

    // 3-bit (values 0-7)
    let entries1_3bit = create_sparse_entries(table_size, density, 7);
    let entries2_3bit = create_sparse_entries(table_size, density, 7);

    let lookup_table1_u8_3bit = simd_lookup::lookup::create_scalar_lookup_table(&entries1_3bit);
    let lookup_table2_u8_3bit = simd_lookup::lookup::create_scalar_lookup_table(&entries2_3bit);
    let lookup_3bit = BitPackedDualVocab::<ThreeBit>::from_u8_tables(&lookup_table1_u8_3bit, &lookup_table2_u8_3bit);

    println!("3-bit dual vocab: {} MB", lookup_3bit.memory_bytes() / 1_000_000);

    group.bench_function("3bit", |b| {
        b.iter(|| {
            let mut result_count = 0u64;
            for (chunk1, chunk2) in test_values1.chunks_exact(500).zip(test_values2.chunks_exact(500)) {
                let mut results = vec![(0u8, 0u8); chunk1.len()];
                lookup_3bit.lookup_batch_conditional(
                    black_box(chunk1),
                    black_box(chunk2),
                    black_box(&mut results)
                );
                for (v1, v2) in results {
                    if v1 & v2 != 0 {
                        result_count += 1;
                    }
                }
            }
            black_box(result_count);
        })
    });

    group.finish();
}

/// Test conditional vs unconditional lookup
fn bench_conditional_vs_unconditional(c: &mut Criterion) {
    let table_size = 15_000_000;
    let density = 20.0;
    let num_values = 1_000_000;

    let entries1 = create_sparse_entries(table_size, density, 3);
    let entries2 = create_sparse_entries(table_size, density, 3);
    let test_values1 = create_test_values(num_values, table_size);
    let test_values2 = create_test_values(num_values, table_size);

    let lookup_table1_u8 = simd_lookup::lookup::create_scalar_lookup_table(&entries1);
    let lookup_table2_u8 = simd_lookup::lookup::create_scalar_lookup_table(&entries2);
    let lookup_2bit = BitPackedDualVocab::<TwoBit>::from_u8_tables(&lookup_table1_u8, &lookup_table2_u8);

    let mut group = c.benchmark_group("conditional_vs_unconditional");
    group.throughput(Throughput::Elements(num_values as u64));

    println!("\nBit-packed 2-bit dual vocab: {} MB", lookup_2bit.memory_bytes() / 1_000_000);

    // Conditional lookup (only table2 if table1 != 0)
    group.bench_function("conditional_20pct", |b| {
        b.iter(|| {
            let mut result_count = 0u64;
            for (chunk1, chunk2) in test_values1.chunks_exact(500).zip(test_values2.chunks_exact(500)) {
                let mut results = vec![(0u8, 0u8); chunk1.len()];
                lookup_2bit.lookup_batch_conditional(
                    black_box(chunk1),
                    black_box(chunk2),
                    black_box(&mut results)
                );
                // Count non-zero ANDs
                for (v1, v2) in results {
                    if v1 & v2 != 0 {
                        result_count += 1;
                    }
                }
            }
            black_box(result_count);
        })
    });

    // Unconditional lookup (always lookup both)
    group.bench_function("unconditional_100pct", |b| {
        b.iter(|| {
            let mut result_count = 0u64;
            for (chunk1, chunk2) in test_values1.chunks_exact(500).zip(test_values2.chunks_exact(500)) {
                let mut results = vec![(0u8, 0u8); chunk1.len()];
                lookup_2bit.lookup_batch_unconditional(
                    black_box(chunk1),
                    black_box(chunk2),
                    black_box(&mut results)
                );
                // Count non-zero ANDs
                for (v1, v2) in results {
                    if v1 & v2 != 0 {
                        result_count += 1;
                    }
                }
            }
            black_box(result_count);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_vocab_comparison,
    bench_dual_vocab_comparison,
    bench_table_size_scaling,
    bench_bit_width_comparison,
    bench_conditional_vs_unconditional
);
criterion_main!(benches);

