use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use simd_lookup::lookup::{HashLookup, Lookup, ScalarLookup, SimdLookup, U8x8};
use simd_lookup::EightValueLookup;
use simd_aligned::arch::u32x8;

fn create_sparse_entries(size: usize, density_percent: f32) -> Vec<(u32, u8)> {
    let num_entries = ((size as f32) * (density_percent / 100.0)) as usize;
    let mut entries = Vec::with_capacity(num_entries);

    // Create sparse entries distributed across the range
    let step = size / num_entries.max(1);
    for i in 0..num_entries {
        let key = (i * step) as u32;
        let value = ((key % 255) + 1) as u8; // Values 1-255, avoiding 0 which is default
        entries.push((key, value));
    }

    entries
}

fn create_lookup_keys(max_key: u32, num_keys: usize) -> Vec<u32> {
    let mut keys = Vec::with_capacity(num_keys);
    for i in 0..num_keys {
        // Mix of valid and invalid keys for realistic testing
        let key = if i % 4 == 0 {
            // 25% invalid keys (beyond range)
            max_key + (i as u32)
        } else {
            // 75% keys within range (some valid, some invalid)
            (i as u32) % max_key
        };
        keys.push(key);
    }
    keys
}

fn bench_single_lookup(c: &mut Criterion) {
    let entries = create_sparse_entries(1_000_000, 2.0); // 2% density
    let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

    let scalar_lookup = ScalarLookup::new(&entries);
    let hash_lookup = HashLookup::new(&entries);
    let simd_lookup = SimdLookup::new(&entries);

    let test_keys = create_lookup_keys(max_key, 1000);

    let mut group = c.benchmark_group("single_lookup");
    group.throughput(Throughput::Elements(test_keys.len() as u64));

    group.bench_function("scalar", |b| {
        b.iter(|| {
            for &key in &test_keys {
                black_box(scalar_lookup.lookup(black_box(key)));
            }
        })
    });

    group.bench_function("hash", |b| {
        b.iter(|| {
            for &key in &test_keys {
                black_box(hash_lookup.lookup(black_box(key)));
            }
        })
    });

    group.bench_function("simd_scalar", |b| {
        b.iter(|| {
            for &key in &test_keys {
                black_box(simd_lookup.lookup_scalar(black_box(key)));
            }
        })
    });

    group.finish();
}

fn bench_batch_lookup(c: &mut Criterion) {
    let entries = create_sparse_entries(1_000_000, 2.0); // 2% density
    let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

    let scalar_lookup = ScalarLookup::new(&entries);
    let hash_lookup = HashLookup::new(&entries);

    let mut group = c.benchmark_group("batch_lookup");

    for batch_size in [64, 256, 1024, 4096] {
        let test_keys = create_lookup_keys(max_key, batch_size);
        let mut results = vec![0u8; batch_size];

        group.throughput(Throughput::Elements(batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    scalar_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hash", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    hash_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
                })
            },
        );
    }

    group.finish();
}

fn bench_simd_u32x8_lookup(c: &mut Criterion) {
    let entries = create_sparse_entries(1_000_000, 2.0); // 2% density
    let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

    let simd_lookup = SimdLookup::new(&entries);

    let test_keys = create_lookup_keys(max_key, 1024);

    // Convert to u32x8 chunks
    let mut u32x8_keys = Vec::new();
    for chunk in test_keys.chunks(8) {
        if chunk.len() == 8 {
            let array: [u32; 8] = chunk.try_into().unwrap();
            u32x8_keys.push(u32x8::from(array));
        }
    }

    let mut results = vec![U8x8::from([0; 8]); u32x8_keys.len()];

    let mut group = c.benchmark_group("simd_lookup");
    // Throughput measured in individual u32 lookups, not u32x8 operations
    group.throughput(Throughput::Elements((u32x8_keys.len() * 8) as u64));

    group.bench_function("u32x8_single_safe", |b| {
        b.iter(|| {
            for &keys in &u32x8_keys {
                black_box(simd_lookup.lookup_u32x8(black_box(keys)));
            }
        })
    });

    group.bench_function("u32x8_single_unchecked", |b| {
        b.iter(|| {
            for &keys in &u32x8_keys {
                black_box(unsafe { simd_lookup.lookup_u32x8_unchecked(black_box(keys)) });
            }
        })
    });

    group.bench_function("u32x8_batch_safe", |b| {
        b.iter(|| {
            simd_lookup.lookup_batch_u32x8(black_box(&u32x8_keys), black_box(&mut results));
        })
    });

    group.finish();
}

fn bench_simd_vs_scalar_comparison(c: &mut Criterion) {
    let entries = create_sparse_entries(1_000_000, 2.0); // 2% density
    let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

    let scalar_lookup = ScalarLookup::new(&entries);
    let simd_lookup = SimdLookup::new(&entries);

    let test_keys = create_lookup_keys(max_key, 1024);

    // Convert to u32x8 chunks for SIMD
    let mut u32x8_keys = Vec::new();
    for chunk in test_keys.chunks(8) {
        if chunk.len() == 8 {
            let array: [u32; 8] = chunk.try_into().unwrap();
            u32x8_keys.push(u32x8::from(array));
        }
    }

    let mut scalar_results = vec![0u8; test_keys.len()];
    let mut simd_results = vec![U8x8::from([0; 8]); u32x8_keys.len()];

    let mut group = c.benchmark_group("simd_vs_scalar");
    // Both measured in individual u32 lookups for fair comparison
    group.throughput(Throughput::Elements(test_keys.len() as u64));

    group.bench_function("scalar_batch", |b| {
        b.iter(|| {
            scalar_lookup.lookup_batch(black_box(&test_keys), black_box(&mut scalar_results));
        })
    });

    group.bench_function("simd_batch_safe", |b| {
        b.iter(|| {
            simd_lookup.lookup_batch_u32x8(black_box(&u32x8_keys), black_box(&mut simd_results));
        })
    });

    group.bench_function("simd_batch_unchecked", |b| {
        b.iter(|| {
            for (i, &keys) in u32x8_keys.iter().enumerate() {
                simd_results[i] = unsafe { simd_lookup.lookup_u32x8_unchecked(keys) };
            }
        })
    });

    group.finish();
}

fn bench_density_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("density_comparison");

    for density in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let entries = create_sparse_entries(1_000_000, density);
        let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

        let scalar_lookup = ScalarLookup::new(&entries);
        let hash_lookup = HashLookup::new(&entries);

        let test_keys = create_lookup_keys(max_key, 1000);
        let mut results = vec![0u8; 1000];

        group.throughput(Throughput::Elements(test_keys.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar", density),
            &density,
            |b, _| {
                b.iter(|| {
                    scalar_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hash", density),
            &density,
            |b, _| {
                b.iter(|| {
                    hash_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
                })
            },
        );
    }

    group.finish();
}

fn bench_memory_usage_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    // Test different table sizes to see cache effects
    for table_size in [10_000, 100_000, 1_000_000, 10_000_000] {
        let entries = create_sparse_entries(table_size, 1.0); // 1% density
        let max_key = entries.iter().map(|(k, _)| *k).max().unwrap_or(0);

        let scalar_lookup = ScalarLookup::new(&entries);
        let hash_lookup = HashLookup::new(&entries);

        let test_keys = create_lookup_keys(max_key, 1000);
        let mut results = vec![0u8; 1000];

        group.throughput(Throughput::Elements(test_keys.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar", table_size),
            &table_size,
            |b, _| {
                b.iter(|| {
                    scalar_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hash", table_size),
            &table_size,
            |b, _| {
                b.iter(|| {
                    hash_lookup.lookup_batch(black_box(&test_keys), black_box(&mut results));
                })
            },
        );
    }

    group.finish();
}

fn bench_eight_value_lookup_single(c: &mut Criterion) {
    let lookup_table = [10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000];
    let simd_lookup = EightValueLookup::new(&lookup_table);

    // Create test values - mix of values in and out of the table
    let mut test_values = Vec::new();
    for &val in &lookup_table {
        test_values.push(val); // Values in the table
        test_values.push(val + 1); // Values not in the table
    }

    let mut group = c.benchmark_group("eight_value_single");

    group.bench_function("simd_position_lookup", |b| {
        b.iter(|| {
            for &val in &test_values {
                black_box(simd_lookup.find_position(black_box(val)));
            }
        })
    });

    group.bench_function("scalar_position_search", |b| {
        b.iter(|| {
            for &val in &test_values {
                let pos = lookup_table.iter().position(|&x| x == black_box(val));
                black_box(pos.map(|p| p as i32).unwrap_or(-1));
            }
        })
    });

    group.bench_function("scalar_iter_enumerate", |b| {
        b.iter(|| {
            for &val in &test_values {
                let mut result = -1i32;
                for (i, &table_val) in lookup_table.iter().enumerate() {
                    if table_val == black_box(val) {
                        result = i as i32;
                        break;
                    }
                }
                black_box(result);
            }
        })
    });

    group.finish();
}

fn bench_eight_value_lookup_batch(c: &mut Criterion) {
    let lookup_table = [5, 15, 25, 35, 45, 55, 65, 75];
    let simd_lookup = EightValueLookup::new(&lookup_table);

    let mut group = c.benchmark_group("eight_value_batch");

    for batch_size in [64, 256, 1024, 4096] {
        // Create test data with 50% hit rate
        let mut test_values = Vec::new();
        for i in 0..batch_size {
            if i % 2 == 0 {
                test_values.push(lookup_table[i % lookup_table.len()]);
            } else {
                test_values.push((i * 7 + 13) as u32); // Likely not in table
            }
        }

        // Convert to u32x8 chunks for SIMD
        let mut u32x8_values = Vec::new();
        for chunk in test_values.chunks(8) {
            if chunk.len() == 8 {
                let array: [u32; 8] = chunk.try_into().unwrap();
                u32x8_values.push(u32x8::from(array));
            }
        }

        let mut simd_results = vec![-1i32; u32x8_values.len() * 8];
        let mut scalar_results = vec![-1i32; test_values.len()];

        group.bench_with_input(
            BenchmarkId::new("simd_batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    for (i, &values) in u32x8_values.iter().enumerate() {
                        let results = simd_lookup.find_positions_batch(black_box(values));
                        let start_idx = i * 8;
                        for (j, result) in results.iter().enumerate() {
                            if start_idx + j < simd_results.len() {
                                simd_results[start_idx + j] = *result;
                            }
                        }
                    }
                    black_box(&simd_results);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar_batch", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    for (i, &val) in test_values.iter().enumerate() {
                        let pos = lookup_table.iter().position(|&x| x == black_box(val));
                        scalar_results[i] = pos.map(|p| p as i32).unwrap_or(-1);
                    }
                    black_box(&scalar_results);
                })
            },
        );
    }

    group.finish();
}

fn bench_eight_value_table_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("eight_value_table_sizes");

    // Test different table sizes (1 to 8 elements)
    for table_size in 1..=8 {
        let lookup_table: Vec<u32> = (0..table_size).map(|i| (i * 1000) as u32).collect();
        let simd_lookup = EightValueLookup::new(&lookup_table);

        // Create test values
        let test_values: Vec<u32> = (0..1000).map(|i| (i * 13 + 7) as u32).collect();

        group.bench_with_input(
            BenchmarkId::new("simd", table_size),
            &table_size,
            |b, _| {
                b.iter(|| {
                    for &val in &test_values {
                        black_box(simd_lookup.find_position(black_box(val)));
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar", table_size),
            &table_size,
            |b, _| {
                b.iter(|| {
                    for &val in &test_values {
                        let pos = lookup_table.iter().position(|&x| x == black_box(val));
                        black_box(pos.map(|p| p as i32).unwrap_or(-1));
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_eight_value_hit_rates(c: &mut Criterion) {
    let lookup_table = [10, 20, 30, 40, 50, 60, 70, 80];
    let simd_lookup = EightValueLookup::new(&lookup_table);

    let mut group = c.benchmark_group("eight_value_hit_rates");

    // Test different hit rates (0%, 25%, 50%, 75%, 100%)
    for hit_rate in [0, 25, 50, 75, 100] {
        let mut test_values = Vec::new();

        for i in 0..1000 {
            if (i * 100 / 1000) < hit_rate {
                // Value in table
                test_values.push(lookup_table[i % lookup_table.len()]);
            } else {
                // Value not in table
                test_values.push((i * 7 + 123) as u32);
            }
        }

        group.bench_with_input(
            BenchmarkId::new("simd", hit_rate),
            &hit_rate,
            |b, _| {
                b.iter(|| {
                    for &val in &test_values {
                        black_box(simd_lookup.find_position(black_box(val)));
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar", hit_rate),
            &hit_rate,
            |b, _| {
                b.iter(|| {
                    for &val in &test_values {
                        let pos = lookup_table.iter().position(|&x| x == black_box(val));
                        black_box(pos.map(|p| p as i32).unwrap_or(-1));
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_single_lookup,
    bench_batch_lookup,
    bench_simd_u32x8_lookup,
    bench_simd_vs_scalar_comparison,
    bench_density_comparison,
    bench_memory_usage_patterns,
    bench_eight_value_lookup_single,
    bench_eight_value_lookup_batch,
    bench_eight_value_table_sizes,
    bench_eight_value_hit_rates
);
criterion_main!(benches);