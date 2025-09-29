use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use simd_aligned::arch::u8x16;
use simd_lookup::table64::Table64;

fn create_test_table() -> [u8; 64] {
    let mut table = [0u8; 64];
    for i in 0..64 {
        table[i] = (i * 3 + 7) as u8; // Some arbitrary pattern
    }
    table
}

fn create_test_indices(size: usize) -> Vec<u8x16> {
    let mut indices = Vec::with_capacity(size);
    for i in 0..size {
        let mut idx_array = [0u8; 16];
        for j in 0..16 {
            // Create indices that are valid (0-63) but varied
            idx_array[j] = ((i * 16 + j) % 64) as u8;
        }
        indices.push(u8x16::from(idx_array));
    }
    indices
}

fn bench_table64_lookup(c: &mut Criterion) {
    let table_data = create_test_table();
    let table = Table64::new(&table_data);

    let mut group = c.benchmark_group("table64_lookup");

    // Test different sizes of data to lookup (each processes 16 bytes)
    for size in [1, 64, 1024, 8192, 31250].iter() { // 31250 * 16 = 500k bytes
        let indices = create_test_indices(*size);
        let mut output = vec![u8x16::splat(0); *size];

        // Set throughput to measure bytes processed per second
        // Each u8x16 processes 16 bytes
        group.throughput(Throughput::Bytes((*size * 16) as u64));

        group.bench_with_input(
            BenchmarkId::new("lookup", format!("{}x16_bytes", size)),
            size,
            |b, _| {
                b.iter(|| {
                    table.lookup(black_box(&indices), black_box(&mut output));
                    black_box(&output);
                });
            },
        );
    }

    group.finish();
}

fn bench_table64_creation(c: &mut Criterion) {
    let table_data = create_test_table();

    c.bench_function("table64_creation", |b| {
        b.iter(|| {
            let table = Table64::new(black_box(&table_data));
            black_box(table);
        });
    });
}

fn bench_scalar_vs_simd_comparison(c: &mut Criterion) {
    let table_data = create_test_table();
    let table = Table64::new(&table_data);

    // Create a large test case for comparison
    let size = 31250; // 31250 * 16 = 500k bytes
    let indices = create_test_indices(size);
    let mut simd_output = vec![u8x16::splat(0); size];

    let mut group = c.benchmark_group("scalar_vs_simd");
    group.throughput(Throughput::Bytes((size * 16) as u64));

    // SIMD lookup
    group.bench_function("simd_lookup", |b| {
        b.iter(|| {
            table.lookup(black_box(&indices), black_box(&mut simd_output));
            black_box(&simd_output);
        });
    });

    // Scalar lookup for comparison
    group.bench_function("scalar_lookup", |b| {
        b.iter(|| {
            let mut scalar_output = vec![0u8; size * 16];
            for (i, idx_vec) in indices.iter().enumerate() {
                let idx_array = idx_vec.to_array();
                for (j, &idx) in idx_array.iter().enumerate() {
                    scalar_output[i * 16 + j] = table_data[idx as usize];
                }
            }
            black_box(scalar_output);
        });
    });

    group.finish();
}

fn bench_different_access_patterns(c: &mut Criterion) {
    let table_data = create_test_table();
    let table = Table64::new(&table_data);
    let size = 31250; // 31250 * 16 = 500k bytes

    let mut group = c.benchmark_group("access_patterns");
    group.throughput(Throughput::Bytes((size * 16) as u64));

    // Sequential access pattern
    let mut sequential_indices = Vec::with_capacity(size);
    for i in 0..size {
        let mut idx_array = [0u8; 16];
        for j in 0..16 {
            idx_array[j] = ((i * 16 + j) % 64) as u8;
        }
        sequential_indices.push(u8x16::from(idx_array));
    }

    // Random access pattern
    let mut random_indices = Vec::with_capacity(size);
    for i in 0..size {
        let mut idx_array = [0u8; 16];
        for j in 0..16 {
            // Simple pseudo-random pattern
            idx_array[j] = ((i * 17 + j * 23 + 13) % 64) as u8;
        }
        random_indices.push(u8x16::from(idx_array));
    }

    // Repeated access pattern (high cache locality)
    let mut repeated_indices = Vec::with_capacity(size);
    for i in 0..size {
        let mut idx_array = [0u8; 16];
        for j in 0..16 {
            idx_array[j] = ((i / 4 + j / 4) % 16) as u8; // Only use first 16 entries
        }
        repeated_indices.push(u8x16::from(idx_array));
    }

    let mut output = vec![u8x16::splat(0); size];

    group.bench_function("sequential", |b| {
        b.iter(|| {
            table.lookup(black_box(&sequential_indices), black_box(&mut output));
            black_box(&output);
        });
    });

    group.bench_function("random", |b| {
        b.iter(|| {
            table.lookup(black_box(&random_indices), black_box(&mut output));
            black_box(&output);
        });
    });

    group.bench_function("repeated", |b| {
        b.iter(|| {
            table.lookup(black_box(&repeated_indices), black_box(&mut output));
            black_box(&output);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_table64_lookup,
    bench_table64_creation,
    bench_scalar_vs_simd_comparison,
    bench_different_access_patterns
);
criterion_main!(benches);

