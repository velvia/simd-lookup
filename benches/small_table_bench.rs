use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use simd_lookup::small_table::{Table2dU8xU8, Table64};
use wide::u8x16;

// =============================================================================
// Test Data Generators
// =============================================================================

fn create_test_table_64() -> [u8; 64] {
    let mut table = [0u8; 64];
    for i in 0..64 {
        table[i] = (i * 3 + 7) as u8;
    }
    table
}

/// Create test indices for 1D lookup (values 0..63)
fn create_test_indices_1d(count: usize) -> Vec<u8x16> {
    let mut indices = Vec::with_capacity(count);
    for i in 0..count {
        let mut idx_array = [0u8; 16];
        for j in 0..16 {
            idx_array[j] = ((i * 16 + j) % 64) as u8;
        }
        indices.push(u8x16::from(idx_array));
    }
    indices
}

/// Create test (row, col) pairs for 2D lookup on 8x8 table
fn create_test_indices_2d_8x8(count: usize) -> (Vec<u8x16>, Vec<u8x16>) {
    let mut rows = Vec::with_capacity(count);
    let mut cols = Vec::with_capacity(count);
    for i in 0..count {
        let mut row_array = [0u8; 16];
        let mut col_array = [0u8; 16];
        for j in 0..16 {
            row_array[j] = ((i * 16 + j) % 8) as u8;
            col_array[j] = ((i * 17 + j * 3) % 8) as u8;
        }
        rows.push(u8x16::from(row_array));
        cols.push(u8x16::from(col_array));
    }
    (rows, cols)
}

/// Create a 2D test table with given dimensions, value = (row * 10 + col) % 256
fn create_test_table_2d(num_rows: usize, num_cols: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_rows * num_cols);
    for r in 0..num_rows {
        for c in 0..num_cols {
            data.push(((r * 10 + c) % 256) as u8);
        }
    }
    data
}

/// Create test (row, col) pairs for Table2dU8xU8 lookup
fn create_test_indices_2d(count: usize, num_rows: usize, num_cols: usize) -> (Vec<u8x16>, Vec<u8x16>) {
    let mut rows = Vec::with_capacity(count);
    let mut cols = Vec::with_capacity(count);
    for i in 0..count {
        let mut row_array = [0u8; 16];
        let mut col_array = [0u8; 16];
        for j in 0..16 {
            row_array[j] = ((i * 16 + j) % num_rows) as u8;
            col_array[j] = ((i * 17 + j * 3) % num_cols) as u8;
        }
        rows.push(u8x16::from(row_array));
        cols.push(u8x16::from(col_array));
    }
    (rows, cols)
}

// =============================================================================
// Table64 Benchmarks
// =============================================================================

/// Benchmark Table64::lookup_one() - single vector lookup
fn bench_table64_lookup_one(c: &mut Criterion) {
    let table_data = create_test_table_64();
    let table = Table64::new(&table_data);

    let mut group = c.benchmark_group("table64_lookup_one");

    for count in [1, 100, 1000, 10000, 100000].iter() {
        let indices = create_test_indices_1d(*count);

        // Throughput: count vectors × 16 bytes each
        group.throughput(Throughput::Bytes((*count * 16) as u64));

        group.bench_with_input(
            BenchmarkId::new("vectors", count),
            count,
            |b, _| {
                b.iter(|| {
                    for idx in &indices {
                        black_box(table.lookup_one(black_box(*idx)));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Table64::lookup_one_2d() - 2D lookup on 8x8 table
fn bench_table64_lookup_one_2d(c: &mut Criterion) {
    let table_data = create_test_table_64();
    let table = Table64::new(&table_data);

    let mut group = c.benchmark_group("table64_lookup_one_2d");

    for count in [1, 100, 1000, 10000, 100000].iter() {
        let (rows, cols) = create_test_indices_2d_8x8(*count);

        group.throughput(Throughput::Bytes((*count * 16) as u64));

        group.bench_with_input(
            BenchmarkId::new("vectors", count),
            count,
            |b, _| {
                b.iter(|| {
                    for (r, c) in rows.iter().zip(cols.iter()) {
                        black_box(table.lookup_one_2d(black_box(*r), black_box(*c)));
                    }
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Table2dU8xU8 Benchmarks
// =============================================================================

/// Benchmark Table2dU8xU8::lookup_one() with various table sizes
fn bench_table2d_lookup_one(c: &mut Criterion) {
    let mut group = c.benchmark_group("table2d_lookup_one");

    // Test different table sizes
    let table_configs: &[(usize, usize, &str)] = &[
        (16, 16, "16x16"),
        (64, 64, "64x64"),
        (128, 128, "128x128"),
        (256, 256, "256x256"),
        (100, 50, "100x50"),
    ];

    let lookup_count = 10000;

    for &(num_rows, num_cols, name) in table_configs {
        let data = create_test_table_2d(num_rows, num_cols);
        let table = Table2dU8xU8::from_flat(&data, num_cols);
        let (rows, cols) = create_test_indices_2d(lookup_count, num_rows, num_cols);

        group.throughput(Throughput::Bytes((lookup_count * 16) as u64));

        group.bench_with_input(
            BenchmarkId::new("table_size", name),
            &(num_rows, num_cols),
            |b, _| {
                b.iter(|| {
                    for (r, c) in rows.iter().zip(cols.iter()) {
                        black_box(table.lookup_one(black_box(*r), black_box(*c)));
                    }
                });
            },
        );
    }

    group.finish();
}

/// Compare Table64 vs Table2dU8xU8 for 8x8 table
fn bench_table64_vs_table2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("table64_vs_table2d_8x8");

    // Create equivalent 8x8 tables
    let table64_data = create_test_table_64();
    let table64 = Table64::new(&table64_data);

    let table2d_data: Vec<u8> = table64_data.to_vec();
    let table2d = Table2dU8xU8::from_flat(&table2d_data, 8);

    let lookup_count = 10000;
    let (rows, cols) = create_test_indices_2d_8x8(lookup_count);

    group.throughput(Throughput::Bytes((lookup_count * 16) as u64));

    // Table64 with lookup_one_2d
    group.bench_function("table64_lookup_one_2d", |b| {
        b.iter(|| {
            for (r, c) in rows.iter().zip(cols.iter()) {
                black_box(table64.lookup_one_2d(black_box(*r), black_box(*c)));
            }
        });
    });

    // Table2dU8xU8 with lookup_one
    group.bench_function("table2d_lookup_one", |b| {
        b.iter(|| {
            for (r, c) in rows.iter().zip(cols.iter()) {
                black_box(table2d.lookup_one(black_box(*r), black_box(*c)));
            }
        });
    });

    // Scalar baseline
    group.bench_function("scalar_lookup", |b| {
        b.iter(|| {
            for (r_vec, c_vec) in rows.iter().zip(cols.iter()) {
                let r_arr = r_vec.to_array();
                let c_arr = c_vec.to_array();
                let mut out = [0u8; 16];
                for i in 0..16 {
                    out[i] = table64_data[(r_arr[i] as usize) * 8 + (c_arr[i] as usize)];
                }
                black_box(u8x16::from(out));
            }
        });
    });

    group.finish();
}

// =============================================================================
// Criterion Setup
// =============================================================================

criterion_group!(
    benches,
    bench_table64_lookup_one,
    bench_table64_lookup_one_2d,
    bench_table2d_lookup_one,
    bench_table64_vs_table2d,
);
criterion_main!(benches);
