use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let comp_bytes = std::fs::read("test_data/index_buffer.compressed.bin").unwrap();

    let num_indices = 54348;
    let num_trianges = num_indices / 3;

    c.bench_function("bench", |b| {
        b.iter(|| {
            let iter = meshopt_decoder::TriangleIterator::new(&comp_bytes, num_indices).unwrap();
            let mut got = 0;

            for tri in iter {
                black_box(tri);
                got += 1;
            }

            assert_eq!(got, num_trianges);
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
