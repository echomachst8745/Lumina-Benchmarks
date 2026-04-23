# Lumina Benchmark Suite

This project benchmarks Lumina against Eigen and Armadillo on a set of small
linear algebra, ODE, and system-style workloads. The benchmark programs write
Google Benchmark JSON files, and the Python script turns those files into CSVs
and plots.

## Build

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Benchmark executables are written to:

```text
build/Release/bench_*.exe
```

## Run Tests

```powershell
ctest --test-dir build -C Release --output-on-failure
```

These tests check that Lumina and Eigen produce close numerical results on the
operations used by the benchmarks.

## Download SuiteSparse Matrices

Run this before using the SuiteSparse benchmark:

```powershell
python scripts\download_matrices.py
```

The files are downloaded into `data/suitesparse/`.

## Run Benchmarks

Run the full benchmark set:

```powershell
.\scripts\run_all.ps1
```

Results are written to a timestamped folder under `results/`.

You can also choose an output folder:

```powershell
.\scripts\run_all.ps1 -OutputDir results\my-run
```

For a quick smoke run:

```powershell
.\scripts\run_all.ps1 -OutputDir results\smoke -MinTime 0.05s -Filter 'double>, 64'
```

## Analyze Results

After a run finishes:

```powershell
python scripts\analyze_results.py results\<run-folder>
```

This creates:

- `analysis/long.csv`
- `analysis/by_op/*.csv`
- `analysis/speedup_vs_lumina.csv`
- `analysis/plots/*.png`

## Optional No-BLAS Build

The default build links Armadillo with OpenBLAS. For a separate run where
Armadillo uses its own C++ kernels for the core operations:

```powershell
cmake -S . -B build-noblas -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DLUMINA_BENCH_ARMA_NO_BLAS=ON
cmake --build build-noblas --config Release -j
```

Only run the core operation benchmarks in this mode:

```powershell
.\scripts\run_all.ps1 -BuildDir build-noblas -OutputDir results\no-blas `
    -Filter '(MatMul|MatVec|Transpose|AddScale|Dot|Norm|Frobenius|Trace|Sum)'
```
