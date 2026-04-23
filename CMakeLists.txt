# Builds the Lumina benchmark suite.

cmake_minimum_required(VERSION 3.20)
project(lumina_benchmarks CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(LuminaBench)

include(FetchContent)
set(BENCHMARK_ENABLE_TESTING        OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS    OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL        OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR         OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1
)
FetchContent_MakeAvailable(benchmark)

set(DEPS_DIR "${CMAKE_SOURCE_DIR}/Dependencies")

add_library(lumina INTERFACE)
target_include_directories(lumina INTERFACE "${DEPS_DIR}/Lumina/include")
target_link_libraries(lumina INTERFACE "${DEPS_DIR}/Lumina/lib/lumina.lib")

add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE "${DEPS_DIR}/eigen-5.0.0")
target_compile_definitions(eigen INTERFACE
    EIGEN_NO_DEBUG
    EIGEN_MPL2_ONLY
)

option(LUMINA_BENCH_ARMA_NO_BLAS
    "Disable Armadillo's BLAS/LAPACK backend so core-op numbers reflect Arma's own C++ kernels. Solver benches break in this mode and must be filtered out at runtime."
    OFF)

set(ARMA_DIR "${DEPS_DIR}/armadillo-15.2.4")
add_library(armadillo INTERFACE)
target_include_directories(armadillo INTERFACE "${ARMA_DIR}/include")

if(LUMINA_BENCH_ARMA_NO_BLAS)
    message(STATUS "Armadillo: kernel-quality mode (no BLAS / no LAPACK)")
    target_compile_definitions(armadillo INTERFACE
        ARMA_DONT_USE_WRAPPER
        ARMA_DONT_USE_BLAS
        ARMA_DONT_USE_LAPACK
        ARMA_DONT_USE_OPENBLAS
        ARMA_NO_DEBUG
    )
else()
    message(STATUS "Armadillo: production config (OpenBLAS linked)")
    target_link_libraries(armadillo INTERFACE
        "${ARMA_DIR}/examples/lib_win64/libopenblas.lib")
    target_compile_definitions(armadillo INTERFACE
        ARMA_DONT_USE_WRAPPER
        ARMA_NO_DEBUG
    )
    set(OPENBLAS_DLL "${ARMA_DIR}/examples/lib_win64/libopenblas.dll")
endif()

add_library(bench_common INTERFACE)
target_include_directories(bench_common INTERFACE "${CMAKE_SOURCE_DIR}/benchmarks")
target_link_libraries(bench_common INTERFACE
    benchmark::benchmark
    lumina
    eigen
    armadillo
)

function(LuminaAddBenchmark name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE bench_common benchmark::benchmark_main)
    LuminaApplyFlags(${name})

    if(WIN32 AND DEFINED OPENBLAS_DLL)
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENBLAS_DLL}"
                "$<TARGET_FILE_DIR:${name}>/libopenblas.dll"
        )
    endif()
endfunction()

LuminaAddBenchmark(bench_matmul      benchmarks/core/bench_matmul.cpp)
LuminaAddBenchmark(bench_matvec      benchmarks/core/bench_matvec.cpp)
LuminaAddBenchmark(bench_transpose   benchmarks/core/bench_transpose.cpp)
LuminaAddBenchmark(bench_add_scale   benchmarks/core/bench_add_scale.cpp)
LuminaAddBenchmark(bench_dot_norm    benchmarks/core/bench_dot_norm.cpp)
LuminaAddBenchmark(bench_reductions  benchmarks/core/bench_reductions.cpp)

LuminaAddBenchmark(bench_lu          benchmarks/linalg/bench_lu.cpp)
LuminaAddBenchmark(bench_cholesky    benchmarks/linalg/bench_cholesky.cpp)
LuminaAddBenchmark(bench_gaussian    benchmarks/linalg/bench_gaussian.cpp)
LuminaAddBenchmark(bench_eigen       benchmarks/linalg/bench_eigen.cpp)
LuminaAddBenchmark(bench_iterative   benchmarks/linalg/bench_iterative.cpp)

LuminaAddBenchmark(bench_euler       benchmarks/ode/bench_euler.cpp)
LuminaAddBenchmark(bench_rk4         benchmarks/ode/bench_rk4.cpp)
LuminaAddBenchmark(bench_rk45        benchmarks/ode/bench_rk45.cpp)

LuminaAddBenchmark(bench_poisson_cg  benchmarks/system/bench_poisson_cg.cpp)
LuminaAddBenchmark(bench_heat_rk4    benchmarks/system/bench_heat_rk4.cpp)
LuminaAddBenchmark(bench_pca_like    benchmarks/system/bench_pca_like.cpp)

LuminaAddBenchmark(bench_suitesparse benchmarks/suitesparse/bench_suitesparse.cpp)
target_compile_definitions(bench_suitesparse PRIVATE
    LUMINA_BENCH_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

enable_testing()

add_library(gtest_shipped INTERFACE)
target_include_directories(gtest_shipped INTERFACE "${DEPS_DIR}/Lumina/include")
target_link_libraries(gtest_shipped INTERFACE
    "${DEPS_DIR}/Lumina/lib/gtest.lib"
    "${DEPS_DIR}/Lumina/lib/gtest_main.lib"
)

add_executable(tests_numerical_equivalence
    tests/test_numerical_equivalence.cpp)
target_link_libraries(tests_numerical_equivalence PRIVATE
    bench_common gtest_shipped)
LuminaApplyFlags(tests_numerical_equivalence)
add_test(NAME numerical_equivalence COMMAND tests_numerical_equivalence)
