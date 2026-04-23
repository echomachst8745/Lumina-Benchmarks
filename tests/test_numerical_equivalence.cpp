// Checks numerical agreement between Lumina and Eigen benchmark operations.

#include <gtest/gtest.h>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "common/adapters.hpp"
#include "common/generators.hpp"

#include <lumina/linearalgebra/cholesky.hpp>
#include <lumina/linearalgebra/eigen_solvers.hpp>
#include <lumina/linearalgebra/gaussian_elimination.hpp>
#include <lumina/linearalgebra/lu_decomposition.hpp>

namespace {

template<class T> constexpr T TolFor();
template<> constexpr double TolFor<double>() { return 1e-10; }
template<> constexpr float  TolFor<float>()  { return 1e-5f; }

using bench::LuminaTag;
using bench::EigenTag;

template<class T, std::size_t N, std::size_t M>
void ExpectMatClose(const lumina::core::Matrix<T, N, M>& L,
                      const Eigen::Matrix<T, int(N), int(M)>& E,
                      T tol = TolFor<T>())
{
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < M; ++j)
            EXPECT_NEAR(L(i, j), E(int(i), int(j)), tol)
                << "mismatch at (" << i << "," << j << ")";
}

template<class T, std::size_t N>
void ExpectVecClose(const lumina::core::Vector<T, N>& L,
                      const Eigen::Matrix<T, int(N), 1>& E,
                      T tol = TolFor<T>())
{
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_NEAR(L[i], E(int(i)), tol) << "mismatch at [" << i << "]";
}

template<class T, std::size_t N>
void CheckMatmul()
{
    auto La = bench::MakeRandomMatrix<LuminaTag<T>, N, N>(1);
    auto Lb = bench::MakeRandomMatrix<LuminaTag<T>, N, N>(2);
    auto Ea = bench::MakeRandomMatrix<EigenTag<T>,  N, N>(1);
    auto Eb = bench::MakeRandomMatrix<EigenTag<T>,  N, N>(2);

    lumina::core::Matrix<T, N, N>    Lc = La * Lb;
    Eigen::Matrix<T, int(N), int(N)> Ec = Ea * Eb;

    ExpectMatClose(Lc, Ec, TolFor<T>() * static_cast<T>(N));
}

template<class T, std::size_t N>
void CheckMatvec()
{
    auto La = bench::MakeRandomMatrix<LuminaTag<T>, N, N>(1);
    auto Lv = bench::MakeRandomVector<LuminaTag<T>, N>(2);
    auto Ea = bench::MakeRandomMatrix<EigenTag<T>,  N, N>(1);
    auto Ev = bench::MakeRandomVector<EigenTag<T>,  N>(2);

    lumina::core::Vector<T, N>    Lr = La * Lv;
    Eigen::Matrix<T, int(N), 1>   Er = Ea * Ev;
    ExpectVecClose(Lr, Er, TolFor<T>() * static_cast<T>(N));
}

template<class T, std::size_t N>
void CheckTranspose()
{
    auto La = bench::MakeRandomMatrix<LuminaTag<T>, N, N>(1);
    auto Ea = bench::MakeRandomMatrix<EigenTag<T>,  N, N>(1);
    lumina::core::Matrix<T, N, N>    Lt = La.Transpose();
    Eigen::Matrix<T, int(N), int(N)> Et = Ea.transpose();
    ExpectMatClose(Lt, Et);
}

template<class T, std::size_t N>
void CheckAddAndScale()
{
    auto La = bench::MakeRandomMatrix<LuminaTag<T>, N, N>(1);
    auto Lb = bench::MakeRandomMatrix<LuminaTag<T>, N, N>(2);
    auto Ea = bench::MakeRandomMatrix<EigenTag<T>,  N, N>(1);
    auto Eb = bench::MakeRandomMatrix<EigenTag<T>,  N, N>(2);

    const T c = T{2.5};
    lumina::core::Matrix<T, N, N>    Ls = (La + Lb) * c;
    Eigen::Matrix<T, int(N), int(N)> Es = (Ea + Eb) * c;
    ExpectMatClose(Ls, Es);
}

template<class T, std::size_t N>
void CheckDotAndNorms()
{
    auto Lu = bench::MakeRandomVector<LuminaTag<T>, N>(1);
    auto Lv = bench::MakeRandomVector<LuminaTag<T>, N>(2);
    auto Eu = bench::MakeRandomVector<EigenTag<T>,  N>(1);
    auto Ev = bench::MakeRandomVector<EigenTag<T>,  N>(2);

    EXPECT_NEAR(Lu.Dot(Lv),  Eu.dot(Ev),                       TolFor<T>() * static_cast<T>(N));
    EXPECT_NEAR(Lu.Norm(),   Eu.norm(),                        TolFor<T>());
    EXPECT_NEAR(Lu.NormL1(), Eu.template lpNorm<1>(),          TolFor<T>());
    EXPECT_NEAR(Lu.NormLInf(), Eu.template lpNorm<Eigen::Infinity>(), TolFor<T>());
}

template<class T, std::size_t N>
void CheckReductions()
{
    auto La = bench::MakeRandomMatrix<LuminaTag<T>, N, N>(1);
    auto Ea = bench::MakeRandomMatrix<EigenTag<T>,  N, N>(1);
    auto Lv = bench::MakeRandomVector<LuminaTag<T>, N>(2);
    auto Ev = bench::MakeRandomVector<EigenTag<T>,  N>(2);

    EXPECT_NEAR(La.Trace(),         Ea.trace(), TolFor<T>() * static_cast<T>(N));
    EXPECT_NEAR(La.FrobeniusNorm(), Ea.norm(),  TolFor<T>() * static_cast<T>(N));
    EXPECT_NEAR(Lv.Sum(),           Ev.sum(),   TolFor<T>() * static_cast<T>(N));
}

template<class T, std::size_t N>
void CheckLuSolve()
{
    auto La = bench::MakeDiagDominantMatrix<LuminaTag<T>, N>(42);
    auto Lb = bench::MakeRandomVector<LuminaTag<T>, N>(43);
    auto Ea = bench::MakeDiagDominantMatrix<EigenTag<T>,  N>(42);
    auto Eb = bench::MakeRandomVector<EigenTag<T>,  N>(43);

    auto Lx = lumina::linearalgebra::LUSolve(La, Lb);
    Eigen::Matrix<T, int(N), 1> Ex = Ea.partialPivLu().solve(Eb);
    ExpectVecClose(Lx, Ex, TolFor<T>() * static_cast<T>(N));
}

template<class T, std::size_t N>
void CheckCholeskySolve()
{
    auto La = bench::MakeSpdMatrix<LuminaTag<T>, N>(42);
    auto Lb = bench::MakeRandomVector<LuminaTag<T>, N>(43);
    auto Ea = bench::MakeSpdMatrix<EigenTag<T>,  N>(42);
    auto Eb = bench::MakeRandomVector<EigenTag<T>,  N>(43);

    auto Lx = lumina::linearalgebra::CholeskySolve(La, Lb);
    Eigen::Matrix<T, int(N), 1> Ex = Ea.llt().solve(Eb);
    ExpectVecClose(Lx, Ex, TolFor<T>() * static_cast<T>(N));
}

template<class T, std::size_t N>
void CheckGaussianSolve()
{
    auto La = bench::MakeDiagDominantMatrix<LuminaTag<T>, N>(42);
    auto Lb = bench::MakeRandomVector<LuminaTag<T>, N>(43);
    auto Ea = bench::MakeDiagDominantMatrix<EigenTag<T>,  N>(42);
    auto Eb = bench::MakeRandomVector<EigenTag<T>,  N>(43);

    auto Lx = lumina::linearalgebra::GaussianSolve(La, Lb);
    Eigen::Matrix<T, int(N), 1> Ex = Ea.partialPivLu().solve(Eb);
    ExpectVecClose(Lx, Ex, TolFor<T>() * static_cast<T>(N));
}

template<class T, std::size_t N>
void CheckInverse()
{
    auto La = bench::MakeDiagDominantMatrix<LuminaTag<T>, N>(42);
    auto Ea = bench::MakeDiagDominantMatrix<EigenTag<T>,  N>(42);

    lumina::core::Matrix<T, N, N>    Li = lumina::linearalgebra::Inverse(La);
    Eigen::Matrix<T, int(N), int(N)> Ei = Ea.inverse();
    ExpectMatClose(Li, Ei, TolFor<T>() * static_cast<T>(N));
}

template<class T, std::size_t N>
void CheckDeterminant()
{

    if constexpr (std::is_same_v<T, float> && N >= 64) {
        GTEST_SKIP() << "float det at N>=64 exceeds dynamic range";
    } else {
        auto La = bench::MakeDiagDominantMatrix<LuminaTag<T>, N>(42);
        auto Ea = bench::MakeDiagDominantMatrix<EigenTag<T>,  N>(42);

        T Ld = lumina::linearalgebra::Determinant(La);
        T Ed = Ea.determinant();
        T rel = std::abs(Ld - Ed) / std::max<T>(std::abs(Ed), T{1});
        EXPECT_LT(rel, TolFor<T>() * static_cast<T>(N));
    }
}

template<class T, std::size_t N>
void CheckDominantEigenvalue()
{
    auto La = bench::MakeSpdMatrix<LuminaTag<T>, N>(42);
    auto Ea = bench::MakeSpdMatrix<EigenTag<T>,  N>(42);

    auto Lres = lumina::linearalgebra::PowerIteration(
        La, lumina::core::Vector<T, N>{}, 1000, T{1e-12});

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, int(N), int(N)>> es(Ea);
    T lambda_max = es.eigenvalues()(int(N) - 1);

    T band = std::max<T>(T{1e-8},
                         TolFor<T>() * std::abs(lambda_max) * static_cast<T>(N));
    EXPECT_NEAR(Lres.eigenvalue, lambda_max, band);
}

#define EQUIV_TEST(NAME, CHECK)                                          \
    TEST(NumericalEquivalence, NAME##_Double_8)  { CHECK<double, 8>();  } \
    TEST(NumericalEquivalence, NAME##_Double_32) { CHECK<double, 32>(); } \
    TEST(NumericalEquivalence, NAME##_Double_64) { CHECK<double, 64>(); } \
    TEST(NumericalEquivalence, NAME##_Float_8)   { CHECK<float,  8>();  } \
    TEST(NumericalEquivalence, NAME##_Float_32)  { CHECK<float,  32>(); } \
    TEST(NumericalEquivalence, NAME##_Float_64)  { CHECK<float,  64>(); }

EQUIV_TEST(Matmul,             CheckMatmul)
EQUIV_TEST(Matvec,             CheckMatvec)
EQUIV_TEST(Transpose,          CheckTranspose)
EQUIV_TEST(AddScale,           CheckAddAndScale)
EQUIV_TEST(DotAndNorms,        CheckDotAndNorms)
EQUIV_TEST(Reductions,         CheckReductions)
EQUIV_TEST(LUSolve,            CheckLuSolve)
EQUIV_TEST(CholeskySolve,      CheckCholeskySolve)
EQUIV_TEST(GaussianSolve,      CheckGaussianSolve)
EQUIV_TEST(Inverse,            CheckInverse)
EQUIV_TEST(Determinant,        CheckDeterminant)
EQUIV_TEST(DominantEigenvalue, CheckDominantEigenvalue)

}
