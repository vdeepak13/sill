
#ifndef _SILL_BLAS_HPP_
#define _SILL_BLAS_HPP_

/**
 * \file blas.hpp  Declarations of BLAS functions.
 */

#include <complex>

namespace blas {

  extern "C" {

    //==========================================================================
    // BLAS 1
    //==========================================================================

    void dswap_(const int* n,
                double* x, const int* incx,
                double* y, const int* incy);

    void zswap_(const int* n,
                std::complex<double>* x, const int* incx,
                std::complex<double>* y, const int* incy);

    void dscal_(const int* n, const double* alpha,
                double* x, const int* incx);

    void zscal_(const int* n, const std::complex<double>* alpha,
                std::complex<double>* x, const int* incx);

    void dcopy_(const int* n,
                const double* x, const int* incx,
                double* y, const int* incy);

    void zcopy_(const int* n,
                const std::complex<double>* x, const int* incx,
                std::complex<double>* y, const int* incy);

    void daxpy_(const int* n, const double* alpha,
                const double* x, const int* incx,
                double* y, const int* incy);

    //==========================================================================
    // BLAS 2
    //==========================================================================

    void dgemv_(const char* transA, const int* m, const int* n,
                const double* alpha,
                const double* A, const int* ldA,
                const double* x, const int* incx,
                const double* beta,
                double* y, const int* incy);

    void zgemv_(const char* transA, const int* m, const int* n,
                const std::complex<double>* alpha,
                const std::complex<double>* A, const int* ldA,
                const std::complex<double>* x, const int* incx,
                const std::complex<double>* beta,
                std::complex<double>* y, const int* incy);

    void dger_(const int* m, const int* n,
               const double* alpha,
               const double* x, const int* incx,
               const double* y, const int* incy,
               double* A, const int* ldA);

    void zgeru_(const int* m, const int* n,
                const std::complex<double>* alpha,
                const std::complex<double>* x, const int* inxx,
                const std::complex<double>* y, const int* incy,
                std::complex<double>* A, const int* ldA);

    void zgerc_(const int* m, const int* n,
                const std::complex<double>* alpha,
                const std::complex<double>* x, const int* inxx,
                const std::complex<double>* y, const int* incy,
                std::complex<double>* A, const int* ldA);

    //==========================================================================
    // BLAS 3
    //==========================================================================

    void dgemm_(const char* transA, const char* transB,
                const int* m, const int* n, const int* k,
                const double* alpha,
                const double* A, const int* ldA,
                const double* B, const int* ldB,
                const double* beta,
                double* C, const int* ldC);

    void zgemm_(const char* transA, const char* transB,
                const int* m, const int* n, const int* k,
                const std::complex<double>* alpha,
                const std::complex<double>* A, const int* ldA,
                const std::complex<double>* B, const int* ldB,
                const std::complex<double>* beta,
                std::complex<double>* C, const int* ldC);

  } // extern "C"

} // namespace blas


#endif // #ifndef _SILL_BLAS_HPP_
