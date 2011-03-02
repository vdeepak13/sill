
/**
 * \file suitesparse_test.cpp
 *       Test of SuiteSparse to see if it will work for SILL.
 */

/* Notes:
   - "cs_di" = CSparse double/int
   - "cs_dl" = CSparse double/UF_long
   - "cs_ci" = CSparse complex/int
   - "cs_cl" = CSparse complex/UF_long
   - int cs_di_cholsol (int order, const cs_di *A, double *b) ;
      - Solve Ax=b with Cholesky.
      - order:  0=natural, 1=Chol, 2=LU, 3=QR
   - cs_din *cs_di_chol (const cs_di *A, const cs_dis *S) ;
      - numeric Cholesky
      - A must be in CSC format.
   - cs_dis *cs_di_schol (int order, const cs_di *A) ;
      - symbolic Cholesky using amd(A+A')
      - A must be in CSC format.
 */

#include <cassert>
#include <iostream>

#include <suitesparse/cs.h>

int main(int argc, char** argv) {

  using namespace std;

  // Make matrix in triplet form.
  cs_di A;
  A.m = 4;
  A.n = A.m;
  A.nzmax = A.m * A.n;
  A.p = new int[A.n + 1];
  A.i = new int[A.nzmax];
  A.x = new double[A.nzmax];
  A.nz = -1;
  assert(A.m * A.n >= A.nzmax);
  for (size_t j = 0; j < A.n; ++j) {
    A.p[j] = A.m * j;
    for (size_t i = 0; i < A.m; ++i) {
      A.i[A.m * j + i] = i;
      if (i > j)
        A.x[A.m * j + i] = A.m * j + i + 1;
      else if (i < j)
        A.x[A.m * j + i] = A.x[A.m * i + j];
      else
        A.x[A.m * j + i] = A.m * 10;
    }
  }
  A.p[A.n] = A.nzmax;

  cout << "Built cs_di matrix A:\n";
  cs_di_print(&A, 0);
  cout << endl;

  int order = 0;
  cs_dis* S = cs_di_schol(order, &A);
  assert(S);
  cs_din* N = cs_di_chol(&A, S);
  assert(N); // This fails if A is not positive definite.
  assert(N->L);

  cout << "Computed chol(A):\n";
  cs_di_print(N->L, 0);
  cout << endl;

  // Free stuff.
  delete [] A.p;
  delete [] A.i;
  delete [] A.x;
  cs_di_sfree(S);
  cs_di_nfree(N);

} // main
