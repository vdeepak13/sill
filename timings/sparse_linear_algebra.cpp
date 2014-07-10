
#include <boost/timer.hpp>

#include <sill/math/linear_algebra/sparse_linear_algebra.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

//! Timing: dense matrix += outer_product(dense vector, sparse vector)
void time_dm_pe_outer_product_dv_sv(int argc);

//! Timing: dense vector += dense matrix * sparse vector
void time_dv_pe_dm_times_sv();

int main(int argc, char** argv) {

  //  time_dv_pe_dm_times_sv();

}

void time_dm_pe_outer_product_dv_sv(int argc) {
  // Timing: dense matrix += outer_product(dense vector, sparse vector)
  size_t m = 100;
  size_t n = 10000000;
  double vec_sparsity = .01;
  mat dm(m,n);
  dm.zeros();
  vec dv(m);
  for (size_t i = 0; i < m; ++i)
    dv[i] = i+1;
  sparse_vector<double> sv(n,(size_t)(n * vec_sparsity));
  for (size_t k = 0; k < sv.num_non_zeros(); ++k) {
    sv.index(k) = (size_t)(.5 / vec_sparsity) * k + 1;
    sv.value(k) = k + 1;
  }
  sv.sort_indices();

  std::cout
    << "Timing: dense matrix += outer_product(dense vector, sparse vector)"
    << "  with m = " << m << ", n = " << n << ", sparsity = " << vec_sparsity
    << std::endl;
  size_t runs = 100;
  boost::timer timer;
  for (size_t run = 0; run < runs; ++run)
    dm += outer_product(dv, sv);
  double elapsed = timer.elapsed();
  std::cout << "Average time (over " << runs << " runs): " << (elapsed / runs)
            << " secs" << std::endl;
  if (argc > 1000)
    std::cout << dm;
}

void time_dv_pe_dm_times_sv() {
  // Timing: dense vector += dense matrix * sparse vector
  // size_t m = 10;
  assert(false); // TO DO
}

#include <sill/macros_undef.hpp>
