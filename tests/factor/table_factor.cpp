#include <iostream>
#include <string>
#include <iterator>
#include <cmath>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int.hpp>

#include <sill/math/gdl_enum.hpp>
#include <sill/variable.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/datastructure/sparse_table.hpp>
#include <sill/map.hpp>
#include <sill/copy_ptr.hpp>
#include <sill/factor/random.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

boost::mt19937 rng;

int main(int argc, char** argv) {

  // Create a source of random numbers.
  boost::mt19937 rng;
  boost::uniform_01<boost::mt19937, double> unif01(rng);

  // Create a universe.
  universe u;

  // Create some variables and an argument set.
  finite_variable* vh[4];
  finite_variable* w_v = vh[0] = u.new_finite_variable(3);
  finite_variable* x_v = vh[1] = u.new_finite_variable(5);
  finite_variable* y_v = vh[2] = u.new_finite_variable(2);
  finite_variable* z_v = vh[3] = u.new_finite_variable(10);
  size_t w, x, y, z;
  finite_domain wxy(vh + 0, vh + 3);

  typedef table_factor< sparse_table_t<double> > sparse_factor;
  typedef table_factor< dense_table<double> > dense_factor;
  finite_assignment a;

//   dense_factor f1 = random_factor<dense_table>(wxy, 0, 1);
//   std::cout << f1 << std::endl;
//   assignment a; a[x_v] = 0;
//   dense_factor f2 = restrict(a, f1);
//   std::cout << f2 << std::endl;

  // Test sparse and dense table combine/collapse factor operations
  // for varying fill factors and default values that correspond to
  // the zero and identity of the multiplication operation.
  std::size_t defaults = 0;
  for (double fill_factor = 0; fill_factor <= 1.0; fill_factor += 0.01) {
    double f_default = (defaults & 0x01) ? 0 : 1;
    double g_default = (defaults & 0x02) ? 0 : 1;

    // Create a sparse table factor over x, y, and z.
    sparse_factor f_sparse =
      random_discrete_factor<sparse_factor>(wxy, rng, f_default, fill_factor);

    // Create an equivalent dense table.
    dense_factor f_dense(f_sparse);

    // Compute the maximum/minimum-likelihood assignment
    finite_assignment amax = f_dense.arg_max();
    assert(f_dense(amax) == f_dense.maximum());

    finite_assignment amin = f_dense.arg_min();
    assert(f_dense(amin) == f_dense.minimum());

    //std::cout << f_sparse << std::endl;

    // Create a sparse table factor over x, y, and z.
    finite_domain xyz(vh + 1, vh + 4);
    sparse_factor g_sparse =
      random_discrete_factor<sparse_factor>(xyz, rng, g_default,1-fill_factor);

    // Create an equivalent dense table factor.
    dense_factor g_dense(g_sparse);

    // Multiply f and g using the sparse representation.
    sparse_factor h_sparse(combine(f_sparse, g_sparse, product_op));

    // Multiply f and g using the dense representation.
    dense_factor  h_dense(combine(f_dense, g_dense, product_op));

    // Compare the answers.
    for (a[w_v] = w = 0; w < 3; a[w_v] = ++w)
      for (a[x_v] = x = 0; x < 5; a[x_v] = ++x)
        for (a[y_v] = y = 0; y < 2; a[y_v] = ++y)
          for (a[z_v] = z = 0; z < 10; a[z_v] = ++z) {
            double dense_val = h_dense(a);
            double sparse_val = h_sparse(a);
            if (fabs(dense_val - sparse_val) > 1e-8) {
              std::cout << "Multiplication in dense and sparse representations not equivalent!"
                        << std::endl
                        << "Left default: " << f_default << "; "
                        << "Right default: " << g_default << std::endl
                        << "Dense value: " << dense_val << "; "
                        << "Sparse value: " << sparse_val << std::endl
                        << "Correct answer: "
                        << f_dense(a) << " * "
                        << g_dense(a) << " = "
                        << f_sparse(a) << " * "
                        << g_sparse(a) << " = "
                        << (f_dense(a) *
                           g_dense(a))
                        << std::endl;
              assert(false);
            }
          }


    // Collapse both factors down to xyz.
    dense_factor h_xyz_dense(collapse(h_dense, sum_op, xyz));
    sparse_factor h_xyz_sparse;
    try {
      h_xyz_sparse = collapse(h_sparse, sum_op, xyz);
    } catch (const std::runtime_error&) {
      std::cout << "Continuing" << std::endl;
      // Aggregation of non-empty sparse tables with non-identity
      // default element not yet implemented.
      continue;
    }

//     if (fill_factor<0.5) {
//       assignment a1;
//       a1[x_v] = 0;
//       dense_factor ff = restrict(a1, h_xyz_dense);
//       std::cout << ff << std::endl;
//       std::cout << h_xyz_dense << std::endl;
//     }


    // Compute the correct answer.
    double sum[5][2][10];
    for (int i = 0; i < 5; ++i) {
      a[x_v] = i;
      for (int j = 0; j < 2; ++j) {
        a[y_v] = j;
        for (int k = 0; k < 10; ++k) {
          a[z_v] = k;
          sum[i][j][k] = 0.0;
          for (int l = 0; l < 3; ++l) {
            a[w_v] = l;
            sum[i][j][k] += h_sparse(a);
          }
        }
      }
    }

    // Compare the answers.
    for (a[x_v] = x = 0; x < 5; a[x_v] = ++x)
      for (a[y_v] = y = 0; y < 2; a[y_v] = ++y)
        for (a[z_v] = z = 0; z < 10; a[z_v] = ++z) {
          double correct = sum[x][y][z];
          assert(fabs(correct - h_xyz_dense(a)) < 1e-8);
          assert(fabs(correct - h_xyz_sparse(a)) < 1e-8);
        }

    // Compare join_aggregate.
    dense_factor diff = dense_factor::join(f_dense, g_dense, abs_difference<double>());
    double correct = collapse(diff, max_op, finite_domain())(finite_assignment());

    // std::cout << correct << " " << std::endl;
    assert(fabs(correct - norm_inf(f_dense, g_dense)) < 1e-8);
    assert(fabs(norm_inf(f_dense, f_dense)) < 1e-10);

  }

  // SPARSE COMBINE SPECIAL CASE TEST
//   {
//     // Create some variables and an argument set.
//     finite_variable* vh[2];
//     const finite_variable*& x_v = vh[0] = u.new_finite_variable(1000);
//     const finite_variable*& y_v = vh[1] = u.new_finite_variable(1000);
//     domain xy(vh + 0, vh + 2);
//
//     // Create two sparse factors.  The first's default is zero, and
//     // the second's default is one.  Choose their explicit elements so
//     // that they do not overlap.
//     sill::copy_ptr<sparse_factor>
//       a_ptr(new sparse_factor(xy, 0.0));
//     sill::copy_ptr<sparse_factor>
//       b_ptr(new sparse_factor(xy, 1.0));
//     for (a[x_v] = x = 0; x < 100; a[x_v] = ++x)
//       for (a[y_v] = y = 0; y < 100; a[y_v] = ++y)
//      a_ptr->set(a, unif01());
//     for (a[x_v] = 100; x < 200; a[x_v] = ++x)
//       for (a[y_v] = 100; y < 200; a[y_v] = ++y)
//      b_ptr->set(a, unif01());
//     // Combine b into (a copy of) a.  This should notice that there is
//     // no overlap in the explicit elements and do no work.
//     sill::copy_ptr<sparse_factor> a_copy_ptr(new sparse_factor(*a_ptr));
//     a_copy_ptr->combine_in(b_ptr, product_op);
//
//     // Create another sparse factor like b, but with a smaller number
//     // of explicit elements that overlap.
//     sill::copy_ptr<sparse_factor>
//       c_ptr(new sparse_factor(xy, 1.0));
//     for (a[x_v] = x = 0; x < 50; a[x_v] = ++x)
//       for (a[y_v] = y = 0; y < 50; a[y_v] = ++y)
//      c_ptr->set(a, 0.0);
//     // Combine c into (a copy of) a.  This should use a linear scan
//     // over the explicit elements of c, joining each element into its
//     // corresponding element of a.
//     a_copy_ptr =
//       sill::copy_ptr<sparse_factor>(new sparse_factor(*a_ptr));
//     a_copy_ptr->combine_in(c_ptr, product_op);
//
//     // Create another sparse factor like c, but with a larger number
//     // of explicit elements that overlap.
//     sill::copy_ptr<sparse_factor>
//       d_ptr(new sparse_factor(xy, 1.0));
//     for (a[x_v] = x = 0; x < 150; a[x_v] = ++x)
//       for (a[y_v] = y = 0; y < 150; a[y_v] = ++y)
//      d_ptr->set(a, 0.0);
//     // Combine d into (a copy of) a.  This should use a linear scan
//     // over the explicit elements of d, joining each element into its
//     // corresponding element of a.
//     a_copy_ptr =
//       sill::copy_ptr<sparse_factor>(new sparse_factor(*a_ptr));
//     a_copy_ptr->combine_in(d_ptr, product_op);
//
//     // Create a sparse factor over just the variable x, whose explicit
//     // elements join with more elements of a than are specified
//     // explicitly.
//     sill::copy_ptr<sparse_factor>
//       e_ptr(new sparse_factor(domain(x_v), 1.0));
//     for (a[x_v] = x = 0; x < 150; a[x_v] = ++x)
//       e_ptr->set(a, 0.0);
//     // Combine e into (a copy of) a.  This should use a linear scan
//     // over the explicit elements of e, joining each element into its
//     // corresponding elements (plural) of a.
//     a_copy_ptr =
//       sill::copy_ptr<sparse_factor>(new sparse_factor(*a_ptr));
//     a_copy_ptr->combine_in(e_ptr, product_op);
//
//     // Create two sparse factors whose default elements are neither
//     // zero nor one.
//     sill::copy_ptr<sparse_factor>
//       f_ptr(new sparse_factor(domain(x_v), 2.0));
//     for (a[x_v] = x = 0; x < 100; a[x_v] = ++x)
//       f_ptr->set(a, unif01());
//     sill::copy_ptr<sparse_factor>
//       g_ptr(new sparse_factor(domain(x_v), -1.0));
//     for (a[x_v] = 100; x < 200; a[x_v] = ++x)
//       g_ptr->set(a, unif01());
//     // Combine f into g.  This should use a linear scan over the
//     // implicit and explicit elements of g, joining each element into
//     // its corresponding elements (plural) of f.
//     g_ptr->combine_in(f_ptr, product_op);
//
//   }

  // Test unrolling and rolling up factors.
  dense_factor f_rolled = random_range_discrete_factor<dense_factor>
    (wxy, rng, 0, 1);
  finite_variable* f_unrolled_v;
  dense_factor f_unrolled;
  boost::tie(f_unrolled_v, f_unrolled) = f_rolled.unroll(u);
  dense_factor f_rolled_again = f_unrolled.roll_up(f_rolled.arg_list());
  foreach(finite_assignment a, f_rolled.assignments())
    if (f_rolled(a) != f_rolled_again(a))
      std::cout << "In test of unrolling and rolling up factors, "
                << " rolling up did not reverse unrolling.\n"
                << "Original (rolled) factor: " << f_rolled
                << "\nUnrolled factor: " << f_unrolled
                << "\nRolled-up factor: " << f_rolled_again << std::endl;

  return EXIT_SUCCESS;
}
