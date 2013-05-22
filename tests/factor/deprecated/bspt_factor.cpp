#include <iostream>
#include <string>
#include <iterator>
#include <cmath>
#include <cstdio>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>

#include <sill/math/gdl_enum.hpp>
#include <sill/variable.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/bspt_factor.hpp>
#include <sill/map.hpp>
#include <sill/copy_ptr.hpp>

/////////////////////////////////////////////////////////////////
// Doesn't compile; needs to be cleaned up
/////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {

  using namespace sill;

  // Create a source of random numbers.
  boost::mt19937 rng;
  boost::uniform_01<boost::mt19937, double> unif01(rng);

  // Create a universe.
  universe univ;

  // Create some variables and an argument set.
  variable_h vh[4];
  const finite_value w_size = 2;
  const finite_value x_size = 3;
  const finite_value y_size = 2;
  const finite_value z_size = 10;
  const variable_h& w_v = vh[0] = univ.new_finite_variable(w_size);
  const variable_h& x_v = vh[1] = univ.new_finite_variable(x_size);
  const variable_h& y_v = vh[2] = univ.new_finite_variable(y_size);
  const variable_h& z_v = vh[3] = univ.new_finite_variable(z_size);
  domain wx(vh + 0, vh + 2);
  domain wxy(vh + 0, vh + 3);

  // Create two dense table factors over w and x and initialize them
  // randomly.
  typedef table_factor<double, dense_table> dense_table_factor_t;
  sill::copy_ptr<dense_table_factor_t>
    f1_ptr(new dense_table_factor_t(wx, 0.0)),
    f2_ptr(new dense_table_factor_t(wx, 0.0));
  sill::finite_value w, x, y, z;
  sill::assignment assignment;
  // Iterator over all assignments to the variables.
  for (assignment[w_v] = w = 0; w < w_size; assignment[w_v] = ++w)
    for (assignment[x_v] = x = 0; x < x_size; assignment[x_v] = ++x) {
      f1_ptr->set(assignment, unif01());
      f2_ptr->set(assignment, unif01());
    }

  // Create single-node BSPT factors for the table factors.
  typedef bspt_factor_t<dense_table_factor_t> factor_t;
  sill::copy_ptr<factor_t> g1_ptr(new factor_t(f1_ptr));
  sill::copy_ptr<factor_t> g2_ptr(new factor_t(f2_ptr));
  std::cout << "Singleton: " << *g1_ptr << std::endl;
  std::cout << "Another singleton: " << *g2_ptr << std::endl;

  // Create a stump (3-node) BSPT factor.
  sill::copy_ptr<factor_t> g_ptr(new factor_t(z_v, 0, *g1_ptr, *g2_ptr));
  std::cout << "Stump: " << *g_ptr << std::endl;

  // Create another stump (3-node) BSPT factor.
  sill::copy_ptr<factor_t> h_ptr(new factor_t(z_v, 4, *g1_ptr, *g2_ptr));
  std::cout << "Another stump: " << *h_ptr << std::endl;

  // Multiply together these two BSPT factors.
  typedef sill::sum_product_tag csr_tag;
  sill::copy_ptr<factor_t> gh_ptr(new factor_t(sill::combine(g_ptr, h_ptr,
                                                        product_tag())));
  std::cout << "Product: " << *gh_ptr << std::endl;

  // Create another stump (3-node) BSPT factor.
  sill::copy_ptr<factor_t> i_ptr(new factor_t(y_v, 0, *g1_ptr, *g2_ptr));
  std::cout << "Stump: " << *i_ptr << std::endl;

  // Create another stump (3-node) BSPT factor.
  sill::copy_ptr<factor_t> j_ptr(new factor_t(y_v, 1, *g1_ptr, *g2_ptr));
  std::cout << "Another stump: " << *j_ptr << std::endl;

  // Multiply together these two BSPT factors.
  sill::copy_ptr<factor_t> ij_ptr(new factor_t(sill::combine(i_ptr, j_ptr,
                                                        product_tag())));
  std::cout << "Product: " << *ij_ptr << std::endl;

  // Multiply together these two BSPT factors.
  sill::copy_ptr<factor_t> ghij_ptr(new factor_t(sill::combine(gh_ptr, ij_ptr,
                                                          product_tag())));
  std::cout << "Product: " << *ghij_ptr << std::endl;

  // Restrict this factor.
  sill::assignment r_assignment;
  r_assignment[z_v] = 0;
  factor_t restriction(sill::restrict(ghij_ptr, r_assignment));
  std::cout << "Restricted by " << r_assignment << ": " << std::endl
            << restriction << std::endl;

  // Multiply this product by two.
  ghij_ptr->combine_in(const_ptr_t<sill::constant_factor<int> >
                       (new sill::constant_factor<int>(2)),
                       product_tag());
  std::cout << "Scaled by 2: " << *ghij_ptr << std::endl;

  // Marginalize out z from the product.
  sill::copy_ptr<factor_t> ghij_wxy_ptr(new factor_t(sill::collapse(ghij_ptr, wxy,
                                                               sum_tag())));
  std::cout << "WXY Marginal: " << *ghij_wxy_ptr << std::endl;

  // Marginalize out y and z from the product.
  sill::copy_ptr<factor_t> ghij_wx_ptr(new factor_t(sill::collapse(ghij_ptr, wx,
                                                              sum_tag())));
  std::cout << "WX Marginal: " << *ghij_wx_ptr << std::endl;

  // Normalize the marginal.
  sill::normalize(ghij_wx_ptr);
  std::cout << "Normalized: " << *ghij_wx_ptr << std::endl;

  return EXIT_SUCCESS;
}
