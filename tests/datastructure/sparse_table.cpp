#include <cmath>
#include <iostream>
#include <functional>
#include <boost/tuple/tuple.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/progress.hpp>
#include <sill/math/gdl_enum.hpp>
#include <sill/datastructure/sparse_table.hpp>


// This file needs to be updated to the new interfaces; does not compile

int main(int argc, char** argv) {

  using namespace sill;

  // Create two large sparse tables of booleans with a dimension in common.
  typedef sill::sparse_table_t<double> sd_table_t;
  const table_size_t d = 2;
  const table_size_t p = 10000;
  const table_size_t q = 100;
  const table_size_t r = 1000;
  table_size_t a_dims[d] = {p, q};
  sd_table_t a_table(a_dims, a_dims + d, 0.0);
  table_size_t b_dims[d] = {r, q};
  sd_table_t b_table(b_dims, b_dims + d, 0.0);

  // Populate k cells in each table with nonzero elements.
  const table_size_t k = 5000;
  boost::mt19937 rng;
  boost::uniform_int<table_size_t> p_unif(0, p - 1);
  boost::uniform_int<table_size_t> q_unif(0, q - 1);
  boost::uniform_int<table_size_t> r_unif(0, r - 1);
  boost::uniform_01<boost::mt19937, double> unif01(rng);
  v_table_index index(d);
  for (table_size_t i = 0; i < k; ++i) {
    index[0] = p_unif(rng);
    index[1] = q_unif(rng);
    a_table.set(index, unif01());
    index[0] = r_unif(rng);
    index[1] = q_unif(rng);
    b_table.set(index, unif01());
  }
  std::cout << "Table a has " << a_table.num_explicit_elts()
            << "/" << a_table.num_elts()
            << " explicit elements." << std::endl;
  std::cout << "Table b has " << b_table.num_explicit_elts()
            << "/" << b_table.num_elts()
            << " explicit elements." << std::endl;

  // Compute the size of the join by explicit cross product.
  std::cout << "Joining a and b should yield..." << std::flush;
  table_size_t size = 0;
  typedef sd_table_t::ielt_it_t ielt_it_t;
  ielt_it_t a_it, a_end;
  v_table_index a_index, b_index;
  bool a_elt, b_elt;
  ielt_it_t b_begin, b_end;
  boost::tie(b_begin, b_end) = b_table.indexed_elements();
  for (boost::tie(a_it, a_end) = a_table.indexed_elements();
       a_it != a_end; ++a_it) {
    boost::tie(a_index, a_elt) = *a_it;
    for (ielt_it_t b_it = b_begin; b_it != b_end; ++b_it) {
      boost::tie(b_index, b_elt) = *b_it;
      if (a_index[1] == b_index[1])
        ++size;
    }
  }
  std::cout << size << "/" << p * q * r << " explicit elements." << std::endl;

  // Compute the join using sort-merge join.
  table_size_t c_dims[3] = {p, q, r};
  sd_table_t c_table(c_dims, c_dims + 3, 0.0);
  table_size_t a_dim_map[2] = {0, 1}; // maps dimensions of a_table to c_table
  table_size_t b_dim_map[2] = {2, 1}; // maps dimensions of b_table to c_table
  boost::timer t;
  join_tables(a_table, b_table, c_table,
              a_dim_map, b_dim_map, product_tag());
  double time = t.elapsed();
  std::cout << "Computed c = join(a, b) [" << c_table.num_explicit_elts()
            << "/" << c_table.num_elts()
            << " explicit elements] in " << time << "s." << std::endl;

  // Check the correctness of the result.
  std::cout << "Checking correctness..." << std::flush;
  v_table_index c_index(3);
  boost::tie(b_begin, b_end) = b_table.indexed_elements();
  for (boost::tie(a_it, a_end) = a_table.indexed_elements();
       a_it != a_end; ++a_it) {
    boost::tie(a_index, a_elt) = *a_it;
    for (ielt_it_t b_it = b_begin; b_it != b_end; ++b_it) {
      boost::tie(b_index, b_elt) = *b_it;
      if (a_index[1] == b_index[1]) {
        c_index[0] = a_index[0];
        c_index[1] = a_index[1];
        c_index[2] = b_index[0];
        assert(fabs(c_table.get(c_index) -
                    a_table.get(a_index) * b_table.get(b_index)) < 1e-8);
      }
    }
  }
  std::cout << "done." << std::endl;

  // Perform an aggregation to sum out the middle dimension.
  std::cout << "Performing aggregation..." << std::flush;
  table_size_t d_dims[3] = {p, r};
  sd_table_t d_table(d_dims, d_dims + 2, 0.0);
  table_size_t d_dim_map[2] = {0, 2}; // maps dimensions of d_table to c_table
  aggregate_table(c_table, d_table, d_dim_map, sum_tag());
  std::cout << "done." << std::endl;

  // Make an example where there is a decision to be made about which
  // dimension to use for the sort-merge.
  const table_size_t u = 300; // side length of grid
  const table_size_t v = 25;  // side length of non-zero corner
  // Think of x as a probability distribution over grid locations.
  table_size_t x_dims[2] = {u, u};
  sd_table_t x_table(x_dims, x_dims + 2, 0.0);
  // Think of y as a motion model.
  table_size_t y_dims[4] = {u, u, u, u};
  sd_table_t y_table(y_dims, y_dims + 4, 0.0);
  std::cout << "Initializing location distribution..." << std::flush;
  v_table_index x_index(2);
  for (x_index[0] = 1; x_index[0] <= v; ++x_index[0])
    for (x_index[1] = 1; x_index[1] <= v; ++x_index[1])
      x_table.set(x_index, 1.0 / static_cast<double>(v * v));
  std::cout << std::endl << "Initializing motion model..." << std::flush;
  v_table_index y_index(4, 0);
  for (y_index[0] = 0; y_index[0] < u; ++y_index[0])
    for (y_index[1] = 0; y_index[1] < u; ++y_index[1])
      for (y_index[2] = std::max<table_size_t>(0, y_index[0] - 1);
           y_index[2] <= std::min<table_size_t>(u - 1, y_index[0] + 1);
           ++y_index[2])
        for (y_index[3] = std::max<table_size_t>(0, y_index[1] - 1);
             y_index[3] <= std::min<table_size_t>(u - 1, y_index[1] + 1);
             ++y_index[3])
          y_table.set(y_index, 1.0 / 9.0); // sort of
  std::cout << std::endl
            << "Evolving location distribution, which should have "
            << v * v * 9 << " explicit elements..."
            << std::flush;
  sd_table_t z_table(y_dims, y_dims + 4, 0.0);
  table_size_t x_dim_map[2] = {0, 1}; // maps dimensions of x to z
  table_size_t y_dim_map[4] = {0, 1, 2, 3}; // maps dimensions of y to z
  boost::timer t2;
  join_tables(x_table, y_table, z_table,
              &x_dim_map[0], &y_dim_map[0], product_tag());
  time = t2.elapsed();
  std::cout << "Computed z = join(x, y) [" << z_table.num_explicit_elts()
            << "/" << z_table.num_elts()
            << " explicit elements] in " << time << "s." << std::endl;

  // Normalize the table.
  sd_table_t::element_it z_it, z_end;
  double sum = 0.0;
  for (boost::tie(z_it, z_end) = z_table.elements(true, 0.0);
       z_it != z_end; ++z_it)
    sum += *z_it;
  for (boost::tie(z_it, z_end) = z_table.elements(true, 0.0);
       z_it != z_end; ++z_it)
    *z_it /= sum;

  return EXIT_SUCCESS;
}
