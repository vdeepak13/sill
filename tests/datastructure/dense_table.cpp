#define BOOST_TEST_MODULE dense_table
#include <boost/test/unit_test.hpp>

#include <functional>

#include <boost/tuple/tuple.hpp>
#include <sill/datastructure/dense_table.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

typedef sill::dense_table<int> table_type;
typedef std::vector<size_t> index_type;

BOOST_AUTO_TEST_CASE(test_read_write) {

#if 0
  // Create a native 12-d array of integers, and time how long it
  // takes to write and read each cell once.
  int a[3][3][3][3][3][3][3][3][3][3][3][3];
  int x = 0;
  double time;
  boost::timer t;
  {
    for (int i0 = 0; i0 < 3; i0++)
      for (int i1 = 0; i1 < 3; i1++)
        for (int i2 = 0; i2 < 3; i2++)
          for (int i3 = 0; i3 < 3; i3++)
            for (int i4 = 0; i4 < 3; i4++)
              for (int i5 = 0; i5 < 3; i5++)
                for (int i6 = 0; i6 < 3; i6++)
                  for (int i7 = 0; i7 < 3; i7++)
                    for (int i8 = 0; i8 < 3; i8++)
                      for (int i9 = 0; i9 < 3; i9++)
                        for (int i10 = 0; i10 < 3; i10++)
                          for (int i11 = 0; i11 < 3; i11++)
                            a[i0][i1][i2][i3][i4][i5][i6][i7][i8][i9][i10][i11] = x++;
    x = 0;
    for (int i0 = 0; i0 < 3; i0++)
      for (int i1 = 0; i1 < 3; i1++)
        for (int i2 = 0; i2 < 3; i2++)
          for (int i3 = 0; i3 < 3; i3++)
            for (int i4 = 0; i4 < 3; i4++)
              for (int i5 = 0; i5 < 3; i5++)
                for (int i6 = 0; i6 < 3; i6++)
                  for (int i7 = 0; i7 < 3; i7++)
                    for (int i8 = 0; i8 < 3; i8++)
                      for (int i9 = 0; i9 < 3; i9++)
                        for (int i10 = 0; i10 < 3; i10++)
                          for (int i11 = 0; i11 < 3; i11++)
                            assert(a[i0][i1][i2][i3][i4][i5][i6][i7][i8][i9][i10][i11] == x++);
    time = t.elapsed();
 }
#endif
  
  const int d = 10;
  unsigned int dims[d] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
  table_type a_table(index_type(dims, dims + d));

  // Number the elements uniquely.
  int x = 0;
  foreach(const table_type::index_type& index, a_table.indices()) {
    a_table(index) = x++;
  }
  // Check to make sure the elements are what we set them to.
  x = 0;
  foreach(const table_type::index_type& index, a_table.indices())
    BOOST_CHECK_EQUAL(a_table(index), x++);
}


BOOST_AUTO_TEST_CASE(test_operations) {
  // Perform a table sum using native arrays.
  const size_t p = 10;
  const size_t q = 8;
  const size_t r = 9;
  int e[p][q];
  int f[q][r];
  int x = 0;
  for (size_t i = 0; i < p; i++)
    for (size_t j = 0; j < q; j++)
      e[i][j] = x++;
  for (size_t i = 0; i < q; i++)
    for (size_t j = 0; j < r; j++)
      f[i][j] = x++;
  int sum[p][q][r];
  for (size_t i = 0; i < p; i++)
    for (size_t j = 0; j < q; j++)
      for (size_t k = 0; k < r; k++)
        sum[i][j][k] = e[i][j] + f[j][k];

  // Now do the same with multidimensional tables.
  unsigned int e_dims[2] = {p, q};
  table_type e_table(index_type(e_dims, e_dims + 2));
  unsigned int f_dims[2] = {q, r};
  table_type f_table(index_type(f_dims, f_dims + 2));
  table_type::index_iterator it, end;
  for (boost::tie(it, end) = e_table.indices(); it != end; ++it)
    e_table(*it) = e[(*it)[0]][(*it)[1]];
  for (boost::tie(it, end) = f_table.indices(); it != end; ++it)
    f_table(*it) = f[(*it)[0]][(*it)[1]];
  unsigned int g_dims[3] = {p, q, r};
  table_type g_table(index_type(g_dims, g_dims + 3));
  index_type e_dim_map; // maps e's dimensions to g's dimensions
  e_dim_map.push_back(0);
  e_dim_map.push_back(1);
  index_type f_dim_map; // maps f's dimensions to g's dimensions
  f_dim_map.push_back(1);
  f_dim_map.push_back(2);
  // Compute the sum.
  g_table.join(e_table, f_table, e_dim_map, f_dim_map, std::plus<int>());
  // Check it is correct.
  table_type::index_type index(3);
  for (index[0] = 0; index[0] < p; index[0]++)
    for (index[1] = 0; index[1] < q; index[1]++)
      for (index[2] = 0; index[2] < r; index[2]++)
        BOOST_CHECK_EQUAL(sum[index[0]][index[1]][index[2]], g_table(index));

  ////////////////////////////////////////////////////////

  // Perform a table aggregation using native arrays.
  int agg[p][r];
  for (size_t i = 0; i < p; i++)
    for (size_t k = 0; k < r; k++)
      agg[i][k] = 0;
  for (size_t i = 0; i < p; i++)
    for (size_t k = 0; k < r; k++)
      for (size_t j = 0; j < q; j++)
        agg[i][k] += sum[i][j][k];

  // Now do the same with multidimensional tables.
  unsigned int h_dims[2] = {p, r};
  table_type h_table(index_type(h_dims, h_dims + 2), 0);
  index_type h_dim_map; // maps h's dimensions to g's dimensions
  h_dim_map.push_back(0);
  h_dim_map.push_back(2);
  // Compute the aggregation.
  h_table.aggregate(g_table, h_dim_map, std::plus<int>());

  table_type::index_type index2(2);
  // Check it is correct.
  for (index2[0] = 0; index2[0] < p; index2[0]++)
    for (index2[1] = 0; index2[1] < r; index2[1]++)
      BOOST_CHECK_EQUAL(agg[index2[0]][index2[1]], h_table(index2));
}
