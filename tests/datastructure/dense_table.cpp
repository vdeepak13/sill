#include <cassert>
#include <iostream>
#include <functional>

#include <boost/tuple/tuple.hpp>
#include <boost/timer.hpp>

#include <prl/datastructure/dense_table.hpp>
#include <prl/functional.hpp>
#include <prl/stl_io.hpp>

#include <prl/macros_def.hpp>

int main(int argc, char** argv) {
  using namespace prl;
  using namespace std;

  boost::timer t;
  // Make sure we can time things.
  //assert(clock() != static_cast<clock_t>(-1));

  std::cerr << "Warning: optimization can make the times below inaccurate."
            << std::endl;

  ////////////////////////////////////////////////////////

  // Create a native 12-d array of integers, and time how long it
  // takes to write and read each cell once.
  int a[3][3][3][3][3][3][3][3][3][3][3][3];
  int x = 0;
  double time;
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
  std::cout << "Wrote and read 3^12 native array cells in "
            << time << "s." << std::endl;

  // Now do the same with multidimensional tables.
  t.restart();

  typedef prl::dense_table<int> table_type;
  typedef std::vector<size_t> shape_type;

  const int d = 12;
  unsigned int dims[d] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
  table_type a_table(shape_type(dims, dims + d));
  cout << a_table.shape() << endl;
  {
    clock_t begin = clock();
    // Number the elements uniquely.
    int x = 0;
    foreach(const table_type::shape_type& index, a_table.indices()) {
      a_table(index) = x++;
    }
    // Check to make sure the elements are what we set them to.
    x = 0;
    foreach(const table_type::shape_type& index, a_table.indices())
      assert(a_table(index) == x++);
    time = t.elapsed();
  }
  std::cout << "Wrote and read 3^12 multidimensional table cells in "
            << time << "s." << std::endl;

  ////////////////////////////////////////////////////////

  // Perform a table sum using native arrays.
  const size_t p = 10;
  const size_t q = 8;
  const size_t r = 9;
  int e[p][q];
  int f[q][r];
  x = 0;
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
  table_type e_table(shape_type(e_dims, e_dims + 2));
  unsigned int f_dims[2] = {q, r};
  table_type f_table(shape_type(f_dims, f_dims + 2));
  table_type::index_iterator it, end;
  for (boost::tie(it, end) = e_table.indices(); it != end; ++it)
    e_table(*it) = e[(*it)[0]][(*it)[1]];
  for (boost::tie(it, end) = f_table.indices(); it != end; ++it)
    f_table(*it) = f[(*it)[0]][(*it)[1]];
  unsigned int g_dims[3] = {p, q, r};
  table_type g_table(shape_type(g_dims, g_dims + 3));
  shape_type e_dim_map; // maps e's dimensions to g's dimensions
  e_dim_map.push_back(0);
  e_dim_map.push_back(1);
  shape_type f_dim_map; // maps f's dimensions to g's dimensions
  f_dim_map.push_back(1);
  f_dim_map.push_back(2);
  // Compute the sum.
  g_table.join(e_table, f_table, e_dim_map, f_dim_map, std::plus<int>());
  // Check it is correct.
  table_type::shape_type index(3);
  for (index[0] = 0; index[0] < p; index[0]++)
    for (index[1] = 0; index[1] < q; index[1]++)
      for (index[2] = 0; index[2] < r; index[2]++)
        assert(sum[index[0]][index[1]][index[2]] == g_table(index));

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
  table_type h_table(shape_type(h_dims, h_dims + 2), 0);
  shape_type h_dim_map; // maps h's dimensions to g's dimensions
  h_dim_map.push_back(0);
  h_dim_map.push_back(2);
  // Compute the aggregation.
  h_table.aggregate(g_table, h_dim_map, std::plus<int>());

  table_type::shape_type index2(2);
  // Check it is correct.
  for (index2[0] = 0; index2[0] < p; index2[0]++)
    for (index2[1] = 0; index2[1] < r; index2[1]++)
      assert(agg[index2[0]][index2[1]] == h_table(index2));

  return EXIT_SUCCESS;
}
