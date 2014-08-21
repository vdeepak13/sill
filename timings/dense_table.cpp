#include <cassert>
#include <iostream>
#include <functional>

#include <boost/timer.hpp>

#include <sill/datastructure/dense_table.hpp>
#include <sill/functional.hpp>
#include <sill/range/io.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char** argv) {

  using namespace std;
  using namespace sill;

  boost::timer t;
  // Make sure we can time things.
  //assert(clock() != static_cast<clock_t>(-1));

  std::cerr << "Warning: optimization can make the times below inaccurate."
            << std::endl;

  ////////////////////////////////////////////////////////

  // Create a native 12-d array of integers, and time how long it
  // takes to write and read each cell once.
  double a[3][3][3][3][3][3][3][3][3][3][3][3];
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
  typedef sill::dense_table<int>::index_type index_type;

  t.restart();
  const int d = 12;
  unsigned int dims[d] = {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
  sill::dense_table<int> a_table(index_type(dims, dims + d));
  cout << a_table.shape() << endl;
  for(int i=0;i<20;i++) {
    // Number the elements uniquely.
    int x = 0;
    foreach(const index_type& index, a_table.indices()) {
      a_table(index) = x++;
    }
    // Check to make sure the elements are what we set them to.
    x = 0;
    foreach(const index_type& index, a_table.indices())
      assert(a_table(index) == x++);
  }
  time = t.elapsed();
  std::cout << "Wrote and read 3^12 multidimensional table cells in "
            << time/20 << "s." << std::endl;

  ////////////////////////////////////////////////////////

  // Perform a table sum using native arrays.
  const size_t p = 10;
  const size_t q = 8;
  const size_t r = 9;
  double e[p][q];
  double f[q][r];
  x = 0;
  for (size_t i = 0; i < p; i++)
    for (size_t j = 0; j < q; j++)
      e[i][j] = x++;
  for (size_t i = 0; i < q; i++)
    for (size_t j = 0; j < r; j++)
      f[i][j] = x++;
  double sum[p][q][r];
  for (size_t i = 0; i < p; i++)
    for (size_t j = 0; j < q; j++)
      for (size_t k = 0; k < r; k++)
        sum[i][j][k] = e[i][j] + f[j][k];

  typedef sill::dense_table<double> double_table;

  // Now do the same with multidimensional tables.
  size_t e_dims[2] = {p, q};
  double_table e_table(index_type(e_dims, e_dims + 2));
  size_t f_dims[2] = {q, r};
  double_table f_table(index_type(f_dims, f_dims + 2));

  foreach(const double_table::index_type& index, e_table.indices()) 
    e_table(index) = e[index[0]][index[1]];

  foreach(const double_table::index_type& index, f_table.indices()) 
    f_table(index) = f[index[0]][index[1]];

  t.restart();
  int N = 10000;
  for(int i=0;i<N;i++) {
    unsigned int g_dims[3] = {p, q, r};
    double_table g_table(index_type(g_dims, g_dims + 3));
    index_type e_dim_map(2); // maps e's dimensions to g's dimensions
    e_dim_map[0] = 0; e_dim_map[1] = 1;
    index_type f_dim_map(2); // maps f's dimensions to g's dimensions
    f_dim_map[0] = 1; f_dim_map[1] = 2;
    // Compute the sum.
    g_table.join(e_table, f_table, e_dim_map, f_dim_map, std::plus<double>());
  }
  cout << "Performed " << N << " joins in " << t.elapsed() << N << "s." << endl;

//   t.restart();
//   for(int i=0;i<N;i++) {
//     unsigned int g_dims[3] = {p, q, r};
//     double_table g_table(index_type(g_dims, g_dims + 3));
//     index_type e_dim_map(2); // maps e's dimensions to g's dimensions
//     e_dim_map[0] = 0; e_dim_map[1] = 1;
//     index_type f_dim_map(2); // maps f's dimensions to g's dimensions
//     f_dim_map[0] = 1; f_dim_map[1] = 2;
//     // Compute the sum.
//     g_table.join(e_table, f_table, e_dim_map, f_dim_map, binary_op<double>(sum_op));
//   }
//   cout << "Performed " << N << " joins using binary_op in " << t.elapsed() 
//         << N << "s." << endl;

  /*
  // Check it is correct.
  index_type index(3);
  for (index[0] = 0; index[0] < p; index[0]++)
    for (index[1] = 0; index[1] < q; index[1]++)
      for (index[2] = 0; index[2] < r; index[2]++)
        assert(sum[index[0]][index[1]][index[2]] == g_table.get(index));

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

  t.restart();
  // Now do the same with multidimensional tables.
  size_t h_dims[2] = {p, r};
  sill::dense_table<int> h_table(h_dims, h_dims + 2, 0);
  size_t h_dim_map[2] = {0, 2}; // maps h's dimensions to g's dimensions
  // Compute the aggregation.
  h_table.aggregate(g_table, h_dim_map, plus<int>());
    
  index_type index2(2);
  // Check it is correct.
  for (index2[0] = 0; index2[0] < p; index2[0]++)
    for (index2[1] = 0; index2[1] < r; index2[1]++)
      assert(agg[index2[0]][index2[1]] == h_table.get(index2));
*/

  return EXIT_SUCCESS;
}
