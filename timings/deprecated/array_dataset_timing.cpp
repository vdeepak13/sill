#include <iostream>
#include <string>

#include <boost/timer.hpp>

#include <prl/variable.hpp>
#include <prl/assignment.hpp>
#include <prl/learning/dataset/array_dataset.hpp>
#include <prl/learning/dataset/data_conversions.hpp>
//#include <prl/learning/dataset/old_array_dataset.hpp>
#include <prl/learning/dataset/vector_dataset.hpp>
#include <prl/factor/table_factor.hpp>

#include <prl/macros_def.hpp>

/**
 * \file array_dataset_timing.cpp Test of the array_dataset.hpp class.
 */
int main(int argc, char* argv[]) {

  using namespace prl;
  using namespace std;

  std::string filename = "../../tests/data/spam.txt";
  std::string sum_filename = "../../tests/data/spam.sum";

  universe u;

//  concept_assert((Dataset<array_data<> >));
  concept_assert((Dataset<array_dataset>));

//  array_data<> data1 = old_load_plain< array_data<> >(filename, v);
  boost::shared_ptr<array_dataset> data2_ptr
    = load_symbolic_dataset<array_dataset>(sum_filename, u);
  dataset& data2 = *data2_ptr;
  boost::shared_ptr<vector_dataset> data3_ptr =
    load_symbolic_dataset<vector_dataset>
    (sum_filename, data2.finite_list(), data2.vector_list(),
     data2.variable_type_order());
  dataset& data3 = *data3_ptr;

  // Print the data
  cout << data2 << endl;

  size_t nruns = 1000000;
  double tmp = 0;
  boost::timer t;
  // Iterate over the data a bunch of times
  cout << "Average time (over " << nruns << " iterations) of getting 1 record "
       << "and using 10 vector values in a sum:" << endl;
/*
  t.restart();
  for (size_t i = 0; i < nruns; i++) {
    const array_data<>::record& r = data1[1];
    for (size_t j = 0; j < 10; j++)
      tmp += r.vector()[j];
  }
  cout << " Old array_data class: " << t.elapsed() / nruns << std::endl;
*/
  t.restart();
  for (size_t i = 0; i < nruns; i++) {
    const record& r = data2[1];
    for (size_t j = 0; j < 10; j++)
      tmp += r.vector()[j];
  }
  cout << " New array_dataset class: " << t.elapsed() / nruns << std::endl
       << endl;
  t.restart();
  for (size_t i = 0; i < nruns; i++) {
    const record& r = data3[1];
    for (size_t j = 0; j < 10; j++)
      tmp += r.vector()[j];
  }
  cout << " vector_dataset class: " << t.elapsed() / nruns << std::endl
       << endl;

  if (argc > 3)
    cout << tmp;

  nruns = 10000000;
  tmp = 0;
  cout << "Average time (over " << nruns << " iterations) of getting 1 finite "
       << "value (for only a specific record and specific variable):" << endl;
/*
  t.restart();
  for (size_t i = 0; i < nruns; i++)
    tmp += data1.finite(0,0);
  cout << " Old array_data class: " << t.elapsed() / nruns << std::endl;
*/
  t.restart();
  for (size_t i = 0; i < nruns; i++)
    tmp += data2.finite(0,0);
  cout << " New array_dataset class: " << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t i = 0; i < nruns; i++)
    tmp += data3.finite(0,0);
  cout << " vector_dataset class: " << t.elapsed() / nruns << std::endl;

  cout << "Average time (over " << nruns << " iterations) of getting 1 vector "
       << "value (for only a specific record and specific variable):" << endl;
/*
  t.restart();
  for (size_t i = 0; i < nruns; i++)
    tmp += data1.vector(0,0);
  cout << " Old array_data class: " << t.elapsed() / nruns << std::endl;
*/
  t.restart();
  for (size_t i = 0; i < nruns; i++)
    tmp += data2.vector(0,0);
  cout << " New array_dataset class: " << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t i = 0; i < nruns; i++)
    tmp += data3.vector(0,0);
  cout << " vector_dataset class: " << t.elapsed() / nruns << std::endl;

  if (argc > 3)
    cout << tmp;

  return 0;
}
