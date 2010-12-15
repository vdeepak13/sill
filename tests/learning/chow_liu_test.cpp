
#include <iostream>

#include <prl/base/universe.hpp>
#include <prl/learning/chow_liu.hpp>
#include <prl/learning/dataset/data_conversions.hpp>
#include <prl/learning/dataset/syn_oracle_majority.hpp>
#include <prl/learning/dataset/statistics.hpp>
#include <prl/learning/dataset/vector_dataset.hpp>

#include <prl/macros_def.hpp>

/**
 * \file chow_liu_test.cpp Test of learning a Bayes net via Chow-Liu.
 */
int main(int argc, char* argv[]) {

  using namespace prl;
  using namespace std;

  universe u;
  syn_oracle_majority majority(create_syn_oracle_majority(10,u));
  cout << "Majority oracle:\n" << majority << std::endl;
  vector_dataset ds(*(oracle2dataset<vector_dataset>(majority, 100)));

  chow_liu<table_factor> chowliu(ds.finite_variables(), ds);
  const decomposable<table_factor>& model = chowliu.model();
  cout << "Ran Chow-Liu to get decomposable model:\n" << model << std::endl;

}
