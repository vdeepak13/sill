#include <iostream>

#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/timer.hpp>

#include <prl/base/universe.hpp>
#include <prl/learning/dataset/dataset_view.hpp>
#include <prl/learning/dataset/data_loader.hpp>
#include <prl/learning/dataset/data_conversions.hpp>
#include <prl/learning/dataset/statistics.hpp>
#include <prl/learning/dataset/syn_oracle_knorm.hpp>
#include <prl/learning/dataset/syn_oracle_majority.hpp>
#include <prl/learning/dataset/vector_dataset.hpp>
#include <prl/learning/discriminative/concepts.hpp>
#include <prl/learning/discriminative/linear_regression.hpp>

#include <prl/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace prl;
  using namespace std;

  // Create a dataset to work with
  universe u;
  size_t ntrain = 1000;
  size_t ntest = 1000;
  size_t nY = 5;
  size_t nX = 10;

  datasource_info_type ds_info;
  for (size_t j(0); j < nX; ++j)
    ds_info.vector_seq.push_back(u.new_vector_variable(1));
  for (size_t j(0); j < nY; ++j) {
    vector_variable* v = u.new_vector_variable(1);
    ds_info.vector_seq.push_back(v);
    ds_info.vector_class_vars.push_back(v);
  }
  ds_info.var_type_order =
    std::vector<variable::variable_typenames>(nY+nX, variable::VECTOR_VARIABLE);
  boost::uniform_real<double> unif_real(-10, 10);
  boost::normal_distribution<double> normal_dist(0, 1);
  boost::lagged_fibonacci607 rng;
  rng.seed(time(NULL));
  mat A(nY, nX, 0);
  for (size_t j(0); j < A.size(); ++j)
    A(j) = unif_real(rng);
  cout << "True A matrix:\n" << A << endl;
  vector_dataset ds_train(ds_info);
  vector_dataset ds_test(ds_info);
  for (size_t i(0); i < ntrain + ntest; ++i) {
    std::vector<size_t> fvals;
    vec vvals(nY+nX, 0);
    for (size_t j(0); j < nX; ++j)
      vvals[j] = unif_real(rng);
    vec tmpvec(A * vvals(irange(0,nX)));
    for (size_t j(0); j < nY; ++j)
      vvals[j + nX] = tmpvec[j] + normal_dist(rng);
    if (i < ntrain)
      ds_train.insert(fvals, vvals);
    else
      ds_test.insert(fvals, vvals);
  }

  linear_regression_parameters params;
  params.objective = 2;
  params.regularization = 2;
  params.lambda = .1;
  params.opt_method = 0;
  params.init_iterations = 100;
  params.debug = 1;
  vec lambdas = "0. .2 .4 .6 .8 1.";
  bool do_lambda_cv = true;

  bool normalize_data = true;
  if (normalize_data) {
    vec means, std_devs;
    boost::tie(means,std_devs) = ds_train.normalize();
    ds_test.normalize(means,std_devs);
    std::cout << "Normalized dataset using:\n  means: " << means
              << "\n  std_devs: " << std_devs << std::endl;
  }

  boost::timer timer;
  if (do_lambda_cv) {
    params.lambda =
      linear_regression::choose_lambda_easy(params, ds_train, 293435);
    std::cout << "LOOCV chose lambda = " << params.lambda << "\n"
              << std::endl;
  }
  linear_regression lr(ds_train, params);
  double elapsed = timer.elapsed();

  std::cout << "Learned linear regression model:\n"
            << lr
            << std::endl;

  std::cout << "Training time = " << elapsed << std::endl;

  std::pair<double, double> results(lr.mean_squared_error(ds_train));
  std::cout << "<mean squared training error, std error> = < "
            << results.first << " , " << results.second << " >" << std::endl;
  results = lr.mean_squared_error(ds_test);
  std::cout << "<mean squared test error, std error> = < "
            << results.first << " , " << results.second << " >" << std::endl;

  lr.set_weights_to_zero();
  std::cout << "With all weights set to 0, this gives:" << std::endl;
  results = lr.mean_squared_error(ds_train);
  std::cout << "<mean squared training error, std error> = < "
            << results.first << " , " << results.second << " >" << std::endl;
  results = lr.mean_squared_error(ds_test);
  std::cout << "<mean squared test error, std error> = < "
            << results.first << " , " << results.second << " >" << std::endl;

}
