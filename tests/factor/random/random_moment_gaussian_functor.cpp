
/**
 * \file random_moment_gaussian_functor.cpp
 *       Test of random_moment_gaussian_functor.
 */

#include <sill/base/universe.hpp>
#include <sill/factor/random/random_moment_gaussian_functor_builder.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

int main(int argc, char** argv) {

  unsigned random_seed;

  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("random_seed",
     po::value<unsigned int>(&random_seed)->default_value(time(NULL)),
     "Random seed (default = time)")
    ("help", "Print this help message.");

  random_moment_gaussian_functor_builder rgff_builder;
  rgff_builder.add_options(desc);

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }

  universe u;
  vector_variable* Y = u.new_vector_variable(1);
  vector_variable* X = u.new_vector_variable(1);

  random_moment_gaussian_functor rmgf(random_seed);
  rmgf.params = rgff_builder.get_parameters();

  std::cout << "Test: random_moment_gaussian_functor\n"
            << "---------------------------------------------" << std::endl;
  moment_gaussian P_Y(rmgf.generate_marginal(Y));
  std::cout << "Generated moment_gaussian P(Y):\n"
            << P_Y << std::endl;
  moment_gaussian P_YX(rmgf.generate_marginal(make_domain(Y,X)));
  std::cout << "Generated moment_gaussian P(Y,X):\n"
            << P_YX << std::endl;
  moment_gaussian P_Y_given_X(rmgf.generate_conditional(Y, X));
  std::cout << "Generated moment_gaussian P(Y|X):\n"
            << P_Y_given_X << std::endl;

} // main

#include <sill/macros_undef.hpp>
