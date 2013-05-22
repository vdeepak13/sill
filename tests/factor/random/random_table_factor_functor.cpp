
/**
 * \file random_table_factor_functor.cpp
 *       Test of random_table_factor_functor.
 */

#include <sill/base/universe.hpp>
#include <sill/factor/random/random_table_factor_functor_builder.hpp>

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

  random_table_factor_functor_builder rtff_builder;
  rtff_builder.add_options(desc);

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cerr << desc << std::endl;
    return 1;
  }

  universe u;
  finite_variable* Y = u.new_finite_variable(2);
  finite_variable* X = u.new_finite_variable(2);

  random_table_factor_functor rtff(random_seed);
  rtff.params = rtff_builder.get_parameters();

  std::cout << "Test: random_table_factor_functor\n"
            << "---------------------------------------------" << std::endl;
  table_factor P_Y(rtff.generate_marginal(Y));
  std::cout << "Generated table_factor P(Y):\n"
            << P_Y << std::endl;
  table_factor P_YX(rtff.generate_marginal(make_domain(Y,X)));
  std::cout << "Generated table_factor P(Y,X):\n"
            << P_YX << std::endl;
  table_factor P_Y_given_X(rtff.generate_conditional(Y, X));
  std::cout << "Generated table_factor P(Y|X):\n"
            << P_Y_given_X << std::endl;

} // main

#include <sill/macros_undef.hpp>
