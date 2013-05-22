
/**
 * \file alternating_factor_functor.cpp
 *       Test of alternating_factor_functor.
 */

#include <sill/base/universe.hpp>
#include <sill/factor/random/alternating_factor_functor_builder.hpp>
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

  alternating_factor_functor_builder<random_table_factor_functor_builder>
    raff_builder;
  raff_builder.add_options(desc);

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

  alternating_factor_functor<random_table_factor_functor>
    raff;
  raff.params = raff_builder.get_parameters();
  raff.seed(random_seed);

  std::cout << "Test: alternating_factor_functor"
            << " with random_table_factor_functor\n"
            << "---------------------------------------------" << std::endl;
  std::cout << "Generated alternating_factor P(Y,X):\n"
            << raff.generate_marginal(make_domain(Y,X)) << std::endl;
  std::cout << "Generated alternating_factor P(Y,X):\n"
            << raff.generate_marginal(make_domain(Y,X)) << std::endl;
  std::cout << "Generated alternating_factor P(Y,X):\n"
            << raff.generate_marginal(make_domain(Y,X)) << std::endl;
  std::cout << "Generated alternating_factor P(Y,X):\n"
            << raff.generate_marginal(make_domain(Y,X)) << std::endl;

} // main

#include <sill/macros_undef.hpp>
