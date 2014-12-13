#define BOOST_TEST_MODULE commutative_semiring
#include <boost/test/unit_test.hpp>

#include <vector>

#include <sill/base/universe.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/util/commutative_semiring.hpp>
#include <sill/factor/table_factor.hpp>

#include <boost/random/mersenne_twister.hpp>

#include <sill/factor/random/uniform_factor_generator.hpp>

using namespace sill;

template class sum_product<table_factor>;
template class max_product<table_factor>;
template class min_sum<table_factor>;
template class max_sum<table_factor>;
template class boolean<table_factor>;

template class sum_product<canonical_gaussian>;
template class max_product<canonical_gaussian>;

BOOST_AUTO_TEST_CASE(test_ops) {
  size_t nvars = 3;
  size_t arity = 2;

  universe u;
  finite_var_vector vec = u.new_finite_variables(nvars, arity);
  finite_domain dom(vec.begin(), vec.end());
  finite_domain var(vec.begin(), vec.begin()+1);

  uniform_factor_generator gen;
  boost::mt19937 rng;
  std::vector<table_factor> f;
  for (size_t i = 0; i < 3; ++i) {
    f.push_back(gen(dom, rng));
  }
  
  sill::sum_product<table_factor> sum_product;
  sill::max_product<table_factor> max_product;
  sill::min_sum<table_factor> min_sum;
  sill::max_sum<table_factor> max_sum;
  // can't test boolean for now
  
  BOOST_CHECK_EQUAL(combine_all(f, sum_product), f[0] * f[1] * f[2]);
  BOOST_CHECK_EQUAL(combine_all(f, max_product), f[0] * f[1] * f[2]);
  BOOST_CHECK_EQUAL(combine_all(f, min_sum), f[0] + f[1] + f[2]);
  BOOST_CHECK_EQUAL(combine_all(f, max_sum), f[0] + f[1] + f[2]);

  BOOST_CHECK_EQUAL(sum_product.combine(f[0], f[1]), f[0] * f[1]);
  BOOST_CHECK_EQUAL(max_product.combine(f[0], f[1]), f[0] * f[1]);
  BOOST_CHECK_EQUAL(min_sum.combine(f[0], f[1]), f[0] + f[1]);
  BOOST_CHECK_EQUAL(max_sum.combine(f[0], f[1]), f[0] + f[1]);
  
  BOOST_CHECK_EQUAL(sum_product.collapse(f[0], var), f[0].marginal(var));
  BOOST_CHECK_EQUAL(max_product.collapse(f[0], var), f[0].maximum(var));
  BOOST_CHECK_EQUAL(min_sum.collapse(f[0], var), f[0].minimum(var));
  BOOST_CHECK_EQUAL(max_sum.collapse(f[0], var), f[0].maximum(var));
}
