#define BOOST_TEST_MODULE diagonal_table_generator
#include <boost/test/unit_test.hpp>

#include <sill/factor/random/diagonal_table_generator.hpp>

#include <sill/base/universe.hpp>
#include <sill/datastructure/finite_index.hpp>
#include <sill/factor/canonical_table.hpp>
#include <sill/factor/probability_table.hpp>

#include <boost/mpl/list.hpp>

namespace sill {
  template class diagonal_table_generator<ctable>;
  template class diagonal_table_generator<ptable>;
}

using namespace sill;

size_t nsamples = 1000;
const double lower = -0.7;
const double upper = +0.5;

typedef boost::mpl::list<ctable,ptable> factor_types;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_all, F, factor_types) {
  universe u;
  finite_variable* x = u.new_finite_variable(3);
  finite_variable* y = u.new_finite_variable(3);
  domain<finite_variable*> xy = {x, y};

  std::mt19937 rng;
  diagonal_table_generator<F> gen(lower, upper);

  // check the marginals
  double sum = 0.0;
  size_t count = 0;
  finite_index shape(2, 3);
  for (size_t i = 0; i < nsamples; ++i) {
    F f = gen(xy, rng);
    finite_index_iterator it(&shape), end(2);
    for (; it != end; ++it) {
      const finite_index& index = *it;
      if (index[0] == index[1]) {
        BOOST_CHECK(f.param(index) >= lower && f.param(index) <= upper);
        sum += f.param(index);
        ++count;
      } else {
        BOOST_CHECK_SMALL(log(f(index)), 1e-8);
      }
    }
  }
  BOOST_CHECK_CLOSE_FRACTION(sum / count, (lower+upper)/2, 0.05);
}
