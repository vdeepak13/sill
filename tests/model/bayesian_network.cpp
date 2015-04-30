#define BOOST_TEST_MODULE bayesian_network
#include <boost/test/unit_test.hpp>

#include <sill/model/bayesian_network.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/probability_table.hpp>
#include <sill/factor/util/operations.hpp>

#include "predicates.hpp"

namespace sill {
  template class bayesian_network<ptable>;
  template class bayesian_network<cgaussian>;
  template class bayesian_network<mgaussian>;
}

using namespace sill;

struct fixture {
  fixture()
    : x(u.new_finite_variables(5, 2)) {

    /* Create factors for a Bayesian network with this structure:
     * 0, 1 (no parents)
     * 1 --> 2
     * 1,2 --> 3
     * 0,3 --> 4
     */

    f0   = ptable({x[0]}, {0.3, 0.7});
    f1   = ptable({x[1]}, {0.5, 0.5});
    f12  = ptable({x[1], x[2]}, {0.8, 0.2, 0.2, 0.8});
    f123 = ptable({x[1], x[2], x[3]}, {0.1, 0.1, 0.3, 0.5, 0.9, 0.9, 0.7, 0.5});
    f034 = ptable({x[0], x[3], x[4]}, {0.6, 0.1, 0.2, 0.1, 0.4, 0.9, 0.8, 0.9});

    bn.add_factor(x[0], f0);
    bn.add_factor(x[1], f1);
    bn.add_factor(x[2], f12);
    bn.add_factor(x[3], f123);
    bn.add_factor(x[4], f034);
  }

  universe u;
  finite_var_vector x;
  ptable f0, f1, f12, f123, f034;
  bayesian_network<ptable> bn;
};

BOOST_FIXTURE_TEST_CASE(test_serialization, fixture) {
  BOOST_CHECK(serialize_deserialize(bn, u));
}

BOOST_FIXTURE_TEST_CASE(test_markov_graph, fixture) {
  typedef std::pair<finite_variable*, finite_variable*> vpair;
  std::vector<vpair> vpairs =
    {vpair(x[1], x[2]), vpair(x[1], x[3]), vpair(x[2], x[3]),
     vpair(x[0], x[3]), vpair(x[0], x[4]), vpair(x[3], x[4])};
  undirected_graph<finite_variable*> mg(vpairs);
  undirected_graph<finite_variable*> mg2;
  bn.markov_graph(mg2);
  BOOST_CHECK_EQUAL(mg, mg2);
}

BOOST_FIXTURE_TEST_CASE(test_conditioning, fixture) {
  finite_assignment a;
  a[x[0]] = 0;
  a[x[1]] = 1;
  double likelihood = bn.condition(a);
  std::vector<ptable> factors = {f0, f1, f12, f123, f034};
  ptable marginal = prod_all(factors).marginal({x[0], x[1]});
  BOOST_CHECK_CLOSE(likelihood, marginal(a), 1e-5);

  bayesian_network<ptable> bn2;
  bn2.add_factor(x[2], f12.restrict(a));
  bn2.add_factor(x[3], f123.restrict(a));
  bn2.add_factor(x[4], f034.restrict(a));
  BOOST_CHECK(model_close_log_likelihoods(bn, bn2, 1e-6));
}
