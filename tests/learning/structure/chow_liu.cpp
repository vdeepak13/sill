#define BOOST_TEST_MODULE chow_liu
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <sill/base/universe.hpp>
#include <sill/factor/random/random_table_factor_functor.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/bayesian_network.hpp>
#include <sill/learning/dataset2/finite_dataset.hpp>
#include <sill/learning/parameter/table_factor_mle.hpp>
#include <sill/learning/structure/chow_liu.hpp>

#include <sill/macros_def.hpp>

/*

Tests Chow-Liu on data generated from a Bayesian network with the 
following structure:

                0
                |
                |
                1
               / \
              /   \
             2     3
                  / \ 
                 /   \
                4     5

*/
               
// int main(int argc, char** argv) {

BOOST_AUTO_TEST_CASE(test_simple) {
  using namespace sill;
  using namespace std;

  size_t nsamples = 1000;

  universe u;
  finite_var_vector v = u.new_finite_variables(6, 3);

  // generate a random Bayesian network with the given structure
  bayesian_network<table_factor> bn;
  random_table_factor_functor gen(0);
  bn.add_factor(v[0], gen.generate_marginal(v[0]));
  bn.add_factor(v[1], gen.generate_conditional(v[1], v[0]));
  bn.add_factor(v[2], gen.generate_conditional(v[2], v[1]));
  bn.add_factor(v[3], gen.generate_conditional(v[3], v[1]));
  bn.add_factor(v[4], gen.generate_conditional(v[4], v[3]));
  bn.add_factor(v[5], gen.generate_conditional(v[5], v[3]));

  //cout << bn << endl;

  // generate a dataset
  boost::mt19937 rng;
  finite_dataset<> data;
  data.initialize(bn.arguments());
  for (size_t i = 0; i < nsamples; ++i) {
    finite_assignment a = bn.sample(rng);
    //cout << i << ": " << a << endl;
    data.insert(a);
  }

  // learn the model
  table_factor_mle<> estim(&data);
  chow_liu<table_factor> learner(make_domain(v), estim);
  decomposable<table_factor> dm = learner.model();
  
  // verify the cliques
  std::set<finite_domain> cliques(dm.cliques().begin(), dm.cliques().end());
  foreach(const finite_domain& clique, cliques) {
    cout << clique << endl;
  }
  
  BOOST_CHECK(cliques.size() == 5);
  BOOST_CHECK(cliques.count(make_domain(v[0], v[1])));
  BOOST_CHECK(cliques.count(make_domain(v[1], v[2])));
  BOOST_CHECK(cliques.count(make_domain(v[1], v[3])));
  BOOST_CHECK(cliques.count(make_domain(v[3], v[4])));
  BOOST_CHECK(cliques.count(make_domain(v[3], v[5])));

  // TODO: flatten, relative entropy for decomposable
  table_factor p = prod_all(bn.factors()).normalize();
  table_factor q = prod_all(dm.factors()).normalize();
  double kl = p.relative_entropy(q);
  cout << "KL divergence: " << kl << endl;
  BOOST_CHECK_SMALL(kl, 0.05);
}
