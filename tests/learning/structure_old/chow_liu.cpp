#define BOOST_TEST_MODULE chow_liu
#include <boost/test/unit_test.hpp>

#include <iostream>

#include <sill/argument/universe.hpp>
#include <sill/factor/random/uniform_factor_generator.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/bayesian_network.hpp>
#include <sill/learning/structure_old/chow_liu.hpp>
#include <sill/learning/dataset_old/vector_dataset.hpp>

#include <boost/random/mersenne_twister.hpp>

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

  size_t nsamples = 1000; // argc > 1 ? atoi(argv[1]) : 10000;

  universe u;
  domain v = u.new_finite_variables(6, 3);

  // generate a random Bayesian network with the given structure
  bayesian_network<table_factor> bn;
  uniform_factor_generator gen;
  boost::mt19937 rng;
  bn.add_factor(v[0], gen(make_domain(v[0]), rng));
  bn.add_factor(v[1], gen(make_domain(v[1]), make_domain(v[0]), rng));
  bn.add_factor(v[2], gen(make_domain(v[2]), make_domain(v[1]), rng));
  bn.add_factor(v[3], gen(make_domain(v[3]), make_domain(v[1]), rng));
  bn.add_factor(v[4], gen(make_domain(v[4]), make_domain(v[3]), rng));
  bn.add_factor(v[5], gen(make_domain(v[5]), make_domain(v[3]), rng));

  //cout << bn << endl;

  // generate a dataset
  vector_dataset_old<> data(v, domain(),
                        std::vector<variable::variable_typenames>());
  for (size_t i = 0; i < nsamples; ++i) {
    finite_assignment a = bn.sample(rng);
    //cout << i << ": " << a << endl;
    data.insert(a);
  }

  // learn the model
  chow_liu<table_factor> learner(v, data);
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
  BOOST_CHECK_SMALL(kl, 0.5);
}
