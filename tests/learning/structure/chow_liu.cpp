#define BOOST_TEST_MODULE chow_liu
#include <boost/test/unit_test.hpp>

#include <sill/learning/structure/chow_liu.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/canonical_table.hpp>
#include <sill/factor/probability_table.hpp>
#include <sill/factor/random/uniform_table_generator.hpp>
#include <sill/learning/dataset/finite_dataset.hpp>
#include <sill/model/bayesian_network.hpp>

#include <iostream>
#include <random>

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

  size_t nsamples = 10000;

  universe u;
  finite_var_vector v = u.new_finite_variables(6, 3);
  typedef domain<finite_variable*> domain_type;

  // generate a random Bayesian network with the given structure
  bayesian_network<ptable> bn;
  uniform_table_generator<ctable> gen(-1.0, 1.0);
  std::mt19937 rng;
  bn.add_factor(v[0], ptable(gen({v[0]}, rng).normalize()));
  bn.add_factor(v[1], ptable(gen({v[1]}, {v[0]}, rng)));
  bn.add_factor(v[2], ptable(gen({v[2]}, {v[1]}, rng)));
  bn.add_factor(v[3], ptable(gen({v[3]}, {v[1]}, rng)));
  bn.add_factor(v[4], ptable(gen({v[4]}, {v[3]}, rng)));
  bn.add_factor(v[5], ptable(gen({v[5]}, {v[3]}, rng)));

  //cout << bn << endl;

  // generate a dataset
  finite_dataset<> data(v, nsamples);
  finite_assignment a;
  for (size_t i = 0; i < nsamples; ++i) {
    bn.sample(rng, a);
    data.insert(a, 1.0);
  }

  // learn the model
  chow_liu<ptable> learner;
  const decomposable<ptable>& dm = learner.fit(data, v).model();

  // print the scores
  for (const auto& p : learner.scores()) {
    std::cout << p.first << " : " << p.second << std::endl;
  }
  
  // verify the cliques
  std::unordered_set<domain_type> cliques;
  for (size_t v : dm.vertices()) {
    const domain_type& clique = dm.clique(v);
    cliques.emplace(clique);
    std::cout << clique << std::endl;
  }
  
  BOOST_CHECK(cliques.size() == 5);
  BOOST_CHECK(cliques.count(domain_type({v[0], v[1]}).unique()));
  BOOST_CHECK(cliques.count(domain_type({v[1], v[2]}).unique()));
  BOOST_CHECK(cliques.count(domain_type({v[1], v[3]}).unique()));
  BOOST_CHECK(cliques.count(domain_type({v[3], v[4]}).unique()));
  BOOST_CHECK(cliques.count(domain_type({v[3], v[5]}).unique()));

  // TODO: flatten, relative entropy for decomposable
  ptable p = prod_all(bn).normalize().marginal(v);
  ptable q = prod_all(dm).normalize().marginal(v);
  double kl = kl_divergence(p, q);
  cout << "KL divergence: " << kl << endl;
  BOOST_CHECK_SMALL(kl, 0.01);
}
