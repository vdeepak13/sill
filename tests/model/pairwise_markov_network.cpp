#define BOOST_TEST_MODULE markov_network
#include <boost/test/unit_test.hpp>

#include <sill/model/pairwise_markov_network.hpp>

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/probability_table.hpp>

namespace sill {
  template class pairwise_markov_network<cgaussian>;
  template class pairwise_markov_network<ptable>;
}
