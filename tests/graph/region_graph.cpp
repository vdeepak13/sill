#define BOOST_TEST_MODULE region_graph
#include <boost/test/unit_test.hpp>

#include <sill/graph/region_graph.hpp>
#include <sill/argument/domain.hpp>

namespace sill {
  template class region_graph<domain<size_t> >;
}

