#define BOOST_TEST_MODULE softmax
#include <boost/test/unit_test.hpp>

#include <sill/factor/softmax.hpp>

namespace sill {
  template class softmax<double>;
  template class softmax<float>;
}

using namespace sill;

