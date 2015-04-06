#define BOOST_TEST_MODULE probability_array_ll
#include <boost/test/unit_test.hpp>

#include <sill/math/likelihood/probability_array_ll.hpp>

namespace sill {
  template class probability_array_ll<double, 1>;
  template class probability_array_ll<double, 2>;
  template class probability_array_ll<float, 1>;
  template class probability_array_ll<float, 2>;
}
