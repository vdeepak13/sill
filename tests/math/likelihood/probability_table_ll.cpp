#define BOOST_TEST_MODULE probability_table_ll
#include <boost/test/unit_test.hpp>

#include <sill/math/likelihood/probability_table_ll.hpp>

namespace sill {
  template class probability_table_ll<double>;
  template class probability_table_ll<float>;
}
