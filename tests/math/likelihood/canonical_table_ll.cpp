#define BOOST_TEST_MODULE canonical_table_ll
#include <boost/test/unit_test.hpp>

#include <sill/math/likelihood/canonical_table_ll.hpp>

namespace sill {
  template class canonical_table_ll<double>;
  template class canonical_table_ll<float>;
}
