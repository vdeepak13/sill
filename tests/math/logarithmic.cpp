#define BOOST_TEST_MODULE logarithmic

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <sstream>
#include <string>

#include <sill/math/logarithmic.hpp>

template class sill::logarithmic<double>;
template class sill::logarithmic<float>;

BOOST_AUTO_TEST_CASE(test_operations) {
  typedef sill::logarithmic<double> log_double;
  using sill::log_tag;

  // Create some values
  double x = 1.0;
  double y = 2.0;
  double z = 3.0;

  // Create some log values
  log_double lx(x);
  log_double ly = y;
  log_double lz = z;
  log_double l1(0, log_tag());

  // Check the constructors and conversion operators
  BOOST_CHECK_CLOSE(double(lx), x, 1e-10);
  BOOST_CHECK_CLOSE(double(ly), y, 1e-10);
  BOOST_CHECK_CLOSE(double(lz), z, 1e-10);
  BOOST_CHECK_CLOSE(double(l1), 1, 1e-10);

  // Check the accessors
  BOOST_CHECK_EQUAL(l1.log_value(), 0.0);

  // Check the binary operations
  BOOST_CHECK_CLOSE(double(lx + ly), 3.0, 1e-10);
  BOOST_CHECK_CLOSE(double(lz - ly), 1.0, 1e-10);
  BOOST_CHECK_CLOSE(double(ly * lz), 6.0, 1e-10);
  BOOST_CHECK_CLOSE(double(lz / ly), 1.5, 1e-10);

  // Check the in-place operations
  log_double tmp;
  tmp = lx; tmp += ly; BOOST_CHECK_CLOSE(double(tmp), 3.0, 1e-10);
  tmp = lz; tmp -= ly; BOOST_CHECK_CLOSE(double(tmp), 1.0, 1e-10);
  tmp = ly; tmp *= lz; BOOST_CHECK_CLOSE(double(tmp), 6.0, 1e-10);
  tmp = lz; tmp /= ly; BOOST_CHECK_CLOSE(double(tmp), 1.5, 1e-10);

  // Check comparisons
  BOOST_CHECK(ly == log_double(y));
  BOOST_CHECK(ly != log_double(z));
  BOOST_CHECK(ly < lz);
  BOOST_CHECK(lz > ly);
  BOOST_CHECK(ly <= ly);
  BOOST_CHECK(ly >= ly);

  // Check logical operators
  BOOST_CHECK(lx || log_double(0.0));
  BOOST_CHECK(lx && ly);
  BOOST_CHECK(!(lx && log_double(0.0)));

  // Check I/O
  std::ostringstream out;
  out << std::setprecision(16) << ly;
  std::istringstream in(out.str());
  log_double ly2;
  in >> ly2;
  BOOST_CHECK_CLOSE(double(ly), double(ly2), 1e-10);
}
