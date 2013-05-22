#ifndef SILL_TEST_FACTOR_PREDICATES_HPP
#define SILL_TEST_FACTOR_PREDICATES_HPP

#include <fstream>
#include <cstdio>

#include <sill/factor/canonical_gaussian.hpp>
#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>

//! Serializes and deserializes a value and verifies they are equal
template <typename T>
boost::test_tools::predicate_result
serialize_deserialize(const T& value, sill::universe& u) {
  char filename[L_tmpnam];
  tmpnam(filename);

  std::ofstream fout(filename, std::fstream::binary);
  sill::oarchive oa(fout);
  oa << value;
  fout.close();

  std::ifstream fin(filename, std::fstream::binary);
  sill::iarchive ia(fin);
  ia.attach_universe(&u);
  T value2;
  ia >> value2;
  fin.close();

  remove(filename);

  if(value != value2) {
    boost::test_tools::predicate_result result(false);
    result.message() << "Serialization did not preserve the value [\n"
                     << value << "!="
                     << value2 << "]";
    return result;
  }
  return true;
}

//! Verifies that two factors are close enough
template <typename F>
boost::test_tools::predicate_result
are_close(const F& a, const F& b, typename F::result_type eps) {
  typename F::result_type norma = a.norm_constant();
  typename F::result_type normb = b.norm_constant();
  if (a.arguments() == b.arguments() &&
      norm_inf(a, b) < eps &&
      (norma > normb ? norma - normb : normb - norma) < eps) {
     return true;
  } else {
    boost::test_tools::predicate_result result(false);
    result.message() << "the two factors differ [\n"
                     << a << "!=" << b << "]";
    return result;
  }
}

template <>
boost::test_tools::predicate_result
are_close(const sill::canonical_gaussian& a,
          const sill::canonical_gaussian& b,
          sill::logarithmic<double> eps) {
  double multa = a.log_multiplier();
  double multb = b.log_multiplier();
  if (a.arguments() == b.arguments() &&
      norm_inf(a, b) < eps &&
      (multa > multb ? multa - multb : multb - multa) < eps) {
     return true;
  } else {
    boost::test_tools::predicate_result result(false);
    result.message() << "the two factors differ [\n"
                     << a << "!=" << b << "]";
    return result;
  }
}

#endif
