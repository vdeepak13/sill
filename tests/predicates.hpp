#ifndef SILL_TEST_PREDICATES_HPP
#define SILL_TEST_PREDICATES_HPP

#include <fstream>
#include <cstdio>

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>

//! Serializes and deserializes a value and verifies they are equal
template <typename T>
boost::test_tools::predicate_result
serialize_deserialize(const T& value) {
  char filename[L_tmpnam];
  tmpnam(filename);

  std::ofstream fout(filename, std::fstream::binary);
  sill::oarchive oa(fout);
  oa << value;
  fout.close();

  std::ifstream fin(filename, std::fstream::binary);
  sill::iarchive ia(fin);
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


#endif
