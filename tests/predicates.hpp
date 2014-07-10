#ifndef SILL_TEST_PREDICATES_HPP
#define SILL_TEST_PREDICATES_HPP

#include <fstream>
#include <cstdio>

#include <sill/serialization/iarchive.hpp>
#include <sill/serialization/oarchive.hpp>

std::string temp_filename() {
  char filename[L_tmpnam+1];
#ifdef WIN32
  filename[0] = '.';
  tmpnam(filename + 1);
#else
  tmpnam(filename);
#endif
  return std::string(filename);
}

//! Serializes and deserializes a value and verifies they are equal
template <typename T>
boost::test_tools::predicate_result
serialize_deserialize(const T& value) {
  std::string filename = temp_filename();
  using std::ios_base;
 
  std::ofstream fout(filename.c_str(), ios_base::binary | ios_base::out);
  assert(fout);
  sill::oarchive oa(fout);
  oa << value;
  fout.close();

  std::ifstream fin(filename.c_str(), ios_base::binary | ios_base::in);
  assert(fin);
  sill::iarchive ia(fin);
  T value2;
  ia >> value2;
  fin.close();

  remove(filename.c_str());

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
  std::string filename = temp_filename();
  using std::ios_base;
 
  std::ofstream fout(filename.c_str(), ios_base::binary | ios_base::out);
  assert(fout);
  sill::oarchive oa(fout);
  oa << value;
  fout.close();

  std::ifstream fin(filename.c_str(), ios_base::binary | ios_base::in);
  assert(fin);
  sill::iarchive ia(fin);
  ia.attach_universe(&u);
  T value2;
  ia >> value2;
  fin.close();

  remove(filename.c_str());

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
