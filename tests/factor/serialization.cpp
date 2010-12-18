#define SILL_PRINT_VARIABLE_ADDRESS

// This file demonstrates serialization / deserialization to different
// archive types.
//
// TODO: at the moment, this file takes a very long time to compile

#include <iostream>
#include <fstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
// #include <boost/archive/xml_oarchive.hpp>
// #include <boost/archive/xml_iarchive.hpp>
// #include <boost/archive/binary_iarchive.hpp>
// #include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/variable.hpp>
#include <sill/stl_io.hpp>
#include <sill/factor/constant_factor.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/gaussian_factors.hpp>
#include <sill/factor/mixture.hpp>
#include <sill/factor/random.hpp>

#include <sill/math/bindings/lapack.hpp>
#include <sill/range/numeric.hpp>

#include <sill/macros_def.hpp>

boost::mt19937 rng;

namespace archive = boost::archive;

// todo: we could switch to polymorphic serialization here
template <typename T, typename Archive>
void save_object(const std::string& filename, const T& object) {
  // could use boost::filesystem here to avoid the .c_str() call
  std::ofstream ofs(filename.c_str());
  Archive oa(ofs);
  oa << serialization_nvp(object);
}

template <typename T, typename Archive>
T load_object(const std::string& filename) {
  std::ifstream ifs(filename.c_str());
  Archive ia(ifs);
  T object;
  ia >> serialization_nvp(object);
  return object;
}

// text archives work very well: portable and concise
template <typename T>
void save_text(const std::string& filename, const T& object) {
  save_object<T, archive::text_oarchive>(filename, object);
}

template <typename T>
T load_text(const std::string& filename) {
  return load_object<T, archive::text_iarchive>(filename);
}

/*
// binary archives are guaranteed to not modify the floating point numbers
// but are not portable.
template <typename T>
void save_bin(const std::string& filename, const T& object) {
  save_object<T, archive::binary_oarchive>(filename, object);
}

template <typename T>
T load_bin(const std::string& filename) {
  return load_object<T, archive::binary_iarchive>(filename);
}

// XML archives are ridiculously verbose but easy to inspect
template <typename T>
void save_xml(const std::string& filename, const T& object) {
  save_object<T, archive::xml_oarchive>(filename, object);
}

template <typename T>
T load_xml(const std::string& filename) {
  return load_object<T, archive::xml_iarchive>(filename);
}
*/

// There's also a slower runtime-polymorphic archive for use in
// shared libraries (it uses any of the above as an implementation),
// see demo_polymorphic.cpp in boost-trunk/libs/serialization/example

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  // Serialize a domain
  typedef std::pair<finite_domain,finite_variable*> domain_variable_pair;
  {
    universe u; // will go out of scope before deserialization
    finite_variable* a = u.new_finite_variable("a", 1);
    finite_variable* b = u.new_finite_variable("b", 5);

    finite_domain vars;
    vars.insert(a);
    vars.insert(b);

    save_text("serialized_vars.txt", std::make_pair(vars, a));
    cout << "Original set of variables: " << vars << endl;
  }

  finite_domain vars;
  finite_variable* a2;
  boost::tie(vars, a2) = load_text<domain_variable_pair>("serialized_vars.txt");
  cout << "Restored set of variables: " << vars << "," << a2 << endl;

  // Check that deserialization correctly deals with copy-on-write
  long K = 3; // number of copies
  typedef std::vector<finite_domain> domain_vector;
  {
    universe u;
    domain_vector domains(K, make_domain(u.new_finite_variable(2),
                                         u.new_finite_variable(5)));
    cout << "Original domains: " << domains << endl;
    cout << "Number of copies: " << domains[0].use_count() << endl;
    assert(domains[0].use_count()==K);
    save_text("serialized_domains.txt", domains);
  }

  domain_vector domains = load_text<domain_vector>("serialized_domains.txt");
  cout << "Restored domain: " << domains << endl;
  cout << "Number of copies: " << domains[0].use_count() << endl;
  assert(domains[0].use_count()==K);

  // Serialize a vector of constant factors
  typedef std::vector<constant_factor > factor_vector;
  {
    factor_vector factors;
    for(double val = 0; val < 10; val++) factors.push_back(val);

    save_text("serialized_factors.txt", factors);
    cout << "Original factors: " << factors << endl;
  }

  factor_vector factors = load_text<factor_vector>("serialized_factors.txt");
  cout << "Restored factors: " << factors << endl;

  // Serialize a table factor through base pointer
  universe u;
  finite_variable* a = u.new_finite_variable(3);
  finite_variable* b = u.new_finite_variable(4);
  // variable_h c = u.new_finite_variable(2);
  typedef tablef table_factor;
//   domain args2 = make_domain(b, c);
//   table_factor f2 = random_discrete_factor< table_factor >(args2, rng);
  {
    finite_domain args1 = make_domain(a, b);
    table_factor f1 = random_discrete_factor< table_factor >(args1, rng);
    save_text("serialized_table_factor.txt", (serializable*)&f1);
    cout << "Original factor: " << f1 << endl;
//     cout << "Op: " << (f1 * f2).marginal(a) << endl;
//     cout << "f1 == 1*f1: " << (f1 == f1*constant_factor(1)) << endl;
  }
  serializable* obj =
    load_text<serializable*>("serialized_table_factor.txt");

  table_factor factor = *dynamic_cast<table_factor*>(obj);
  cout << "Restored factor: " << factor << endl;

  // Serialize a pair of Gaussian factors
  vector_variable* x = u.new_vector_variable(1);
  vector_variable* y = u.new_vector_variable(2);
  vector_domain xy = make_domain(x, y);
  vector_var_vector xy_vec(xy.begin(), xy.end());

  using sill::math::bindings::lapack::double_matrix;
  using sill::math::bindings::lapack::double_vector;
  typedef canonical_gaussian<double_matrix,double_vector> canonical_gaussian;
  typedef moment_gaussian<double_matrix,double_vector> moment_gaussian;
  typedef std::pair<canonical_gaussian, moment_gaussian> factor_pair;
  {
    canonical_gaussian cg(xy, identity_matrix<double>(3), ones<double>(3));
    moment_gaussian mg(xy_vec, scalars<double>(3,3.14159),
                       identity_matrix<double>(3));
    cout << "Original Gaussians: " << make_pair(cg,mg) << endl;
    save_text("serialized_gaussians.txt", make_pair(cg, mg));
    /* save_xml("serialized_gaussians.xml", make_pair(cg, mg));
       save_bin("serialized_gaussians.bin", make_pair(cg, mg)); */
  }

  factor_pair gpair_text = load_text<factor_pair>("serialized_gaussians.txt");
  cout << "Restored Gaussians (text): " << gpair_text << endl;

  /*
  factor_pair gpair_xml = load_xml<factor_pair>("serialized_gaussians.xml");
  cout << "Restored Gaussians (xml): " << gpair_xml << endl;

  factor_pair gpair_bin = load_bin<factor_pair>("serialized_gaussians.bin");
  cout << "Restored Gaussians (bin): " << gpair_bin << endl;
  */

  // Test serialization of mixtures of Gaussians
  typedef mixture<moment_gaussian> gaussian_mixture;
  {
    moment_gaussian mg(xy_vec, scalars<double>(3,3.14159),
                       identity_matrix<double>(3));
    moment_gaussian mg1(xy_vec, scalars<double>(3,1),
                       identity_matrix<double>(3));    
    gaussian_mixture mix(2, mg);
    //mix[1] = mg1;
    // there is an issue with serializing / deserializing the mixture
    // when it contains two copies of the same component
    // -- we fail at the shared pointer / sill::map
    cout << "Original mixture: " << mix << endl;
    save_text("serialized_mixture.txt", mix);
  }
  
  gaussian_mixture mix = load_text<gaussian_mixture>("serialized_mixture.txt");
  cout << "Restored mixture: " << mix << endl;

  return EXIT_SUCCESS;
}
