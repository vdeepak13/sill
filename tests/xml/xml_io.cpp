// tests the XML input/output

#include <fstream>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>

#include <sill/archive/xml_oarchive.hpp>
#include <sill/archive/xml_iarchive.hpp>

#include <sill/factor/xml/table_factor.hpp>
#include <sill/factor/xml/constant_factor.hpp>
#include <sill/factor/xml/gaussian_factors.hpp>
#include <sill/factor/xml/decomposable_fragment.hpp>

#include <sill/graph/grid_graphs.hpp>
#include <sill/model/markov_network.hpp>
#include <sill/model/random.hpp>
#include <sill/inference/junction_tree_inference.hpp>

#include <sill/math/bindings/lapack.hpp>

typedef sill::math::bindings::lapack::double_matrix matrix_type;
typedef sill::math::bindings::lapack::double_vector vector_type;

typedef sill::tablef table_factor;
typedef sill::constant_factor constant_factor;
typedef sill::canonical_gaussian<matrix_type, vector_type> canonical_gaussian;
typedef sill::decomposable_fragment<table_factor> decomposable_fragment;

namespace sill {
  map<std::string, xml_iarchive::deserializer*> xml_iarchive::deserializers;
}

/*
decomposable_fragment
generate_fragment(std::size_t m, std::size_t n, sill::universe& u) {
  boost::mt19937 rng;

  sill::pairwise_markov_network< table_factor > mn;
  sill::var_vector variables = u.new_finite_variables(m*n, 2);
  make_grid_graph(m, n, mn, variables);
  random_ising_model(mn, rng);

  sill::shafer_shenoy<table_factor> ss(mn);
  ss.calibrate();
  ss.normalize();

  return decomposable_fragment(ss.clique_beliefs());
}
*/

int main()
{
  using namespace std;
  using namespace sill;

  sill::universe u;
  {
    ofstream out("factors.xml");
    xml_oarchive arch(out);
    var_vector vf = u.new_finite_variables(2, 2);
    var_vector vv = u.new_vector_variables(2, 1);

    ::table_factor table(vf, 1);
    ::constant_factor constant(2);
    ::canonical_gaussian gaussian(vv, sill::identity_matrix<double>(2),
                                  sill::ones<double>(2));
    ::decomposable_fragment fragment; // = generate_fragment(2, 3, u);

    arch << table;
    arch << constant;
    arch << gaussian;
    arch << fragment;
    arch << domain(vf);
  }

  {
    xml_iarchive in("factors.xml", u);
    xml_iarchive::register_type<domain>("domain");

    ::table_factor table;
    ::constant_factor constant;
    ::canonical_gaussian gaussian;
    ::decomposable_fragment fragment;
    domain d;

    in >> table;
    in >> constant;
    in >> gaussian;
    in >> fragment;
    in >> d;

    cout << table << endl;
    cout << constant << endl;
    cout << gaussian << endl;
    cout << fragment << endl;
    cout << d << endl;
  }

}
