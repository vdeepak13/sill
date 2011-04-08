#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include <boost/array.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/uniform_int.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/random/random.hpp>
#include <sill/model/crf_graph.hpp>
#include <sill/serialization/serialize.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

/**
 * \file crf_graph_test.cpp CRF graph test
 */
int main() {

  using boost::array;

  // Create a universe.
  universe u;

  // Create an empty CRF graph
  crf_graph<finite_variable, variable, variable> crf;
  std::cout << "Empty CRF graph: " << std::endl;
  crf.print(std::cout);

  // Create variables
  size_t nvars(4);
  finite_var_vector Y(nvars);
  for(size_t i = 0; i < Y.size(); ++i)
    Y[i] = u.new_finite_variable("Variable: " + to_string(i), 2);
  finite_var_vector Xfinite(nvars);
  for(size_t i = 0; i < Xfinite.size(); ++i)
    Xfinite[i] = u.new_finite_variable("Variable: " + to_string(i+nvars), 2);
  vector_var_vector Xvector(nvars);
  for(size_t i = 0; i < Xvector.size(); ++i)
    Xvector[i] = u.new_vector_variable("Variable: " + to_string(i+2*nvars), 1);

  // Create some unary factors
  for (size_t i = 0; i < Y.size() / 2; ++i)
    crf.add_factor(make_domain<finite_variable>(Y[i]),
                   make_domain<variable>(Xfinite[i], Xvector[i]));

  // Create some larger factors
  for (size_t i = 1; i < Y.size(); i += 2)
    crf.add_factor(make_domain<finite_variable>(Y[i-1], Y[i]),
                   make_domain<variable>(Xfinite[i-1], Xfinite[i],
                                         Xvector[i-1], Xvector[i]));

  std::cout << "Print the CRF graph:" << std::endl;
  crf.print(std::cout);

}
