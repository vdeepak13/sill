#ifndef PRL_STANFORD_IO_HPP
#define PRL_STANFORD_IO_HPP

#include <iosfwd>
#include <sstream>

#include <prl/model/markov_network.hpp>

#include <prl/macros_def.hpp>

/**
 * \file io.hpp Functions for reading data in Stanford-like syntax.
 *              (this file may be renamed in the future to be more specific).
 */

namespace prl {

  /**
   * Reads a graphical model in Stanford-like syntax.
   * For now, only discrete models are supported.
   * @tparam F Factor type
   * \ingroup model
   */
  template <typename F>
  bool read_model(std::istream& in,
                  pairwise_markov_network<F>& mn,
                  universe& u) {
    using namespace std;
    typedef typename F::variable_type variable_type;

    string line;
    string dummy;    
    finite_var_vector variables;
    
    getline(in, line);
    // load the variables
    assert(line == "@Variables");
    
    getline(in, line); // comment out to check assertions
    while (line != "@End") {
      istringstream ss(line);
      string name;
      size_t size;
      ss >> name >> size;
      variables.push_back(u.new_finite_variable(name, size));
      assert(!(ss >> dummy));
      getline(in, line);
    }
    // cout << variables << endl;
    getline(in, line);
    
    // skip white space
    while(line.empty() && in.good()) getline(in, line);
    
    // load the potentials
    finite_domain vars(variables);
    mn.add_nodes(variables);
    // prl::pairwise_markov_network<F> mn(vars);
    
    assert(line == "@Potentials");
    getline(in, line);
    while(line != "@End") {
      // cout << line << endl;
      istringstream ss(line);
      size_t n, index;
      ss >> n;
      finite_var_vector args(n);
      for(size_t i = 0; i<n; i++) {
        ss >> index; assert(index<vars.size());
        args[i] = variables[index];
      }
      F f(args, 0);
      foreach(double& value, f.values()) ss >> value;
      mn.add_factor(f);
      assert(!(ss >> dummy));
      getline(in, line);
    }

    // return mn;
    return true;
  }
  
} // namespace prl


#include <prl/macros_undef.hpp>

#endif
