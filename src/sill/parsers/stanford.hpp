#ifndef SILL_STANFORD_IO_HPP
#define SILL_STANFORD_IO_HPP

#include <sill/base/universe.hpp>
#include <sill/model/markov_network.hpp>

#include <iosfwd>
#include <sstream>

namespace sill {

  /**
   * Reads a graphical model in Stanford-like syntax.
   * For now, only discrete models are supported.
   * @tparam F Factor type
   * \ingroup model
   */
  template <typename F>
  bool parse_stanford(std::istream& in,
                      pairwise_markov_network<F>& mn,
                      universe& u) {
    using namespace std;

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
    finite_domain vars = make_domain(variables);
    mn.add_nodes(vars);
    // sill::pairwise_markov_network<F> mn(vars);
    
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
      for (double& value : f.param()) { ss >> value; }
      mn.add_factor(f);
      assert(!(ss >> dummy));
      getline(in, line);
    }

    // return mn;
    return true;
  }
  
} // namespace sill

#endif
