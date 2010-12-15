#ifndef PRL_RANDOM_SAT_MODEL_HPP
#define PRL_RANDOM_SAT_MODEL_HPP

#include <algorithm>
#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

#include <prl/base/variable.hpp>
#include <prl/base/universe.hpp>
#include <prl/math/random.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/factor/log_table_factor.hpp>
#include <prl/model/factor_graph_model.hpp>

#include <prl/macros_def.hpp>
namespace prl {


template <typename F>
void random_k_sat(universe &u, factor_graph_model<F> &model, 
                  size_t numvars, 
                  size_t k,
                  double alpha = 4.0,    // #clauses = alpha * numvars.
                                        // hard problems are where
                                        // alpha in [3.92, 4.2667]
                                        //
                                        // See Survey and Belief Propagation 
                                        // on Random K-Sat (A.Braunstein,
                                        // R.Zecchina)
                   
                  double log_ising_coupling = -1.0 //  agreement is =log(0)
                                                  //   disagreement is 
                                                  //   log_ising_coupling
                  ) {

    std::vector<finite_variable*> variables;
    variables.resize(numvars);
    boost::mt19937 rng;               // Mersenne Twister
    boost::uniform_real<double> unifrand; // random number between 0 and 1

    for (size_t i = 0;i < numvars; ++i) {
      // name the variable
      std::string varname = boost::lexical_cast<std::string>(i);
      variables[i] = u.new_finite_variable(varname, 2);
    }

    size_t num_clauses = size_t(numvars * alpha);
    
    for (size_t i = 0;i < num_clauses; ++i) {
      finite_var_vector factordomain;
      factordomain.resize(k);
      // I need to random sample 3
      for (size_t j = 0;j < k; ++j) {
        size_t selectedvar = (size_t)(unifrand(rng) * (numvars - j)); 
        if (selectedvar >= numvars - j) selectedvar = numvars - j - 1;
        
        factordomain[j] = variables[selectedvar];
        std::swap(variables[numvars-1-j], variables[selectedvar]);
      }
      // recall that the clause is a bunch of "Ors"
      // pick a random assignment to be "false"
      finite_assignment falseassg;
      foreach(finite_variable* f, factordomain) {
        falseassg[f] = (unifrand(rng) >= 0.5);
      }
      
      F factor(factordomain, logarithmic<double>(0.0, log_tag()));
      factor.set_logv(falseassg, log_ising_coupling);
      model.add_factor(factor);
    }
  }



template <typename F>
void random_satisfiable_k_sat(universe &u, factor_graph_model<F> &model,
                  size_t numvars,
                  size_t k,
                  double alpha = 4.0,    // #clauses = alpha * numvars.
                                        // hard problems are where
                                        // alpha in [3.92, 4.2667]
                                        //
                                        // See Survey and Belief Propagation
                                        // on Random K-Sat (A.Braunstein,
                                        // R.Zecchina)

                  double log_ising_coupling = -1.0 //  agreement is =log(0)
                                                  //   disagreement is
                                                  //   log_ising_coupling
                  ) {

    std::vector<finite_variable*> variables;
    variables.resize(numvars);
    boost::mt19937 rng;               // Mersenne Twister
    boost::uniform_real<double> unifrand; // random number between 0 and 1

    finite_assignment sat;
    for (size_t i = 0;i < numvars; ++i) {
      // name the variable
      std::string varname = boost::lexical_cast<std::string>(i);
      variables[i] = u.new_finite_variable(varname, 2);
      sat[variables[i]] = (unifrand(rng) >= 0.5);
    }

    size_t num_clauses = size_t(numvars * alpha);

    for (size_t i = 0;i < num_clauses; ++i) {
      finite_var_vector factordomain;
      factordomain.resize(k);
      // I need to random sample 3
      for (size_t j = 0;j < k; ++j) {
        size_t selectedvar = (size_t)(unifrand(rng) * (numvars - j));
        if (selectedvar >= numvars - j) selectedvar = numvars - j - 1;

        factordomain[j] = variables[selectedvar];
        std::swap(variables[numvars-1-j], variables[selectedvar]);
      }
      // recall that the clause is a bunch of "Ors"
      // pick a random assignment to be "false"
      finite_assignment falseassg;
      foreach(finite_variable* f, factordomain) {
        falseassg[f] = (1 - sat[f]);
      }

      F factor(factordomain, logarithmic<double>(0.0, log_tag()));
      factor.set_logv(falseassg, log_ising_coupling);
      model.add_factor(factor);
    }
  }
};
#include <prl/macros_undef.hpp>

#endif
