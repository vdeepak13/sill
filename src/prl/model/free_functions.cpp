#include <prl/model/free_functions.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  std::pair
  <pairwise_markov_network<table_factor >,
   std::map<finite_variable*, std::vector<finite_variable*> > >
  fm2pairwise_markov_network
  (const factorized_model<table_factor >& fm, universe& u) {
    typedef table_factor factor_type;
    typedef factorized_model<factor_type> fm_type;
    typedef pairwise_markov_network<factor_type> pmn_type;
    finite_domain vars(fm.arguments());
    std::list<factor_type> factor_list;
    std::map<finite_variable*, std::vector<finite_variable*> > var_mapping;
    foreach(factor_type f, fm.factors()) {
      const std::vector<finite_variable*>& f_arg_list = f.arg_list();
      if (f_arg_list.size() < 2)
        factor_list.push_back(f);
      else {
        // Create new variable (new_v) and a potential for it (new_f).
        finite_variable* new_v;
        factor_type new_f;
        boost::tie(new_v, new_f) = f.unroll(u);
        var_mapping[new_v] = f.arg_list();
        vars.insert(new_v);
        factor_list.push_back(new_f);
        // Create indicator potentials linking new_v and v in f_arg_list.
        // The potentials will have the form f(v, new_v).
        // If v has index i in f_arg_list, find product P of domains of all
        //  variables in f_arg_list with indices > i.  Then every P values of
        //  new_v, switch to the next value of v.
        std::vector<factor_type> pot_fs;
        std::map<finite_variable*, size_t> f_arg_list_inv;
        for (size_t i = 0; i < f_arg_list.size(); ++i) {
          std::vector<finite_variable*> pot_vars;
          pot_vars.push_back(f_arg_list[i]);
          pot_vars.push_back(new_v);
          pot_fs.push_back(factor_type(pot_vars, 0));
          f_arg_list_inv[f_arg_list[i]] = i;
        }
        finite_assignment pot_a;   // used for assignments to potential factors
        size_t new_v_val = 0;
        foreach(finite_assignment a, f.assignments()) {
          foreach(finite_variable* v, f_arg_list) {
            size_t i = f_arg_list_inv[v];
            // a iterates over assignments to f_arg_list
            // new_v_val is the value of new_v
            // i is the index of v in f_arg_list
            pot_a[new_v] = new_v_val;
            pot_a[v] = a[v];
            pot_fs[i].set_v(pot_a, 1);
          }
          ++new_v_val;
        }
        factor_list.insert(factor_list.end(), pot_fs.begin(), pot_fs.end());
      }
    }
    pmn_type pmn(vars);
    // Set all node factors to be constant 1 factors.
    foreach(pmn_type::vertex v, pmn.vertices())
      pmn.factor(v) = factor_type(make_domain(v), 1);
    /*
    // Why does this commented-out code not work?
    foreach(factor_type f, pmn.node_factors())
    f = factor_type(f.arguments(), 1);
    */
    // Add the actual factors.
    foreach(factor_type f, factor_list)
      pmn.add_factor(f);
    return std::make_pair(pmn, var_mapping);
  }

} // namespace prl

#include <prl/macros_undef.hpp>
