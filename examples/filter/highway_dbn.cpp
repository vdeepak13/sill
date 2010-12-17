#include "highway_dbn.hpp"

#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>

// The values of the prior distribution
boost::array<double, 3> prior_values =      // tail: empty
  {{0.3, 0.5, 0.2}};

// The values of the transition model for the last lane
boost::array<double, 9> transition_last =   // tail: vars_t[i]
  {{0.8, 0.2, 0.0,   // slow
    0.3, 0.5, 0.2,   // medium
    0.2, 0.3, 0.5}}; // fast

/* 

The original values from Mark's implementation made s'[i] independent
of s[i+1] given s[i]

// The values of the transition model for all other lanes
boost::array<double, 27> transition_values = // tail: vars_t1[i+1], vars_t[i]
  {{0.8, 0.2, 0.0,   // slow, slow
    0.8, 0.2, 0.0,   // slow, medium
    0.8, 0.2, 0.0,   // slow, fast
    0.3, 0.5, 0.2,   // medium, slow
    0.3, 0.5, 0.2,   // medium, medium
    0.3, 0.5, 0.2,   // medium, fast
    0.2, 0.3, 0.5,   // fast, slow
    0.2, 0.3, 0.5,   // fast, medium
    0.2, 0.3, 0.5}}; // fast, fast
*/

// The values of the transition model for all other lanes
boost::array<double, 27> transition_values = // tail: vars_t1[i+1], vars_t[i]
  {{0.8, 0.2, 0.0,   // slow, slow
    0.7, 0.3, 0.0,   // slow, medium
    0.5, 0.3, 0.2,   // slow, fast
    0.5, 0.5, 0.0,   // medium, slow
    0.2, 0.5, 0.3,   // medium, medium
    0.1, 0.5, 0.4,   // medium, fast
    0.2, 0.5, 0.3,   // fast, slow
    0.2, 0.3, 0.5,   // fast, medium
    0.1, 0.3, 0.6}}; // fast, fast

// The values of the observation model
boost::array<double, 9> observation_values = // tail: vars_t[i]
  {{0.8, 0.2, 0.0,   // slow
    0.1, 0.8, 0.1,   // medium
    0.0, 0.2, 0.8}}; // fast

typedef prl::dynamic_bayesian_network<prl::table_factor> dbn_type;

void highway_dbn(std::size_t n, 
                 dbn_type& dbn,
                 std::vector<prl::finite_timed_process*>& procs) {
  using namespace prl;
  dbn.clear();
  procs.clear();
  
  // Create the processes and obtain their variables
  finite_var_vector vars_t;
  finite_var_vector vars_t1;
  for(size_t i = 0; i < n; i++) {
    std::string istr = boost::lexical_cast<std::string>(i);
    finite_timed_process* p = new finite_timed_process("S" + istr, 3);
    procs.push_back(p);
    vars_t.push_back(p->current());
    vars_t1.push_back(p->next());
  }

//   finite_timed_process* obs_up = new finite_timed_process("Z_up", 3);
//   finite_timed_process* obs_dn = new finite_timed_process("Z_dn", 3);
  
  // Setup the prior model
  for(size_t i = 0; i < n; i++) {
    table_factor f = make_dense_table_factor(make_vector(vars_t[i]), prior_values);
    dbn.add_factor(vars_t[i], f);
  }

  // Setup the transition model
  for(size_t i = 0; i < n; i++) {
    table_factor f;
    if (i == n - 1) {
      finite_var_vector args = make_vector(vars_t1[i],vars_t[i]);
      // last lane only depends upon its previous speed
      f = make_dense_table_factor(args, transition_last);
    } else {
      // other segments also depend on the current speed of the downstream
      // segment
      finite_var_vector args = make_vector(vars_t1[i], vars_t[i], vars_t1[i+1]);
      f = make_dense_table_factor(args, transition_values);
    }
    dbn.add_factor(procs[i], f);
  }

//   // Add the observation model
//   {
//     finite_variable* obs = obs_up->current();
//     finite_var_vector args = make_vector(obs, vars_t[n-1]);
//     dbn.add_observation(make_dense_table_factor(args, obs_values), obs);

//     obs = obs_dn->current();
//     args[0] = obs;
//     dbn.add_observation(make_dense_table_factor(args, obs_values), obs);
//   }

}

