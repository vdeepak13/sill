#include <boost/bind.hpp>

#include <sill/math/logarithmic.hpp>
#include <sill/base/universe.hpp>
#include <sill/factor/norms.hpp>
#include <sill/factor/log_table_factor.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/serialization/map.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  // Serialization
  //============================================================================
  void log_table_factor::save(oarchive& ar) const {
    ar << args << arg_seq << table_data << var_index;
  }

  void log_table_factor::load(iarchive& ar) {
    ar >> args >> arg_seq >> table_data >> var_index;
  }

  // Accessors
  //============================================================================
  
  void log_table_factor::swap(log_table_factor& f) {
    args.swap(f.args);
    var_index.swap(f.var_index);
    arg_seq.swap(f.arg_seq);
    
    dense_table<result_type> temp = table_data;
    table_data = f.table_data;
    f.table_data = temp;
  }


  finite_assignment_range log_table_factor::assignments() const {
    return std::make_pair(finite_assignment_iterator(arg_seq),
                          finite_assignment_iterator());
  }



  bool log_table_factor::operator==(const log_table_factor& other) const {
    if (arguments() == other.arguments()) {
      if (arg_seq == other.arg_seq) // can directly compare the tables
        return table() == other.table();
      else // revert to combine
        return !combine_find(*this, other, std::not_equal_to<result_type>());
    }
    else return false;
  }

//   bool log_table_factor::operator<(const log_table_factor& other) const {
//     if (this->arguments() < other.arguments()) return true;

//     if (this->arguments() == other.arguments()) {
//       // we need to always perform the combine, even if the argument sequences
//       // are the same, in order to traverse the table in the right order
//       boost::optional< std::pair<result_type,result_type> > values =
//         combine_find(*this, other, std::not_equal_to<result_type>());
//       return values && (values->first < values->second);
//     }
//     else return false;
//   }


  log_table_factor
  log_table_factor::restrict(const finite_assignment& a) const {
    finite_domain retained;
    //more efficient set difference, the domain of the factor is
    //supposed to be small, but evidence size can be very large
    foreach(finite_variable* v, arguments()){
      if(a.find(v) == a.end())
        retained.insert(v);
    }

    //non of the variables of this factors are assigned, return a copy
    if(retained.size() == arguments().size())
      return *this;

    log_table_factor factor(retained, result_type());
    // TODO: only need to initialize the table's dimensions, not its elements

    factor.table_data.restrict(table(),
                           make_restrict_map(arg_seq, a),
                           make_dim_map(factor.arg_seq, var_index));
    return factor;
  }

  log_table_factor&
  log_table_factor::subst_args(const finite_var_map& var_map) {
    args = subst_vars(args, var_map);
    // Compute the new var_index of the arguments, so it matches up
    // with the current var_index.
    for (size_t i = 0; i < arg_seq.size(); ++i) {
      arg_seq[i] = safe_get(var_map, arg_seq[i]);
    }
    var_index = rekey(var_index, var_map);
    return *this;
  }

  void
  log_table_factor::marginal(log_table_factor& f,
                             const finite_domain& retain) const {
    collapse(std::plus<result_type>(), 0, retain, f);
  }

  log_table_factor& log_table_factor::normalize() {
      result_type max_value = maximum();

      if (std::isinf(max_value.log_value())) return *this;
      assert( !std::isnan(max_value.log_value()) );
      // scale and compute normalizing constant
      double Z = 0.0;
      foreach(logarithmic<double>& d, values()) {
        d /= max_value;
        Z += double(d);
      }

      logarithmic<double> logZ = Z;
      // Normalize
      foreach(logarithmic<double>& d, values()) {
        d /= logZ;
      }
      return *this;
  }
 
  // TODO (Stano): this should be cleaned up to not use assignment_iterator
  double log_table_factor::mutual_information(const finite_domain& fd1,
                                            const finite_domain& fd2) const {
    assert(set_disjoint(fd1, fd2));
    assert(includes(args, fd1) && includes(args, fd2));
    finite_assignment_iterator fa_it(set_union(fd1, fd2));
    finite_assignment_iterator fa_end;
    double mi = 0;
    if (args.size() > fd1.size() + fd2.size()) {
      log_table_factor fctr(marginal(set_union(fd1,fd2)));
      log_table_factor fctr1(fctr.marginal(fd1));
      log_table_factor fctr2(fctr.marginal(fd2));
      while (fa_it != fa_end) {
        const finite_assignment& fa = *fa_it;
        mi += fctr.v(fa) *
          (std::log(fctr.v(fa)) - fctr1.logv(fa) - fctr2.logv(fa));
        ++fa_it;
      }
    } else {
      log_table_factor fctr1(marginal(fd1));
      log_table_factor fctr2(marginal(fd2));
      while (fa_it != fa_end) {
        const finite_assignment& fa = *fa_it;
        mi += v(fa) * (logv(fa)
                                - fctr1.logv(fa) - fctr2.logv(fa));
        ++fa_it;
      }
    }
    return mi;
  }

  double log_table_factor::bp_msg_derivative_ub(variable_type* x,
                                            variable_type* y) const {
    double result = 1;
    // get it in index coordinates
    size_t v = safe_get(var_index, x);
    size_t w = safe_get(var_index, y);
    foreach(const index_type& a_b_g, table_data.indices()) {
      index_type ap_b_g(a_b_g);
      foreach(const index_type& ap_bp_gp, table_data.indices()) {
        index_type a_bp_gp(ap_bp_gp);
        //in the notation of the paper,
        //f_a_b_g is alpha beta gamma
        //f_ap_bp_gp is alpha' beta' gamma'

        if(a_b_g[v] != ap_bp_gp[v] && a_b_g[w] != ap_bp_gp[w])
        {
          //f_ap_b_g is alpha' beta gamma
          ap_b_g[v] = ap_bp_gp[v];

          //f_a_bp_gp is alpha beta' gamma'
          a_bp_gp[v] = a_b_g[v];

          result = std::max(result,
          double(table_data(a_b_g) *
           table_data(ap_bp_gp) /
           (table_data(ap_b_g) *
            table_data(a_bp_gp))));
        }
      }
    }
    return tanh(std::log(result) * 0.25);
  }


  #ifndef SWIG // std::pair<variable, log_table_factor> not supported at the moment
  std::pair<finite_variable*, log_table_factor >
  log_table_factor::unroll(universe& u) const {
    size_t new_v_size = 1;
    foreach(finite_variable* v, arg_seq)
      new_v_size *= v->size();
    finite_variable* new_v = u.new_finite_variable(new_v_size);
    finite_var_vector new_args;
    new_args.push_back(new_v);
    log_table_factor newf
      (new_args, std::vector<result_type>(values().first, values().second));
    return std::make_pair(new_v, newf);
  }
  #endif

  log_table_factor
  log_table_factor::roll_up(const finite_var_vector& orig_arg_list) const {
    assert(args.size() == 1);
    finite_variable* arg = *(args.begin());
    size_t s = 1;
    foreach(finite_variable* v, orig_arg_list)
      s *= v->size();
    assert(s == arg->size());
    log_table_factor newf
      (orig_arg_list,
       std::vector<result_type>(values().first, values().second));
    return newf;
  }


  finite_assignment
  log_table_factor::assignment(const index_type& index) const {
    finite_assignment a;
    assert(index.size() == arg_seq.size());
    for(size_t i = 0; i < index.size(); i++)
      a[arg_seq[i]] = index[i];
    return a;
  }

  // Private helper functions
  //==========================================================================

  void log_table_factor::initialize(
                          const forward_range<finite_variable*>& arguments,
                          result_type default_value) {
    using namespace boost;
// I replaced this line with a manual copy, which is much faster.
//    arg_seq.assign(boost::begin(arguments), boost::end(arguments));
    forward_range<finite_variable*>::const_iterator arg_end(arguments.end());
    for (forward_range<finite_variable*>::const_iterator
           arg_it(arguments.begin());
         arg_it != arg_end; ++arg_it)
      arg_seq.push_back(*arg_it);
    var_index.clear();
    index_type geometry(arg_seq.size());
    for (size_t i = 0; i < arg_seq.size(); ++i) {
      var_index[arg_seq[i]] = i;
      geometry[i] = arg_seq[i]->size();
    }
    // Allocate the table.
    table_data = dense_table<result_type>(geometry, default_value);
  }

  log_table_factor::index_type
  log_table_factor::make_dim_map(const finite_var_vector& vars,
                                    const var_index_map& to_map) {
    // return make_vector(to_map.values(vars)); <-- slow
    dense_table<result_type>::index_type map(vars.size());
    for(size_t i = 0; i < vars.size(); i++) {
      map[i] = safe_get(to_map, vars[i]);
    }
    return map;
  }

  log_table_factor::index_type
  log_table_factor::make_restrict_map(const finite_var_vector& vars,
                                         const finite_assignment& a) {
    size_t retained = std::numeric_limits<size_t>::max();
    dense_table<result_type>::index_type map(vars.size(), retained);
    for(size_t i = 0; i < vars.size(); i++) {
      finite_assignment::const_iterator it = a.find(vars[i]);
      if (it != a.end()) map[i] = it->second;
    }
    return map;
  }

  std::map<finite_variable*, size_t>
  log_table_factor::make_index_map(const finite_domain& vars) {
    std::map<finite_variable*, size_t> var_index;
    size_t i = 0;
    foreach(finite_variable* v, vars) var_index[v] = i++;
    return var_index;
  }

  // Free functions
  //============================================================================
  std::ostream& operator<<(std::ostream& out, const log_table_factor& f) {
    out << f.arg_vector() << std::endl;
    out << f.table();
    return out;
  }

  double norm_1(const log_table_factor& x, const log_table_factor& y) {
    return log_table_factor::combine_collapse
      (x, y, abs_difference<log_table_factor::result_type>(), 
       std::plus<log_table_factor::result_type>(), 0.0);
  }

  double norm_inf(const log_table_factor& x, const log_table_factor& y) {
    return log_table_factor::combine_collapse
      (x, y, abs_difference<log_table_factor::result_type>(),
       maximum<log_table_factor::result_type>(), 0);
  }

  
  double norm_inf_log(const log_table_factor& x, const log_table_factor& y) {
    return log_table_factor::combine_collapse
      (x, y, 
       abs_difference_log<log_table_factor::result_type>(), 
       maximum<log_table_factor::result_type>(), 
       -std::numeric_limits<double>::infinity());
  }

  
  double norm_1_log(const log_table_factor& x, const log_table_factor& y) {
    return log_table_factor::combine_collapse
         (x, y, 
         abs_difference_log<log_table_factor::result_type>(), 
         std::plus<log_table_factor::result_type>(), 0.0);
  }


  
  log_table_factor weighted_update(const log_table_factor& f1,
                                      const log_table_factor& f2,
                                      double a) {
    return log_table_factor::combine(f1, f2, 
           weighted_plus<log_table_factor::result_type>(1-a, a));
  }

  
  log_table_factor pow(const log_table_factor& f, double a) {
    using std::pow;
    log_table_factor result(f);
    foreach(log_table_factor::result_type& x, result.table()) {
      x = pow(x, a);
    }
    return result;
  }

  finite_assignment arg_max(const log_table_factor& f) {
    log_table_factor::table_type::const_iterator it = max_element(f.values());
    return f.assignment(f.table().index(it));
  }

  finite_assignment arg_min(const log_table_factor& f) {
    log_table_factor::table_type::const_iterator it = min_element(f.values());
    return f.assignment(f.table().index(it));
  }
  // Operator Overloads
  // =====================================================================
  log_table_factor& log_table_factor::operator+=(const log_table_factor& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      std::plus<log_table_factor::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, std::plus<log_table_factor::result_type>());
    }
    return *this;
  }
  
  log_table_factor& log_table_factor::operator-=(const log_table_factor& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      std::minus<log_table_factor::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, std::minus<log_table_factor::result_type>());
    }
    return *this;
  }

  log_table_factor& log_table_factor::operator*=(const log_table_factor& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      std::multiplies<log_table_factor::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, std::multiplies<log_table_factor::result_type>());
    }
    return *this;
  }

  log_table_factor& log_table_factor::operator/=(const log_table_factor& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                           safe_divides<log_table_factor::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, safe_divides<log_table_factor::result_type>());
    }
    return *this;
  }

  log_table_factor& log_table_factor::logical_and(const log_table_factor& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      sill::logical_and<log_table_factor::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, sill::logical_and<log_table_factor::result_type>());
    }
    return *this;
  }


  log_table_factor& log_table_factor::logical_or(const log_table_factor& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      sill::logical_or<log_table_factor::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, sill::logical_or<log_table_factor::result_type>());
    }
    return *this;
  }
  
  log_table_factor& log_table_factor::max(const log_table_factor& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      sill::maximum<log_table_factor::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, sill::maximum<log_table_factor::result_type>());
    }
    return *this;
  }

  log_table_factor& log_table_factor::min(const log_table_factor& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      sill::minimum<log_table_factor::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, sill::minimum<log_table_factor::result_type>());
    }
    return *this;
  }


} // namespace sill

#include <sill/macros_undef.hpp>
