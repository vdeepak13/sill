#include <boost/bind.hpp>

#include <sill/math/logarithmic.hpp>
#include <sill/base/universe.hpp>
#include <sill/factor/util/norms.hpp>
#include <sill/factor/canonical_table.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/serialization/map.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  // Serialization
  //============================================================================
  void canonical_table::save(oarchive& ar) const {
    ar << args << arg_seq << table_data << var_index;
  }

  void canonical_table::load(iarchive& ar) {
    ar >> args >> arg_seq >> table_data >> var_index;
  }

  // Accessors
  //============================================================================
  
  void canonical_table::swap(canonical_table& f) {
    args.swap(f.args);
    var_index.swap(f.var_index);
    arg_seq.swap(f.arg_seq);
    
    dense_table<result_type> temp = table_data;
    table_data = f.table_data;
    f.table_data = temp;
  }


  finite_assignment_range canonical_table::assignments() const {
    return std::make_pair(finite_assignment_iterator(arg_seq),
                          finite_assignment_iterator());
  }



  bool canonical_table::operator==(const canonical_table& other) const {
    if (arguments() == other.arguments()) {
      if (arg_seq == other.arg_seq) // can directly compare the tables
        return table() == other.table();
      else // revert to combine
        return !combine_find(*this, other, std::not_equal_to<result_type>());
    }
    else return false;
  }

//   bool canonical_table::operator<(const canonical_table& other) const {
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


  canonical_table
  canonical_table::restrict(const finite_assignment& a) const {
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

    canonical_table factor(retained, result_type());
    // TODO: only need to initialize the table's dimensions, not its elements

    factor.table_data.restrict(table(),
                           make_restrict_map(arg_seq, a),
                           make_dim_map(factor.arg_seq, var_index));
    return factor;
  }

  canonical_table&
  canonical_table::subst_args(const finite_var_map& var_map) {
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
  canonical_table::marginal(canonical_table& f,
                             const finite_domain& retain) const {
    collapse(std::plus<result_type>(), 0, retain, f);
  }

  canonical_table& canonical_table::normalize() {
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
  double canonical_table::mutual_information(const finite_domain& fd1,
                                            const finite_domain& fd2) const {
    assert(set_disjoint(fd1, fd2));
    assert(includes(args, fd1) && includes(args, fd2));
    finite_assignment_iterator fa_it(set_union(fd1, fd2));
    finite_assignment_iterator fa_end;
    double mi = 0;
    if (args.size() > fd1.size() + fd2.size()) {
      canonical_table fctr(marginal(set_union(fd1,fd2)));
      canonical_table fctr1(fctr.marginal(fd1));
      canonical_table fctr2(fctr.marginal(fd2));
      while (fa_it != fa_end) {
        const finite_assignment& fa = *fa_it;
        mi += fctr.v(fa) *
          (std::log(fctr.v(fa)) - fctr1.logv(fa) - fctr2.logv(fa));
        ++fa_it;
      }
    } else {
      canonical_table fctr1(marginal(fd1));
      canonical_table fctr2(marginal(fd2));
      while (fa_it != fa_end) {
        const finite_assignment& fa = *fa_it;
        mi += v(fa) * (logv(fa)
                                - fctr1.logv(fa) - fctr2.logv(fa));
        ++fa_it;
      }
    }
    return mi;
  }

  double canonical_table::bp_msg_derivative_ub(variable_type* x,
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


  std::pair<finite_variable*, canonical_table >
  canonical_table::unroll(universe& u) const {
    size_t new_v_size = 1;
    foreach(finite_variable* v, arg_seq)
      new_v_size *= v->size();
    finite_variable* new_v = u.new_finite_variable(new_v_size);
    finite_var_vector new_args;
    new_args.push_back(new_v);
    canonical_table newf
      (new_args, std::vector<result_type>(values().first, values().second));
    return std::make_pair(new_v, newf);
  }


  canonical_table
  canonical_table::roll_up(const finite_var_vector& orig_arg_list) const {
    assert(args.size() == 1);
    finite_variable* arg = *(args.begin());
    size_t s = 1;
    foreach(finite_variable* v, orig_arg_list)
      s *= v->size();
    assert(s == arg->size());
    canonical_table newf
      (orig_arg_list,
       std::vector<result_type>(values().first, values().second));
    return newf;
  }


  finite_assignment
  canonical_table::assignment(const index_type& index) const {
    finite_assignment a;
    assert(index.size() == arg_seq.size());
    for(size_t i = 0; i < index.size(); i++)
      a[arg_seq[i]] = index[i];
    return a;
  }

  // Private helper functions
  //==========================================================================

  void canonical_table::initialize(
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

  canonical_table::index_type
  canonical_table::make_dim_map(const finite_var_vector& vars,
                                    const var_index_map& to_map) {
    // return make_vector(to_map.values(vars)); <-- slow
    dense_table<result_type>::index_type map(vars.size());
    for(size_t i = 0; i < vars.size(); i++) {
      map[i] = safe_get(to_map, vars[i]);
    }
    return map;
  }

  canonical_table::index_type
  canonical_table::make_restrict_map(const finite_var_vector& vars,
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
  canonical_table::make_index_map(const finite_domain& vars) {
    std::map<finite_variable*, size_t> var_index;
    size_t i = 0;
    foreach(finite_variable* v, vars) var_index[v] = i++;
    return var_index;
  }

  // Free functions
  //============================================================================
  std::ostream& operator<<(std::ostream& out, const canonical_table& f) {
    out << f.arg_vector() << std::endl;
    out << f.table();
    return out;
  }

  double norm_1(const canonical_table& x, const canonical_table& y) {
    return canonical_table::combine_collapse
      (x, y, abs_difference<canonical_table::result_type>(), 
       std::plus<canonical_table::result_type>(), 0.0);
  }

  double norm_inf(const canonical_table& x, const canonical_table& y) {
    return canonical_table::combine_collapse
      (x, y, abs_difference<canonical_table::result_type>(),
       maximum<canonical_table::result_type>(), 0);
  }

  
  double norm_inf_log(const canonical_table& x, const canonical_table& y) {
    return canonical_table::combine_collapse
      (x, y, 
       abs_difference_log<canonical_table::result_type>(), 
       maximum<canonical_table::result_type>(), 
       -std::numeric_limits<double>::infinity());
  }

  
  double norm_1_log(const canonical_table& x, const canonical_table& y) {
    return canonical_table::combine_collapse
         (x, y, 
         abs_difference_log<canonical_table::result_type>(), 
         std::plus<canonical_table::result_type>(), 0.0);
  }


  
  canonical_table weighted_update(const canonical_table& f1,
                                      const canonical_table& f2,
                                      double a) {
    return canonical_table::combine(f1, f2, 
           weighted_plus<canonical_table::result_type>(1-a, a));
  }

  
  canonical_table pow(const canonical_table& f, double a) {
    using std::pow;
    canonical_table result(f);
    foreach(canonical_table::result_type& x, result.table()) {
      x = pow(x, a);
    }
    return result;
  }

  finite_assignment arg_max(const canonical_table& f) {
    canonical_table::table_type::const_iterator it = max_element(f.values());
    return f.assignment(f.table().index(it));
  }

  finite_assignment arg_min(const canonical_table& f) {
    canonical_table::table_type::const_iterator it = min_element(f.values());
    return f.assignment(f.table().index(it));
  }
  // Operator Overloads
  // =====================================================================
  canonical_table& canonical_table::operator+=(const canonical_table& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      std::plus<canonical_table::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, std::plus<canonical_table::result_type>());
    }
    return *this;
  }
  
  canonical_table& canonical_table::operator-=(const canonical_table& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      std::minus<canonical_table::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, std::minus<canonical_table::result_type>());
    }
    return *this;
  }

  canonical_table& canonical_table::operator*=(const canonical_table& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      std::multiplies<canonical_table::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, std::multiplies<canonical_table::result_type>());
    }
    return *this;
  }

  canonical_table& canonical_table::operator/=(const canonical_table& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                           safe_divides<canonical_table::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, safe_divides<canonical_table::result_type>());
    }
    return *this;
  }

  canonical_table& canonical_table::logical_and(const canonical_table& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      sill::logical_and<canonical_table::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, sill::logical_and<canonical_table::result_type>());
    }
    return *this;
  }


  canonical_table& canonical_table::logical_or(const canonical_table& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      sill::logical_or<canonical_table::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, sill::logical_or<canonical_table::result_type>());
    }
    return *this;
  }
  
  canonical_table& canonical_table::max(const canonical_table& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      sill::maximum<canonical_table::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, sill::maximum<canonical_table::result_type>());
    }
    return *this;
  }

  canonical_table& canonical_table::min(const canonical_table& y) { 
    if (includes(this->arguments(), y.arguments())) {
      // We can implement the combination efficiently.
      table_data.join_with(y.table(), make_dim_map(y.arg_seq, var_index),
                      sill::minimum<canonical_table::result_type>());
    } else {
      // Revert to the standard implementation
      *this = combine(*this, y, sill::minimum<canonical_table::result_type>());
    }
    return *this;
  }


} // namespace sill

#include <sill/macros_undef.hpp>
