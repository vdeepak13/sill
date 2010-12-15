#ifndef PRL_GAUSSIAN_FACTOR_HPP
#define PRL_GAUSSIAN_FACTOR_HPP

#include <map>

#include <prl/base/stl_util.hpp>
#include <prl/base/vector_assignment.hpp>
#include <prl/factor/factor.hpp>
#include <prl/learning/dataset/vector_record.hpp>
#include <prl/math/irange.hpp>
#include <prl/math/logarithmic.hpp>
#include <prl/math/vector.hpp>
#include <prl/range/concepts.hpp>
#include <prl/range/forward_range.hpp>
#include <prl/serialization/serialize.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * The base class of all Guassians
   *
   * \ingroup factor_types
   */
  class gaussian_factor : public factor {
  public:

    // Public type declarations 
    //==========================================================================
  public:
    //! implements Factor::result_type
    typedef logarithmic<double> result_type;

    //! implements Factor::domain_type
    typedef vector_domain domain_type;

    //! implements Factor::variable_type
    typedef vector_variable variable_type;

    //! implements Factor::assignment_type
    typedef vector_assignment assignment_type;

    //! implements Factor::record_type
    typedef vector_record record_type;

    // Protected member data and member functions
    //==========================================================================
  protected:

    //! The map from each variable to its index range
    std::map<vector_variable*, irange> var_range;

    //! The argument set of this factor
    vector_domain args;

  protected:
    //! Default constructor
    gaussian_factor() { }

    //! Initializes the given set of arguments
    gaussian_factor(const vector_domain& vars) : args(vars) { }

    //! Initializes the given set of arguments
    gaussian_factor(const vector_var_vector& vars)
      : args(vars.begin(), vars.end()) { }

    //! Initializes the given set of arguments
    gaussian_factor(const forward_range<vector_variable*>& vars)
      : args(vars.begin(), vars.end()) { }

    //! Assigns an index range to each variable in vars in an increasing order.
    void compute_indices(const vector_var_vector& vars) {
      size_t n = 0;
      foreach(vector_variable* v, vars) {
        var_range[v] = irange(n, n + v->size());
        n = n + v->size();
      }
    }

    //! Renames the arguments and the variable-index range map
    void subst_args(const vector_var_map& map) {
      args = subst_vars(args, map);
      var_range = rekey(var_range, map);
    }

    // Public member functions
    //==========================================================================
  public:
    //! Returns the argument set of this factor
    const vector_domain& arguments() const {
      return args;
    }

    //! Returns an array of indices for the given list of variables
    ivec indices(const vector_var_vector& vars) const {
      ivec ind(vector_size(vars));
      size_t n = 0;
      foreach(vector_variable* v, vars) {
        const irange& range = safe_get(var_range, v);
        for(size_t i = 0; i < range.size(); i++)
          ind[n++] = range(i);
      }
      return ind;
    }

    //! Sets 'ind' to an array of indices for the given list of variables.
    void indices(const vector_var_vector& vars, ivec& ind) const {
      size_t n(vector_size(vars));
      if (ind.size() != n)
        ind.resize(n);
      n = 0;
      foreach(vector_variable* v, vars) {
        const irange& range = safe_get(var_range, v);
        for(size_t i = 0; i < range.size(); i++)
          ind[n++] = range(i);
      }
    }

    //! Returns an array of indices for the given set of variables
    ivec indices(const vector_domain& vars) const {
      ivec ind(vector_size(vars));
      size_t n = 0;
      foreach(vector_variable* v, vars) {
        const irange& range = safe_get(var_range, v);
        for(size_t i = 0; i < range.size(); i++)
          ind[n++] = range(i);
      }
      return ind;
    }

    //! Returns an array of indices for the given set of variables
    void indices(const vector_domain& vars, ivec& ind) const {
      size_t n(vector_size(vars));
      if (ind.size() != n)
        ind.resize(n);
      n = 0;
      foreach(vector_variable* v, vars) {
        const irange& range = safe_get(var_range, v);
        for(size_t i = 0; i < range.size(); i++)
          ind[n++] = range(i);
      }
    }

  }; // class gaussian_factor

}

#include <prl/macros_undef.hpp>

#endif
