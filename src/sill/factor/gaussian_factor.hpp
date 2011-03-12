#ifndef SILL_GAUSSIAN_FACTOR_HPP
#define SILL_GAUSSIAN_FACTOR_HPP

#include <map>

#include <sill/base/stl_util.hpp>
#include <sill/base/vector_assignment.hpp>
#include <sill/factor/factor.hpp>
#include <sill/learning/dataset/vector_record.hpp>
#include <sill/math/irange.hpp>
#include <sill/math/logarithmic.hpp>
#include <sill/math/vector.hpp>
#include <sill/range/concepts.hpp>
#include <sill/range/forward_range.hpp>
#include <sill/serialization/serialize.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * The base class of all Guassians
   *
   * \ingroup factor_types
   */
  class gaussian_factor : public factor {

    // Public type declarations 
    //==========================================================================
  public:
    //! implements Factor::result_type
    typedef logarithmic<double> result_type;

    //! implements Factor::variable_type
    typedef vector_variable variable_type;

    //! implements Factor::domain_type
    typedef vector_domain domain_type;

    typedef vector_var_vector var_vector_type;
    typedef vector_var_map    var_map_type;

    //! implements Factor::assignment_type
    typedef vector_assignment assignment_type;

    //! implements Factor::record_type
    typedef vector_record<dense_linear_algebra<> > record_type;

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
      ivec ind;
      indices(vars, ind);
      return ind;
    }

    //! Sets 'ind' to an array of indices for the given list of variables.
    //! @param strict  If true, require that this factor include all of vars.
    //!                (default = true)
    void
    indices(const vector_var_vector& vars, ivec& ind, bool strict = true) const{
      size_t n = 0;
      if (strict) {
        n = vector_size(vars);
      } else {
        foreach(vector_variable* v, vars) {
          if (args.count(v) != 0)
            n += v->size();
        }
      }
      if (ind.size() != n)
        ind.resize(n);
      n = 0;
      foreach(vector_variable* v, vars) {
        std::map<vector_variable*, irange>::const_iterator
          it(var_range.find(v));
        if (it == var_range.end()) {
          if (strict) {
            throw std::runtime_error
              (std::string("gaussian_factor::indices(vars,ind,strict)") +
               " called with some vars not included in this factor and" +
               " strict = true.");
          }
        } else {
          const irange& range = it->second;
          for(size_t i = 0; i < range.size(); i++)
            ind[n++] = range(i);
        }
      }
    }

    //! Returns an array of indices for the given set of variables.
    ivec indices(const vector_domain& vars) const {
      ivec ind;
      indices(vars, ind);
      return ind;
    }

    //! Sets 'ind' to an array of indices for the given set of variables.
    //! @param strict  If true, require that this factor include all of vars.
    //!                (default = true)
    void
    indices(const vector_domain& vars, ivec& ind, bool strict = true) const {
      size_t n = 0;
      if (strict) {
        n = vector_size(vars);
      } else {
        foreach(vector_variable* v, vars) {
          if (args.count(v) != 0)
            n += v->size();
        }
      }
      if (ind.size() != n)
        ind.resize(n);
      n = 0;
      foreach(vector_variable* v, vars) {
        std::map<vector_variable*, irange>::const_iterator
          it(var_range.find(v));
        if (it == var_range.end()) {
          if (strict) {
            throw std::runtime_error
              (std::string("gaussian_factor::indices(vars,ind,strict)") +
               " called with some vars not included in this factor and" +
               " strict = true.");
          }
        } else {
          const irange& range = it->second;
          for(size_t i = 0; i < range.size(); i++)
            ind[n++] = range(i);
        }
      }
    }

  }; // class gaussian_factor

}

#include <sill/macros_undef.hpp>

#endif
