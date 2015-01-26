#ifndef SILL_GAUSSIAN_FACTOR_HPP
#define SILL_GAUSSIAN_FACTOR_HPP

#include <map>

#include <sill/base/stl_util.hpp>
#include <sill/base/vector_assignment.hpp>
#include <sill/factor/base/factor.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/dataset_old/vector_record.hpp>
#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/math/logarithmic.hpp>
#include <sill/range/concepts.hpp>
#include <sill/serialization/serialize.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * The base class of all Guassians
   *
   * \ingroup factor_types
   */
  class gaussian_base : public factor {

    // Public type declarations 
    //==========================================================================
  public:
    // Factor concept types
    typedef logarithmic<double> result_type;
    typedef double              real_type;
    typedef vector_variable     variable_type;
    typedef vector_domain       domain_type;
    typedef vector_var_vector   var_vector_type;
    typedef vector_assignment   assignment_type;

    // IndexableFactor concept types
    typedef vec index_type;

    // LearnableFactor concept types
    typedef vector_dataset<double> dataset_type;
    typedef vector_record_old<dense_linear_algebra<> > record_type;

    // Protected member data and member functions
    //==========================================================================
  protected:

    //! The map from each variable to its index span
    std::map<vector_variable*, span> var_span;

    //! The argument set of this factor
    vector_domain args;

  protected:
    //! Default constructor
    gaussian_base() { }

    //! Initializes the given set of arguments
    gaussian_base(const vector_domain& vars) : args(vars) { }

    //! Initializes the given set of arguments
    gaussian_base(const vector_var_vector& vars)
      : args(vars.begin(), vars.end()) { }

    //! Assigns an index span to each variable in vars in an increasing order.
    void compute_indices(const vector_var_vector& vars) {
      size_t n = 0;
      foreach(vector_variable* v, vars) {
        var_span[v] = span(n, n + v->size() - 1);
        n = n + v->size();
      }
    }

    //! Renames the arguments and the variable-index span map
    void subst_args(const vector_var_map& map) {
      args = subst_vars(args, map);
      var_span = rekey(var_span, map);
    }

    // Public member functions
    //==========================================================================
  public:
    //! Returns the argument set of this factor
    const vector_domain& arguments() const {
      return args;
    }

    //! Returns an array of indices for the given list of variables
    uvec indices(const vector_var_vector& vars) const {
      uvec ind;
      indices(vars, ind);
      return ind;
    }

    //! Sets 'ind' to an array of indices for the given list of variables.
    //! @param strict  If true, require that this factor include all of vars.
    //!                (default = true)
    void
    indices(const vector_var_vector& vars, uvec& ind, bool strict = true) const{
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
        ind.set_size(n);
      n = 0;
      foreach(vector_variable* v, vars) {
        std::map<vector_variable*, span>::const_iterator
          it(var_span.find(v));
        if (it == var_span.end()) {
          if (strict) {
            throw std::runtime_error
              (std::string("gaussian_base::indices(vars,ind,strict)") +
               " called with some vars not included in this factor and" +
               " strict = true.");
          }
        } else {
          const span& s = it->second;
          for(size_t i = s.a; i <= s.b; i++)
            ind[n++] = i;
        }
      }
    }

    //! Returns an array of indices for the given set of variables.
    uvec indices(const vector_domain& vars) const {
      uvec ind;
      indices(vars, ind);
      return ind;
    }

    //! Sets 'ind' to an array of indices for the given set of variables.
    //! @param strict  If true, require that this factor include all of vars.
    //!                (default = true)
    void
    indices(const vector_domain& vars, uvec& ind, bool strict = true) const {
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
        ind.set_size(n);
      n = 0;
      foreach(vector_variable* v, vars) {
        std::map<vector_variable*, span>::const_iterator
          it(var_span.find(v));
        if (it == var_span.end()) {
          if (strict) {
            throw std::runtime_error
              (std::string("gaussian_base::indices(vars,ind,strict)") +
               " called with some vars not included in this factor and" +
               " strict = true.");
          }
        } else {
          const span& s = it->second;
          for(size_t i = s.a; i <= s.b; i++)
            ind[n++] = i;
        }
      }
    }

  }; // class gaussian_base

}

#include <sill/macros_undef.hpp>

#endif
