#ifndef SILL_GAUSSIAN_FACTOR_HPP
#define SILL_GAUSSIAN_FACTOR_HPP

#include <sill/argument/domain.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/datastructure/vector_map.hpp>
#include <sill/factor/base/factor.hpp>
#include <sill/math/eigen/matrix_index.hpp>

namespace sill {

  /**
   * The base class of all Guassian factors.
   *
   * \ingroup factor_types
   */
  class gaussian_factor : public factor {
  public:
    typedef domain<vector_variable*> domain_type;

    // Constructors and indexing
    //==========================================================================

    //! Default constructor.
    gaussian_factor() { }

    //! Initializes the argument index to the given arguments.
    explicit gaussian_factor(const domain_type& args) {
      compute_start(args);
    }

    //! Initializes the argument index to the given domains.
    gaussian_factor(const domain_type& args1, const domain_type& args2) {
      compute_start(args1, args2);
    }

    //! Returns the start of a single variable.
    size_t start(vector_variable* v) const {
      auto it = start_.find(v);
      if (it == start_.end()) {
        throw std::invalid_argument("Could not find variable " + v->str());
      }
      return it->second;
    }

    //! Returns the indices of arguments of this corresponding to dom.
    matrix_index index_map(const domain_type& dom) const {
      matrix_index result;
      for (vector_variable* v : dom) {
        result.append(start(v), v->size());
      }
      return result;
    }
    
  protected:
    // Protected members
    //==========================================================================

    //! Assigns a starting index span to each argument in an increasing order.
    size_t compute_start(const domain_type& args) {
      start_.clear();
      start_.reserve(args.size());
      size_t m = insert_start(args);
      start_.sort();
      return m;
    }

    //! Assigns a starting index span to each argument in an increasing order.
    std::pair<size_t,size_t>
    compute_start(const domain_type& args1, const domain_type& args2) {
      start_.clear();
      start_.reserve(args1.size() + args2.size());
      size_t m = insert_start(args1);
      size_t n = insert_start(args2);
      start_.sort();
      return std::make_pair(m, n);
    }

    //! Renames the arguments and the variable-index span map
    void subst_args(const vector_var_map& map) {
      start_.subst_keys(map);
    }

    //! The base implementation of swap.
    void base_swap(gaussian_factor& other) {
      swap(start_, other.start_);
    }

    //! The map from each variable to its index span
    vector_map<vector_variable*, size_t> start_;

  private:
    //! Inserts a domain into the start structure.
    size_t insert_start(const domain_type& args) {
      size_t n = 0;
      for (vector_variable* v : args) {
        start_.emplace(v, n);
        n += v->size();
      }
      return n;
    }

  }; // class gaussian_factor

} // namespace sill

#endif
