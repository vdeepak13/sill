
#ifndef SILL_MUTABLE_DATASET_HPP
#define SILL_MUTABLE_DATASET_HPP

#include <string>
#include <vector>

#include <boost/iterator.hpp>
#include <boost/random/uniform_int.hpp>

#include <sill/assignment.hpp>
#include <sill/datastructure/dense_table.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset/dataset.hpp>
#include <sill/range/forward_range.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class for mutable datasets.
   * \author Joseph Bradley
   * \ingroup learning_dataset
   * \todo serialization
   *
   * THIS IS INCOMPLETE.  I NEED TO DECIDE HOW TO ORGANIZE THE CLASS HIERARCHY.
   */
  class mutable_dataset : public dataset {

    typedef dataset base;

    // Constructors
    //==========================================================================
  public:

    //! Constructs an empty mutable_dataset
    mutable_dataset() : base() { }

    //! Constructs the mutable_dataset with the given variables.
    //! Note this does not let the mutable_dataset know the ordering of the
    //! variables in the data file, so var_order() cannot be used to load
    //! another mutable_dataset over the same variables.
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    mutable_dataset(const finite_domain& finite_vars,
                    const vector_domain& vector_vars)
      : base(finite_vars, vector_vars) {
    }

    //! Constructs the datset with the given sequence of variables
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    mutable_dataset(const finite_var_vector& finite_vars,
                    const vector_var_vector& vector_vars,
                    const std::vector<variable::variable_typenames>& var_type_order)
      : base(finite_vars, vector_vars, var_type_order) {
    }

    //! Constructs the datset with the given sequence of variables
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    mutable_dataset
    (const forward_range<finite_variable*>& finite_vars,
     const forward_range<vector_variable*>& vector_vars,
     const std::vector<variable::variable_typenames>& var_type_order)
      : base(finite_vars, vector_vars, var_type_order) {
    }

    virtual ~mutable_dataset() { }

    // Mutating operations
    //==========================================================================

    //! Adds a new record
    virtual void insert(const assignment& a) = 0;

    //! Adds a new record with finite variable values fvals and vector variable
    //! values vvals.
    virtual void
    insert(const std::vector<size_t>& fvals, const vec& vvals) = 0;

    //! Set all weights
    void set_weights(const vec& weights_) {
      assert(weights_.size() == size());
      this->weights_ = weights_;
    }

    //! Set a single weight of record i (with bound checking)
    void set_weight(size_t i, double weight_) {
      assert(i < size());
      weights_[i] = weight_;
    }

  };  // class mutable_dataset

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
