
#ifndef SILL_ORACLE_HPP
#define SILL_ORACLE_HPP

// Note: Include dataset.hpp because of an issue with record_iterator being
//  within the dataset class, and record requiring dataset being defined so
//  that record_iterator may be declared a friend.  This could be fixed by
//  moving record_iterator to outside of the dataset class.
#include <sill/learning/dataset/dataset.hpp>
#include <sill/copy_ptr.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Data oracle interface.
   * \author Joseph Bradley
   * \ingroup learning_dataset
   */
  class oracle : public datasource {

    // Public type declarations
    //==========================================================================
  public:

    //! Base type (datasource)
    typedef datasource base;

    // Constructors
    //==========================================================================
  public:

    //! Constructs the oracle with the given sequence of variables
    //! NOTE: next() must be called to load the first record.
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    oracle(const finite_var_vector& finite_vars,
           const vector_var_vector& vector_vars,
           const std::vector<variable::variable_typenames>& var_type_order)
      : base(finite_vars, vector_vars, var_type_order) {
    }

    //! Constructs the oracle with the given sequence of variables
    //! NOTE: next() must be called to load the first record.
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    oracle(const forward_range<finite_variable*>& finite_vars,
           const forward_range<vector_variable*>& vector_vars,
           const std::vector<variable::variable_typenames>& var_type_order)
      : base(finite_vars, vector_vars, var_type_order) {
    }

    //! Constructs the datasource with the given sequence of variables.
    //! @param info    info from calling datasource_info()
    oracle(const datasource_info_type& info)
      : base(info) {
    }

    virtual ~oracle() { }

    // Getters and helpers
    //==========================================================================

    //! Returns the current record.
    virtual const record& current() const = 0;

    //! Returns the weight of the current example.
    //! Note: This does not have to be implemented.
    //! @todo Fix all oracles to support this!
    virtual double weight() const {
      return 1;
    }

    //! Returns the example limit of the oracle, or 0 if none/unknown.
    virtual size_t limit() const {
      return 0;
    }

    // Mutating operations
    //==========================================================================

    //! Increments the oracle to the next record.
    //! This returns true iff the operation was successful; false indicates
    //! a depleted oracle.
    //! NOTE: This must be explicitly called after the oracle is constructed.
    virtual bool next() = 0;

  }; // class oracle

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_ORACLE_HPP
