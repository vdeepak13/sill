#ifndef SILL_FINITE_DATASET_HPP
#define SILL_FINITE_DATASET_HPP

#include <sill/base/finite_assignment.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/learning/dataset/finite_record.hpp>
#include <sill/learning/dataset/raw_record_iterators.hpp>

#include <iostream>
#include <vector>

#include <boost/random/uniform_int.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  // forward declaration
  template <typename BaseDS> class slice_view;

  /**
   * A dataset that stores observations only for finite variables.
   * \see Dataset
   */
  class finite_dataset {
  public:
    // types for the Dataset concept
    typedef finite_variable   argument_type;
    typedef finite_domain     domain_type;
    typedef finite_var_vector vector_type;
    typedef finite_assignment assignment_type;
    typedef finite_record     record_type;

    typedef raw_record_iterator<finite_dataset>       record_iterator;
    typedef raw_const_record_iterator<finite_dataset> const_record_iterator;

    // other helper types
    typedef size_t elem_type;
    typedef double weight_type;

    //! Creates an uninitialized dataset
    finite_dataset() { }

    //! Destructor
    virtual ~finite_dataset() { }

    //! Returns the number of datapoints in the dataset.
    virtual size_t size() const = 0;

    //! Returns true if the dataset has no datapoints.
    bool empty() const { return size() == 0; }

    //! Returns the columns of this dataset.
    virtual finite_domain arguments() const = 0;

    //! Returns a single data point in the dataset's natural ordering.
    virtual finite_record record(size_t row) const = 0;

    //! Returns a single data point for a subset of the variables.
    virtual finite_record
    record(size_t row, const finite_var_vector& vars) const = 0;

    //! Returns mutable records for the specified finite variables.
    std::pair<record_iterator, record_iterator>
    records(const finite_var_vector& vars) {
      return std::make_pair(record_iterator(this, vars),
                            record_iterator(size()));
    }

    //! Returns immutable records for the specified finite variables.
    std::pair<const_record_iterator, const_record_iterator>
    records(const finite_var_vector& vars) const {
      return std::make_pair(const_record_iterator(this, vars),
                            const_record_iterator(size()));
    }

    //! Draws a random sample from this dataset.
    template <typename RandomNumberGenerator>
    finite_record sample(const finite_var_vector& vars,
                          RandomNumberGenerator& rng) const {
      assert(!empty());
      boost::uniform_int<size_t> uniform(0, size() - 1);
      return record(uniform(rng), vars);
    }

    // Utility functions, invoked by the iterators and subclasses
    //========================================================================
  protected:
    typedef raw_record_iterator_state<finite_dataset> iterator_state_type;

    //! initializes the data structures in the record iterator
    virtual aux_data* init(const finite_var_vector& args,
                           iterator_state_type& state) const = 0;

    //! advances the internal pointer in data by the given difference
    virtual void advance(ptrdiff_t diff,
                         iterator_state_type& state,
                         aux_data* data) const = 0;

    //! loads at most n rows
    virtual size_t load(size_t n,
                        iterator_state_type& state,
                        aux_data* data) const = 0;

    //! saves the previously loaded data
    virtual void save(iterator_state_type& state, aux_data* data) = 0;

    //! prints the summary of this dataset to a stream
    virtual void print(std::ostream& out) const = 0;

    // friends
    friend class raw_record_iterator<finite_dataset>;
    friend class raw_const_record_iterator<finite_dataset>;
    friend class slice_view<finite_dataset>;
    friend std::ostream& operator<<(std::ostream&, const finite_dataset&);

  }; // class finite_dataset

  //! \relates finite_dataset
  inline std::ostream&
  operator<<(std::ostream& out, const finite_dataset& ds) {
    ds.print(out);
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
