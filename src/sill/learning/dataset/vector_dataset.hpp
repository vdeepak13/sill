#ifndef SILL_VECTOR_DATASET_HPP
#define SILL_VECTOR_DATASET_HPP

#include <sill/base/vector_assignment.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/learning/dataset/raw_record_iterators.hpp>
#include <sill/learning/dataset/vector_record.hpp>

#include <iostream>
#include <vector>

#include <boost/random/uniform_int.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  // forward declaration
  template <typename BaseDS> class slice_view;

  /**
   * A dataset that stores observations only for vector variables.
   * 
   * \tparam T the internal storage of the vector values. This should match the
   *         storage type of the learned factors.
   * \see Dataset
  */
  template <typename T = double>
  class vector_dataset {
  public:
    // types for the Dataset concept
    typedef vector_variable   argument_type;
    typedef vector_domain     domain_type;
    typedef vector_var_vector vector_type;
    typedef vector_assignment assignment_type;
    typedef vector_record<T>  record_type;

    typedef raw_record_iterator<vector_dataset>       record_iterator;
    typedef raw_const_record_iterator<vector_dataset> const_record_iterator;

    // other helper types
    typedef T elem_type;
    typedef T weight_type;

    //! Creates an uninitialized dataset
    vector_dataset() { }

    //! Destructor
    virtual ~vector_dataset() { }

    //! Returns the number of datapoints in the dataset.
    virtual size_t size() const = 0;

    //! Returns true if the dataset has no records.
    bool empty() const { return size() == 0; }

    //! Returns the columns of this dataset.
    virtual vector_domain arguments() const = 0;

    //! Returns a single data point in the dataset's natural ordering.
    virtual vector_record<T> record(size_t row) const = 0;

    //! Returns a single data point for a subset of the variables.
    virtual vector_record<T>
    record(size_t row, const vector_var_vector& vars) const = 0;

    //! Returns mutable records for the specified vector variables.
    std::pair<record_iterator, record_iterator>
    records(const vector_var_vector& vars) {
      return std::make_pair(record_iterator(this, vars),
                            record_iterator(size()));
    }

    //! Returns immutable records for the specified vector variables.
    std::pair<const_record_iterator, const_record_iterator>
    records(const vector_var_vector& vars) const {
      return std::make_pair(const_record_iterator(this, vars),
                            const_record_iterator(size()));
    }

    //! Draws a random sample from this dataset.
    template <typename RandomNumberGenerator>
    vector_record<T> sample(const vector_var_vector& vars,
                             RandomNumberGenerator& rng) const {
      assert(!empty());
      boost::uniform_int<size_t> uniform(0, size() - 1);
      return record(uniform(rng), vars);
    }

    // Utility functions, invoked by the iterators and public functions
    //========================================================================
  protected:
    typedef raw_record_iterator_state<vector_dataset> iterator_state_type;

    //! initializes the data structures in the record iterator
    virtual aux_data* init(const vector_var_vector& args,
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
    friend class raw_record_iterator<vector_dataset>;
    friend class raw_const_record_iterator<vector_dataset>;
    friend class slice_view<vector_dataset>;
    friend std::ostream& operator<<(std::ostream&, const vector_dataset&);

  }; // class vector_dataset

  //! \relates vector_dataset
  template <typename T>
  inline std::ostream&
  operator<<(std::ostream& out, const vector_dataset<T>& ds) {
    ds.print(out);
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
