
#ifndef SILL_VECTOR_ASSIGNMENT_DATASET_HPP
#define SILL_VECTOR_ASSIGNMENT_DATASET_HPP

#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include <boost/iterator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <sill/base/assignment.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset/dataset.hpp>
#include <sill/range/forward_range.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that represents a dataset stored both in vectors and in
   * assignments.
   * THIS IS A HACK TO SUPPORT BOTH RECORDS AND ASSIGNMENTS EFFICIENTLY
   * AND SHOULD BE ELIMINATED ONCE I CONVINCE THEM TO MAKE ASSIGNMENTS
   * AN INTERFACE INSTEAD OF A TYPEDEF FROM STD::MAP.
   *
   * \author Joseph Bradley
   * \ingroup learning_dataset
   */
  class vector_assignment_dataset : public dataset {

    // Public type declarations
    //==========================================================================
  public:

    //! Base class
    typedef dataset base;

    //! Import public typedefs and functions from base class
    typedef base::record_iterator record_iterator;

    // Protected data members
    //==========================================================================
  protected:

    //! Table of finite values.
    //! (# data points x # finite variables)
    //! Note: This must be mutable for record_iterator to be efficient.
    mutable std::vector<std::vector<size_t> > finite_data;

    //! Table of vector values.
    //! (# data points x total dimensionality of vector variables)0
    //! Note: This must be mutable for record_iterator to be efficient.
    mutable std::vector<vec> vector_data;

    //! The same data, stored in assignments.
    mutable std::vector<sill::assignment> data_vector;

    // Protected methods required by record
    //==========================================================================

    //! Load datapoint i into assignment a
    void load_assignment(size_t i, sill::assignment& a) const;

    //! Load record i into r
    void load_record(size_t i, record& r) const;

    //! Load finite data for datapoint i into findata
    void load_finite(size_t i, std::vector<size_t>& findata) const {
      findata = finite_data[i];
    }

    //! Load vector data for datapoint i into vecdata
    void load_vector(size_t i, vec& vecdata) const {
      vecdata = vector_data[i];
    }

    //! ONLY for datasets which use assignments as native types:
    //!  Load the pointer to datapoint i into (*a).
    void load_assignment_pointer(size_t i, assignment** a) const;

    // Protected methods
    //==========================================================================

    void init(size_t nreserved);

    // Constructors
    //==========================================================================
  public:

    //! Constructor for empty dataset.
    vector_assignment_dataset() : base() { }

    //! Constructs the dataset with the given sequence of variables
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    vector_assignment_dataset
    (const finite_var_vector& finite_vars,
     const vector_var_vector& vector_vars,
     const std::vector<variable::variable_typenames>& var_type_order,
     size_t nreserved = 1)
      : base(finite_vars, vector_vars, var_type_order) {
      init(nreserved);
    }

    //! Constructs the dataset with the given sequence of variables
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    vector_assignment_dataset
    (const forward_range<finite_variable*>& finite_vars,
     const forward_range<vector_variable*>& vector_vars,
     const std::vector<variable::variable_typenames>& var_type_order,
     size_t nreserved = 1)
      : base(finite_vars, vector_vars, var_type_order) {
      init(nreserved);
    }

    //! Constructs the datasource with the given sequence of variables.
    //! @param info    info from calling datasource_info()
    explicit vector_assignment_dataset(const datasource_info_type& info,
                                       size_t nreserved = 1)
      : base(info) {
      init(nreserved);
    }

    // Getters and helpers
    //==========================================================================

    //! Return capacity
    size_t capacity() const {
      return finite_data.size();
    }

    //! Element access: record i, finite variable j (in the order finite_list())
    //! NOTE: This is slower than record_iterator.
    size_t finite(size_t i, size_t j) const {
      assert(i < nrecords && j < num_finite());
      return finite_data[i][j];
    }

    //! Element access: record i, vector variable j (in the order finite_list(),
    //! but with n-value vector variables using n indices j)
    //! NOTE: This is slower than record_iterator.
    double vector(size_t i, size_t j) const {
      assert(i < nrecords && j < dvector);
      return vector_data[i][j];
    }

    //! Returns a range over the records of this dataset
    //! Eventually, will be able to provide a set of variables
    std::pair<record_iterator, record_iterator> records() const;

    //! Returns an iterator over the records of this dataset
    record_iterator begin() const;

    //! Returns a range over the records of this dataset, as assignments.
    //! Eventually, will be able to provide a set of variables.
    std::pair<assignment_iterator, assignment_iterator>
    assignments() const {
      return std::make_pair(make_assignment_iterator(0, false),
                            make_assignment_iterator(nrecords, false));
    }

    //! Returns an iterator over the records of this dataset, as assignments.
    assignment_iterator begin_assignments() const {
      return make_assignment_iterator(0, false);
    }

    // Mutating operations
    //==========================================================================

    //! Increases the capacity in anticipation of adding new elements.
    void reserve(size_t n);

    //! Sets record with index i to this value and weight.
    void set_record(size_t i, const assignment& a, double w = 1);

    //! Sets record with index i to this value and weight.
    void set_record(size_t i, const std::vector<size_t>& fvals,
                    const vec& vvals, double w = 1);

    using base::normalize;

    //! Normalizes the vector data so that the empirical mean and variance
    //! are 0 and 1, respectively.
    //! This takes record weights into account.
    //! @return pair<means, std_devs>
    std::pair<vec, vec> normalize();

    //! Normalizes the vector data using the given means and std_devs
    //!  (which are assumed to be correct).
    void normalize(const vec& means, const vec& std_devs);

    //! Normalizes the vector data using the given means and std_devs
    //!  (which are assumed to be correct).
    //! @param vars  Only apply normalization to these variables.
    void normalize(const vec& means, const vec& std_devs,
                   const vector_var_vector& vars);

    using base::normalize2;

    //! Normalizes the vector data so that each record's vector values lie
    //! on the unit sphere.
    //! @param vars  Only apply normalization to these variables.
    void normalize2(const vector_var_vector& vars);

    //! Clears the dataset of all records.
    //! NOTE: This should not be called if views of the data exist!
    //!       (This is unsafe but very useful for avoiding reallocation.)
    void clear() {
      nrecords = 0;
    }

    using base::randomize;

    //! Randomly reorders the dataset (this is a mutable operation)
    void randomize(double random_seed);

  };  // class vector_assignment_dataset

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
