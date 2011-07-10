
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
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<double,size_t>)
   */
  template <typename LA = dense_linear_algebra<> >
  class vector_assignment_dataset : public dataset<LA> {

    // Public type declarations
    //==========================================================================
  public:

    //! Base class
    typedef dataset<LA> base;

    typedef LA la_type;

    //! Import public typedefs and functions from base class
    typedef typename base::vector_type     vector_type;
    typedef typename base::record_type     record_type;
    typedef typename base::record_iterator_type record_iterator_type;
    typedef typename base::assignment_iterator assignment_iterator;

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

    // From datasource
    using base::num_finite;
    using base::num_vector;
    using base::variable_type_order;
    using base::var_order;
    //    using base::var_order_index;
    //    using base::variable_index;
    using base::record_index;
    using base::vector_indices;
    using base::finite_numbering;
    using base::finite_numbering_ptr;
    using base::vector_numbering;
    using base::vector_numbering_ptr;

    // From dataset
    using base::size;
    using base::weight;

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

    //! Returns an iterator over the records of this dataset
    record_iterator_type begin() const;

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

    // Protected data members
    //==========================================================================
  protected:

    // From datasource
    //    using base::finite_vars;
    using base::finite_seq;
    using base::finite_numbering_ptr_;
    using base::dfinite;
    using base::finite_class_vars;
    //    using base::vector_vars;
    using base::vector_seq;
    using base::vector_numbering_ptr_;
    using base::dvector;
    using base::vector_class_vars;
    using base::var_type_order;
    //    using base::var_order_map;
    //    using base::vector_var_order_map;

    // From dataset
    using base::nrecords;
    using base::weighted;
    using base::weights_;

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
    void load_record(size_t i, record_type& r) const;

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

    // From datasource:
    using base::convert_finite_record2assignment;
    using base::convert_vector_record2assignment;
    using base::convert_finite_assignment2record;
    using base::convert_vector_assignment2record;

    // From dataset:
    using base::make_assignment_iterator;
    using base::make_record_iterator;

    void init(size_t nreserved);

  };  // class vector_assignment_dataset

  //============================================================================
  // Implementations of methods in vector_assignment_dataset
  //============================================================================

  // Getters and helpers
  //==========================================================================

  template <typename LA>
  typename vector_assignment_dataset<LA>::record_iterator_type
  vector_assignment_dataset<LA>::begin() const {
    if (nrecords > 0)
      return make_record_iterator(0, &(finite_data[0]), &(vector_data[0]));
    else
      return make_record_iterator(0);
  }

  // Mutating operations
  //==========================================================================

  template <typename LA>
  void vector_assignment_dataset<LA>::reserve(size_t n) {
    size_t n_previous = finite_data.size();
    if (n > capacity()) {
      finite_data.resize(n);
      for (size_t i = n_previous; i < n; ++i)
        finite_data[i].resize(num_finite());
      vector_data.resize(n);
      for (size_t i = n_previous; i < n; ++i)
        vector_data[i].set_size(dvector);
      data_vector.resize(n);
      if (weighted)
        weights_.reshape(n,1);
    }
  }

  template <typename LA>
  void vector_assignment_dataset<LA>::set_record(size_t i, const assignment& a, double w) {
    assert(i < nrecords);
    convert_finite_assignment2record(a.finite(), finite_data[i]);
    convert_vector_assignment2record(a.vector(), vector_data[i]);
    data_vector[i].clear();
    foreach(finite_variable* v, finite_seq) {
      finite_assignment::const_iterator it(a.finite().find(v));
      assert(it != a.finite().end());
      data_vector[i].finite()[v] = it->second;
    }
    foreach(vector_variable* v, vector_seq) {
      vector_assignment::const_iterator it(a.vector().find(v));
      assert(it != a.vector().end());
      data_vector[i].vector()[v] = it->second;
    }
    if (weighted) {
      weights_[i] = w;
    } else if (w != 1) {
      std::cerr << "vector_assignment_dataset::set_record() called with weight w != 1"
                << " on an unweighted dataset." << std::endl;
        assert(false);
    }
  }

  template <typename LA>
  void vector_assignment_dataset<LA>::set_record(size_t i, const std::vector<size_t>& fvals,
                                             const vec& vvals, double w) {
    assert(i < nrecords);
    sill::copy(fvals, finite_data[i].begin());
    sill::copy(vvals, vector_data[i].begin());
    convert_finite_record2assignment(fvals, data_vector[i].finite());
    convert_vector_record2assignment(vvals, data_vector[i].vector());
    if (weighted) {
      weights_[i] = w;
    } else if (w != 1) {
      std::cerr << "vector_assignment_dataset::set_record() called with weight w != 1"
                << " on an unweighted dataset." << std::endl;
      assert(false);
    }
  }

  template <typename LA>
  std::pair<vec, vec> vector_assignment_dataset<LA>::normalize() {
    vec means(dvector,0);
    vec std_devs(dvector,0);
    double total_ds_weight(0);
    vec tmpvec(dvector,0);
    for (size_t i = 0; i < nrecords; ++i) {
      means += weight(i) * vector_data[i];
      tmpvec = vector_data[i];
      tmpvec *= tmpvec;
      std_devs += weight(i) * tmpvec;
      total_ds_weight += weight(i);
    }
    means /= total_ds_weight;
    std_devs /= total_ds_weight;
    for (size_t j = 0; j < dvector; ++j)
      std_devs[j] = sqrt(std_devs[j] - means[j] * means[j]);
    normalize(means, std_devs);
    return std::make_pair(means, std_devs);
  }

  template <typename LA>
  void vector_assignment_dataset<LA>::normalize(const vec& means,
                                            const vec& std_devs) {
    assert(means.size() == dvector);
    assert(std_devs.size() == dvector);
    vec stddevs(std_devs);
    for (size_t j(0); j < dvector; ++j) {
      if (stddevs[j] < 0)
        assert(false);
      if (stddevs[j] == 0)
        stddevs[j] = 1;
    }
    for (size_t i(0); i < nrecords; ++i) {
      vector_data[i] -= means;
      vector_data[i] /= stddevs;
      convert_vector_record2assignment(vector_data[i], data_vector[i].vector());
    }
  }

  template <typename LA>
  void vector_assignment_dataset<LA>::normalize(const vec& means,
                                            const vec& std_devs,
                                            const vector_var_vector& vars) {
    foreach(vector_variable* v, vars)
      assert(this->has_variable(v));
    uvec vars_inds(vector_indices(vars));
    assert(means.size() == vars_inds.size());
    assert(std_devs.size() == vars_inds.size());
    vec stddevs(std_devs);
    for (size_t j(0); j < stddevs.size(); ++j) {
      if (stddevs[j] < 0)
        assert(false);
      if (stddevs[j] == 0)
        stddevs[j] = 1;
    }
    for (size_t i(0); i < nrecords; ++i) {
      for (size_t j(0); j < stddevs.size(); ++j) {
        size_t j2(vars_inds[j]);
        vector_data[i][j2] -= means[j];
        vector_data[i][j2] /= stddevs[j];
      }
      convert_vector_record2assignment(vector_data[i], data_vector[i].vector());
    }
  }

  template <typename LA>
  void
  vector_assignment_dataset<LA>::normalize2(const vector_var_vector& vars) {
    foreach(vector_variable* v, vars)
      assert(this->has_variable(v));
    uvec vars_inds(vector_indices(vars));
    for (size_t i(0); i < nrecords; ++i) {
      double normalizer = norm(vector_data[i](vars_inds),2);
      if (normalizer == 0)
        continue;
      foreach(size_t j2, vars_inds)
        vector_data[i][j2] /= normalizer;
      convert_vector_record2assignment(vector_data[i], data_vector[i].vector());
    }
  }

  template <typename LA>
  void vector_assignment_dataset<LA>::randomize(double random_seed) {
    boost::mt11213b rng(static_cast<unsigned>(random_seed));
    std::vector<size_t> fin_tmp(finite_seq.size());
    vec vec_tmp;
    vec_tmp.set_size(dvector);
    assignment tmpa;
    double weight_tmp;
    for (size_t i = 0; i < nrecords-1; ++i) {
      size_t j = (size_t)(boost::uniform_int<int>(i,nrecords-1)(rng));
      sill::copy(finite_data[i], fin_tmp.begin());
      sill::copy(finite_data[j], finite_data[i].begin());
      sill::copy(fin_tmp, finite_data[j].begin());
      sill::copy(vector_data[i], vec_tmp.begin());
      sill::copy(vector_data[j], vector_data[i].begin());
      sill::copy(vec_tmp, vector_data[j].begin());
      tmpa = data_vector[i];
      data_vector[i] = data_vector[j];
      data_vector[j] = tmpa;
      if (weighted) {
        weight_tmp = weights_[i];
        weights_[i] = weights_[j];
        weights_[j] = weight_tmp;
      }
    }
  }

  // Protected methods required by record
  //==========================================================================

  template <typename LA>
  void vector_assignment_dataset<LA>::load_assignment(size_t i, sill::assignment& a) const {
    assert(i < nrecords);
    a = data_vector[i];
  }

  template <typename LA>
  void vector_assignment_dataset<LA>::load_record(size_t i, record_type& r) const {
    if (r.fin_own)
      r.fin_ptr->operator=(finite_data[i]);
    else
      r.fin_ptr = &(finite_data[i]);
    if (r.vec_own)
      r.vec_ptr->operator=(vector_data[i]);
    else
      r.vec_ptr = &(vector_data[i]);
  }

  template <typename LA>
  void vector_assignment_dataset<LA>::load_assignment_pointer(size_t i, assignment** a) const {
    assert(i < nrecords);
    *a = &(data_vector[i]);
  }

  // Protected methods
  //==========================================================================

  template <typename LA>
  void vector_assignment_dataset<LA>::init(size_t nreserved) {
    assert(nreserved > 0);
    finite_data.resize(nreserved);
    for (size_t i = 0; i < nreserved; ++i)
      finite_data[i].resize(num_finite());
    vector_data.resize(nreserved);
      for (size_t i = 0; i < nreserved; ++i)
        vector_data[i].set_size(dvector);
    data_vector.resize(nreserved);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
