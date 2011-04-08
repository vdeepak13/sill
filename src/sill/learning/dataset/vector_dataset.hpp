
#ifndef SILL_VECTOR_DATASET_HPP
#define SILL_VECTOR_DATASET_HPP

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
#include <sill/math/norms.hpp>
#include <sill/range/forward_range.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that represents a dataset stored in a vectors which may be
   * efficiently loaded into records.
   *
   * Based on tests/dataset_view_timing, it seems that this class is much
   * faster than any other dataset, save for the original array_data class on
   * single element access (e.g., vector(i,j) to get a single real value).
   * It takes up about the same amount of memory.
   *
   * \author Joseph Bradley
   * \ingroup learning_dataset
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  class vector_dataset : public dataset<LA> {

    // Public type declarations
    //==========================================================================
  public:

    //! Base class
    typedef dataset<LA> base;

    typedef LA la_type;

    //! Import stuff from base class
    typedef typename base::record_type record_type;
    typedef typename base::vector_type vector_type;
    typedef typename base::record_iterator record_iterator;

    // Constructors
    //==========================================================================
  public:

    //! Constructor for empty dataset.
    vector_dataset();

    //! Constructs the dataset with the given sequence of variables
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    vector_dataset
    (const finite_var_vector& finite_vars,
     const vector_var_vector& vector_vars,
     const std::vector<variable::variable_typenames>& var_type_order,
     size_t nreserved = 1);

    //! Constructs the dataset with the given sequence of variables
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    vector_dataset
    (const forward_range<finite_variable*>& finite_vars,
     const forward_range<vector_variable*>& vector_vars,
     const std::vector<variable::variable_typenames>& var_type_order,
     size_t nreserved = 1);

    //! Constructs the datasource with the given sequence of variables.
    //! @param info    info from calling datasource_info()
    explicit vector_dataset(const datasource_info_type& info);

    //! Constructs the datasource with the given sequence of variables.
    //! @param info    info from calling datasource_info()
    vector_dataset(const datasource_info_type& info,
                   size_t nreserved);

    void save(oarchive& a) const;

    void load(iarchive& a);

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
    size_t capacity() const;

    //! Element access: record i, finite variable j (in the order finite_list())
    //! NOTE: This is slower than record_iterator.
    size_t finite(size_t i, size_t j) const;

    //! Element access: record i, vector variable j (in the order finite_list(),
    //! but with n-value vector variables using n indices j)
    //! NOTE: This is slower than record_iterator.
    double vector(size_t i, size_t j) const;

    //! Returns a range over the records of this dataset
    //! Eventually, will be able to provide a set of variables
    std::pair<record_iterator, record_iterator> records() const;

    //! Returns an iterator over the records of this dataset
    record_iterator begin() const;

    using base::load;

    // Mutating operations
    //==========================================================================

    //! Increases the capacity in anticipation of adding new elements.
    void reserve(size_t n);

    //! Sets record with index i to this value and weight.
    void set_record(size_t i, const assignment& a, double w = 1);

    //! Sets record with index i to this value and weight.
    void set_record(size_t i, const std::vector<size_t>& fvals,
                    const vector_type& vvals, double w = 1);

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

    using base::randomize;

    //! Randomly reorders the dataset (this is a mutable operation)
    void randomize(double random_seed);

    // Protected data members
    //==========================================================================
  protected:

    //! The type for storing data points' finite variable values.
    //! (# data points x # finite variables)
    typedef std::vector<std::vector<size_t> > finite_array;

    //! Type for storing data points' vector variable values.
    //! (# data points x total dimensionality of vector variables
    typedef std::vector<vector_type> vector_array;

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
    //! Note: This must be mutable for record_iterator to be efficient.
    mutable finite_array finite_data;

    //! Table of vector values.
    //! Note: This must be mutable for record_iterator to be efficient.
    mutable vector_array vector_data;

    // Protected methods required by record
    //==========================================================================

    //! Load datapoint i into assignment a
    void load_assignment(size_t i, sill::assignment& a) const;

    //! Load record i into r
    void load_record(size_t i, record_type& r) const;

    //! Load finite data for datapoint i into findata
    void load_finite(size_t i, std::vector<size_t>& findata) const;

    //! Load vector data for datapoint i into vecdata
    void load_vector(size_t i, vector_type& vecdata) const;

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

  };  // class vector_dataset

  //============================================================================
  // Implementations of methods in vector_dataset
  //============================================================================

  // Constructors
  //==========================================================================

  template <typename LA>
  vector_dataset<LA>::vector_dataset() : base() { }

  template <typename LA>
  vector_dataset<LA>::vector_dataset
  (const finite_var_vector& finite_vars,
   const vector_var_vector& vector_vars,
   const std::vector<variable::variable_typenames>& var_type_order,
   size_t nreserved)
    : base(finite_vars, vector_vars, var_type_order) {
    reserve(nreserved);
  }

  template <typename LA>
  vector_dataset<LA>::vector_dataset
  (const forward_range<finite_variable*>& finite_vars,
   const forward_range<vector_variable*>& vector_vars,
   const std::vector<variable::variable_typenames>& var_type_order,
   size_t nreserved)
    : base(finite_vars, vector_vars, var_type_order) {
    reserve(nreserved);
  }

  template <typename LA>
  vector_dataset<LA>::vector_dataset(const datasource_info_type& info)
    : base(info) {
    reserve(1);
  }

  template <typename LA>
  vector_dataset<LA>::vector_dataset(const datasource_info_type& info,
                                 size_t nreserved)
    : base(info) {
    reserve(nreserved);
  }

  template <typename LA>
  void vector_dataset<LA>::save(oarchive& a) const {
    base::save(a);
    a << finite_data << vector_data;
  }

  template <typename LA>
  void vector_dataset<LA>::load(iarchive& a) {
    base::load(a);
    a >> finite_data >> vector_data;
  }

  // Getters and helpers
  //==========================================================================

  template <typename LA>
  size_t vector_dataset<LA>::capacity() const {
    return finite_data.size();
  }

  template <typename LA>
  size_t vector_dataset<LA>::finite(size_t i, size_t j) const {
    assert(i < nrecords && j < num_finite());
    return finite_data[i][j];
  }

  template <typename LA>
  double vector_dataset<LA>::vector(size_t i, size_t j) const {
    assert(i < nrecords && j < dvector);
    return vector_data[i][j];
  }

  template <typename LA>
  std::pair<typename vector_dataset<LA>::record_iterator,
            typename vector_dataset<LA>::record_iterator>
  vector_dataset<LA>::records() const {
    if (nrecords > 0)
      return std::make_pair
        (make_record_iterator(0, &(finite_data[0]), &(vector_data[0])),
         make_record_iterator(nrecords));
    else
      return std::make_pair
        (make_record_iterator(0), make_record_iterator(nrecords));
  }

  template <typename LA>
  typename vector_dataset<LA>::record_iterator
  vector_dataset<LA>::begin() const {
    if (nrecords > 0)
      return make_record_iterator(0, &(finite_data[0]), &(vector_data[0]));
    else
      return make_record_iterator(0);
  }

  // Mutating operations
  //==========================================================================

  template <typename LA>
  void vector_dataset<LA>::reserve(size_t n) {
    if (n > capacity()) {
//      size_t n_previous = finite_data.size();
      finite_data.resize(n);
//      for (size_t i = n_previous; i < n; ++i)
//        finite_data[i].resize(num_finite());
      vector_data.resize(n);
//      for (size_t i = n_previous; i < n; ++i)
//        vector_data[i].resize(dvector);
      if (weighted)
        weights_.resize(n, true);
    }
  }

  template <typename LA>
  void vector_dataset<LA>::set_record(size_t i, const assignment& a, double w) {
    assert(i < nrecords);
    convert_finite_assignment2record(a.finite(), finite_data[i]);
    convert_vector_assignment2record(a.vector(), vector_data[i]);
    if (weighted) {
      weights_[i] = w;
    } else if (w != 1) {
      std::cerr << "vector_dataset::set_record() called with weight w != 1"
                << " on an unweighted dataset." << std::endl;
      assert(false);
    }
  }

  template <typename LA>
  void
  vector_dataset<LA>::set_record(size_t i, const std::vector<size_t>& fvals,
                                 const vector_type& vvals, double w) {
    assert(i < nrecords);
    assert(fvals.size() == num_finite());
    assert(vvals.size() == dvector);
    finite_data[i] = fvals;
    vector_data[i] = vvals;
    if (weighted) {
      weights_[i] = w;
    } else if (w != 1) {
      std::cerr << "vector_dataset::set_record() called with weight w != 1"
                << " on an unweighted dataset." << std::endl;
      assert(false);
    }
  }

  template <typename LA>
  std::pair<vec, vec> vector_dataset<LA>::normalize() {
    vec means(dvector,0);
    vec std_devs(dvector,0);
    double total_ds_weight(0);
    vector_type tmpvec(dvector,0);
    for (size_t i = 0; i < nrecords; ++i) {
      means += weight(i) * vector_data[i];
      tmpvec = vector_data[i];
      tmpvec.elem_mult(tmpvec);
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
  void vector_dataset<LA>::normalize(const vec& means,
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
    }
  }

  template <typename LA>
  void vector_dataset<LA>::normalize(const vec& means, const vec& std_devs,
                                     const vector_var_vector& vars) {
    foreach(vector_variable* v, vars)
      assert(this->has_variable(v));
    ivec vars_inds(vector_indices(vars));
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
        vector_data[i](j2) -= means(j);
        vector_data[i](j2) /= stddevs(j);
      }
    }
  }

  template <typename LA>
  void vector_dataset<LA>::normalize2(const vector_var_vector& vars) {
    foreach(vector_variable* v, vars)
      assert(this->has_variable(v));
    ivec vars_inds(vector_indices(vars));
    for (size_t i(0); i < nrecords; ++i) {
      double normalizer(norm_2(vector_data[i](vars_inds)));
      if (normalizer == 0)
        continue;
      foreach(size_t j2, vars_inds)
        vector_data[i](j2) /= normalizer;
    }
  }

  template <typename LA>
  void vector_dataset<LA>::randomize(double random_seed) {
    boost::mt11213b rng(static_cast<unsigned>(random_seed));
    for (size_t i = 0; i < nrecords-1; ++i) {
      size_t j = (size_t)(boost::uniform_int<int>(i,nrecords-1)(rng));
      finite_data[i].swap(finite_data[j]);
      vector_data[i].swap(vector_data[j]);
      if (weighted)
        std::swap(weights_[i], weights_[j]);
    }
  }

  // Protected methods required by record
  //==========================================================================

  template <typename LA>
  void
  vector_dataset<LA>::load_assignment(size_t i, sill::assignment& a) const {
    assert(i < nrecords);
    convert_finite_record2assignment(finite_data[i], a.finite());
    convert_vector_record2assignment(vector_data[i], a.vector());
  }

  template <typename LA>
  void vector_dataset<LA>::load_record(size_t i, record_type& r) const {
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
  void
  vector_dataset<LA>::load_finite(size_t i, std::vector<size_t>& findata) const{
    findata = finite_data[i];
  }

  template <typename LA>
  void vector_dataset<LA>::load_vector(size_t i, vector_type& vecdata) const {
    vecdata = vector_data[i];
  }

  template <typename LA>
  void
  vector_dataset<LA>::load_assignment_pointer(size_t i, assignment** a) const {
    assert(false);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
