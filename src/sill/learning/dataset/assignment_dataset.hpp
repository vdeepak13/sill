
#ifndef SILL_ASSIGNMENT_DATASET_HPP
#define SILL_ASSIGNMENT_DATASET_HPP

#include <string>
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
   * A class that represents a dataset stored in assignments.
   * (This should be used with PRL graphical models, not with PRL
   *  discriminative learners.)
   *
   * \author Joseph Bradley
   * \ingroup learning_dataset
   * \todo serialization
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<double,size_t>)
   */
  template <typename LA = dense_linear_algebra<> >
  class assignment_dataset : public dataset<LA> {

    // Public type declarations
    //==========================================================================
  public:

    typedef LA la_type;

    //! Base class
    typedef dataset<la_type> base;

    //! Import public typedefs and functions from base class
    typedef typename base::vector_type         vector_type;
    typedef typename base::record_type         record_type;
    typedef typename base::assignment_iterator assignment_iterator;
    typedef typename base::record_iterator     record_iterator;

    // Constructors
    //==========================================================================
  public:

    //! Constructor for empty dataset.
    assignment_dataset() : base() { }

    //! Constructs the dataset with the given sequence of variables
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    assignment_dataset(const finite_var_vector& finite_vars,
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
    assignment_dataset
    (const forward_range<finite_variable*>& finite_vars,
     const forward_range<vector_variable*>& vector_vars,
     const std::vector<variable::variable_typenames>& var_type_order,
     size_t nreserved = 1)
      : base(finite_vars, vector_vars, var_type_order) {
      init(nreserved);
    }

    //! Constructs the datasource with the given sequence of variables.
    //! @param info    info from calling datasource_info()
    assignment_dataset(const datasource_info_type& info, size_t nreserved = 1)
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
    using base::var_order_index;
    using base::variable_index;
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
    size_t capacity() const { return data_vector.size(); }

    //! Element access: record i, finite variable j (in the order finite_list())
    //! NOTE: This is slower than record_iterator.
    size_t finite(size_t i, size_t j) const {
      assert(i < nrecords && j < num_finite());
      return safe_get(data_vector[i].finite(), finite_seq[j]);
    }

    //! Element access: record i, vector variable j (in the order finite_list(),
    //! but with n-value vector variables using n indices j)
    //! NOTE: This is slower than record_iterator.
    double vector(size_t i, size_t j) const {
      assert(i < nrecords && j < dvector);
      const std::pair<vector_variable*, size_t>& ipair = vector_i2pair[j];
      const vector_type& v =safe_get(data_vector[i].vector(), ipair.first);
      return v[ipair.second];
    }

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
                    const vector_type& vvals, double w = 1);

    using base::normalize;

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
    using base::finite_vars;
    using base::finite_seq;
    using base::finite_numbering_ptr_;
    using base::dfinite;
    using base::finite_class_vars;
    using base::vector_vars;
    using base::vector_seq;
    using base::vector_numbering_ptr_;
    using base::dvector;
    using base::vector_class_vars;
    using base::var_type_order;
    using base::var_order_map;
    using base::vector_var_order_map;

    // From dataset
    using base::nrecords;
    using base::weighted;
    using base::weights_;

    //! Data
    mutable std::vector<sill::assignment> data_vector;

    //! vector_i2pair[j] = <vector variable, index in value> for value j
    //! in vector record data
    std::vector<std::pair<vector_variable*, size_t> > vector_i2pair;

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

    // Protected functions
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

  };  // class assignment_dataset

  //============================================================================
  // Implementations of methods in assignment_dataset
  //============================================================================

  // Protected methods required by record
  //==========================================================================

  template <typename LA>
  void assignment_dataset<LA>::load_assignment(size_t i, sill::assignment& a) const {
    assert(i < nrecords);
    a = data_vector[i];
  }

  template <typename LA>
  void assignment_dataset<LA>::load_record(size_t i, record_type& r) const {
    if (!r.fin_own) {
      r.fin_own = true;
      r.fin_ptr = new std::vector<size_t>(finite_numbering_ptr_->size());
    }
    if (!r.vec_own) {
      r.vec_own = true;
      r.vec_ptr = new vector_type(dvector);
    }
    convert_finite_assignment2record(data_vector[i].finite(), *(r.fin_ptr));
    convert_vector_assignment2record(data_vector[i].vector(), *(r.vec_ptr));
  }

  template <typename LA>
  void
  assignment_dataset<LA>::load_finite(size_t i,
                                  std::vector<size_t>& findata) const {
    convert_finite_assignment2record(data_vector[i].finite(), findata);
  }

  template <typename LA>
  void assignment_dataset<LA>::load_vector(size_t i, vector_type& vecdata) const {
    convert_vector_assignment2record(data_vector[i].vector(), vecdata);
  }

  template <typename LA>
  void assignment_dataset<LA>::load_assignment_pointer(size_t i, assignment** a) const {
    assert(i < nrecords);
    *a = &(data_vector[i]);
  }

  // Protected functions
  //==========================================================================

  template <typename LA>
  void assignment_dataset<LA>::init(size_t nreserved) {
    assert(nreserved > 0);
    data_vector.resize(nreserved);
    for (size_t j(0); j < vector_seq.size(); ++j)
      for (size_t k(0); k < vector_seq[j]->size(); ++k)
        vector_i2pair.push_back(std::make_pair(vector_seq[j], k));
  }

  // Mutating operations
  //==========================================================================

  template <typename LA>
  void assignment_dataset<LA>::reserve(size_t n) {
    if (n > capacity()) {
      data_vector.resize(n);
      if (weighted)
        weights_.resize(n, true);
    }
  }

  template <typename LA>
  void assignment_dataset<LA>::set_record(size_t i, const assignment& a, double w) {
    assert(i < nrecords);
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
//    data_vector[i] = a;
    if (weighted) {
      weights_[i] = w;
    } else if (w != 1) {
      std::cerr << "assignment_dataset::set_record() called with weight"
                << " w != 1 on an unweighted dataset." << std::endl;
      assert(false);
    }
  }

  template <typename LA>
  void
  assignment_dataset<LA>::set_record(size_t i, const std::vector<size_t>& fvals,
                                 const vector_type& vvals, double w) {
    assert(i < nrecords);
    convert_finite_record2assignment(fvals, data_vector[i].finite());
    convert_vector_record2assignment(vvals, data_vector[i].vector());
    if (weighted) {
      weights_[i] = w;
    } else if (w != 1) {
      std::cerr << "assignment_dataset::set_record() called with weight"
                << " w != 1 on an unweighted dataset." << std::endl;
      assert(false);
    }
  }

  template <typename LA>
  void assignment_dataset<LA>::normalize(const vec& means,
                                     const vec& std_devs) {
    assert(means.size() == dvector);
    assert(std_devs.size() == dvector);
    for (size_t j(0); j < dvector; ++j) {
      const std::pair<vector_variable*, size_t>& ipair = vector_i2pair[j];
      if (std_devs[j] < 0)
        assert(false);
      double std_dev(std_devs[j] == 0 ? 1 : std_devs[j]);
      for (size_t i(0); i < nrecords; ++i)
        data_vector[i].vector()[ipair.first][ipair.second] =
          (data_vector[i].vector()[ipair.first][ipair.second] - means[j])
          / std_dev;
    }
  }

  template <typename LA>
  void assignment_dataset<LA>::normalize(const vec& means, const vec& std_devs,
                                     const vector_var_vector& vars) {
    foreach(vector_variable* v, vars)
      assert(vector_vars.count(v) != 0);
    ivec vars_inds(vector_indices(vars));
    assert(means.size() == vars_inds.size());
    assert(std_devs.size() == vars_inds.size());
    for (size_t j(0); j < vars_inds.size(); ++j) {
      size_t j2(static_cast<size_t>(vars_inds[j]));
      const std::pair<vector_variable*, size_t>& ipair = vector_i2pair[j2];
      if (std_devs[j] < 0)
        assert(false);
      double std_dev(std_devs[j] == 0 ? 1 : std_devs[j]);
      for (size_t i(0); i < nrecords; ++i)
        data_vector[i].vector()[ipair.first][ipair.second] =
          (data_vector[i].vector()[ipair.first][ipair.second] - means[j])
          / std_dev;
    }
  }

  template <typename LA>
  void assignment_dataset<LA>::normalize2(const vector_var_vector& vars) {
    foreach(vector_variable* v, vars)
      assert(vector_vars.count(v) != 0);
    ivec vars_inds(vector_indices(vars));
    for (size_t i(0); i < nrecords; ++i) {
      double normalizer(0);
      foreach(vector_variable* v, vars) {
        const vector_type& tmpval = data_vector[i].vector()[v];
        normalizer += tmpval.inner_prod(tmpval);
      }
      if (normalizer == 0)
        continue;
      normalizer = sqrt(normalizer);
      foreach(vector_variable* v, vars)
        data_vector[i].vector()[v] /= normalizer;
    }
  }

  template <typename LA>
  void assignment_dataset<LA>::randomize(double random_seed) {
    boost::mt11213b rng(static_cast<unsigned>(random_seed));
    assignment tmpa;
    double weight_tmp;
    for (size_t i = 0; i < nrecords-1; ++i) {
      size_t j = (size_t)(boost::uniform_int<int>(i,nrecords-1)(rng));
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

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
