#ifndef SILL_DATASET_HPP
#define SILL_DATASET_HPP

#include <string>
#include <vector>
#include <ctime>

#include <boost/iterator.hpp>
#include <boost/random/uniform_int.hpp>

#include <sill/base/assignment.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/datastructure/dense_table.hpp>
#include <sill/learning/dataset_old/datasource.hpp>
#include <sill/learning/dataset_old/record.hpp>
#include <sill/learning/dataset_old/record_iterator.hpp>
#include <sill/range/concepts.hpp>
#include <sill/range/forward_range.hpp>

#include <sill/macros_def.hpp>

/**
 * \file dataset.hpp This specifies a base class for datasets.
 *
 * Design thoughts:
 *  - Since virtual function calls seem to be cheap, we are using inheritance
 *    to simplify interfaces.  We could switch to static typing at some point
 *    if necessary.
 *  - Note that datasets should not be shrunk since doing so might corrupt
 *    views.
 *  - It might be worthwhile to put mutation into the record class, e.g.:
 *    - record::operator=(const assignment& a)
 *    - record::set(variable_h v, size_t value)
 *    - record::set(variable_h v, const V& v)
 *
 * Concept hierarchy:
 *  - Dataset (without mutating operations)
 *    - MutableDataset
 * Class hierarchy:
 *  - dataset (Dataset)
 *    - dataset_view (Dataset) (Stores a view of a const dataset&)
 *    - vector_dataset_old (MutableDataset)
 *    - assignment_dataset (MutableDataset)
 */

namespace sill {

  // Forward declarations
  template <typename LA> class ds_oracle;
  template <typename LA> class record_iterator;

  /**
   * A base class for datasets.
   * This supports weighted datasets, but dataset types inheriting from this
   * are unweighted by default (constant weight 1).
   *
   * \author Joseph Bradley, Stanislav Funiak
   * \ingroup learning_dataset
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  class dataset : public datasource {

    // Public type declarations
    //==========================================================================
  public:

    //! Base type (datasource)
    typedef datasource base;

    typedef LA la_type;

    typedef record<LA>        record_type;
    typedef finite_record_old     finite_record_old_type;
    typedef vector_record_old<LA> vector_record_old_type;
    typedef record_iterator<LA> record_iterator_type;

    typedef typename la_type::matrix_type matrix_type;
    typedef typename la_type::vector_type vector_type;
    typedef typename la_type::value_type  value_type;

    /**
     * An iterator over the records of a dataset (in assignment format).
     * NOTE: This is to make datasets more efficient for iterating
     *  over assignments.  It is not analogous to record_iterator;
     *  at some point, we should consider reorganizing one or both.
     */
    class assignment_iterator
      : public std::iterator<std::forward_iterator_tag, const assignment> {

      //! associated dataset
      const dataset* data;

      //! current index into the dataset's records
      size_t i;

    protected:
      friend class dataset;
      friend class ds_oracle<LA>;

      //! True if this iterator owns its data.
      bool own;

      //! Assignment used if this iterator does not own its data.
      mutable assignment* a;

      //! Constructs an iterator pointed to record i.
      //! @param own  If true, then this will own its data.
      assignment_iterator(const dataset* data, size_t i, bool own)
        : data(data), i(i), own(own), a(NULL) {
        if (own)
          a = new assignment();
        if (i < data->size()) {
          if (own)
            data->load_assignment(i, *a);
          else
            data->load_assignment_pointer(i, &a);
        }
      }

    public:

      //! Copy constructor.
      assignment_iterator(const assignment_iterator& a_it)
        : data(a_it.data), i(a_it.i) {
        if (a_it.own)
          a = new assignment(*(a_it.a));
        else
          a = a_it.a;
        own = a_it.own;
      }

      ~assignment_iterator() {
        if (own) {
          assert(a != NULL);
          delete(a);
        }
      }

      assignment_iterator& operator=(const assignment_iterator& a_it) {
        if (a_it.own) {
          if (own)
            (*a) = *(a_it.a);
          else
            a = new assignment(*(a_it.a));
        } else {
          if (own)
            delete(a);
          a = a_it.a;
        }
        own = a_it.own;
        data = a_it.data;
        i = a_it.i;
        return *this;
      }

      const assignment& operator*() const {
        return *a;
      }

      const assignment* const operator->() const {
        return a;
      }

      assignment_iterator& operator++() {
        if (i+1 < data->size()) {
          ++i;
          if (own)
            data->load_assignment(i, *a);
          else
            data->load_assignment_pointer(i, &a);
        } else
          i = data->size();
        return *this;
      }

      assignment_iterator operator++(int) {
        assignment_iterator copy(*this);
        if (i+1 < data->size()) {
          ++i;
          if (own)
            data->load_assignment(i, *a);
          else
            data->load_assignment_pointer(i, &a);
        } else
          i = data->size();
        return copy;
      }

      bool operator==(const assignment_iterator& it) const {
        return i == it.i;
      }

      bool operator!=(const assignment_iterator& it) const {
        return i != it.i;
      }

      //! Returns the weight of the current example, or 0 if the iterator
      //! does not point to an example.
      //! @todo Make this safer!
      value_type weight() const {
        return data->weight(i);
      }

      //! Resets this assignment iterator to the first record.
      void reset() {
        i = 0;
        if (i < data->size()) {
          if (own)
            data->load_assignment(i, *a);
          else
            data->load_assignment_pointer(i, &a);
        }
      }

    }; // class assignment_iterator

    // Constructors
    //==========================================================================
  public:

    //! Constructs an empty dataset
    dataset()
      : base(), nrecords(0), weighted(false) { }

    //! Constructs the dataset with the given sequence of variables
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    dataset(const finite_var_vector& finite_vars,
            const vector_var_vector& vector_vars,
            const std::vector<variable::variable_typenames>& var_type_order)
      : base(finite_vars, vector_vars, var_type_order), nrecords(0),
        weighted(false) {
    }

    //! Constructs the dataset with the given sequence of variables
    //! @param finite_vars     finite variables in data
    //! @param vector_vars     vector variables in data
    //! @param var_type_order  Order of variable types in datasource's
    //!                        natural order
    dataset(const forward_range<finite_variable*>& finite_vars,
            const forward_range<vector_variable*>& vector_vars,
            const std::vector<variable::variable_typenames>& var_type_order)
      : base(finite_vars, vector_vars, var_type_order), nrecords(0),
        weighted(false) {
    }

    //! Constructs the datasource with the given sequence of variables.
    //! @param info    info from calling datasource_info()
    explicit dataset(const datasource_info_type& info)
      : base(info), nrecords(0), weighted(false) {
    }

    virtual ~dataset() { }

    virtual void save(oarchive& a) const;

    virtual void load(iarchive& a);

    //! This method is like a constructor but is virtualized.
    //! @param info    info from calling datasource_info()
    //! @todo This method should really be a pure virtual function.
    virtual void reset(const datasource_info_type& info) {
      base::reset(info);
      nrecords = 0;
      weighted = false;
    }

    // Getters and queries
    //==========================================================================

    //! True iff no records in dataset.
    bool empty() const {
      return nrecords == 0;
    }

    //! Returns the number of records in the dataset.
    size_t size() const {
      return nrecords;
    }

    //! Returns the current allocated capacitiy of the dataset.
    virtual size_t capacity() const = 0;

    //! Element access which does not do range checking
    //! Note: use records() for more efficient record access.
    //! @todo Make this more efficient.
    record_type operator[](size_t i) const;

    //! Load datapoint i into assignment a
    virtual void load_assignment(size_t i, sill::assignment& a) const = 0;

    //! Element access which does range checking
    //! Note: use records() for more efficient record access.
    record_type at(size_t i) const;

    //! Element access which does range checking.
    assignment at_assignment(size_t i) const;

    //! Element access: record i, finite variable j (in the order finite_list())
    //! Note: Full record retrievals are more efficient than this function.
    virtual size_t finite(size_t i, size_t j) const = 0;

    //! Element access: record i, vector variable j (in the order finite_list(),
    //! but with n-value vector variables using n indices j)
    //! Note: Full record retrievals are more efficient than this function.
    virtual value_type vector(size_t i, size_t j) const = 0;

    //! Element access: record i, finite variable v
    virtual size_t finite(size_t i, finite_variable* v) const;

    /*
    //! Element access: record i, vector variable v.
    virtual vector_type vector(size_t i, vector_variable* v) const;
    */

    //! Element access: record i, vector variable v, element j.
    virtual value_type vector(size_t i, vector_variable* v, size_t j) const;

    //! Returns an element, sampled uniformly at random
    //! \todo Fix this to use record weights.
    template <typename Engine>
    record_type sample(Engine& engine) const {
      boost::uniform_int<> unif(0, size() - 1);
      return operator[](unif(engine));
    }

    //! Returns a range over the records of this dataset
    //! Eventually, will be able to provide a set of variables
    std::pair<record_iterator_type, record_iterator_type> records() const {
      return std::make_pair(begin(), end());
    }

    //! Returns an iterator over the records of this dataset
    virtual record_iterator_type begin() const {
      return record_iterator_type(this, 0);
    }

    //! Returns an end iterator over the records of this dataset
    virtual record_iterator_type end() const {
      return record_iterator_type(this, nrecords);
    }

    //! Returns a range over the records of this dataset, as assignments.
    //! Eventually, will be able to provide a set of variables.
    //! Note: This only needs to be re-implemented by datasets which
    //!       use assignments as their native types.
    virtual std::pair<assignment_iterator, assignment_iterator>
    assignments() const {
      return std::make_pair(make_assignment_iterator(0, true),
                            make_assignment_iterator(nrecords, true));
    }

    //! Returns an iterator over the records of this dataset, as assignments.
    //! Note: This only needs to be re-implemented by datasets which
    //!       use assignments as their native types.
    virtual assignment_iterator begin_assignments() const {
      return make_assignment_iterator(0, true);
    }

    //! Returns an end iterator over the records of this dataset, as assignments
    //! Note: This only needs to be re-implemented by datasets which
    //!       use assignments as their native types.
    virtual assignment_iterator end_assignments() const {
      return make_assignment_iterator(nrecords, true);
    }

    //! Returns true iff this dataset's records are weighted.
    bool is_weighted() const {
      return weighted;
    }

    //! Returns the weight of record i (without bound checking)
    value_type weight(size_t i) const;

    //! Returns the weight of record i (with bound checking)
    value_type weight_at(size_t i) const;

    //! Returns the vector of weights (or an empty vector if the dataset is
    //! not weighted).
    const vec& weights() const {
      return weights_;
    }

    /**
     * Sets the given (not necessarily pre-allocated) matrix to hold
     * the given set of values for all records.
     * @param X         Set to be a (nrecords x variable vector size) matrix.
     * @param add_ones  If this parameter is set to true, then a constant
     *                  ones vector is added, so X is set to be a
     *                  (nrecords x variable vector size + 1) matrix.
     *                   (default = false)
     */
    void get_value_matrix(matrix_type& X, const vector_var_vector& vars,
                          bool add_ones = false) const;

    /**
     * Compute unnormalized training data log likelihood according to model.
     *
     * @param model  model over dataset's variables
     * @param base   base of the log, default e
     * \todo Fix this to use record weights.
     */
    template <typename D>
    value_type log_likelihood(const D& model, value_type base = exp(1.)) const {
      // concept_assert((Distribution<D>));
      assert(nrecords > 0);
      value_type loglike = 0;
      for(size_t i = 0; i < nrecords; i++)
        loglike += model.log_likelihood(operator[](i), base);
      return loglike;
    }

    /**
     * Returns the expected value of the given functor w.r.t. this dataset.
     * Returns 0 if this dataset is empty.
     *
     * @tparam F   Functor type implementing: double operator()(record type)
     */
    template <typename F>
    value_type expected_value(F f) const {
      value_type sum = 0;
      value_type total_weight = 0;
      if (nrecords == 0)
        return 0;
      size_t i = 0;
      foreach(const record_type& r, records()) {
        sum += weight(i) * f(r);
        total_weight += weight(i);
        ++i;
      }
      sum /= total_weight;
      return sum;
    }

    /**
     * Returns the <expected value, stderr> of the given functor
     * w.r.t. this dataset.
     * Returns <0,0> if this dataset is empty.
     *
     * @tparam F   Functor type implementing: double operator()(record type)
     */
    template <typename F>
    std::pair<value_type, value_type> expected_value_and_stderr(F f) const {
      value_type sum = 0;
      value_type sum2 = 0;
      value_type total_weight = 0;
      if (nrecords == 0)
        return std::make_pair(0,0);
      size_t i = 0;
      foreach(const record_type& r, records()) {
        value_type val = weight(i) * f(r);
        sum += val;
        sum2 += val * val;
        total_weight += weight(i);
        ++i;
      }
      sum /= total_weight;
      sum2 = sqrt((sum2 / total_weight) - (sum * sum)) / sqrt(total_weight);
      return std::make_pair(sum, sum2);
    }

    /**
     * Compute the empirical mean of a set X of vector variables.
     *
     * @param  mu   (Return value.) Empirical mean.
     */
    void mean(vec& mu, const vector_var_vector& X) const;

    /**
     * Compute the empirical covariance matrix for a set X of vector variables.
     * This computes: (1/(n-1)) \sum_{i=1}^n (x_i - mu)(x_i - mu)^T
     * where mu is the mean of X.
     *
     * @param  cov   (Return value.) Empirical covariance matrix.
     */
    void covariance(matrix_type& cov, const vector_var_vector& X) const;

    /**
     * Computes the empirical mean and covariance for vector variables X.
     * This is more efficient than calling mean(), covariance() separately.
     * 
     * @param  mu   (Return value.) Empirical mean.
     * @param  cov  (Return value.) Empirical covariance matrix.
     */
    void mean_covariance(vec& mu, matrix_type& cov, const vector_var_vector& X) const;

    /**
     * Writes a human-readable representation of the dataset.
     * The 'format' parameter can be one of these formats:
     *  - "default": Print variables, then data in compact, readable format.
     *  - "vars": Print 3 columns: variable name, type, arity.
     *  - "tabbed": Print tab-delimited data, as for Matlab.
     *  - "tabbed_weighted": Print tab-delimited data, as for Matlab, and
     *       include record weights as a last column.
     */
    std::ostream&
    print(std::ostream& out, const std::string& format = "default") const;

    /**
     * Loads data from an input stream.
     * Existing data is erased.
     *
     * The 'format' parameter can be one of these formats:
     *  - "default": Print variables, then data in compact, readable format.
     *     - NOT YET IMPLEMENTED
     *  - "vars": Print 3 columns: variable name, type, arity.
     *     - NOT YET IMPLEMENTED
     *  - "tabbed": Print tab-delimited data, as for Matlab.
     *     - This uses the dataset's current variable layout;
     *       the tabbed data must match this layout.
     *  - "tabbed_weighted": Print tab-delimited data, as for Matlab, and
     *       include record weights as a last column.
     *     - NOT YET IMPLEMENTED
     */
    std::istream&
    load(std::istream& in, const std::string& format);

    // Mutating operations
    //==========================================================================

    //! Increases the capacity in anticipation of adding new elements.
    virtual void reserve(size_t n) = 0;

    //! Adds a new record with weight w (default = 1)
    void insert(const assignment& a, value_type w = 1);

    //! Adds a new record with weight w (default = 1)
    template <typename OtherLA>
    void insert(const record<OtherLA>& r, value_type w = 1) {
      insert(r.finite(), r.vector(), w);
    }

    //! Adds a new record with weight w (default = 1).
    //! This version fails if this dataset has any vector variables.
    void insert(const finite_record_old_type& r, value_type w = 1) {
      assert(num_vector() == 0);
      insert(r.finite(), vector_type(), w);
    }

    //! Adds a new record with weight w (default = 1).
    //! This version fails if this dataset has any finite variables.
    template <typename OtherLA>
    void insert(const vector_record_old<OtherLA>& r, value_type w = 1) {
      assert(num_finite() == 0);
      insert(std::vector<size_t>(), r.vector(), w);
    }

    //! Adds a new record with finite variable values fvals and vector variable
    //! values vvals, with weight w (default = 1).
    void insert(const std::vector<size_t>& fvals, const vector_type& vvals,
                value_type w = 1) {
      if (nrecords == capacity())
        reserve(std::max<size_t>(1, 2*nrecords));
      size_t i = nrecords;
      ++nrecords;
      set_record(i, fvals, vvals, w);
    }

    //! Adds a new record with finite variable values fvals and vector variable
    //! values vvals, with weight w (default = 1).
    template <typename OtherVecType>
    void insert(const std::vector<size_t>& fvals, const OtherVecType& vvals,
                value_type w = 1) {
      insert(fvals, vector_type(vvals), w);
    }

    //! Adds a new record with all values set to 0, with weight w (default = 1).
    void insert_zero_record(value_type w = 1);

    //! Sets record with index i to this value and weight.
    virtual void set_record(size_t i, const assignment& a, value_type w = 1) =0;

    //! Sets record with index i to this value and weight.
    virtual void set_record(size_t i, const std::vector<size_t>& fvals,
                            const vector_type& vvals, value_type w = 1) = 0;

    /**
     * Normalizes the vector data so that the empirical mean and variance
     * are 0 and 1, respectively.
     * This takes record weights into account.
     * @return pair<means, std_devs>
     */
    virtual std::pair<vec, vec> normalize();

    //! Normalizes the vector data using the given means and std_devs
    //!  (which are assumed to be correct).
    virtual void normalize(const vec& means, const vec& std_devs);

    /**
     * Normalizes the vector data so that the empirical mean and variance
     * are 0 and 1, respectively.
     * This takes record weights into account.
     * @param vars  Only apply normalization to these variables.
     * @return pair<means, std_devs>
     */
    virtual std::pair<vec, vec> normalize(const vector_var_vector& vars);

    //! Normalizes the vector data using the given means and std_devs
    //!  (which are assumed to be correct).
    //! @param vars  Only apply normalization to these variables.
    virtual void normalize(const vec& means, const vec& std_devs,
                           const vector_var_vector& vars) = 0;

    //! Normalizes the vector data so that each record's vector values lie
    //! on the unit sphere.
    virtual void normalize2();

    //! Normalizes the vector data so that each record's vector values lie
    //! on the unit sphere.
    //! @param vars  Only apply normalization to these variables.
    virtual void normalize2(const vector_var_vector& vars) = 0;

    //! Randomly reorders the dataset view (this is a mutable operation)
    //! using time as the random seed
    void randomize();

    //! Randomly reorders the dataset view (this is a mutable operation)
    virtual void randomize(value_type random_seed) = 0;

    //! Makes this dataset weighted with all weights set to w (default 1).
    //! This may only be called on an unweighted dataset.
    void make_weighted(value_type w = 1);

    //! Set all weights.
    //! This may be called on an unweighted dataset.
    void set_weights(const vec& weights_);

    //! Set a single weight of record i (with bound checking).
    //! This may only be called if the dataset is already weighted.
    void set_weight(size_t i, value_type weight_);

    //! Clears the dataset of all records.
    //! NOTE: This should not be called if views of the data exist!
    void clear() {
      nrecords = 0;
    }

    // Protected data members
    //==========================================================================
  protected:

    friend class record_iterator<LA>;
    friend class assignment_iterator;

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

    //! Number of data points in this dataset.
    size_t nrecords;

    //! Indicates if this is a weighted dataset.
    bool weighted;

    //! Holds the weights for the dataset if it is weighted.
    vec weights_;

    // Protected functions
    //==========================================================================

    //! Constructs an iterator which owns its data pointed to record i
    record_iterator_type make_record_iterator(size_t i) const {
      return record_iterator_type(this, i);
    }

    //! Constructs an iterator which does not own its data pointed to record i
    //! @param fin_ptr  pointer to data
    //! @param vec_ptr  pointer to data
    record_iterator_type make_record_iterator
    (size_t i, std::vector<size_t>* fin_ptr, vector_type* vec_ptr) const {
      return record_iterator_type(this, i, fin_ptr, vec_ptr);
    }

    //! Constructs an assignment iterator pointed to record i.
    //! @param own  Passed to assignment_iterator constructor.
    assignment_iterator make_assignment_iterator(size_t i, bool own) const {
      return assignment_iterator(this, i, own);
    }

    // Do NOT use these methods unless you know what you are doing!
    //==========================================================================
  public:

    //! Load record i into r.
    //! Record r is assumed to be the correct size for this dataset.
    void load_record(size_t i, record_type& r) const {
      load_finite_record_old(i, r);
      load_vector_record_old(i, r);
    }

    //! Load finite record i into r.
    //! Record r is assumed to be the correct size for this dataset.
    virtual void load_finite_record_old(size_t i, finite_record_old& r) const = 0;

    //! Load vector record i into r.
    //! Record r is assumed to be the correct size for this dataset.
    virtual void load_vector_record_old(size_t i,vector_record_old<la_type>& r) const=0;

    //! Load finite data for datapoint i into findata.
    //! findata is assumed to be the correct size for this dataset.
    virtual void load_finite(size_t i, std::vector<size_t>& findata) const = 0;

    //! Load vector data for datapoint i into vecdata.
    //! vecdata is assumed to be the correct size for this dataset.
    virtual void load_vector(size_t i, vector_type& vecdata) const = 0;

    //! ONLY for datasets which use assignments as native types:
    //!  Load the pointer to datapoint i into (*a).
    virtual void load_assignment_pointer(size_t i, assignment** a) const = 0;

  }; // class dataset

  // Free functions
  //==========================================================================

  //! Writes a human-readable representation of the dataset.
  template <typename LA>
  std::ostream& operator<<(std::ostream& out, const dataset<LA>& ds) {
    ds.print(out);
    return out;
  }

  //============================================================================
  // Implementations of methods in dataset
  //============================================================================

  // Constructors
  //==========================================================================

  template <typename LA>
  void dataset<LA>::save(oarchive& a) const {
    base::save(a);
    a << nrecords << weighted << weights_;
  }

  template <typename LA>
  void dataset<LA>::load(iarchive& a) {
    base::load(a);
    a >> nrecords >> weighted >> weights_;
  }

  // Getters and queries
  //==========================================================================

  template <typename LA>
  record<LA> dataset<LA>::operator[](size_t i) const {
    record_type r(finite_numbering_ptr_, vector_numbering_ptr_, dvector);
    load_finite(i, *(r.fin_ptr));
    load_vector(i, *(r.vec_ptr));
    return r;
  }

  template <typename LA>
  record<LA> dataset<LA>::at(size_t i) const {
    assert(i < nrecords);
    return operator[](i);
  }

  template <typename LA>
  assignment dataset<LA>::at_assignment(size_t i) const {
    assert(i < nrecords);
    assignment a;
    load_assignment(i, a);
    return a;
  }

  template <typename LA>
  size_t dataset<LA>::finite(size_t i, finite_variable* v) const {
    return finite(i, safe_get(*finite_numbering_ptr_, v));
  }

  /*
    vector_type dataset::vector(size_t i, vector_variable* v) const {
    assert(v);
    vector_type val(v.size());
    for (size_t k(0); k < v.size(); ++k)
    val[k] = vector(i, safe_get(*vector_numbering_ptr_, v) + k);
    return val;
    }
  */

  template <typename LA>
  typename dataset<LA>::value_type
  dataset<LA>::vector(size_t i, vector_variable* v, size_t j) const {
    assert(v && (j < v.size()));
    return vector(i, safe_get(*vector_numbering_ptr_, v) + j);
  }

  template <typename LA>
  typename dataset<LA>::value_type
  dataset<LA>::weight(size_t i) const {
    if (weighted)
      return weights_[i];
    else
      return 1;
  }

  template <typename LA>
  typename dataset<LA>::value_type
  dataset<LA>::weight_at(size_t i) const {
    assert(i < size());
    return weight(i);
  }

  template <typename LA>
  void
  dataset<LA>::get_value_matrix(matrix_type& X, const vector_var_vector& vars,
                                bool add_ones) const {
    foreach(vector_variable* v, vars)
      assert(this->has_variable(v));
    size_t vars_size(vector_size(vars));
    if (add_ones) {
      if (X.n_rows != nrecords || X.n_cols != vars_size + 1)
        X.set_size(nrecords, vars_size + 1);
      X.col(vars_size).fill(1);
    } else {
      if (X.n_rows != nrecords || X.n_cols != vars_size)
        X.set_size(nrecords, vars_size);
    }
    for (size_t i(0); i < nrecords; ++i) {
      size_t l2(0); // index into a row in X
      for (size_t j(0); j < vars.size(); ++j) {
        for (size_t k(0); k < vars[j]->size(); ++k) {
          size_t l(safe_get(*vector_numbering_ptr_, vars[j]) + k);
          X(i, l2) = vector(i, l);
          ++l2;
        }
      }
    }
  }

  template <typename LA>
  void dataset<LA>::mean(vec& mu, const vector_var_vector& X) const {
    size_t Xsize(0);
    foreach(vector_variable* v, X) {
      if (!has_variable(v))
        throw std::invalid_argument
          ("dataset::covariance() given variable not in dataset");
      Xsize += v.size();
    }
    if (mu.size() != Xsize)
      mu.zeros(Xsize);
    if (nrecords == 0)
      return;
    foreach(const record_type& r, records()) {
      size_t l(0);
      for (size_t j(0); j < X.size(); ++j) {
        size_t ind(safe_get(*vector_numbering_ptr_, X[j]));
        for (size_t k(0); k < X[j]->size(); ++k) {
          mu[l] += r.vector(ind + k);
          ++l;
        }
      }
    }
    mu /= nrecords;
  }

  template <typename LA>
  void dataset<LA>::covariance(matrix_type& cov, const vector_var_vector& X) const {
    vec mu;
    mean_covariance(mu, cov, X);
  }

  template <typename LA>
  void dataset<LA>::mean_covariance(vec& mu, matrix_type& cov,
                                const vector_var_vector& X) const {
    size_t Xsize(0);
    foreach(vector_variable* v, X) {
      if (!has_variable(v))
        throw std::invalid_argument
          ("dataset::covariance() given variable not in dataset");
      Xsize += v.size();
    }
    cov.zeros(Xsize, Xsize);
    if (nrecords <= 1)
      return;
    mean(mu, X);
    vec tmpvec(zeros<vec>(Xsize));
    foreach(const record_type& r, records()) {
      r.vector_values(tmpvec, X);
      tmpvec -= mu;
      cov += outer_product(tmpvec, tmpvec);
    }
    cov /= (nrecords - 1);
  }

  template <typename LA>
  std::ostream&
  dataset<LA>::print(std::ostream& out, const std::string& format) const {
    if (format == "default") {
      out << "Data set (";
      out << finite_seq << " "
          << vector_seq << ")" << std::endl;
      for(size_t i = 0; i < nrecords; i++) {
        record_type r(operator[](i));
        foreach(size_t f, r.finite())
          out << f << " ";
        out << "| ";
        foreach(value_type v, r.vector())
          out << v << " ";
        out << std::endl;
      }
    } else if (format == "vars") {
      foreach(finite_variable* v, finite_seq)
        out << v.name() << "\t" << v.type() << "\t"
            << v.size() << "\n";
      foreach(vector_variable* v, vector_seq)
        out << v.name() << "\t" << v.type() << "\t"
            << v.size() << "\n";
    } else if (format == "tabbed") {
      foreach(const record_type& r, records()) {
        foreach(size_t f, r.finite())
          out << f << "\t";
        foreach(value_type v, r.vector())
          out << v << "\t";
        out << "\n";
      }
    } else if (format == "tabbed_weighted") {
      size_t i(0);
      foreach(const record_type& r, records()) {
        foreach(size_t f, r.finite())
          out << f << "\t";
        foreach(value_type v, r.vector())
          out << v << "\t";
        out << weight(i) << "\n";
      }
    } else {
      throw std::invalid_argument
        ("dataset::print() given invalid format parameter: " + format);
    }
    return out;
  } // print

  template <typename LA>
  std::istream&
  dataset<LA>::load(std::istream& in, const std::string& format) {
    if (format == "default") {
      throw std::runtime_error("dataset::load NOT YET FULLY IMPLEMENTED.");
    } else if (format == "vars") {
      throw std::runtime_error("dataset::load NOT YET FULLY IMPLEMENTED.");
    } else if (format == "tabbed") {
      clear();
      std::string line;
      std::istringstream is;
      value_type d;
      size_t s;
      std::vector<size_t> fvals(num_finite());
      vector_type vvals(vector_dim());
      while (in.good()) {
        getline(in, line);
        if (line.size() == 0)
          continue;
        is.clear();
        is.str(line);
        size_t f_i = 0;
        size_t v_i = 0;
        for (size_t i = 0; i < num_variables(); ++i) {
          bool bad_parse = false;
          switch (var_type_order[i]) {
          case variable::FINITE_VARIABLE:
            if (!(is >> s) ||
                s >= finite_seq[f_i]->size()) {
              bad_parse = true;
              break;
            }
            fvals[f_i] = s;
            ++f_i;
            break;
          case variable::VECTOR_VARIABLE:
            for (size_t j = 0; j < vector_seq[v_i]->size(); ++j) {
              if (!(is >> d)) {
                bad_parse = true;
                break;
              }
              vvals[v_i + j] = d;
            }
            ++v_i;
            break;
          default:
            assert(false);
          }
          if (bad_parse) {
            throw std::runtime_error("dataset::load (tabbed) had bad parse!");
          }
        }
        if (f_i == 0 && v_i == 0)
          continue;
        if (f_i != num_finite() || v_i != num_vector()) {
          throw std::runtime_error
            ("dataset::load (tabbed) had bad parse (incomplete record)!");
        }
        insert(fvals, vvals);
      }
    } else if (format == "tabbed_weighted") {
      throw std::runtime_error("dataset::load NOT YET FULLY IMPLEMENTED.");
    } else {
      throw std::invalid_argument
        ("dataset::load() given invalid format parameter: " + format);
    }
    return in;
  } // load

  // Mutating operations
  //==========================================================================

  template <typename LA>
  void dataset<LA>::insert(const assignment& a, value_type w) {
    if (nrecords == capacity())
      reserve(std::max<size_t>(1, 2*nrecords));
    size_t i(nrecords);
    ++nrecords;
    set_record(i, a, w);
  }

  template <typename LA>
  void dataset<LA>::insert_zero_record(value_type w) {
    if (nrecords == capacity())
      reserve(std::max<size_t>(1, 2*nrecords));
    size_t i(nrecords);
    ++nrecords;
    set_record
      (i, std::vector<size_t>(num_finite(),0), zeros<vec>(vector_dim()), w);
  }

  template <typename LA>
  std::pair<vec, vec> dataset<LA>::normalize() {
    vec means(zeros<vec>(dvector));
    vec std_devs(zeros<vec>(dvector));
    value_type total_ds_weight = 0;
    for (size_t i = 0; i < nrecords; ++i) {
      for (size_t j = 0; j < dvector; ++j) {
        means[j] += weight(i) * vector(i,j);
        std_devs[j] += weight(i) * vector(i,j) * vector(i,j);
      }
      total_ds_weight += weight(i);
    }
    if (total_ds_weight == 0) {
      means.zeros();
      std_devs.zeros();
      return std::make_pair(means, std_devs);
    }
    means /= total_ds_weight;
    std_devs /= total_ds_weight;
    std_devs = sqrt(std_devs - means % means);
    normalize(means, std_devs);
    return std::make_pair(means, std_devs);
  }

  template <typename LA>
  void dataset<LA>::normalize(const vec& means, const vec& std_devs) {
    normalize(means, std_devs, vector_seq);
  }

  template <typename LA>
  std::pair<vec, vec> dataset<LA>::normalize(const vector_var_vector& vars) {
    foreach(vector_variable* v, vars)
      assert(this->has_variable(v));
    vec means(zeros<vec>(vector_size(vars)));
    vec std_devs(zeros<vec>(vector_size(vars)));
    value_type total_ds_weight = 0;
    uvec vars_inds(vector_indices(vars));
    for (size_t i = 0; i < nrecords; ++i) {
      for (size_t j = 0; j < vars_inds.size(); ++j) {
        size_t j2(vars_inds[j]);
        means[j] += weight(i) * vector(i,j2);
        std_devs[j] += weight(i) * vector(i,j2) * vector(i,j2);
      }
      total_ds_weight += weight(i);
    }
    if (total_ds_weight == 0) {
      means.zeros();
      std_devs.zeros();
      return std::make_pair(means, std_devs);
    }
    means /= total_ds_weight;
    std_devs /= total_ds_weight;
    std_devs = sqrt(std_devs - means % means);
    normalize(means, std_devs, vars);
    return std::make_pair(means, std_devs);
  }

  template <typename LA>
  void dataset<LA>::normalize2() {
    normalize2(vector_seq);
  }

  template <typename LA>
  void dataset<LA>::randomize() {
    std::time_t time_tmp;
    time(&time_tmp);
    randomize(time_tmp);
  }

  template <typename LA>
  void dataset<LA>::make_weighted(value_type w) {
    assert(weighted == false);
    weights_.set_size(capacity());
    for (size_t i = 0; i < nrecords; ++i)
      weights_[i] = w;
    weighted = true;
  }

  template <typename LA>
  void dataset<LA>::set_weights(const vec& weights_) {
    assert(weights_.size() == size());
    weighted = true;
    this->weights_ = weights_;
  }

  template <typename LA>
  void dataset<LA>::set_weight(size_t i, value_type weight_) {
    assert(weighted == true);
    assert(i < size());
    weights_[i] = weight_;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_DATASET_HPP
