
#ifndef PRL_DATASET_HPP
#define PRL_DATASET_HPP

#include <string>
#include <vector>
#include <ctime>

#include <boost/iterator.hpp>
#include <boost/random/uniform_int.hpp>

#include <prl/base/assignment.hpp>
#include <prl/base/stl_util.hpp>
#include <prl/datastructure/dense_table.hpp>
#include <prl/learning/dataset/datasource.hpp>
#include <prl/learning/dataset/record.hpp>
#include <prl/math/matrix.hpp>
#include <prl/range/algorithm.hpp>
#include <prl/range/concepts.hpp>
#include <prl/range/forward_range.hpp>

#include <prl/macros_def.hpp>

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
 *    - vector_dataset (MutableDataset)
 *    - assignment_dataset (MutableDataset)
 */

namespace prl {

  /**
   * A base class for datasets.
   * This supports weighted datasets, but dataset types inheriting from this
   * are unweighted by default (constant weight 1).
   *
   * \author Joseph Bradley, Stanislav Funiak
   * \ingroup learning_dataset
   * \todo serialization
   */
  class dataset : public datasource {

    friend class dataset_view;

    // Public type declarations
    //==========================================================================
  public:

    //! Base type (datasource)
    typedef datasource base;

    /**
     * An iterator over the records of a dataset (in record format).
     */
    class record_iterator
      : public std::iterator<std::forward_iterator_tag, const record> {

      //! indicates if the current record is valid
      mutable bool r_valid;

      //! associated dataset
      const dataset* data;

      //! current index into the dataset's records
      size_t i;

    protected:
      friend class dataset;
      friend class ds_oracle;

      mutable record r;

      //! Constructs an iterator which owns its data pointed to record i
      record_iterator(const dataset* data, size_t i)
        : r_valid(false), data(data), i(i),
          r(data->finite_numbering_ptr(), data->vector_numbering_ptr(),
            data->vector_dim()) {
      }

      //! Constructs an iterator which does not own its data pointed to record i
      //! @param fin_ptr  pointer to data
      //! @param vec_ptr  pointer to data
      record_iterator(const dataset* data, size_t i,
                      std::vector<size_t>* fin_ptr, vec* vec_ptr)
        : r_valid(false), data(data), i(i),
          r(data->finite_numbering_ptr(), data->vector_numbering_ptr(),
            fin_ptr, vec_ptr) {
      }

      //! Loads the current record if necessary
      void load_cur_record() const {
        if (!r_valid) {
          assert(data);
          assert(i < data->size());
          data->load_record(i, r);
          r_valid = true;
        }
      }

    public:

      //! Constructs an iterator which acts as an end iterator.
      record_iterator() : r_valid(false), data(NULL), i(0) { }

      //! Copy constructor.
      record_iterator(const record_iterator& it)
        : r_valid(it.r_valid), data(it.data), i(it.i) {
        r.finite_numbering_ptr = it.r.finite_numbering_ptr;
        r.vector_numbering_ptr = it.r.vector_numbering_ptr;
        if (it.r.fin_own) {
          r.fin_ptr->operator=(*(it.r.fin_ptr));
        } else {
          r.fin_own = false;
          delete(r.fin_ptr);
          r.fin_ptr = it.r.fin_ptr;
        }
        if (it.r.vec_own) {
          r.vec_ptr->operator=(*(it.r.vec_ptr));
        } else {
          r.vec_own = false;
          delete(r.vec_ptr);
          r.vec_ptr = it.r.vec_ptr;
        }
      }

      //! Assignment operator.
      record_iterator& operator=(const record_iterator& rec_it) {
        r.finite_numbering_ptr = rec_it.r.finite_numbering_ptr;
        r.vector_numbering_ptr = rec_it.r.vector_numbering_ptr;
        if (rec_it.r.fin_own) {
          if (r.fin_own) {
            r.fin_ptr->operator=(*(rec_it.r.fin_ptr));
          } else {
            r.fin_own = true;
            r.fin_ptr = new std::vector<size_t>(*(rec_it.r.fin_ptr));
          }
        } else {
          if (r.fin_own) {
            r.fin_own = false;
            delete(r.fin_ptr);
          }
          r.fin_ptr = rec_it.r.fin_ptr;
        }
        if (rec_it.r.vec_own) {
          if (r.vec_own) {
            r.vec_ptr->operator=(*(rec_it.r.vec_ptr));
          } else {
            r.vec_own = true;
            r.vec_ptr = new vec(*(rec_it.r.vec_ptr));
          }
        } else {
          if (r.vec_own) {
            r.vec_own = false;
            delete(r.vec_ptr);
          }
          r.vec_ptr = rec_it.r.vec_ptr;
        }
        r_valid = rec_it.r_valid;
        data = rec_it.data;
        i = rec_it.i;
        return *this;
      }

      const record& operator*() const {
        load_cur_record();
        return r;
      }

      const record* const operator->() const {
        load_cur_record();
        return &r;
      }

      record_iterator& operator++() {
        if (data) {
          ++i;
          r_valid = false;
        }
        return *this;
      }

      record_iterator operator++(int) {
        record_iterator copy(*this);
        if (data) {
          ++i;
          r_valid = false;
        }
        return copy;
      }

      bool operator==(const record_iterator& it) const {
        if (data) {
          if (it.data)
            return i == it.i;
          else
            return i == data->size();
        } else {
          if (it.data)
            return it.i == data->size();
          else
            return true;
        }
      }

      bool operator!=(const record_iterator& it) const {
        return i != it.i;
      }

      //! Returns the weight of the current example, or 0 if the iterator
      //! does not point to an example.
      //! @todo Make this safer!
      double weight() const {
        assert(data);
        return data->weight(i);
      }

      //! Resets this record iterator to the first record.
      void reset() {
        i = 0;
        r_valid = false;
      }

      //! Resets this record iterator to record j;
      //! This permits more efficient access to datasets which use records
      //! as native types (than using operator[]).
      void reset(size_t j) {
        assert(data);
        assert(j < data->size());
        i = j;
        r_valid = false;
      }

    }; // class record_iterator

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
      friend class ds_oracle;

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
      double weight() const {
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

    // Protected data members
    //==========================================================================
  protected:

    friend class record_iterator;
    friend class assignment_iterator;

    //! Number of data points in this dataset.
    size_t nrecords;

    //! Indicates if this is a weighted dataset.
    bool weighted;

    //! Holds the weights for the dataset if it is weighted.
    vec weights_;

    // Protected functions
    //==========================================================================

    //! Constructs an iterator which owns its data pointed to record i
    record_iterator make_record_iterator(size_t i) const {
      return record_iterator(this, i);
    }

    //! Constructs an iterator which does not own its data pointed to record i
    //! @param fin_ptr  pointer to data
    //! @param vec_ptr  pointer to data
    record_iterator make_record_iterator
    (size_t i, std::vector<size_t>* fin_ptr, vec* vec_ptr) const {
      return record_iterator(this, i, fin_ptr, vec_ptr);
    }

    //! Constructs an assignment iterator pointed to record i.
    //! @param own  Passed to assignment_iterator constructor.
    assignment_iterator make_assignment_iterator(size_t i, bool own) const {
      return assignment_iterator(this, i, own);
    }

    //! Load record i into r.
    //! Record r is assumed to be the correct size for this dataset.
    virtual void load_record(size_t i, record& r) const = 0;

    //! Load finite data for datapoint i into findata.
    //! findata is assumed to be the correct size for this dataset.
    virtual void load_finite(size_t i, std::vector<size_t>& findata) const = 0;

    //! Load vector data for datapoint i into vecdata.
    //! vecdata is assumed to be the correct size for this dataset.
    virtual void load_vector(size_t i, vec& vecdata) const = 0;

    //! ONLY for datasets which use assignments as native types:
    //!  Load the pointer to datapoint i into (*a).
    virtual void load_assignment_pointer(size_t i, assignment** a) const = 0;

    // Constructors
    //==========================================================================
  public:

    //! Constructs an empty dataset
    dataset() : base(), weighted(false) { }

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
    record operator[](size_t i) const;

    //! Load datapoint i into assignment a
    virtual void load_assignment(size_t i, prl::assignment& a) const = 0;

    //! Element access which does range checking
    //! Note: use records() for more efficient record access.
    record at(size_t i) const;

    //! Element access which does range checking.
    assignment at_assignment(size_t i) const;

    //! Element access: record i, finite variable j (in the order finite_list())
    //! Note: Full record retrievals are more efficient than this function.
    virtual size_t finite(size_t i, size_t j) const = 0;

    //! Element access: record i, vector variable j (in the order finite_list(),
    //! but with n-value vector variables using n indices j)
    //! Note: Full record retrievals are more efficient than this function.
    virtual double vector(size_t i, size_t j) const = 0;

    //! Element access: record i, finite variable v
    virtual size_t finite(size_t i, finite_variable* v) const;

    /*
    //! Element access: record i, vector variable v.
    virtual vec vector(size_t i, vector_variable* v) const;
    */

    //! Element access: record i, vector variable v, element j.
    virtual double vector(size_t i, vector_variable* v, size_t j) const;

    //! Returns an element, sampled uniformly at random
    //! \todo Fix this to use record weights.
    template <typename Engine>
    record sample(Engine& engine) const {
      boost::uniform_int<> unif(0, size() - 1);
      return operator[](unif(engine));
    }

    //! Returns a range over the records of this dataset
    //! Eventually, will be able to provide a set of variables
    virtual std::pair<record_iterator, record_iterator> records() const {
      return std::make_pair(record_iterator(this, 0),
                            record_iterator(this, nrecords));
    }

    //! Returns an iterator over the records of this dataset
    virtual record_iterator begin() const {
      return record_iterator(this, 0);
    }

    //! Returns an end iterator over the records of this dataset
    virtual record_iterator end() const {
      return record_iterator(this, nrecords);
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
    double weight(size_t i) const;

    //! Returns the weight of record i (with bound checking)
    double weight_at(size_t i) const;

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
    void get_value_matrix(mat& X, const vector_var_vector& vars,
                          bool add_ones = false) const;

    /**
     * Compute unnormalized training data log likelihood according to model.
     *
     * @param model  model over dataset's variables
     * @param base   base of the log, default e
     * \todo Fix this to use record weights.
     */
    template <typename D>
    double log_likelihood(const D& model, double base = exp(1.)) const {
      // concept_assert((Distribution<D>));
      assert(nrecords > 0);
      double loglike = 0;
      for(size_t i = 0; i < nrecords; i++)
        loglike += model.log_likelihood(operator[](i), base);
      return loglike;
    }

    /**
     * Compute the expected value and standard error of the given function
     * w.r.t. this dataset.
     *
     * @tparam Function type; this must take a record and return a real number.
     *
     * @todo Replace the above function with this one.
     */
    template <typename F>
    std::pair<double, double> expected_value(F f) const {
      double sum(0);
      double sum2(0);
      double total_weight(0);
      if (nrecords == 0)
        return std::make_pair(0,0);
      size_t i(0);
      foreach(const record& r, records()) {
        double val(weight(i) * f(r));
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
    void covariance(mat& cov, const vector_var_vector& X) const;

    /**
     * Computes the empirical mean and covariance for vector variables X.
     * This is more efficient than calling mean(), covariance() separately.
     * 
     * @param  mu   (Return value.) Empirical mean.
     * @param  cov  (Return value.) Empirical covariance matrix.
     */
    void mean_covariance(vec& mu, mat& cov, const vector_var_vector& X) const;

    /**
     * Writes a human-readable representation of the dataset.
     * The 'format' parameter can be one of these formats:
     *  - "default": Print variables, then data in compact, readable format.
     *  - "vars": Print 3 columns: variable name, type, arity.
     *  - "tabbed": Print tab-delimited data, as for Matlab.
     *  - "tabbed_weighted": Print tab-delimited data, as for Matlab, and
     *       include record weights as a last column.
     */
    template <typename Char, typename Traits>
    std::basic_ostream<Char,Traits>&
    print(std::basic_ostream<Char,Traits>& out,
          const std::string& format = "default") const {
      if (format == "default") {
        out << "Data set (";
        out << finite_seq << " "
            << vector_seq << ")" << std::endl;
        for(size_t i = 0; i < nrecords; i++) {
          record r(operator[](i));
          foreach(size_t f, r.finite())
            out << f << " ";
          out << "| ";
          foreach(double v, r.vector())
            out << v << " ";
          out << std::endl;
        }
      } else if (format == "vars") {
        foreach(finite_variable* v, finite_seq)
          out << v->name() << "\t" << v->get_variable_type() << "\t"
              << v->size() << "\n";
        foreach(vector_variable* v, vector_seq)
          out << v->name() << "\t" << v->get_variable_type() << "\t"
              << v->size() << "\n";
      } else if (format == "tabbed") {
        foreach(const record& r, records()) {
          foreach(size_t f, r.finite())
            out << f << "\t";
          foreach(double v, r.vector())
            out << v << "\t";
          out << "\n";
        }
      } else if (format == "tabbed_weighted") {
        size_t i(0);
        foreach(const record& r, records()) {
          foreach(size_t f, r.finite())
            out << f << "\t";
          foreach(double v, r.vector())
            out << v << "\t";
          out << weight(i) << "\n";
        }
      } else {
        throw std::invalid_argument
          ("dataset::print() given invalid format parameter: " + format);
      }
      return out;
    }

    // Mutating operations
    //==========================================================================

    //! Increases the capacity in anticipation of adding new elements.
    virtual void reserve(size_t n) = 0;

    //! Adds a new record with weight w (default = 1)
    void insert(const assignment& a, double w = 1);

    //! Adds a new record with weight w (default = 1)
    void insert(const record& r, double w = 1) {
      insert(r.finite(), r.vector(), w);
    }

    //! Adds a new record with finite variable values fvals and vector variable
    //! values vvals, with weight w (default = 1).
    void insert(const std::vector<size_t>& fvals, const vec& vvals,
                double w = 1);

    //! Adds a new record with all values set to 0, with weight w (default = 1).
    void insert_zero_record(double w = 1);

    //! Sets record with index i to this value and weight.
    virtual void set_record(size_t i, const assignment& a, double w = 1) = 0;

    //! Sets record with index i to this value and weight.
    virtual void set_record(size_t i, const std::vector<size_t>& fvals,
                            const vec& vvals, double w = 1) = 0;

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
    virtual void randomize(double random_seed) = 0;

    //! Makes this dataset weighted with all weights set to w (default 1).
    //! This may only be called on an unweighted dataset.
    void make_weighted(double w = 1);

    //! Set all weights.
    //! This may be called on an unweighted dataset.
    void set_weights(const vec& weights_);

    //! Set a single weight of record i (with bound checking).
    //! This may only be called if the dataset is already weighted.
    void set_weight(size_t i, double weight_);

  }; // class dataset

  // Free functions
  //==========================================================================

  //! Writes a human-readable representation of the dataset.
  std::ostream& operator<<(std::ostream& out, const dataset& data);

} // namespace prl

#include <prl/macros_undef.hpp>

#endif
