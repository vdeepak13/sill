#ifndef SILL_VECTOR_DATASET_HPP
#define SILL_VECTOR_DATASET_HPP

#include <sill/base/vector_assignment.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/learning/dataset2/basic_record_iterators.hpp>
#include <sill/learning/dataset2/basic_sample_iterator.hpp>
#include <sill/learning/dataset2/vector_record.hpp>
#include <sill/math/permutations.hpp>

#include <boost/shared_ptr.hpp>

#include <stdexcept>
#include <vector>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A dataset that stores observations only for vector variables.
   * Models VectorDataset and InsertableDataset.
   *
   * \tparam T the internal storage of the vector values. This should match the
   *         storage type of the learned factors.
   */
  template <typename T = double> 
  class vector_dataset {
  public:
    typedef T value_type;

    typedef vector_variable   variable_type;
    typedef vector_domain     domain_type;
    typedef vector_var_vector var_vector_type;
    typedef vector_assignment assignment_type;
    typedef vector_record2<T>  record_type;

    typedef basic_record_iterator<vector_dataset>       record_iterator;
    typedef basic_const_record_iterator<vector_dataset> const_record_iterator;
    typedef basic_sample_iterator<vector_dataset>       sample_iterator;

    //! Creates an uninitialized dataset
    vector_dataset() { }

    //! Initializes the dataset with the given sequence of variables.
    //! It is an error to call initialize() more than once.
    void initialize(const vector_var_vector& variables) {
      if (table_ptr) {
        throw std::logic_error("Attempt to call initialize() more than once.");
      }
      table_ptr.reset(new table(variables));
    }

    //! Initializes the dataset with the given vector domain.
    //! It is an error to call initialize() more than once.
    void initialize(const vector_domain& variables) {
      initialize(vector_var_vector(variables.begin(), variables.end()));
    }

    //! Returns the number of datapoints in the dataset.
    size_t size() const {
      return ordering.size();
    }

    //! Returns the columns of this dataset.
    vector_domain arguments() const {
      check_initialized();
      return make_domain(table_ptr->variables);
    }

    //! Returns mutable records for the specified vector variables.
    std::pair<record_iterator, record_iterator>
    records(const vector_var_vector& vars) {
      check_initialized();
      return std::make_pair(record_iterator(this, get_indices(vars)),
                            record_iterator(this));
    }

    //! Returns immutable records for the specified vector variables.
    std::pair<const_record_iterator, const_record_iterator>
    records(const vector_var_vector& vars) const {
      check_initialized();
      return std::make_pair(const_record_iterator(this, get_indices(vars)),
                            const_record_iterator(this));
    }

    //! Returns an iterator that generates samples drawn from this dataset
    sample_iterator
    samples(const vector_var_vector& vars, unsigned seed = 0) const {
      check_initialized();
      return sample_iterator(this, get_indices(vars), seed);
    }

    //! Returns a single sample drawn from this dataset
    template <typename RandomNumberGenerator>
    vector_record2<T>
    sample(const vector_var_vector& vars, RandomNumberGenerator& rng) const {
      return *samples(vars, rng());
      // TODO: possibly move the load function to the dataset
    }

    //! Returns a view of the dataset for a range of the rows.
    vector_dataset subset(size_t begin, size_t end) const {
      check_initialized();
      assert(begin <= size());
      assert(end <= size());
      return vector_dataset(*this, begin, end);
    }

    //! Inserts the values in this dataset's ordering.
    virtual void insert(const vector_record2<T>& r) {
      check_initialized();
      insert(r.values, r.weight); // protected function
    }
 
    //! Inserts a new row from an assignment (all variables must be present).
    virtual void insert(const vector_assignment& a, T weight = 1.0) {
      check_initialized();
      arma::Col<T> values(table_ptr->ncols());
      typedef std::pair<vector_variable*, arma::span> var_span;
      foreach(const var_span& vs, table_ptr->var_span) {
        values(vs.second) = safe_get(a, vs.first);
      }
      insert(values, weight); // protected function
    }

    //! Inserts the given number of rows with unit weights and "undefined" values.
    virtual void insert(size_t nrows) {
      check_initialized();
      for (size_t i = 0; i < nrows; ++i) {
        ordering.push_back(table_ptr->weights.size());
        table_ptr->weights.push_back(1.0);
      }
      table_ptr->data.insert(table_ptr->data.end(),
                             nrows * table_ptr->ncols(),
                             std::numeric_limits<T>::quiet_NaN());
    }

    //! Randomizes the ordering of the records in this dataset.
    template <typename RandomNumberGenerator>
    void shuffle(RandomNumberGenerator& rng) {
      check_initialized();
      permute(ordering, rng);
    }

    // Private data members
    //========================================================================
  private:
    struct table {
      vector_var_vector variables; // the ordering of variables in the table
      std::map<vector_variable*, arma::span> var_span; // indices of each var
      std::vector<T> data;
      std::vector<T> weights;
      size_t ncols_;
      table(const vector_var_vector& variables)
        : variables(variables), ncols_(0) {
        foreach(vector_variable* v, variables) {
          var_span[v] = arma::span(ncols_, ncols_ + v->size() - 1);
          ncols_ += v->size();
        }
      }
      size_t ncols() const { return ncols_; }
    };

    //! the actual data plus some indexing stuff. null until initialization
    boost::shared_ptr<table> table_ptr;

    //! the subset of datapoints that this dataset logically consists of
    std::vector<size_t> ordering;

    // Private member functions
    //========================================================================
  private:
    //! Creates a derived dataset
    vector_dataset(const boost::shared_ptr<table>& table_ptr, 
                   const std::vector<size_t>& ordering)
      : table_ptr(table_ptr), ordering(ordering) { }

    //! Creates a derived dataset
    vector_dataset(const vector_dataset& other, size_t begin, size_t end)
      : table_ptr(other.table_ptr),
        ordering(other.ordering.begin() + begin,
                 other.ordering.begin() + end) { }
 
    //! Throws an exception if the dataset is not initialized
    void check_initialized() const {
      if (!table_ptr) {
        throw std::logic_error("The vector dataset is not initialized!");
      }
    }

    //! Returns the column indices for the specified variables
    std::vector<size_t> get_indices(const vector_var_vector& variables) const {
      std::vector<size_t> indices;
      indices.reserve(vector_size(variables));
      foreach(vector_variable* v, variables) {
        arma::span span = safe_get(table_ptr->var_span, v);
        for (size_t i = span.a; i <= span.b; ++i) {
          indices.push_back(i);
        }
      }
      return indices;
    }
    
    //! Common implementation of the insert() function
    void insert(const arma::Col<T>& values, T weight) {
      assert(values.size() == table_ptr->ncols());
      ordering.push_back(table_ptr->weights.size());
      table_ptr->data.insert(table_ptr->data.end(), values.begin(), values.end());
      table_ptr->weights.push_back(weight);
    }

    //! Returns an iterator to the underlying data starting at a row
    typename std::vector<T>::const_iterator row_begin(size_t row) const {
      return table_ptr->data.begin() += ordering[row] * table_ptr->ncols();
    }

    //! Returns an iterator to the underlying data starting at a row
    typename std::vector<T>::iterator row_begin(size_t row) {
      return table_ptr->data.begin() += ordering[row] * table_ptr->ncols();
    }

    //! Returns the weight for the given row
    T weight(size_t row) const {
      return table_ptr->weights[ordering[row]];
    }
    
    // C++11 conversion feature
    // friend record_iterator;
    // friend const_record_iterator;
    // friend sample_iterator;
    friend class basic_record_iterator<vector_dataset>;
    friend class basic_const_record_iterator<vector_dataset>;
    friend class basic_sample_iterator<vector_dataset>;

  }; // class vector_dataset

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
