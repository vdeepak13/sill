#ifndef SILL_FINITE_DATASET_HPP
#define SILL_FINITE_DATASET_HPP

#include <sill/base/finite_assignment.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/learning/dataset2/basic_record_iterators.hpp>
#include <sill/learning/dataset2/finite_record.hpp>

#include <boost/shared_ptr.hpp>

#include <stdexcept>
#include <vector>

#include <sill/macros_def.hpp>

namespace sill {

  // models FiniteDataset and InsertableDataset
  template <typename T = uint32_t> 
  class finite_dataset {
  public:
    typedef T value_type;

    typedef finite_variable   variable_type;
    typedef finite_var_vector vector_type;
    typedef finite_domain     domain_type;
    typedef finite_assignment assignment_type;
    typedef finite_record2     record_type;

    typedef basic_record_iterator<finite_dataset> record_iterator;
    typedef basic_const_record_iterator<finite_dataset> const_record_iterator;

    //! Creates an uninitialized dataset
    finite_dataset() { }

    //! Initializes the dataset with the given sequence of variables.
    //! It is an error to call initialize() more than once.
    void initialize(const finite_var_vector& variables) {
      if (table_ptr) {
        throw std::logic_error("Attempt to call initialize() more than once.");
      }
      table_ptr.reset(new table(variables));
    }

    //! Initializes the dataset with the given finite domain.
    //! It is an error to call initialize() more than once.
    void initialize(const finite_domain& variables) {
      initialize(finite_var_vector(variables.begin(), variables.end()));
    }

    //! Returns the number of datapoints in the dataset.
    size_t size() const {
      return ordering.size();
    }

    //! Returns the columns of this dataset.
    finite_domain arguments() const {
      check_initialized();
      return make_domain(table_ptr->variables);
    }

    //! Returns mutable records for the specified finite variables.
    std::pair<record_iterator, record_iterator>
    records(const finite_var_vector& vars) {
      check_initialized();
      return std::make_pair(record_iterator(this, get_indices(vars)),
                            record_iterator(this));
    }

    //! Returns immutable records for the specified finite variables.
    std::pair<const_record_iterator, const_record_iterator>
    records(const finite_var_vector& vars) const {
      check_initialized();
      return std::make_pair(const_record_iterator(this, get_indices(vars)),
                            const_record_iterator(this));
    }

    //! Returns a view of the dataset for a range of the rows.
    finite_dataset subset(size_t begin, size_t end) const {
      check_initialized();
      assert(begin <= size());
      assert(end <= size());
      return finite_dataset(*this, begin, end);
    }

    //! Returns a view whose records match the given assignment.
    finite_dataset restrict(const finite_assignment& a) const {
      check_initialized();

      // extract the indices of the fixed variables and the values
      std::vector<size_t> indices;
      std::vector<T> values;
      foreach(finite_assignment::const_reference p, a) {
        if (table_ptr->var_index.count(p.first)) {
          indices.push_back(safe_get(table_ptr->var_index, p.first));
          values.push_back(p.second);
        }
      }

      // iterate through the records and extract ones with correct values
      const_record_iterator it(this, indices);
      const_record_iterator end(this);
      std::vector<size_t> new_ordering;
      for(; it != end; ++it) {
        if (it->index == values) {
          new_ordering.push_back(ordering[it.current_row()]);
        }
      }
        
      // return the view
      return finite_dataset(table_ptr, new_ordering);
    }

    //! Inserts the values in this dataset's ordering.
    virtual void insert(const finite_record2& r) {
      check_initialized();
      insert(r.values, r.weight); // protected function
    }
 
    //! Inserts a new row from an assignment (all variables must be present).
    virtual void insert(const finite_assignment& a, double weight = 1.0) {
      check_initialized();
      std::vector<size_t> values;
      values.reserve(table_ptr->variables.size());
      foreach(finite_variable* v, table_ptr->variables) {
        values.push_back(safe_get(a, v));
      }
      insert(values, weight); // protected function
    }

    //! Inserts the given number of rows with unit weights and values -1.
    virtual void insert(size_t nrows) {
      check_initialized();
      for (size_t i = 0; i < nrows; ++i) {
        ordering.push_back(table_ptr->weights.size());
        table_ptr->weights.push_back(1.0);
      }
      table_ptr->data.insert(table_ptr->data.end(),
                             nrows * table_ptr->ncols(),
                             -1);
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
      finite_var_vector variables; // the ordering of variables in the table
      std::map<finite_variable*, size_t> var_index; // index of each var
      std::vector<T> data;
      std::vector<double> weights;
      size_t ncols() const { return variables.size(); }
      table(const finite_var_vector& variables)
        : variables(variables) {
        for (size_t i = 0; i < variables.size(); ++i) {
          var_index[variables[i]] = i;
        }
      }
    };

    //! the actual data plus some indexing stuff. null until initialization
    boost::shared_ptr<table> table_ptr;

    //! the subset of datapoints that this dataset logically consists of
    std::vector<size_t> ordering;

    // Private member functions
    //========================================================================
  private:
    //! Creates a derived dataset
    finite_dataset(const boost::shared_ptr<table>& table_ptr, 
                   const std::vector<size_t>& ordering)
      : table_ptr(table_ptr), ordering(ordering) { }

    //! Creates a derived dataset
    finite_dataset(const finite_dataset& other, size_t begin, size_t end)
      : table_ptr(other.table_ptr),
        ordering(other.ordering.begin() + begin,
                 other.ordering.end() + end) { }
 
    //! Throws an exception if the dataset is not initialized
    void check_initialized() const {
      if (!table_ptr) {
        throw std::logic_error("The finite dataset is not initialized!");
      }
    }

    //! Returns the column indices for the specified variables
    std::vector<size_t> get_indices(const finite_var_vector& variables) const {
      std::vector<size_t> indices;
      indices.reserve(variables.size());
      foreach(finite_variable* v, variables) {
        indices.push_back(safe_get(table_ptr->var_index, v));
      }
      return indices;
    }
    
    //! Common implementation of the insert() function
    void insert(const std::vector<size_t>& values, double weight) {
      assert(values.size() == table_ptr->variables.size());
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
    double weight(size_t row) const {
      return table_ptr->weights[ordering[row]];
    }
    
    // C++11 conversion feature
    // friend record_iterator;
    // friend const_record_iterator;
    friend class basic_record_iterator<finite_dataset>;
    friend class basic_const_record_iterator<finite_dataset>;

  }; // class finite_dataset

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
