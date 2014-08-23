#ifndef SILL_FINITE_MEMORY_DATASET_HPP
#define SILL_FINITE_MEMORY_DATASET_HPP

#include <sill/learning/dataset/finite_dataset.hpp>
#include <sill/learning/dataset/slice_view.hpp>

#include <algorithm>
#include <stdexcept>

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // TODO: replace boost::shared_ptr with std::unique_ptr
  class finite_memory_dataset : public finite_dataset, boost::noncopyable {
  public:
    // SliceableDataset concept typedefs
    typedef slice_view<finite_dataset> slice_view_type;
    
    //! Creates an uninitialized dataset
    finite_memory_dataset() { } 

    /**
     * Initializes the dataset with the given sequence of variables
     * and pre-allocates memory for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const finite_var_vector& variables, size_t capacity = 1) {
      if (data) {
        throw std::logic_error("Attempt to call initialize() more than once.");
      }
      args = variables;
      num_allocated = std::max(capacity, size_t(1));
      num_inserted = 0;
      num_cols = variables.size();
      data.reset(new size_t[num_allocated * num_cols]);
      weights.reset(new double[num_allocated]);
      col_ptr.resize(variables.size());
      for (size_t i = 0; i < variables.size(); ++i) {
        arg_index[variables[i]] = i;
        col_ptr[i] = data.get() + num_allocated * i;
      }
    }

    /**
     * Initializes the dataset with the given sequence of variables
     * and pre-allocates memory for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const finite_domain& variables, size_t capacity = 1) {
      initialize(make_vector(variables), capacity);
    }

    size_t size() const {
      return num_inserted;
    }

    size_t capacity() const {
      return num_allocated;
    }

    void reserve(size_t new_capacity) {
      if (new_capacity > num_allocated) {
        reallocate(new_capacity); // private function
      }
    }

    finite_domain arguments() const {
      return make_domain(args);
    }

    finite_record record(size_t row) const {
      return record(row, args);
    }

    finite_record record(size_t row, const finite_var_vector& vars) const {
      assert(row < num_inserted);
      finite_record result(vars.size(), weights[row]);
      for (size_t i = 0; i < vars.size(); ++i) {
        result.values[i] = col_ptr[safe_get(arg_index, vars[i])][row];
      }
      return result;
    }

    //! Returns a view representing a contiguous range of rows
    slice_view<finite_dataset> subset(size_t begin, size_t end) {
      return slice_view<finite_dataset>(this, slice(begin, end));
    }

    //! Returns a view representing a contiguous range of rows
    slice_view<finite_dataset> subset(const slice& s) {
      return slice_view<finite_dataset>(this, s);
    }

    //! Returns a view of representing a union of row ranges
    slice_view<finite_dataset> subset(const std::vector<slice>& s) {
      return slice_view<finite_dataset>(this, s);
    }

    //! Inserts the values in this dataset's ordering.
    virtual void insert(const finite_record& r) {
      check_initialized();
      insert(r.values, r.weight); // protected function
    }
 
    //! Inserts a new row from an assignment (all variables must be present).
    virtual void insert(const finite_assignment& a, double weight = 1.0) {
      check_initialized();
      std::vector<size_t> values;
      values.reserve(num_cols);
      foreach(finite_variable* v, args) {
        values.push_back(safe_get(a, v));
      }
      insert(values, weight); // protected function
    }

    //! Inserts the given number of rows with unit weights and "undefined" values.
    virtual void insert(size_t nrows) {
      check_initialized();

      // compute the special "undefined" value for each variable
      std::vector<size_t> values;
      foreach(finite_variable* v, args) {
        values.push_back(v->size());
      }

      // insert the rows
      for (size_t i = 0; i < nrows; ++i) {
        insert(values, 1.0); // protected function
      }
    }

    // Protected functions
    //========================================================================
  protected:
    //! Throws an exception if the dataset is not initialized
    void check_initialized() const {
      if (!data) {
        throw std::logic_error("The finite dataset is not initialized!");
      }
    }

    //! The common implementation of the insert() function
    void insert(const std::vector<size_t>& values, double weight) {
      assert(num_inserted <= num_allocated);
      if (num_inserted == num_allocated) {
        reallocate(2 * num_allocated);
      }

      assert(values.size() == num_cols);
      for (size_t i = 0; i < num_cols; ++i) {
        col_ptr[i][num_inserted] = values[i];
      }
      weights[num_inserted] = weight;
      ++num_inserted;
    }

    aux_data* init(const finite_var_vector& args,
                   iterator_state_type& state) const {
      check_initialized();
      state.elems.resize(args.size());
      for (size_t i = 0; i < args.size(); ++i) {
        state.elems[i] = col_ptr[safe_get(arg_index, args[i])];
      }
      state.weights = weights.get();
      state.e_step.assign(args.size(), 1);
      state.w_step = 1;
      return NULL;
    }
    
    void advance(ptrdiff_t diff,
                 iterator_state_type& state,
                 aux_data* data) const {
      for (size_t i = 0; i < state.elems.size(); ++i) {
        state.elems[i] += diff; // step is always one for finite_memory_dataset
      }
      state.weights += diff;
    }
    
    size_t load(size_t n,
                iterator_state_type& state,
                aux_data* data) const {
      return std::min(n, size_t(weights.get() + num_inserted - state.weights));
    }

    void save(iterator_state_type& state, aux_data* data) { }

    void print(std::ostream& out) const {
      out << "finite_dataset[N=" << size() << ", args=" << args << "]";
    }

    // Private data members
    //========================================================================
  private:
    // increases the storage capacity to new_capacity and copies the data
    void reallocate(size_t new_capacity) {
      // allocate the new data
      size_t* new_data = new size_t[new_capacity * num_cols];
      double* new_weights = new double[new_capacity];
      std::vector<size_t*> new_col_ptr(args.size());
      for (size_t i = 0; i < args.size(); ++i) {
        new_col_ptr[i] = new_data + new_capacity * i;
      }

      // copy the elements and weights to the new locations
      for (size_t i = 0; i < args.size(); ++i) {
        std::copy(col_ptr[i], col_ptr[i] + num_inserted, new_col_ptr[i]);
      }
      std::copy(weights.get(), weights.get() + num_inserted, new_weights);

      // swap the old and the new data
      data.reset(new_data);
      weights.reset(new_weights);
      col_ptr.swap(new_col_ptr);
      num_allocated = new_capacity;
    }

    finite_var_vector args;  // the ordering of variables in the table
    std::map<finite_variable*, size_t> arg_index; // the index of each var
    boost::shared_ptr<size_t[]> data;    // the data storage
    boost::shared_ptr<double[]> weights; // the weights storage
    std::vector<size_t*> col_ptr;        // pointers to the elements
    size_t num_allocated;                // the number of allocated rows
    size_t num_inserted;                 // the number of inserted rows
    size_t num_cols;                     // the number of columns

  }; // class finite_memory_dataset

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
