#ifndef SILL_VECTOR_MEMORY_DATASET_HPP
#define SILL_VECTOR_MEMORY_DATASET_HPP

#include <sill/learning/dataset/slice_view.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>

#include <algorithm>
#include <stdexcept>

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forwad declarations
  template <typename T> class hybrid_memory_dataset;

  /**
   * A dataset that stores observations for vector variables in memory.
   * Models Dataset, InsertableDataset, and SliceableDataset.
   *
   * \tparam T the internal storage of the vector values. This should match the
   *         storage type of the learned factors.
   */
  template <typename T = double> 
  class vector_memory_dataset : public vector_dataset<T>, boost::noncopyable {
  public:
    // SliceableDataset concept typedefs
    typedef slice_view<vector_dataset<T> > slice_view_type;

    // Bring the record(row) implementation up to this class
    using vector_dataset<T>::record;
    
    //! Creates an uninitialized dataset
    vector_memory_dataset() { } 

    /**
     * Initializes the dataset with the given sequence of variables
     * and allocates memory for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const vector_var_vector& variables, size_t capacity = 1) {
      if (data) {
        throw std::logic_error("Attempt to call initialize() more than once.");
      }
      vector_dataset<T>::initialize(variables);
      num_allocated = std::max(capacity, size_t(1));
      num_inserted = 0;
      num_cols = vector_size(variables);
      data.reset(new T[num_allocated * num_cols]);
      weights.reset(new T[num_allocated]);
      col_ptr.resize(variables.size());
      for (size_t i = 0, col = 0; i < variables.size(); ++i) {
        arg_index[variables[i]] = i;
        col_ptr[i] = data.get() + num_allocated * col;
        col += variables[i]->size();
      }
    }

    /**
     * Initializes the dataset with the given sequence of variables
     * and pre-allocates memory for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const vector_domain& variables, size_t capacity = 1) {
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

    vector_record<T> record(size_t row, const vector_var_vector& vars) const {
      assert(row < num_inserted);
      vector_record<T> result(vector_size(vars), weights[row]);
      size_t col = 0;
      foreach(vector_variable* v, vars) {
        T* begin = col_ptr[safe_get(arg_index, v)] + row * v->size();
        std::copy(begin, begin + v->size(), &result.values[col]);
        col += v->size();
      }
      return result;
    }

    //! Returns a view representing a contiguous range of rows
    slice_view<vector_dataset<T> > subset(size_t begin, size_t end) {
      return slice_view<vector_dataset<T> >(this, slice(begin, end));
    }

    //! Returns a view representing a contiguous range of rows
    slice_view<vector_dataset<T> > subset(const slice& s) {
      return slice_view<vector_dataset<T> >(this, s);
    }

    //! Returns a view of representing a union of row ranges
    slice_view<vector_dataset<T> > subset(const std::vector<slice>& s) {
      return slice_view<vector_dataset<T> >(this, s);
    }

    //! Inserts the values in this dataset's ordering.
    void insert(const vector_record<T>& r) {
      check_initialized();
      insert(r.values, r.weight); // protected function
    }
 
    //! Inserts a new row from an assignment (all variables must be present).
    void insert(const vector_assignment& a, T weight = 1.0) {
      check_initialized();
      arma::Col<T> values(num_cols);
      size_t col = 0;
      foreach(vector_variable* v, args) {
        size_t vsize = v->size();
        values(arma::span(col, col + vsize - 1)) = safe_get(a, v);
        col += vsize;
      }
      insert(values, weight); // protected function
    }

    //! Inserts the given number of rows with unit weights and "undefined" values.
    void insert(size_t nrows) {
      check_initialized();
      arma::Col<T> values(num_cols);
      values.fill(std::numeric_limits<T>::quiet_NaN());
      for (size_t i = 0; i < nrows; ++i) {
        insert(values, 1.0); // protected function
      }
    }

    // Protected functions
    //========================================================================
  protected:
    typedef raw_record_iterator_state<vector_record<T> > iterator_state_type;
    using vector_dataset<T>::args;

    //! Throws an exception if the dataset is not initialized
    void check_initialized() const {
      if (!data) {
        throw std::logic_error("The dataset is not initialized!");
      }
    }

    //! The common implementation of the insert() function
    void insert(const arma::Col<T>& values, double weight) {
      assert(num_inserted <= num_allocated);
      if (num_inserted == num_allocated) {
        reallocate(2 * num_allocated);
      }

      assert(values.size() == num_cols);
      for (size_t i = 0, col = 0; i < args.size(); ++i) {
        size_t vsize = args[i]->size();
        T* begin = col_ptr[i] + num_inserted * vsize;
        std::copy(&values[col], &values[col+vsize], begin);
        col += vsize;
      }
      weights[num_inserted] = weight;
      ++num_inserted;
    }

    aux_data* init(const vector_var_vector& args,
                   iterator_state_type& state) const {
      check_initialized();
      state.elems.reserve(num_cols);
      state.e_step.reserve(num_cols);
      foreach(vector_variable* v, args) {
        size_t vsize = v->size();
        T* col_begin = col_ptr[safe_get(arg_index, v)];
        for (size_t j = 0; j < vsize; ++j) {
          state.elems.push_back(col_begin++);
          state.e_step.push_back(vsize);
        }
      }
      state.weights = weights.get();
      state.w_step = 1;
      return NULL;
      // hybrid_memory_dataset depends on this being NULL
      // if this ever changes, so should hybrid_memory_dataset
    }
    
    void advance(ptrdiff_t diff,
                 iterator_state_type& state,
                 aux_data* data) const {
      for (size_t i = 0; i < state.elems.size(); ++i) {
        state.elems[i] += diff * state.e_step[i];
      }
      state.weights += diff; // weight step is always 1 for this dataset type
    }
    
    size_t load(size_t n,
                iterator_state_type& state,
                aux_data* data) const {
      return std::min(n, size_t(weights.get() + num_inserted - state.weights));
    }

    void save(iterator_state_type& state, aux_data* data) { }

    void print(std::ostream& out) const {
      out << "vector_memory_dataset(N=" << size() << ", args=" << args << ")";
    }

    // friends
    friend class hybrid_memory_dataset<T>;

    // Private data members
    //========================================================================
  private:
    // increases the storage capacity to new_capacity and copies the data
    void reallocate(size_t new_capacity) {
      // allocate the new data
      T* new_data = new T[new_capacity * num_cols];
      T* new_weights = new T[new_capacity];
      std::vector<T*> new_col_ptr(args.size());
      for (size_t i = 0, col = 0; i < args.size(); ++i) {
        new_col_ptr[i] = new_data + new_capacity * col;
        col += args[i]->size();
      }

      // copy the elements and weights to the new locations
      for (size_t i = 0; i < args.size(); ++i) {
        std::copy(col_ptr[i], col_ptr[i] + num_inserted * args[i]->size(),
                  new_col_ptr[i]);
      }
      std::copy(weights.get(), weights.get() + num_inserted, new_weights);

      // swap the old and the new data
      data.reset(new_data);
      weights.reset(new_weights);
      col_ptr.swap(new_col_ptr);
      num_allocated = new_capacity;
    }

    // vector_var_vector args;  // moved to the base class
    std::map<vector_variable*, size_t> arg_index; // the index of each var
    boost::shared_ptr<T[]> data;    // the data storage
    boost::shared_ptr<T[]> weights; // the weights storage
    std::vector<T*> col_ptr;        // pointers to the elements
    size_t num_allocated;           // the number of allocated rows
    size_t num_inserted;            // the number of inserted rows
    size_t num_cols;                // the number of columns

  }; // class vector_memory_dataset

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
