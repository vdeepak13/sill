#ifndef SILL_HYBRID_MEMORY_DATASET_HPP
#define SILL_HYBRID_MEMORY_DATASET_HPP

#include <sill/learning/dataset/hybrid_dataset.hpp>
#include <sill/learning/dataset/finite_memory_dataset.hpp>
#include <sill/learning/dataset/vector_memory_dataset.hpp>
#include <sill/learning/dataset/slice_view.hpp>

#include <algorithm>
#include <stdexcept>

#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename T = double>
  class hybrid_memory_dataset : public hybrid_dataset<T>, boost::noncopyable {
  public:
    // SliceableDataset concept typedefs
    typedef slice_view<hybrid_dataset<T> > slice_view_type;
    
    // Bring the record(row) implementation up to this class
    using hybrid_dataset<T>::record;

    //! Creates an uninitialized dataset
    hybrid_memory_dataset() { } 

    /**
     * Initializes the dataset with the given sequence of variables
     * and pre-allocates memory for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const var_vector& variables, size_t capacity = 1) {
      finite_var_vector finite_vars;
      vector_var_vector vector_vars;
      split(variables, finite_vars, vector_vars);
      finite_ds.initialize(finite_vars, capacity);
      vector_ds.initialize(vector_vars, capacity);
      hybrid_dataset<T>::initialize(variables);
    }

    /**
     * Initializes the dataset with the given sequences of finite and
     * vector variables and pre-allocates memory for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const finite_var_vector& finite_vars,
                    const vector_var_vector& vector_vars,
                    size_t capacity = 1) {
      finite_ds.initialize(finite_vars, capacity);
      vector_ds.initialize(vector_vars, capacity);
      hybrid_dataset<T>::initialize(concat(finite_vars, vector_vars));
    }

    /**
     * Initializes the dataset with the given sequence of variables
     * and pre-allocates memory for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const domain& variables, size_t capacity = 1) {
      initialize(make_vector(variables), capacity);
    }
    
    size_t size() const {
      return finite_ds.size();
    }

    size_t finite_cols() const {
      return finite_ds.num_cols;
    }

    size_t vector_cols() const {
      return vector_ds.num_cols;
    }

    size_t capacity() const {
      return finite_ds.capacity();
    }

    void reserve(size_t new_capacity) {
      finite_ds.reserve(new_capacity);
      vector_ds.reserve(new_capacity);
    }

    hybrid_record<T> record(size_t row, const var_vector& variables) const {
      finite_var_vector finite_vars;
      vector_var_vector vector_vars;
      split(variables, finite_vars, vector_vars);
      finite_record    fr = finite_ds.record(row, finite_vars);
      vector_record<T> vr = vector_ds.record(row, vector_vars);
      return hybrid_record<T>(fr.values, vr.values, vr.weight);
    }

    //! Returns a view representing a contiguous range of rows
    slice_view<hybrid_dataset<T> > subset(size_t begin, size_t end) {
      return slice_view<hybrid_dataset<T> >(this, slice(begin, end));
    }

    //! Returns a view representing a contiguous range of rows
    slice_view<hybrid_dataset<T> > subset(const slice& s) {
      return slice_view<hybrid_dataset<T> >(this, s);
    }

    //! Returns a view of representing a union of row ranges
    slice_view<hybrid_dataset<T> > subset(const std::vector<slice>& s) {
      return slice_view<hybrid_dataset<T> >(this, s);
    }

    //! Inserts the values in this dataset's ordering.
    void insert(const hybrid_record<T>& r) {
      finite_ds.insert(r.values.finite, r.weight);
      vector_ds.insert(r.values.vector, r.weight);
    }
 
    //! Inserts a new row from an assignment (all variables must be present).
    void insert(const assignment& a, T weight = 1.0) {
      finite_ds.insert(a, weight);
      vector_ds.insert(a, weight);
    }

    //! Inserts the given number of rows with unit weights and "undefined" values.
    void insert(size_t nrows) {
      finite_ds.insert(nrows);
      vector_ds.insert(nrows);
    }

    // Protected functions
    //========================================================================
  protected:
    typedef typename hybrid_dataset<T>::iterator_state_type iterator_state_type;
    using hybrid_dataset<T>::args;

    aux_data* init(const var_vector& args, iterator_state_type& state) const {
      finite_var_vector finite_vars;
      vector_var_vector vector_vars;
      split(args, finite_vars, vector_vars);
      if (state.finite && state.vector) {
        finite_ds.init(finite_vars, *state.finite);
        vector_ds.init(vector_vars, *state.vector);
      } else if (state.finite && !state.vector) {
        finite_ds.init(finite_vars, *state.finite);
        assert(vector_vars.empty());
      } else if (!state.finite && state.vector) {
        assert(finite_vars.empty());
        vector_ds.init(vector_vars, *state.vector);
      } else {
        assert(false);
      }
      return NULL;
    }

    void advance(ptrdiff_t diff, iterator_state_type& state, aux_data*) const {
      if (state.finite) {
        finite_ds.advance(diff, *state.finite, NULL);
      }
      if (state.vector) {
        vector_ds.advance(diff, *state.vector, NULL);
      }
    }
    
    size_t load(size_t n, iterator_state_type& state, aux_data*) const {
      size_t count = 0;
      if (state.finite) {
        count = finite_ds.load(n, *state.finite, NULL);
      }
      if (state.vector) {
        count = vector_ds.load(n, *state.vector, NULL);
      }
      return count;
    }

    void save(iterator_state_type& state, aux_data*) {
      if (state.finite) {
        finite_ds.save(*state.finite, NULL);
      }
      if (state.vector) {
        vector_ds.save(*state.vector, NULL);
      }
    }

    void print(std::ostream& out) const {
      out << "hybrid_memory_dataset(N=" << size() << ", args=" << args << ")";
    }

    // Private data members
    //========================================================================
  private:
    // var_vector args;  // moved to the base class
    finite_memory_dataset    finite_ds;
    vector_memory_dataset<T> vector_ds;

  }; // class hybrid_memory_dataset

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
