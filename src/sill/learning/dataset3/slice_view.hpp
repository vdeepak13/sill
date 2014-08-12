#ifndef SILL_SLICE_VIEW_HPP
#define SILL_SLICE_VIEW_HPP

#include <sill/learning/dataset3/aux_data.hpp>
#include <sill/learning/dataset3/slice.hpp>

#include <vector>

#include <boost/shared_ptr.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // TODO: replace boost::shared_ptr with std::unique_ptr

  /**
   * A basic slice view that supports contiguous datasets
   * like finite_dataset and vector_dataset.
   * \see Dataset
   */
  template <typename BaseDS>
  class slice_view : public BaseDS {
  public:
    typedef typename BaseDS::domain_type domain_type;
    typedef typename BaseDS::vector_type vector_type;
    typedef typename BaseDS::record_type record_type;

    //! Default constructor. Creates an uninitialized view.
    slice_view()
      : dataset_(NULL), size_(0) { }

    //! Constructs a view of a dataset with a single slice
    slice_view(BaseDS* dataset, const slice& s)
      : dataset_(dataset), size_(0) {
      initialize(std::vector<slice>(1, s));
    }

    //! Constructs a view of a dataset with multiple slices
    slice_view(BaseDS* dataset, const std::vector<slice>& s)
      : dataset_(dataset), size_(0) {
      initialize(s);
    }
      
    //! Returns the logical number of rows in this view
    size_t size() const {
      return size_;
    }

    //! Returns the total number of slices in this view
    size_t num_slices() const {
      return slices_.size();
    }
    
    //! Returns the columns of this dataset
    domain_type arguments() const {
      check_initialized();
      return dataset_->arguments();
    }
    
    //! Returns a single data point in the base dataset's natural ordering
    record_type record(size_t row) const {
      assert(row < size_);
      foreach(const slice& s, slices_) {
        if (row < s.size()) {
          return dataset_->record(s.begin + row);
        } else {
          row -= s.size();
        }
      }
      assert(false); // inconsistent state
    }

    //! Returns a single data point for a subset of arguments (variables)
    record_type record(size_t row, const vector_type& args) const {
      assert(row < size_);
      foreach(const slice& s, slices_) {
        if (row < s.size()) {
          return dataset_->record(s.begin + row, args);
        } else {
          row -= s.size();
        }
      }
      assert(false); // inconsistent state
    }

    // Protected functions (invoked by the iterators and public functions)
    //========================================================================
  protected:
    typedef typename BaseDS::record_iterator::state_type iterator_state_type;

    struct view_data : public aux_data {
      size_t slice_index; // the index of the current slice
      size_t rows_left;   // the number of rows left in the current slice
      boost::shared_ptr<aux_data> ds_data_ptr;
      view_data(aux_data* ds_data)
        : slice_index(-1), rows_left(0), ds_data_ptr(ds_data) { }
      aux_data* ds_data() { return ds_data_ptr.get(); }
    };
  
    // initializes the data structures in the record iterator
    aux_data* init(const vector_type& args,
                   iterator_state_type& state) const {
      check_initialized();
      return new view_data(dataset_->init(args, state));
    }
  
    // advances the internal pointer in data by the given difference
    void advance(ptrdiff_t diff,
                 iterator_state_type& state,
                 aux_data* data) const {
      throw std::logic_error("slice_view does not suport advance()");
    }

    // loads at most n rows
    size_t load(size_t n,
                iterator_state_type& state,
                aux_data* data) const {
      view_data& d = cast(data);

      // advance if needed
      if (d.rows_left == 0) {
        ++d.slice_index;
        if (d.slice_index >= slices_.size()) {
          return 0;
        }
        slice s = slices_[d.slice_index];
        if (d.slice_index > 0) {
          ptrdiff_t diff = s.begin - slices_[d.slice_index-1].end;
          dataset_->advance(diff, state, d.ds_data());
        } else {
          dataset_->advance(s.begin, state, d.ds_data());
        }
        d.rows_left = s.size();
      }
      
      // load more data
      size_t to_load = std::min(n, d.rows_left);
      size_t nloaded = dataset_->load(to_load, state, d.ds_data());
      d.rows_left -= nloaded;
      
      return nloaded;
    }
    
    // saves the previously loaded data
    void save(iterator_state_type& state, aux_data* data) {
      dataset_->save(state, cast(data).ds_data());
    }

    // prints the summary of this view to a stream
    void print(std::ostream& out) const {
      out << "slice_view[N=" << size() << ",slices=" << num_slices() << "] "
          << "base dataset: " << dataset_;
    }

    // Private functions
    //========================================================================
  private:
    //! Throws an exception if the dataset is not initialized
    void check_initialized() const {
      if (!dataset_) {
        throw std::logic_error("The slice_vew is not initialized!");
      }
    }

    // initializes the slice vector and the view size
    void initialize(const std::vector<slice>& slices) {
      // first, validate the slices
      size_t ds_size = dataset_->size();
      foreach(const slice& s, slices) {
        assert(s.begin <= s.end);
        assert(s.end <= ds_size);
      }

      // now initialize the slices and the cached view size
      foreach(const slice& s, slices) {
        if (!s.empty()) {
          slices_.push_back(s);
          size_ += s.size();
        }
      }
    }

    // downcasts the auxiliary data to this dataset's data type
    static view_data& cast(aux_data* data) {
      view_data* result = dynamic_cast<view_data*>(data);
      assert(result);
      return *result;
    }

    // Data members
    //========================================================================
  private:
    BaseDS* dataset_;            // underlying dataset
    std::vector<slice> slices_;  // list of slices (unsorted)
    size_t size_;                // cached number of rows

  }; // class slice_view

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

