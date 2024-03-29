#ifndef SILL_SLIDING_VIEW_HPP
#define SILL_SLIDING_VIEW_HPP

#include <sill/base/discrete_process.hpp>
#include <sill/learning/dataset/aux_data.hpp>

#include <vector>

#include <sill/macros_def.hpp>

namespace sill {

  // forward declaration
  template <typename BaseDS> class sequence_dataset;

  /**
   * A view of sequence datasets over a moving, fixed-size sliding window
   * over the sequence data.
   * \todo allow mutations?
   */
  template <typename BaseDS>
  class sliding_view : public BaseDS {
  public:
    // Bring in some types from BaseDS
    typedef typename BaseDS::argument_type   argument_type;
    typedef typename BaseDS::var_vector_type var_vector_type;
    typedef typename BaseDS::record_type     record_type;

    // helper typedefs
    typedef discrete_process<argument_type> process_type;
    typedef typename BaseDS::iterator_state_type iterator_state_type;
    typedef typename BaseDS::sequence_record_type sequence_record_type;
    typedef typename sequence_record_type::var_indices_type var_indices_type;

    //! Default constructor. Creates an uninitialized view
    sliding_view()
      : dataset_(NULL), window_(0) { }

    //! Constructs a sliding view for the given sequence dataset
    sliding_view(const sequence_dataset<BaseDS>* dataset, size_t window)
      : dataset_(dataset), window_(window) {
      assert(window <= 1); // need to clean up discrete_process to support >1

      // initialize the variables
      var_vector_type vars;
      vars.reserve((window+1) * dataset->num_arguments());
      foreach (process_type* proc, dataset->arg_vector()) {
        vars.push_back(proc->current());
        if (window == 1) vars.push_back(proc->next());
      }
      BaseDS::initialize(vars);

      // compute the (cumulative) size for each record in the underlying dataset
      size_t sum = 0;
      cum_size_.reserve(dataset->size());
      std::vector<process_type*> no_procs;
      foreach(const sequence_record_type& r, dataset->records(no_procs)) {
        if (r.num_steps() > window) {
          sum += r.num_steps() - window;
        }
        cum_size_.push_back(sum);
      }
    }

    //! Returns the logical number of rows in this view
    size_t size() const {
      return cum_size_.empty() ? 0 : cum_size_.back();
    }

    //! Returns a single data point for a subset of variables
    record_type record(size_t row, const var_vector_type& vars) const {
      assert(row < size());
      size_t ds_row =
        std::upper_bound(cum_size_.begin(), cum_size_.end(), row) - 
        cum_size_.begin();
      size_t offset = // TODO this needs to be fixed after process refactor
        (ds_row > 0) ? row - cum_size_[ds_row-1] : row;
      var_indices_type var_indices;
      dataset_->index_mapping().indices(vars, offset, var_indices);
      record_type result(vars);
      dataset_->record(ds_row).extract(var_indices, result);
      return result;
    }
    
  protected:
    struct view_data : public aux_data {
      //! The underlying iterator range
      typename sequence_dataset<BaseDS>::const_record_iterator it, end;
      //! True if we are at the beginning of the range
      bool start;
      //! The indices of the variables in the underlying records
      var_indices_type indices;
    };

    // initializes the data structures in the record iterator
    aux_data* init(const var_vector_type& vars,
                   iterator_state_type& state) const {
      view_data& d = *(new view_data);
      std::vector<process_type*> procs =
        make_vector(discrete_processes(make_domain(vars)));
      boost::tie(d.it, d.end) = dataset_->records(procs);
      d.start = true;
      typename sequence_record_type::index_map_type index_map(procs);
      index_map.indices(vars, d.indices);
      return &d;
    }

    void advance(ptrdiff_t diff,
                 iterator_state_type& state,
                 aux_data* data) const {
      throw std::logic_error("sliding_view does not support advance()");
    }

    // loads rows (n is ignored)
    size_t load(size_t n,
                iterator_state_type& state,
                aux_data* data) const {
      view_data& d = cast(data);

      // advance to the next valid position
      do {
        if (d.start) { d.start = false; } else { ++d.it; }
      } while (d.it != d.end && d.it->num_steps() <= window_);

      // if not at the end, extract the state
      if (d.it == d.end) {
        return 0;
      } else {
        d.it->extract(d.indices, state);
        return d.it->num_steps() - window_;
      }
    }
  
    // noop, data is not saved yet. in the future, this will be done by load()
    void save(iterator_state_type& state, aux_data* data) { }

    // prints the summary of this view to a stream
    void print(std::ostream& out) const {
      out << "sliding_view(N=" << size() << ", window=" << window_ << ") "
          << "base dataset: " << dataset_;
    }
    
  private:
    // downcasts the auxiliary data to this dataset's data type
    static view_data& cast(aux_data* data) {
      view_data* result = dynamic_cast<view_data*>(data);
      assert(result);
      return *result;
    }

  private:
    const sequence_dataset<BaseDS>* dataset_;
    size_t window_;
    std::vector<size_t> cum_size_;

  }; // class sliding_view

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
