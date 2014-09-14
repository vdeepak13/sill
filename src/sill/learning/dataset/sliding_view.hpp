#ifndef SILL_SLIDING_VIEW_HPP
#define SILL_SLIDING_VIEW_HPP

namespace sill {

  /**
   * A view of sequence datasets over a moving, fixed-size sliding window
   * over the sequence data. 
   * \todo allow mutations
   */
  template <typename BaseDS>
  class sliding_view : public BaseDS {

    // Helper types
    typedef typename BaseDS::variable_type variable_type;
    typedef discrete_process<variable_type> process_type;

    //! Default constructor. Creates an uninitialized view
    sliding_view()
      : dataset_(NULL), window(0) { }

    //! Constructs a sliding view for the given sequence dataset
    sliding_view(const sequence_dataset<BaseDS>* dataset, size_t window)
      : dataset_(dataset), window_(window) {
      // TODO: initialize the base dataset
    }

    //! Returns the logical number of rows in this view
    size_t size() const {
      return dataset_->total_steps() - dataset_->size() * window_;
    }

    //! Returns a single data point for a subset of variables (slow)
    record_type record(size_t row, const var_vector_type& args) const {
      size_t sequence, step;
      dataset_->find_step(row, sequence, step); // fix this
      assignment_type a;
      dataset_->record(sequence, processes(args)).extract(step, a);
      return record_type(a);
      assignment_type a;
    }
    
  protected:
    typedef typename BaseDS::iterator_state_type iterator_state_type;

    struct view_data : public aux_data {
      std::vector<size_t> index;
      std::vector<size_t> offset;
      typename sequence_dataset<BaseDS>::const_record_iterator it, end;
    };

    // initializes the data structures in the record iterator
    aux_data* init(const var_vector_type& vars,
                   iterator_state_type& state) const {
      view_data* data = new view_data;
      std::vector<process_type*> processes;
      foreach(variable_type* v, vars) {
        process_type* process = var->process();
        size_t index = processes.find(process);
        if (index == end) {
          processes.push_back(process);
        }
        data->index.push_back(index);
        data->offset.push_back(get_index(v->index()));
        // todo get the right index (need to clean up the indices first)
        // todo generalize this to different variable types
      }
      boost::tie(data->it, data->end) = dataset_->records(processes);
      return data;
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
      if (d.first) {
        d.first = false;
      } else {
        ++d.it; // this may save data
      }
      for (size_t i = 0; i < d.size(); ++i) {
        size_t sequence = d.sequences[i];
        size_t offset = d.offsets[i];
        state.elems[i] = advance(d.it->values[sequence], offset); // todo
        state.e_step[i] = 1;
      }
      state.weights = &it->weight;
      state.w_step = 0;
      return ...;
    }
  
    // noop (data is saved when load advances the iterator)
    void save(iterator_state_type& state, aux_data* data) { }

    // prints the summary of this view to a stream
    void print(std::ostream& out) const {
      out << "sliding_view(N=" << size() << ", history=" << history_ << ") "
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
    size_t history_;

  }; // class sliding_view

} // namespace sill

#endif
