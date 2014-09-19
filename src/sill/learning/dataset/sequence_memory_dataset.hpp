#ifndef SILL_SEQUENCE_MEMORY_DATASET_HPP
#define SILL_SEQUENCE_MEMORY_DATASET_HPP

#include <sill/learning/dataset/sequence_dataset.hpp>
#include <sill/learning/dataset/slice_view.hpp>
#include <sill/math/permutations.hpp>

#include <boost/noncopyable.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A dataset that stores observations for a sequence of variables
   * in memory. Models Dataset, InsertableDataset, and SliceableDataset.
   */
  template <typename BaseDS>
  class sequence_memory_dataset
    : public sequence_dataset<BaseDS>, boost::noncopyable {
  public:
    // types for the SliceableDataset concept
    typedef slice_view<sequence_dataset<BaseDS> > slice_view_type;

    // bring some types up from the base class
    typedef typename BaseDS::argument_type        variable_type;
    typedef discrete_process<variable_type>       argument_type;
    typedef std::set<argument_type*>              domain_type;
    typedef std::vector<argument_type*>           arg_vector_type;
    typedef typename BaseDS::assignment_type      assignment_type;
    typedef typename BaseDS::sequence_record_type record_type;

    // some useful typedefs
    typedef typename record_type::weight_type weight_type;

    // bring the record(row) implementation up to this class
    using sequence_dataset<BaseDS>::record;

    //! Creates an uninitialized dataset
    sequence_memory_dataset() { }

    //! Releases the memory owned by the records in this dataset
    ~sequence_memory_dataset() {
      foreach(record_type& r, records_) {
        r.free_memory();
      }
    }

    /**
     * Initializes the dataset with the given sequence of processes
     * and pre-allocates memory for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const arg_vector_type& procs, size_t capacity = 1) {
      if (this->num_arguments() > 0) {
        throw std::logic_error("Attempt to call initialize() more than once.");
      }
      sequence_dataset<BaseDS>::initialize(procs);
      records_.reserve(capacity);
    }

    /**
     * Initializes the dataset with the given sequence of processes
     * and pre-allocates memory for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const domain_type& procs, size_t capacity = 1) {
      initialize(make_vector(procs), capacity);
    }

    size_t size() const {
      return records_.size();
    }

    size_t capacity() const {
      return records_.capacity();
    }

    void reserve(size_t new_capacity) {
      records_.reserve(new_capacity);
    }

    record_type record(size_t row, const arg_vector_type& args) const {
      assert(row < records_.size());
      typename record_type::proc_indices_type indices;
      this->index_map.indices(args, indices);
      record_type result(args);
      result.load(records_[row], indices);
      return result;
    }

    //! Returns a view representing a contiguous range of rows
    slice_view_type subset(size_t begin, size_t end) {
      return slice_view_type(this, slice(begin, end));
    }

    //! Returns a view representing a contiguous range of rows
    slice_view_type subset(const slice& s) {
      return slice_view_type(this, s);
    }

    //! Returns a view representing a union of row ranges
    slice_view_type subset(const std::vector<slice>& s) {
      return slice_view_type(this, s);
    }

    //! Inserts the values in this dataset's ordering.
    void insert(const record_type& r) {
      r.check_compatible(this->args);
      records_.push_back(r);
    }

    //! Inserts the values from an assignment (all variables must be
    //! present, and the time steps must form a contiguous range [0; t).
    void insert(const assignment_type& a, weight_type weight = 1.0) {
      record_type r(this->args);
      r.assign(a, weight);
      records_.push_back(r);
    }

    // does this make sense?
//     //! Inserts the given number of rows with empty sequences.
//     void insert(size_t nrows) {
//       /// tood:
//     }

    //! Randomly permutes the sequences
    template <typename RandomNumberGenerator>
    void shuffle(RandomNumberGenerator& rng) {
      permute(records_, rng);
    }

    // Protected functions
    //========================================================================
  protected:
    typedef typename sequence_dataset<BaseDS>::iterator_state_type
      iterator_state_type;

    aux_data* init(const arg_vector_type& args,
                   iterator_state_type& state) const {
      this->index_map.indices(args, state.indices);
      state.records = &records_[0];
      return NULL;
    }

    void advance(ptrdiff_t diff,
                 iterator_state_type& state,
                 aux_data* data) const {
      state.records += diff;
    }

    size_t load(size_t n,
                iterator_state_type& state,
                aux_data* data) const {
      return std::min(n, size_t(&records_[0] + records_.size() - state.records));
    }

    void save(iterator_state_type& state, aux_data* data) { } 

    void print(std::ostream& out) const {
      out << "sequence_memory_dataset(N=" << size()
          << ", args=" << this->args << ")";
    }

    // Private data members
    //========================================================================
  private:
    // args and index_map are in the base class
    std::vector<record_type> records_;
    
  }; // class sequence_memory_dataset

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
