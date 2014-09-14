#ifndef SILL_SEQUENCE_MEMORY_DATASET_HPP
#define SILL_SEQUENCE_MEMORY_DATASET_HPP

#include <sill/learning/dataset/sequence_dataset.hpp>
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
    typedef slice_view<sequence_dataset<BaseDS> > slice_view_type;

    // Bring the record(row) implementation up to this class
    using sequence_dataset<BaseDS>::record;

    //! Creates an uninitialized dataset
    sequence_memory_dataset() { }

    //! Frees the memory associated with the dataset
    ~sequence_memory_dataset() {
      foreach(record_type& r, records) {
        r.free_memory();
      }
    }
    
    /**
     * Initializes the dataset with the given sequence of processes
     * and pre-allocates memory for the given number of rows.
     * It is an error to call initialize() more than once.
     */
    void initialize(const arg_vector_type& procs, size_t capacity = 1) {
      if (data) {
        throw std::logic_error("Attempt to call initialize() more than once.");
      }
      
      sequence_dataset<BaseDS>::initialize(procs);
      data.reserve(capacity);
      for (size_t i = 0; i < procs.size(); ++i) {
        arg_index[procs[i]] = i;
      }
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
      return records.size();
    }

    size_t capacity() const {
      return records.capacity();
    }

    void reserve(size_t new_capacity) {
      records.reserve(new_capacity);
    }

    record_type record(size_t row, const arg_vector_type& args) const {
      assert(row < records.size());
      record_type result(args);
      std::vector<size_t> indices(args.size());
      for (size_t i = 0; i < args.size(); ++i) {
        indices[i] = safe_get(arg_index, args[i]);
      }
      result.load(records[row], indices);
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
      records.push_back(r);
    }

    //! Inserts the values from an assignment (all variables and indices
    //! must be present).
    void insert(const assignment_type& a, weight_type weight = 1.0) {
      records.push_back(record_type(this->args, a, weight));
    }

    //! Inserts the given number of rows with empty sequences.
    void insert(size_t nrows) {
      /// tood:
    }
    //! Randomly permutes the sequences
    template <typename RandomNumberGenerator>
    void shuffle(RandomNumberGenerator& rng) {
      permute(records, rng);
    }

    // Protected functions
    //========================================================================
    aux_data* init(const arg_vector_type& args,
                   iterator_state_type& state) const {
      state.indices.reserve(args.size());
      for (size_t i = 0; i < args.size() ; ++i) {
        state.indices[i] = safe_get(arg_index, args[i]);
      }
      state.records = &records[0];
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
      return std::min(n, size_t(&records[0] + records.size() - state.records));
    }

    void save(iterator_state_type& state, aux_data* data) { } 

    void print(std::ostream& out) const {
      out << "sequence_memory_dataset(N=" << size() << ", args=" << args << ")";
    }

    // Private data members
    //========================================================================
  private:
    // arg_vector_type args;  // in the base class
    std::map<argument_type*, size_t> arg_index; // the index of each process
    std::vector<record_type> records; // the data in row-major format
    
  }; // class sequence_memory_dataset

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
