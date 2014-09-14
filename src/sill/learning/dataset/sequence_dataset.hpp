#ifndef SILL_SEQUENCE_DATASET_HPP
#define SILL_SEQUENCE_DATASET_HPP

#include <sill/base/discrete_process.hp>

#include <boost/random/uniform_int.hpp>

namespace sill {
  
  /**
   * A dataset where each datapoint is a sequence of random variables.
   * \tparam BaseDS Base (static) dataset that determines the variable type
   *         and storage (see finite_dataset, vector_dataset, or hybrid_dataset)
   */
  template <typename BaseDS>
  class sequence_dataset {

    typedef typename BaseDS::argument_type        variable_type;
    typedef discrete_process<variable_type>       argument_type;
    typedef std::set<argument_type*>              domain_type;
    typedef std::vector<argument_type*>           arg_vector_type;
    typedef typename BaseDS::assignment_type      assignment_type;
    typedef typename BaseDS::sequence_record_type record_type;
    
    class record_iterator;
    class const_record_iterator;

    //! Default constructor
    sequence_dataset() { }

    //! Returns the number of datapoints in the dataset.
    virtual size_t size() const = 0;

    //! Returns true if the dataset has no datapoints.
    bool empty() const { return size() == 0; }

    //! Returns the columns of this dataset.
    const domain_type arguments() const { return make_domain(args); }

    //! Returns the columns of this dataset.
    const arg_vector_type& arg_vector() const { return args; }

    //! Returns a single data point in the dataset's natural ordering.
    record_type record(size_t row) const { return record(row, args); }

    //! Returns a single data point for a subset of the processes.
    virtual record_type
    record(size_t row, const arg_vector_type& args) const = 0;

    //! Returns mutable records for the specified processes.
    std::pair<record_iterator, record_iterator>
    records(const arg_vector_type& args) {
      return std::make_pair(record_iterator(this, args),
                            record_iterator(size()));
    }

    //! Returns immutable records for the specified processes.
    std::pair<const_record_iterator, const_record_iterator>
    records(const arg_vector_type& args) {
      return std::make_pair(const_record_iterator(this, args),
                            const_record_iterator(size()));
    }

    //! Returns a sliding view that exposes subsequences of each datapoint
    sliding_view<BaseDS>
    sliding(size_t history) const {
      return sliding_view<BaseDS>(this, history);
    }

    //! Returns a fixed view that exposes a fixed subsequence of each datapoint
    fixed_view<BaseDS>
    fixed(size_t start, size_t stop = start + 1) const {
      return fixed_view<BaseDS>(this, start, stop);
    }
    
    //! Draws a random sample from this dataset.
    template <typename RandomNumberGenerator>
    record_type sample(const arg_vector_type& args,
                       RandomNumberGenerator& rng) const {
      assert(!empty());
      boost::uniform_int<size_t> uniform(0, size() - 1);
      return record(uniform(rng), vars);
    }

    // Utility functions, invoked by the iterators and subclasses
    //========================================================================
  protected:
    //! datastructure used internally by the iterators to store
    //! to store the iterator state visible to the dataset
    struct iterator_state_type {
      std::vector<size_t> indices; // the processes to extract
      record_type*        records; // the pointer to the next record
    };
    
    //! initializes the data structures in the record iterator
    virtual aux_data* init(const arg_vector_type& args,
                           iterator_state_type& state) const = 0;

    //! advances the internal pointer in data by the given difference
    virtual void advance(ptrdiff_t diff,
                         iterator_state_type& state,
                         aux_data* data) const = 0;

    //! loads at most n rows
    virtual size_t load(size_t n,
                        iterator_state_type& state,
                        aux_data* data) const = 0;

    //! saves the previously loaded data
    virtual void save(iterator_state_type& state, aux_data* data) = 0;

    //! prints the summary of this dataset to a stream
    virtual void print(std::ostream& out) const = 0;

    //! initializes the variables in this dataset
    void initialize(const arg_vector_type& vars) { args = vars; }

    //! The variables in the dataset's internal ordering of columns.
    arg_vector_type args;

  public:
    //! iterator over sequence records with mutable access
    class record_iterator
      : public std::iterator<std::forward_iterator_tag, record_type> {
    public:
      // singular and past-the-end constructor
      explicit record_iterator(size_t endrow = 0)
        : dataset(NULL), row(endrow) { }

      // begin constructor
      record_iterator(sequence_dataset* dataset,
                      const arg_vector_type& args)
        : dataset(dataset), row(0), rows_left(0) {
        aux.reset(dataset->init(args, state));
        load_advance();
      }

      size_t current_row() const {
        return row;
      }

      record_type& operator*() {
        return record;
      }

      record_type* operator->() {
        return &record;
      }

      record_iterator& operator++() {
        ++row;
        load_advance();
        return *this;
      }

      record_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("record iterators do not support postincrement");
      }
      
      bool operator==(const record_iterator& other) const {
        return row == other.row;
      }
    
      bool operator!=(const record_iterator& other) const {
        return row != other.row;
      }
      
      bool operator==(const const_record_iterator& other) const {
        return row == other.row;
      }
      
      bool operator!=(const const_record_iterator& other) const {
        return row != other.row;
      }

    private:
      sequence_dataset* dataset;       // the underlying dataset
      size_t row;                      // the logical row in the dataset
      size_t rows_left;                // 0 indicates we need to fetch more data
      state_type state;                // the iterator state visible to the dataset
      boost::shared_ptr<aux_data> aux; // auxiliary data used by the dataset
      record_type record;              // user-facing data
      
      void load_advance() {
        if (rows_left == 0) {
          dataset->save(state, aux.get());
          rows_left = dataset->load(size_t(-1), state, aux.get());
          if (rows_left == 0) return; // no more data left
        }
        record.load(*state.records, state.indices);
        ++state.records;
        --rows_left;
      }

      friend class const_record_iterator;

    }; // class record_iterator

    //! iterator over sequence records with const access
    class const_record_iterator
      : public std::iterator<std::forward_iterator_tag, record_type> {
    public:
      // singular and past-the-end constructor
      explicit const_record_iterator(size_t endrow = 0)
        : dataset(NULL), row(endrow) { }

      // begin constructor
      const_record_iterator(sequence_dataset* dataset,
                            const arg_vector_type& args)
        : dataset(dataset), row(0), rows_left(0) {
        aux.reset(dataset->init(args, state));
        load_advance();
      }

      // record iterator conversions are expensive, so we make them explicit
      explicit const_record_iterator(const recorrd_iterator& it)
        : datset(it.dataset),
          row(it.row),
          rows_left(it.rows_left),
          state(it.state),
          aux(it.aux),
          record(it.record) { }

      // mutable iterator can be assigned to a const iterator
      const_record_iterator& operator=(const record_iterator& it) {
        dataset = it.dataset;
        row = it.row;
        rows_left = it.rows_left;
        state = it.state;
        aux = it.aux;
        record = it.record;
        return *this;
      }

      size_t current_row() const {
        return row;
      }

      const record_type& operator*() {
        return record;
      }

      const record_type* operator->() {
        return &record;
      }

      const_record_iterator& operator++() {
        ++row;
        load_advance();
        return *this;
      }

      const_record_iterator operator++(int) {
        // this operation is too expensive and is not supported
        throw std::logic_error("record iterators do not support postincrement");
      }
      
      bool operator==(const const_record_iterator& other) const {
        return row == other.row;
      }
      
      bool operator!=(const const_record_iterator& other) const {
        return row != other.row;
      }

      bool operator==(const record_iterator& other) const {
        return row == other.row;
      }
    
      bool operator!=(const record_iterator& other) const {
        return row != other.row;
      }
      
    private:
      const sequence_dataset* dataset; // the underlying dataset
      size_t row;                      // the logical row in the dataset
      size_t rows_left;                // 0 indicates we need to fetch more data
      state_type state;                // the iterator state visible to the dataset
      boost::shared_ptr<aux_data> aux; // auxiliary data used by the dataset
      record_type record;              // user-facing data
      
      void load_advance() {
        if (rows_left == 0) {
          rows_left = dataset->load(size_t(-1), state, aux.get());
          if (rows_left == 0) return; // no more data left
        }
        record.load(*state.records, state.indices);
        ++state.records;
        --rows_left;
      }

      friend class record_iterator;

    }; // class const_record_iterator

  }; // classs sequence_dataset

} // namespace sill

#endif
