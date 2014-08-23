#ifndef SILL_RAW_RECORD_ITERATORS_HPP
#define SILL_RAW_RECORD_ITERATORS_HPP

#include <sill/learning/dataset/aux_data.hpp>

#include <iterator>

#include <boost/shared_ptr.hpp>

namespace sill {

  /**
   * Datastructure used internally by raw_record_iterator and
   * raw_const_record_iterator to store most of the iterator state
   * visible to the dataset.
   */
  template <typename Dataset>
  struct raw_record_iterator_state {
    typedef typename Dataset::elem_type   elem_type;
    typedef typename Dataset::weight_type weight_type;

    std::vector<elem_type*> elems;   // the pointers to the next values
    const weight_type*      weights; // the pointer to the next weight
    std::vector<size_t>     e_step;  // the step size for each column
    size_t                  w_step;  // the step size for the weight
  };
  
  // =======================================================================

  // forward declaration
  template <typename Dataset> class raw_const_record_iterator;

  /**
   * Iterator for dense datasets over single element type that
   * store the data in chunks. Provides mutable access to the 
   * underlying dataset (data is saved on iterator increment).
   *
   * Note that even though the datapoint weight is exposed as
   * a mutable member of the record, it is not saved at increment
   * and thus cannot be modified through this interface. This is
   * an intentional design decision to avoid datasets with shared
   * data modifying each other's weights.
   *
   * \todo should this be std::input_iterator_tag? swap?
   */
  template <typename Dataset>
  class raw_record_iterator
    : public std::iterator<std::forward_iterator_tag,
                           typename Dataset::record_type> {
  public:
    typedef typename Dataset::vector_type      vector_type;
    typedef typename Dataset::record_type      record_type;
    typedef raw_record_iterator_state<Dataset> state_type;

    // singular and past-the-end constructor
    raw_record_iterator(size_t endrow = 0)
      : dataset(NULL), row(endrow) { }

    // begin constructor
    raw_record_iterator(Dataset* dataset, const vector_type& args)
      : dataset(dataset), row(0), rows_left(0) {
      aux.reset(dataset->init(args, state));
      record.resize(state.elems.size());
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

    raw_record_iterator& operator++() {
      ++row;
      save_record();
      load_advance();
      return *this;
    }

    raw_record_iterator operator++(int) {
      // this operation is too expensive and is not supported
      throw std::logic_error("record iterators do not support postincrement");
    }
    
    bool operator==(const raw_record_iterator& other) const {
      return row == other.row;
    }
    
    bool operator!=(const raw_record_iterator& other) const {
      return row != other.row;
    }
    
    bool operator==(const raw_const_record_iterator<Dataset>& other) const {
      return row == other.row;
    }
    
    bool operator!=(const raw_const_record_iterator<Dataset>& other) const {
      return row != other.row;
    }
    
  private:
    Dataset* dataset;                // the underlying dataset
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
      for (size_t i = 0; i < state.elems.size(); ++i) {
        record.values[i] = *state.elems[i];
        state.elems[i] += state.e_step[i];
      }
      record.weight = *state.weights;
      state.weights += state.w_step;
      --rows_left;
    }

    // to maintain compatibility with raw_const_record_iterator,
    // we advance in load_advance() and then go back in save_record()
    void save_record() {
      for (size_t i = 0; i < state.elems.size(); ++i) {
        *(state.elems[i] - state.e_step[i]) = record.values[i];
      }
    }

    friend class raw_const_record_iterator<Dataset>;

  }; // class raw_record_iterator

  // =======================================================================

  /**
   * Iterator for datasets over single element type that store the
   * data in an std::vector. Provides const access to the 
   * underlying dataset.
   */
  template <typename Dataset>
  class raw_const_record_iterator
    : public std::iterator<std::forward_iterator_tag,
                           const typename Dataset::record_type> {
  public:
    typedef typename Dataset::vector_type      vector_type;
    typedef typename Dataset::record_type      record_type;
    typedef raw_record_iterator_state<Dataset> state_type;

    // default and end constructor
    raw_const_record_iterator(size_t endrow = 0)
      : dataset(NULL), row(endrow) { }

    // begin constructor
    raw_const_record_iterator(const Dataset* dataset, const vector_type& args)
      : dataset(dataset), row(0), rows_left(0) {
      aux.reset(dataset->init(args, state));
      record.resize(state.elems.size());
      load_advance();
    }

    // record iterator conversions are expensive, so we make them explicit
    explicit raw_const_record_iterator(const raw_record_iterator<Dataset>& it)
      : dataset(it.dataset),
        row(it.row),
        rows_left(it.rows_left),
        state(it.state),
        aux(it.aux),
        record(it.record) { }

    // mutable iterator can be assigned to a const iterator
    raw_const_record_iterator&
    operator=(const raw_record_iterator<Dataset>& it) {
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
    
    const record_type& operator*() const {
      return record;
    }

    const record_type* operator->() const {
      return &record;
    }

    raw_const_record_iterator& operator++() {
      ++row;
      load_advance();
      return *this;
    }

    raw_const_record_iterator operator++(int) {
      // this operation is ridiculously expensive and is not supported
      throw std::logic_error("record iterators do not support postincrement");
    }
    
    bool operator==(const raw_const_record_iterator& other) const {
      return row == other.row;
    }
    
    bool operator!=(const raw_const_record_iterator& other) const {
      return row != other.row;
    }

    bool operator==(const raw_record_iterator<Dataset>& other) const {
      return row == other.row;
    }

    bool operator!=(const raw_record_iterator<Dataset>& other) const {
      return row != other.row;
    }
    
  private:
    const Dataset* dataset;          // the underlying dataset
    size_t row;                      // the logical row in the dataset
    size_t rows_left;                // 0 indicates we need to fetch more data
    state_type state;                // the iterator state visible to the dataset
    boost::shared_ptr<aux_data> aux; // auxiliary data used for fetching
    record_type record;              // user-facing data

    void load_advance() {
      if (rows_left == 0) {
        rows_left = dataset->load(size_t(-1), state, aux.get());
        if (rows_left == 0) return; // no more data left
      }
      for (size_t i = 0; i < state.elems.size(); ++i) {
        record.values[i] = *state.elems[i];
        state.elems[i] += state.e_step[i];
      }
      record.weight = *state.weights;
      state.weights += state.w_step;
      --rows_left;
    }

    friend class raw_record_iterator<Dataset>;

  }; // class raw_const_record_iterator

} // namespace sill

#endif
