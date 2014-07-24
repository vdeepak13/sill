#ifndef SILL_BASIC_RECORD_ITERATORS_HPP
#define SILL_BASIC_RECORD_ITERATORS_HPP

#include <iterator>

namespace sill {
  
  // forward declaration
  template <typename Dataset> class basic_const_record_iterator;

  /**
   * Iterator for datasets over single element type that store the
   * data in an std::vector. Provides mutable access to the 
   * underlying dataset (data is saved on iterator increment).
   *
   * Note that even though the datapoint weight is exposed as
   * a mutable member of the record, it is not saved at increment
   * and thus cannot be modified through this interface. This is
   * an intentional design decision to avoid datasets with shared
   * data modifying each other's weights.
   */
  template <typename Dataset>
  class basic_record_iterator
    : public std::iterator<std::forward_iterator_tag,
                           typename Dataset::record_type> {
    typedef typename Dataset::record_type record_type;

  public:
    basic_record_iterator()
      : dataset(NULL) { }
    
    basic_record_iterator(Dataset* dataset,
                          const std::vector<size_t>& indices)
      : dataset(dataset), row(0), indices(indices), record(indices.size()) {
      load_record();
    }
    
    explicit basic_record_iterator(Dataset* dataset)
      : dataset(dataset), row(dataset->size()) { }
    
    size_t current_row() const {
      return row;
    }

    record_type& operator*() {
      return record;
    }

    record_type* operator->() {
      return &record;
    }

    basic_record_iterator& operator++() {
      assert(row < dataset->size());
      save_record();
      ++row;
      load_record();
      return *this;
    }

    basic_record_iterator operator++(int) {
      // this operation is ridiculously expensive and is not supported
      throw std::logic_error("record iterators do not support postincrement");
    }
    
    bool operator==(const basic_record_iterator& other) const {
      return row == other.row;
    }
    
    bool operator!=(const basic_record_iterator& other) const {
      return row != other.row;
    }
    
    bool operator==(const basic_const_record_iterator<Dataset>& other) const {
      return row == other.row;
    }
    
    bool operator!=(const basic_const_record_iterator<Dataset>& other) const {
      return row != other.row;
    }
    
  private:
    Dataset* dataset;
    size_t row;
    std::vector<size_t> indices;
    record_type record;

    typedef typename Dataset::value_type elem_type;
    
    void load_record() {
      if (row >= dataset->size()) return;
      typename std::vector<elem_type>::const_iterator row_begin =
        dataset->row_begin(row);
      for (size_t i = 0; i < indices.size(); ++i) {
        record.values[i] = row_begin[indices[i]];
      }
      record.weight = dataset->weight(row);
    }

    void save_record() {
      typename std::vector<elem_type>::iterator row_begin =
        dataset->row_begin(row);
      for (size_t i = 0; i < indices.size(); ++i) {
        row_begin[indices[i]] = record.values[i];
      }
    }

    friend class basic_const_record_iterator<Dataset>;

  }; // class basic_record_iterator


  /**
   * Iterator for datasets over single element type that store the
   * data in an std::vector. Provides const access to the 
   * underlying dataset.
   */
  template <typename Dataset>
  class basic_const_record_iterator
    : public std::iterator<std::forward_iterator_tag,
                           const typename Dataset::record_type> {
    typedef typename Dataset::record_type record_type;

  public:
    basic_const_record_iterator()
      : dataset(NULL) { }
    
    basic_const_record_iterator(const Dataset* dataset,
                                const std::vector<size_t>& indices)
      : dataset(dataset), row(0), indices(indices), record(indices.size()) {
      load_record();
    }
    
    explicit basic_const_record_iterator(const Dataset* dataset)
      : dataset(dataset), row(dataset->size()) { }

    // record iterator conversions are expensive, so we make them explicit
    explicit basic_const_record_iterator(const basic_record_iterator<Dataset>& it)
      : dataset(it.dataset), row(it.row), indices(it.indices), record(it.record) { }

    // mutable iterator can be assigned to a const iterator
    basic_const_record_iterator&
    operator=(const basic_record_iterator<Dataset>& it) {
      dataset = it.dataset;
      row = it.row;
      indices = it.indices;
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

    basic_const_record_iterator& operator++() {
      assert(row < dataset->size());
      ++row;
      load_record();
      return *this;
    }

    basic_const_record_iterator operator++(int) {
      // this operation is ridiculously expensive and is not supported
      throw std::logic_error("record iterators do not support postincrement");
    }
    
    bool operator==(const basic_const_record_iterator& other) const {
      return row == other.row;
    }
    
    bool operator!=(const basic_const_record_iterator& other) const {
      return row != other.row;
    }

    bool operator==(const basic_record_iterator<Dataset>& other) const {
      return row == other.row;
    }

    bool operator!=(const basic_record_iterator<Dataset>& other) const {
      return row != other.row;
    }
    
  private:
    const Dataset* dataset;
    size_t row;
    std::vector<size_t> indices;
    record_type record;

    typedef typename Dataset::value_type elem_type;
    
    void load_record() {
      if (row >= dataset->size()) return;
      typename std::vector<elem_type>::const_iterator row_begin =
        dataset->row_begin(row);
      for (size_t i = 0; i < indices.size(); ++i) {
        record.values[i] = row_begin[indices[i]];
      }
      record.weight = dataset->weight(row);
    }

    friend class basic_record_iterator<Dataset>;

  }; // class basic_const_record_iterator

} // namespace sill

#endif
