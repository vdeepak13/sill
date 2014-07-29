#ifndef SILL_BASIC_SAMPLE_ITERATOR_HPP
#define SILL_BASIC_SAMPLE_ITERATOR_HPP

#include <iterator>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>

namespace sill {

  /**
   * Iterator that randomly draws records from datasets that store their data
   * in an std::vector.
   */
  template <typename Dataset>
  class basic_sample_iterator
    : public std::iterator<std::forward_iterator_tag,
                           const typename Dataset::record_type> {
    typedef typename Dataset::record_type record_type;

  public:
    basic_sample_iterator()
      : dataset(NULL) { }
    
    basic_sample_iterator(const Dataset* dataset,
                          const std::vector<size_t>& indices,
                          unsigned seed)
      : dataset(dataset),
        indices(indices),
        rng(seed),
        unif(0, dataset->size() - 1),
        record(indices.size()) {
      assert(dataset->size() > 0);
      row = unif(rng);
      load_record();
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

    basic_sample_iterator& operator++() {
      row = unif(rng);
      load_record();
      return *this;
    }

    basic_sample_iterator operator++(int) {
      // this operation is ridiculously expensive and is not supported
      throw std::logic_error("sample iterators do not support postincrement");
    }
    
    bool operator==(const basic_sample_iterator& other) const {
      return row == other.row;
    }
    
    bool operator!=(const basic_sample_iterator& other) const {
      return row != other.row;
    }

  private:
    const Dataset* dataset;
    std::vector<size_t> indices;
    boost::mt19937 rng;
    boost::random::uniform_int_distribution<size_t> unif;
    size_t row;
    record_type record;

    typedef typename Dataset::value_type elem_type;
    
    void load_record() {
      typename std::vector<elem_type>::const_iterator row_begin =
        dataset->row_begin(row);
      for (size_t i = 0; i < indices.size(); ++i) {
        record.values[i] = row_begin[indices[i]];
      }
      record.weight = dataset->weight(row);
    }

  }; // class basic_sample_iterator

} // namespace sill

#endif
