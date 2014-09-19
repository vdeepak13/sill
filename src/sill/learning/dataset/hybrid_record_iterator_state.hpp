#ifndef SILL_HYBRID_RECORD_ITERATOR_STATE_HPP
#define SILL_HYBRID_RECORD_ITERATOR_STATE_HPP

#include <sill/learning/dataset/finite_record.hpp>
#include <sill/learning/dataset/vector_record.hpp>
#include <sill/learning/dataset/raw_record_iterators.hpp>

namespace sill {

  template <typename T>
  struct hybrid_record_iterator_state {
    typedef raw_record_iterator_state<finite_record>     finite_state_type;
    typedef raw_record_iterator_state<vector_record<T> > vector_state_type;

    finite_state_type* finite;
    vector_state_type* vector;
    explicit hybrid_record_iterator_state(finite_state_type* finite)
      : finite(finite), vector(NULL) { }

    explicit hybrid_record_iterator_state(vector_state_type* vector)
      : finite(NULL), vector(vector) { }
    
    hybrid_record_iterator_state(finite_state_type* finite,
                                 vector_state_type* vector)
      : finite(finite), vector(vector) { }

  }; // struct hybrid_record_iterator_state

} // namespace sill

#endif

