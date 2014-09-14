#ifndef SILL_VECTOR_SEQUENCE_RECORD_HPP
#define SILL_VECTOR_SEQUENCE_RECORD_HPP

#include <sill/base/discrete_process.hpp>

#include <armadillo>

namespace sill {

  template <typename T = double>
  struct vector_sequence_record {
    typedef discrete_process<finite_variable> process_type;

    std::vector<process_type*> processes;
    std::vector<T*> values;
    size_t num_steps;
    T weight;

    vector_sequence_record()
      : weight(0.0) { }

    vector_sequence_record(const std::vector<process_type*>& processes,
                           size_t num_steps,
                           T weight = 0.0)
      : processes(processes), num_steps(num_steps), weight(weight) { }

    // extracts the entire time series
    void extract(assignment_type& a) {
      for (size_t i = 0; i < processes.size(); ++i) {
        for (size_t t = 0; t < num_steps; ++t) {
          something::extract(processes[i].at(t), values[i], t, a);
        }
      }
    }

    // extracts a single time point
    void extract(size_t time, assignment_type& a) {
      for (size_t i = 0; i < processes.size(); ++i) {
        something::extract(processes[i].current(), values[i], time, a);
      }
    }

    // adds the data for a sequence
    template <typename T>
    void set(size_t i, const std::vector<T>& sequence) {
      T* copy = new T[sequence.size()];
      std::copy(sequence.begin(), sequence.end(), copy);
      values[i] = copy;
    }

  }; // struct vector_sequence_record

} // namespace sill

#endif
