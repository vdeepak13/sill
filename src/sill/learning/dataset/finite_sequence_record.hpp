#ifndef SILL_FINITE_SEQUENCE_RECORD_HPP
#define SILL_FINITE_SEQUENCE_RECORD_HPP

#include <sill/base/discrete_process.hpp>

namespace sill {

  struct finite_sequence_record {
    typedef discrete_process<finite_variable> process_type;

    std::vector<process_type*> processes;
    std::vector<size_t*> values;
    size_t num_steps;
    double weight;

    finite_sequence_record()
      : weight(0.0) { }

    finite_sequence_record(const std::vector<process_type*>& processes,
                           size_t num_steps,
                           double weight = 0.0)
      : processes(processes), num_steps(num_steps), weight(weight) { }

    /**
     * Frees the memory associated with this record. Do not call this
     * function explicitly unless you own the memory pointed to 
     * (e.g., in sequence_memory_dataset).
     */
    void free_memory() {
      if (!values.empty()) {
        delete values[0];
        values.clear();
      }
    }

    /**
     * Extracts the entire time series into an assignment. The assignment
     * will contain a variable for each process, for each time step in the
     * sequence.
     */
    void extract(finite_assignment& a) {
      for (size_t i = 0; i < processes.size(); ++i) {
        for (size_t t = 0; t < num_steps; ++t) {
          a[processes[i].at(t)] = values[i][t];
        }
      }
    }

    /**
     * Extracts the a single time point
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

  }; // struct finite_sequence_record

} // namespace sill

#endif
