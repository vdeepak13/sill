#ifndef SILL_VECTOR_SEQUENCE_RECORD_HPP
#define SILL_VECTOR_SEQUENCE_RECORD_HPP

#include <sill/base/discrete_process.hpp>
#include <sill/base/vector_assignment.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/learning/dataset/raw_record_iterators.hpp>
#include <sill/learning/dataset/simple_process_index_map.hpp>
#include <sill/learning/dataset/symbolic_format.hpp>
#include <sill/learning/dataset/vector_record.hpp>

#include <algorithm>
#include <fstream>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  /**
   * A class that represents a sequence of vector values for a number of
   * discrete processes. By default, the values are not owned by the
   * record; the copies of the record are shallow. Each call to assign()
   * is guaranteed to create a fresh copy of the data. It is the caller's
   * responsibility to make sure that allocated meory is released using
   * using free_memory(); this is typically performed by classes such as
   * sequence_memory_dataset.
   *
   * \tparam T the storage type of the record
   */
  template <typename T = double>
  class vector_sequence_record {
  public:
    /**
     * The type of processes handled by this record.
     */
    typedef discrete_process<vector_variable> process_type;

    /**
     * The type that specifies a sequence of process indices in the record.
     */
    typedef std::vector<size_t> proc_indices_type;

    /**
     * The type that specifies a sequence of variable indices in the record.
     * Each index is a pair (i, t) such that values[i][t] is the value
     * corresponding to the index (i, t).
     */
    typedef std::vector<std::pair<size_t, size_t> > var_indices_type;

    /**
     * The type that specifies a mapping from each process to its index.
     */
    typedef simple_process_index_map<vector_variable> index_map_type;

    /**
     * The type that represents the weight.
     */
    typedef T weight_type;

    /**
     * Creates an empty record.
     */
    vector_sequence_record()
      : num_steps_(0), weight_(0.0) { }

    /**
     * Creates a record with the given vector of processes and weight.
     */
    explicit vector_sequence_record(const std::vector<process_type*>& processes,
                                    T weight = 1.0) {
      initialize(processes, weight);
    }

    /**
     * Initializes the record to the given vector of processes and weight.
     */
    void initialize(const std::vector<process_type*>& processes,
                    T weight = 1.0) {
      assert(values_.empty());
      processes_.reset(new process_type*[processes.size()]);
      std::copy(processes.begin(), processes.end(), processes_.get());
      values_.assign(processes.size(), NULL);
      num_steps_ = 0;
      num_dims_ = sill::vector_size(processes);
      weight_ = weight;
    }

    /**
     * Releases the memory associated with the record. The memory must be
     * owned by the record.
     */
    void free_memory() {
      if (!values_.empty() && values_.front()) {
        delete[] values_.front();
        std::fill(values_.begin(), values_.end(), (T*)NULL);
      }
    }

    /**
     * Returns the number of processes in this record.
     */
    size_t num_processes() const {
      return values_.size();
    }

    /**
     * Returns the number of columns (sum of vector sizes).
     */
    size_t num_dims() const {
      return num_dims_;
    }

    /**
     * Returns the number of steps in this record.
     */
    size_t num_steps() const {
      return num_steps_;
    }

    /**
     * Returns the total number of values in this record.
     */
    size_t size() const {
      return num_dims_ * num_steps_;
    }

    /**
     * Returns the pointer to the array of processes in the record.
     */
    process_type** processes() const {
      return processes_.get();
    }
    
    /**
     * Returns the pointer to the first value for the given process and time.
     */
    const T* value_ptr(size_t proc, size_t time) const {
      return values_[proc] + time;
    }

    /**
     * Returns the record weight.
     */
    const T& weight() const {
      return weight_;
    }

    /**
     * Checks if the record is compatible with the given argument vector.
     * \throw invalid_argument exception if the arguments do not match
     */
    void check_compatible(const std::vector<process_type*>& procs) const {
      if (procs.size() != num_processes() ||
          !std::equal(procs.begin(), procs.end(), processes_.get())) {
        throw std::invalid_argument("Record incompatible with given processes.");
      }
    }

    /**
     * Extracts the entire time series into an assignment. The assignment
     * will contain a variable for each process and for each time step in
     * the sequence.
     */
    void extract(vector_assignment& a) const {
      for (size_t i = 0; i < values_.size(); ++i) {
        size_t len = processes_[i]->size();
        for (size_t t = 0; t < num_steps_; ++t) {
          copy(values_[i] + t, len, a[processes_[i]->at(t)]); // private function
        }
      }
    }

    /**
     * Extracts a single time point into an assignment. The variables are
     * indexed with "current" time index.
     */
    void extract(size_t t, vector_assignment& a) const {
      assert(t < num_steps_);
      for (size_t i = 0; i < values_.size(); ++i) {
        size_t len = processes_[i]->size();
        copy(values_[i] + t, len, a[processes_[i]->current()]); // private function
      }
    }

    /**
     * Extracts a single time step into a vector.
     */
    void extract(size_t t, arma::Col<T>& values) const {
      assert(t < num_steps_);
      values.set_size(num_dims_);
      T* dest = values.begin();
      for (size_t i = 0; i < values_.size(); ++i) {
        size_t len = processes_[i]->size();
        copy(values_[i] + t, len, dest); // private function
        dest += len;
      }
    }

    /**
     * Extracts a closed range of time steps into a vector.
     */
    void extract(size_t t1, size_t t2, arma::Col<T>& values) const {
      assert(t1 <= t2 && t2 < num_steps());
      values.set_size(num_dims_ * (t2-t1+1));
      T* dest = values.begin();
      for (size_t t = t1; t <= t2; ++t) {
        for (size_t i = 0; i < values_.size(); ++i) {
          size_t len = processes_[i]->size();
          copy(values_[i] + t, len, dest); // private function
          dest += len;
        }
      }
    }

    /**
     * Extracts the values for the variables with given indices.
     */
    void extract(const var_indices_type& indices, arma::Col<T>& values) const {
      values.set_size(vector_size(indices)); // vector_size is a private function
      for (size_t i = 0, dest = 0; i < indices.size(); ++i) {
        std::pair<size_t,size_t> index = indices[i];
        size_t len = processes_[index.first]->size();
        copy(values_[index.first] + index.second, len, &values[dest]); // private fn
        dest += len;
      }
    }

    /**
     * Extracts the weight and values for the variables with given indices.
     */
    void extract(const var_indices_type& indices, vector_record<T>& r) const {
      r.weight = weight_;
      extract(indices, r.values);
    }

    /**
     * Extracts the weight and pointers to the variables with the given indices.
     */
    void extract(const var_indices_type& indices,
                 raw_record_iterator_state<vector_record<T> >& state) const {
      state.elems.resize(vector_size(indices)); // vector_size is a private fn
      for (size_t i = 0, dest = 0; i < indices.size(); ++i) {
        std::pair<size_t,size_t> index = indices[i];
        size_t len = processes_[index.first]->size();
        T* ptr = values_[index.first] + index.second;
        for (size_t j = 0; j < len; ++j) {
          state.elems[dest++] = ptr;
          ptr += num_steps_;
        }
      }
      state.weights = &weight_;
      state.e_step.assign(state.elems.size(), 1);
      state.w_step = 0;
    }

    /**
     * Assigns data to this record. The data is stored in process-major order,
     * that is, first the entire sequence for the first process, then for the
     * second process, etc. The size of the values vector must be divisible by
     * the number of processes. The values are copied into a freshly allocated
     * block of memory.
     */
    void assign(const std::vector<T>& values, T weight) {
      if (values.empty()) {
        assign(NULL, 0, weight);
      } else {
        assert(num_dims_ != 0 && values.size() % num_dims_ == 0);
        T* data = new T[values.size()];
        std::copy(values.begin(), values.end(), data);
        assign(data, values.size() / num_dims_, weight);
      }
    }

    /**
     * Allocates memory for the given number of steps and sets the weight.
     * Does not initialize the values.
     */
    void assign(size_t num_steps, T weight) {
      assign(new T[num_dims_ * num_steps], num_steps, weight);
    }

    /**
     * Assigns data to this record. The data is stored in process-major order,
     * that is, first the entire sequence for the first process, then for the
     * second process, etc. The value count must be divisible by the number
     * of processes. The values pointer becomes owned by this record (i.e.,
     * the data is not copied).
     */
    void assign(T* data, size_t num_steps, T weight) {
      num_steps_ = num_steps;
      weight_ = weight;
      for (size_t i = 0; i < num_processes(); ++i) {
        values_[i] = data;
        data += num_steps_ * processes_[i]->size();
      }
      if (num_processes() == 0 && data) {
        delete[] data;
      }
    }

    /**
     * Assigns data to this record. The given assignment must contain variables
     * for all the processes in the record and, if max_t is the maximum absolute
     * time index, all the <= maxt.
     */
    void assign(const vector_assignment& a, T weight) {
      int maxt = -1;
      foreach(const vector_assignment::value_type& p, a) {
        int t = boost::any_cast<int>(p.first->index());
        assert(t >= 0);
        maxt = std::max(maxt, t);
      }
      if (maxt >= 0) {
        assign(new T[(maxt + 1) * num_dims_], maxt + 1, weight);
        for(size_t i = 0; i < num_processes(); ++i) {
          size_t len = processes_[i]->size();
          for(int t = 0; t <= maxt; ++t) {
            copy(safe_get(a, processes_[i]->at(t)), len, values_[i] + t);
          }
        }
      } else {
        assign(NULL, 0, weight);
      }
    }

    /**
     * Sets the values for the given time step.
     */
    void set(size_t t, const arma::Col<T>& values) {
      assert(values.size() == num_dims_);
      const T* src = values.begin();
      for (size_t i = 0; i < num_processes(); ++i) {
        size_t len = processes_[i]->size();
        T* dest = values_[i] + t;
        for (size_t k = 0; k < len; ++k) {
          *dest = *src++;
          dest += num_steps_;
        }
      }
    }

    /**
     * Loads the data from another record. The records will share the storage.
     * \param other the record to load the data from
     * \param indices the subset of process indices to copy
     */
    void load(const vector_sequence_record& other,
              const proc_indices_type& indices) {
      assert(num_processes() == indices.size());
      num_steps_ = other.num_steps_;
      weight_ = other.weight_;
      for (size_t i = 0; i < indices.size(); ++i) {
        values_[i] = other.values_[indices[i]];
      }
    }

  private:
    void copy(const T* ptr, size_t len, arma::Col<T>& dest) const {
      dest.set_size(len);
      for (size_t i = 0; i < len; ++i) {
        dest[i] = *ptr;
        ptr += num_steps_;
      }
    }

    void copy(const arma::Col<T>& src, size_t len, T* dest) const {
      assert(src.size() == len);
      for (size_t i = 0; i < len; ++i) {
        *dest = src[i];
        dest += num_steps_;
      }
    }
    
    void copy(const T* ptr, size_t len, T* dest) const {
      for (size_t i = 0; i < len; ++i) {
        dest[i] = *ptr;
        ptr += num_steps_;
      }
    }

    size_t vector_size(const var_indices_type& indices) const {
      size_t len = 0;
      for (size_t i = 0; i < indices.size(); ++i) {
        len += processes_[indices[i].first]->size();
      }
      return len;
    }

    boost::shared_ptr<process_type*[]> processes_;
    std::vector<T*> values_;
    size_t num_steps_;
    size_t num_dims_;
    T weight_;

  }; // class vector_sequence_record


  // Free functions
  //========================================================================

  /**
   * Prints the record to an output stream.
   * \relates vector_sequence_record
   */
  template <typename T>
  std::ostream&
  operator<<(std::ostream& out, const vector_sequence_record<T>& r) {
    out << "[" << r.num_steps() << " x " << r.num_dims() << "]" << std::endl;
    for (size_t t = 0; t < r.num_steps(); ++t) {
      for (size_t i = 0; i < r.num_processes(); ++i) {
        size_t len = r.processes()[i]->size();
        const T* ptr = r.value_ptr(i, t);
        for (size_t j = 0; j < len; ++j) {
          out << "\t" << *ptr;
          ptr += r.num_steps();
        }
      }
      out << std::endl;
    }
    return out;
  }

  /**
   * Loads a single sequence in a tabular format.
   * \relates vector_sequence_record
   */
  template <typename T>
  void load_tabular(const std::string& filename,
                    const symbolic_format& format,
                    vector_sequence_record<T>& record) {
    if (!format.is_vector_discrete()) {
      throw std::domain_error("The format contains process(es) that are not vector");
    }
    if (record.num_processes() != format.discrete_procs.size()) {
      throw std::logic_error("The record and format must have the same processes.");
    }

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }
    
    // read the table
    size_t num_dims = record.num_dims();
    std::vector<std::vector<T> > values(num_dims);
    std::string line;
    size_t line_number = 0;
    while (std::getline(in, line)) {
      std::vector<const char*> tokens;
      if (format.parse(num_dims, line, line_number, tokens)) {
        for (size_t i = 0; i < num_dims; ++i) {
          values[i].push_back(parse_string<T>(tokens[i + format.skip_cols]));
        }
      }
    }

    // concatenate the values and store them in the record
    size_t num_steps = values[0].size();
    T* data = new T[num_dims * num_steps];
    for (size_t i = 0; i < num_dims; ++i) {
      std::copy(values[i].begin(), values[i].end(), data + i * num_steps);
    }
    record.assign(data, num_steps, 1.0);
  }

  /**
   * Saves a single sequence in a tabular format.
   * \relates vector_sequence_record
   */
  template <typename T>
  void save_tabular(const std::string& filename,
                    const symbolic_format& format,
                    const vector_sequence_record<T>& record) {
    if (!format.is_vector_discrete()) {
      throw std::domain_error("The format contains process(es) that are not vector");
    }
    if (record.num_processes() != format.discrete_procs.size()) {
      throw std::logic_error("The record and format must have the same processes.");
    }

    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    for (size_t i = 0; i < format.skip_rows; ++i) {
      out << std::endl;
    }

    size_t num_processes = record.num_processes();
    size_t num_steps = record.num_steps();
    std::string separator = format.separator.empty() ? " " : format.separator;
    for (size_t t = 0; t < record.num_steps(); ++t) {
      for (size_t i = 0; i < format.skip_cols; ++i) {
        out << "0" << separator;
      }
      for (size_t i = 0; i < num_processes; ++i) {
        const T* ptr = record.value_ptr(i, t);
        size_t len = record.processes()[i]->size();
        for (size_t j = 0; j < len; ++j) {
          if (i || j) { out << separator; }
          out << *ptr;
          ptr += num_steps;
        }
      }
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
