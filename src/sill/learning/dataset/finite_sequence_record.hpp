#ifndef SILL_FINITE_SEQUENCE_RECORD_HPP
#define SILL_FINITE_SEQUENCE_RECORD_HPP

#include <sill/base/finite_assignment.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/discrete_process.hpp>
#include <sill/learning/dataset/finite_record.hpp>
#include <sill/learning/dataset/raw_record_iterators.hpp>
#include <sill/learning/dataset/simple_process_index_map.hpp>
#include <sill/learning/dataset/symbolic_format.hpp>

#include <algorithm>
#include <fstream>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that represents a sequence of finite values for a number of
   * discrete processes. By default, the values are not owned by the
   * record; the copies of the record are shallow. Each call to assign()
   * is guaranteed to create a fresh copy of the data. It is the caller's
   * responsibility to make sure that allocated meory is released using
   * using free_memory(); this is typically performed by classes such as
   * sequence_memory_dataset.
   */
  class finite_sequence_record {
  public:
    /**
     * The type of processes handled by this record.
     */
    typedef discrete_process<finite_variable> process_type;

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
    typedef simple_process_index_map<finite_variable> index_map_type;

    /**
     * The type that represents the weight.
     */
    typedef double weight_type;

    /**
     * Creates an empty record.
     */
    finite_sequence_record()
      : num_steps_(0), weight_(0.0) { }

    /**
     * Creates a record with the given vector of processes and weight.
     */
    explicit finite_sequence_record(const std::vector<process_type*>& processes,
                                    double weight = 1.0) {
      initialize(processes, weight);
    }

    /**
     * Initializes the record to the given vector of processes and weight.
     */
    void initialize(const std::vector<process_type*>& processes,
                    double weight = 1.0) {
      assert(values_.empty());
      processes_.reset(new process_type*[processes.size()]);
      std::copy(processes.begin(), processes.end(), processes_.get());
      values_.assign(processes.size(), NULL);
      num_steps_ = 0;
      weight_ = weight;
    }

    /**
     * Releases the memory associated with the record. The memory must be
     * owned by the record.
     */
    void free_memory() {
      if (!values_.empty() && values_.front()) {
        delete[] values_.front();
        std::fill(values_.begin(), values_.end(), (size_t*)NULL);
      }
    }

    /**
     * Returns the number of processes in this record.
     */
    size_t num_processes() const {
      return values_.size();
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
      return values_.size() * num_steps_;
    }

    /**
     * Returns the pointer to the array of processes in the record.
     */
    process_type** processes() const {
      return processes_.get();
    }
    
    /**
     * Returns the value.
     */
    const size_t& operator()(size_t proc, size_t time) const {
      return values_[proc][time];
    }

    /**
     * Returns the record weight.
     */
    const double& weight() const {
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
     * will contain a variable for each process, for each time step in the
     * sequence.
     */
    void extract(finite_assignment& a) const {
      for (size_t i = 0; i < values_.size(); ++i) {
        for (size_t t = 0; t < num_steps_; ++t) {
          a[processes_[i]->at(t)] = values_[i][t];
        }
      }
    }

    /**
     * Extracts a single time point into an assignment. The variables are
     * indexed with "current" time index.
     */
    void extract(size_t t, finite_assignment& a) const {
      assert(t < num_steps_);
      for (size_t i = 0; i < values_.size(); ++i) {
        a[processes_[i]->current()] = values_[i][t];
      }
    }

    /**
     * Extracts a single time step into a vector of values.
     */
    void extract(size_t t, std::vector<size_t>& values) const {
      assert(t < num_steps_);
      values.resize(num_processes());
      for (size_t i = 0; i < values_.size(); ++i) {
        values[i] = values_[i][t];
      }
    }

    /**
     * Extracts a closed range of time steps into a vector of values.
     */
    void extract(size_t t1, size_t t2, std::vector<size_t>& values) const {
      assert(t1 <= t2 && t2 < num_steps());
      values.resize(num_processes() * (t2-t1+1));
      size_t* dest = &values[0];
      for (size_t t = t1; t <= t2; ++t) {
        for (size_t i = 0; i < values_.size(); ++i) {
          *dest++ = values_[i][t];
        }
      }
    }

    /**
     * Extracts the values for the variables with given indices.
     */
    void extract(const var_indices_type& indices,
                 std::vector<size_t>& values) const {
      values.resize(indices.size());
      for (size_t i = 0; i < indices.size(); ++i) {
        std::pair<size_t, size_t> index = indices[i];
        values[i] = values_[index.first][index.second];
      }
    }

    /**
     * Extracts the weight and values for the variables with given indices.
     */
    void extract(const var_indices_type& indices, finite_record& r) const {
      r.weight = weight_;
      extract(indices, r.values);
    }

    /**
     * Extracts the weight and pointers to the variables with the given indices.
     */
    void extract(const var_indices_type& indices,
                 raw_record_iterator_state<finite_record>& state) const {
      state.elems.resize(indices.size());
      for (size_t i = 0; i < indices.size(); ++i) {
        std::pair<size_t, size_t> index = indices[i];
        state.elems[i] = values_[index.first] + index.second;
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
    void assign(const std::vector<size_t>& values, double weight = 1.0) {
      if (values.empty()) {
        assign(NULL, 0, weight);
      } else {
        assert(num_processes() != 0 && values.size() % num_processes() == 0);
        size_t* data = new size_t[values.size()];
        std::copy(values.begin(), values.end(), data);
        assign(data, values.size() / num_processes(), weight);
      }
    }

    /**
     * Allocates memory for the given number of steps and sets the weight.
     * Does not initialize the values.
     */
    void assign(size_t num_steps, double weight) {
      assign(new size_t[num_processes() * num_steps], num_steps, weight);
    }

    /**
     * Assigns data to this record. The data is stored in process-major order,
     * that is, first the entire sequence for the first process, then for the
     * second process, etc. The values pointer becomes owned by this record
     * (i.e., the data is not copied).
     */
    void assign(size_t* data, size_t num_steps, double weight = 1.0) {
      num_steps_ = num_steps;
      weight_ = weight;
      for (size_t i = 0; i < values_.size(); ++i) {
        values_[i] = data;
        data += num_steps;
      }
      if (values_.size() == 0 && data) {
        delete[] data;
      }
    }

    /**
     * Assigns data to this record. The given assignment must contain variables
     * for all the processes in the record and, if max_t is the maximum absolute
     * time index, all the <= maxt.
     */
    void assign(const finite_assignment& a, double weight = 1.0) {
      int maxt = -1;
      foreach(const finite_assignment::value_type& p, a) {
        int t = boost::any_cast<int>(p.first->index());
        assert(t >= 0);
        maxt = std::max(maxt, t);
      }
      if (maxt >= 0) {
        assign(new size_t[(maxt + 1) * num_processes()], maxt + 1, weight);
        for(size_t i = 0; i < values_.size(); ++i) {
          for(int t = 0; t <= maxt; ++t) {
            values_[i][t] = safe_get(a, processes_[i]->at(t));
          }
        }
      } else {
        assign(NULL, 0, weight);
      }
    }

    /**
     * Sets the values for the given time step.
     */
    void set(size_t t, const std::vector<size_t>& values) {
      assert(values.size() == num_processes());
      for (size_t i = 0; i < num_processes(); ++i) {
        values_[i][t] = values[i];
      }
    }

    /**
     * Loads the data from another record. The records will share the storage.
     * The argument count of this record must be equal to the number of indices.
     * \param other the record to load the data from
     * \param indices the subset of process indices to copy
     */
    void load(const finite_sequence_record& other,
              const proc_indices_type& indices) {
      assert(num_processes() == indices.size());
      num_steps_ = other.num_steps_;
      weight_ = other.weight_;
      for (size_t i = 0; i < indices.size(); ++i) {
        values_[i] = other.values_[indices[i]];
      }
    }

  private:
    boost::shared_ptr<process_type*[]> processes_;
    std::vector<size_t*> values_;
    size_t num_steps_;
    double weight_;

  }; // class finite_sequence_record


  // Free functions
  //========================================================================

  /**
   * Prints the record to an output stream.
   * \relates finite_sequence_record
   */
  inline std::ostream&
  operator<<(std::ostream& out, const finite_sequence_record& r) {
    out << "[" << r.num_steps() << " x " << r.num_processes() << "]" << std::endl;
    for (size_t t = 0; t < r.num_steps(); ++t) {
      for (size_t i = 0; i < r.num_processes(); ++i) {
        out << "\t" << r(i, t);
      }
      out << std::endl;
    }
    return out;
  }

  /**
   * Loads a single sequence in a tabular format.
   * \relates finite_sequence_record
   */
  inline void load_tabular(const std::string& filename,
                           const symbolic_format& format,
                           finite_sequence_record& record) {
    //record.check_compatible(format.finite_processes());

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }
    
    // read the table
    size_t num_processes = record.num_processes();
    std::vector<std::vector<size_t> > values(num_processes);
    std::string line;
    size_t line_number = 0;
    while (std::getline(in, line)) {
      std::vector<const char*> tokens;
      if (format.parse(num_processes, line, line_number, tokens)) {
        for (size_t i = 0; i < num_processes; ++i) {
          values[i].push_back(format.vars[i].parse(tokens[i + format.skip_cols]));
        }
      }
    }

    // concatenate the values and store them in the record
    size_t num_steps = values[0].size();
    size_t* data = new size_t[num_processes * num_steps];
    for (size_t i = 0; i < num_processes; ++i) {
      std::copy(values[i].begin(), values[i].end(), data + i * num_steps);
    }
    record.assign(data, num_steps, 1.0);
  }

  /**
   * Saves a single sequence in a tabular format.
   * \relates finite_sequence_record
   */
  inline void save_tabular(const std::string& filename,
                           const symbolic_format& format,
                           const finite_sequence_record& record) {
    //record.check_compatible(format.finite_processes());
    
    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    for (size_t i = 0; i < format.skip_rows; ++i) {
      out << std::endl;
    }

    size_t num_processes = record.num_processes();
    std::string separator = format.separator.empty() ? " " : format.separator;
    for (size_t t = 0; t < record.num_steps(); ++t) {
      for (size_t i = 0; i < format.skip_cols; ++i) {
        out << "0" << separator;
      }
      for (size_t i = 0; i < num_processes; ++i) {
        if (i > 0) { out << separator; }
        format.vars[i].print(out, record(i, t));
      }
      out << std::endl;
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
