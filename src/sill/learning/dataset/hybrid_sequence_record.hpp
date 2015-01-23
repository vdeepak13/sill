#ifndef SILL_HYBRID_SEQUENCE_RECORD_HPP
#define SILL_HYBRID_SEQUENCE_RECORD_HPP

#include <sill/base/discrete_process.hpp>
#include <sill/base/variable.hpp>
#include <sill/base/variable_utils.hpp>
#include <sill/learning/dataset/finite_sequence_record.hpp>
#include <sill/learning/dataset/vector_sequence_record.hpp>
#include <sill/learning/dataset/simple_process_index_map.hpp>
#include <sill/learning/dataset/hybrid_record.hpp>
#include <sill/learning/dataset/hybrid_record_iterator_state.hpp>

#include <fstream>
#include <vector>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that represents a sequence of finite and vector values
   * for a set of discrete processes.
   * \tparam T the storage type of the record
   */
  template <typename T = double>
  class hybrid_sequence_record {
  public:
    /**
     * The type of processes handled by this record.
     */
    typedef discrete_process<variable> process_type;

    /**
     * The type that specifies a sequence of process indices in the record.
     */
    struct proc_indices_type {
      std::vector<size_t> finite;
      std::vector<size_t> vector;
    };

    /**
     * The type that specifies a sequence of variable indices in the record.
     */
    struct var_indices_type {
      std::vector<std::pair<size_t,size_t> > finite;
      std::vector<std::pair<size_t,size_t> > vector;
    };

    /**
     * The type that specifies a mapping from each process to its index.
     */
    class index_map_type {
    public:
      index_map_type() { }

      index_map_type(const std::vector<process_type*>& procs) {
        initialize(procs);
      }

      void initialize(const std::vector<process_type*>& procs) {
        std::vector<finite_discrete_process*> finite_procs;
        std::vector<vector_discrete_process*> vector_procs;
        split(procs, finite_procs, vector_procs);
        finite.initialize(finite_procs);
        vector.initialize(vector_procs);
      }

      void indices(const std::vector<process_type*>& procs,
                   proc_indices_type& result) const {
        std::vector<finite_discrete_process*> finite_procs;
        std::vector<vector_discrete_process*> vector_procs;
        split(procs, finite_procs, vector_procs);
        finite.indices(finite_procs, result.finite);
        vector.indices(vector_procs, result.vector);
      }

      void indices(const var_vector& vars,
                   var_indices_type& result) const {
        indices(vars, 0, result);
      }

      void indices(const var_vector& vars,
                   size_t offset,
                   var_indices_type& result) const {
        finite_var_vector finite_vars;
        vector_var_vector vector_vars;
        split(vars, finite_vars, vector_vars);
        finite.indices(finite_vars, offset, result.finite);
        vector.indices(vector_vars, offset, result.vector);
      }

    private:
      simple_process_index_map<finite_variable> finite;
      simple_process_index_map<vector_variable> vector;

    }; // class index_map_type

    /**
     * The type that represents the weight.
     */
    typedef T weight_type;

    /**
     * Creates an empty record.
     */
    hybrid_sequence_record() { }

    /**
     * Creates a record with the given vector of processes and weight/
     */
    explicit hybrid_sequence_record(const std::vector<process_type*>& procs,
                                    T weight = 1.0) {
      initialize(procs, weight);
    }

    /**
     * Initializes the record to the given vector of processes and weight.
     */
    void initialize(const std::vector<process_type*>& procs, T weight = 1.0) {
      std::vector<finite_discrete_process*> finite_procs;
      std::vector<vector_discrete_process*> vector_procs;
      split(procs, finite_procs, vector_procs);
      finite_.initialize(finite_procs, weight);
      vector_.initialize(vector_procs, weight);
    }

    /**
     * Releases the memory associated with the record. The memory must be
     * owned by the record.
     */
    void free_memory() {
      finite_.free_memory();
      vector_.free_memory();
    }

    /**
     * Returns the number of processes in this record.
     */
    size_t num_processes() const {
      return finite_.num_processes() + vector_.num_processes();
    }

    /**
     * Returns the number of steps in this record.
     */
    size_t num_steps() const {
      return finite_.num_steps();
    }

    /**
     * Returns the total number of values in this record.
     */
    size_t size() const {
      return finite_.size() + vector_.size();
    }

    /**
     * Returns the record weight.
     */
    const T& weight() const {
      return vector_.weight();
    }

    /**
     * Returns the finite component.
     */
    const finite_sequence_record& finite() const {
      return finite_;
    }

    /**
     * Returns the vector component.
     */
    const vector_sequence_record<T>& vector() const {
      return vector_;
    }

    /**
     * Checks if the record is compatible with the given argument vector.
     * \throw invalid_argument if the arguments do not match
     */
    void check_compatible(const std::vector<process_type*>& procs) const {
      std::vector<finite_discrete_process*> finite_procs;
      std::vector<vector_discrete_process*> vector_procs;
      split(procs, finite_procs, vector_procs);
      finite_.check_compatible(finite_procs);
      vector_.check_compatible(vector_procs);
    }
    
    /**
     * Extracts the entire time series into an assignment. The assignment
     * will contain a variable for each process and for each time step in
     * the sequence.
     */
    void extract(assignment& a) const {
      finite_.extract(a);
      vector_.extract(a);
    }

    /**
     * Extracts a single time point into an assignment. THe variables are
     * indexed with teh "current" time index.
     */
    void extract(size_t t, assignment& a) const {
      finite_.extract(t, a);
      vector_.extract(t, a);
    }

    /**
     * Extracts the values for the variables with given indices.
     */
    void extract(const var_indices_type& indices, hybrid_index<T>& values) const {
      finite_.extract(indices.finite, values.finite);
      vector_.extract(indices.vector, values.vector);
    }

    /**
     * Extracts the weight and values for the variables with given indices.
     */
    void extract(const var_indices_type& indices, hybrid_record<T>& r) const {
      r.weight = weight();
      extract(indices, r.values);
    }

    /**
     * Extracts the weight and pointers to the variables with given indices.
     */
    void extract(const var_indices_type& indices,
                 hybrid_record_iterator_state<T>& state) const {
      if (state.finite) { finite_.extract(indices.finite, *state.finite); }
      if (state.vector) { vector_.extract(indices.vector, *state.vector); }
    }

    /**
     * Assigns data to this record. The values are copied into a
     * newly-allocated block of memory.
     */
    void assign(const std::vector<size_t>& finite_vals,
                const std::vector<T>& vector_vals,
                T weight) {
      finite_.assign(finite_vals, weight);
      vector_.assign(vector_vals, weight);
      assert(finite_.num_steps() == vector_.num_steps());
    }

    /**
     * Assigns data to this record. The value pointers become owned by this record.
     */
    void assign(size_t* finite_data,
                T* vector_data,
                size_t num_steps,
                T weight) {
      finite_.assign(finite_data, num_steps, weight);
      vector_.assign(vector_data, num_steps, weight);
    }

    /**
     * Assigns data to this record.
     */
    void assign(const assignment& a, T weight) {
      finite_.assign(a, weight);
      vector_.assign(a, weight);
      assert(finite_.num_steps() == vector_.num_steps());
    }

    /**
     * Loads the data from another record. The records will share the storage.
     * \param other the record to load the data from
     * \param indices the subset of process indices to copy
     */
    void load(const hybrid_sequence_record& other,
              const proc_indices_type& indices) {
      finite_.load(other.finite_, indices.finite);
      vector_.load(other.vector_, indices.vector);
    }

  private:
    finite_sequence_record finite_;
    vector_sequence_record<T> vector_;
      
  }; // class hybrid_sequence_record

  // Free functions
  //========================================================================

  template <typename T>
  std::ostream&
  operator<<(std::ostream& out, const hybrid_sequence_record<T>& record) {
    out << record.finite() << record.vector();
    return out;
  }

  /**
   * Loads a single sequence in a tabular format.
   * \relates hybrid_sequence_record
   */
  template <typename T>
  void load_tabular(const std::string& filename,
                    const symbolic_format& format,
                    hybrid_sequence_record<T>& record) {
    if (record.num_processes() != format.discrete_procs.size()) {
      throw std::logic_error("The record and format must have the same processes.");
    }

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }
    
    // read the table
    size_t finite_cols = record.finite().num_processes();
    size_t vector_cols = record.vector().num_dims();
    std::vector<std::vector<size_t> > finite_vals(finite_cols);
    std::vector<std::vector<T> >      vector_vals(vector_cols);
    std::string line;
    size_t line_number = 0;
    size_t num_steps = 0;
    while (std::getline(in, line)) {
      std::vector<const char*> tokens;
      if (format.parse(finite_cols + vector_cols, line, line_number, tokens)) {
        size_t col = format.skip_cols;
        size_t fi = 0;
        size_t vi = 0;
        foreach(const symbolic_format::discrete_process_info& info,
                format.discrete_procs) {
          if (info.is_finite()) {
            finite_vals[fi++].push_back(info.parse(tokens[col++]));
          } else if (info.is_vector()) {
            size_t len = info.size();
            for (size_t j = 0; j < len; ++j) {
              vector_vals[vi++].push_back(parse_string<T>(tokens[col++]));
            }
          } else {
            throw std::logic_error("Unsupported variable type " + info.name());
          }
        }
        assert(finite_cols == fi);
        assert(vector_cols == vi);
        ++num_steps;
      }
    }

    // concatenate the values and store them in the record
    size_t* finite_data = new size_t[finite_vals.size() * num_steps];
    for (size_t i = 0; i < finite_vals.size(); ++i) {
      std::copy(finite_vals[i].begin(),
                finite_vals[i].end(),
                finite_data + i * num_steps);
    }
    T* vector_data = new T[vector_vals.size() * num_steps];
    for (size_t i = 0; i < vector_vals.size(); ++i) {
      std::copy(vector_vals[i].begin(),
                vector_vals[i].end(),
                vector_data + i * num_steps);
    }
    record.assign(finite_data, vector_data, num_steps, 1.0);
  }  // load_tabular()

  /**
   * Saves a single sequence in a tabular format.
   * \relates hybrid_sequence_record
   */
  template <typename T>
  void save_tabular(const std::string& filename,
                    const symbolic_format& format,
                    const hybrid_sequence_record<T>& record) {
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

    std::string separator = format.separator.empty() ? " " : format.separator;
    size_t num_steps = record.num_steps();
    for (size_t t = 0; t < num_steps; ++t) {
      for (size_t i = 0; i < format.skip_cols; ++i) {
        out << "0" << separator;
      }
      size_t fi = 0;
      size_t vi = 0;
      for(size_t i = 0; i < format.discrete_procs.size(); ++i) {
        const symbolic_format::discrete_process_info& info = format.discrete_procs[i];
        if (info.is_finite()) {
          if (i > 0) { out << separator; }
          info.print(out, record.finite()(fi++, t));
        } else {
          const T* ptr = record.vector().value_ptr(vi++, t);
          size_t len = info.size();
          for (size_t j = 0; j < len; ++j) {
            if (i || j) { out << separator; }
            out << *ptr;
            ptr += num_steps;
          }
        }
      }
      out << std::endl;
    }
  }  // save_tabular()

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
