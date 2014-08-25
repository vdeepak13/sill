#ifndef SILL_HYBRID_DATASET_HPP
#define SILL_HYBRID_DATASET_HPP

#include <sill/base/variable_utils.hpp>
#include <sill/learning/dataset/aux_data.hpp>
#include <sill/learning/dataset/finite_dataset.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/dataset/hybrid_record.hpp>

#include <iterator>

#include <boost/shared_ptr.hpp>

namespace sill {

  /**
   * A dataset that can store observations for both finite and vector variables.
   * This class can be used in two ways: it can be either passed to functions that
   * expect finite_dataset or vector_dataset (because it extends these two classes),
   * and everything will behave as expected (the functions will see the finite or
   * the vector portion of the dataset, respectively). Alternatively, this class
   * can be used by functions that require simultaneous access to both finite and
   * vector variables in the dataset, by modeling the Dataset concept for the 
   * generic variable classes and hybrid_record.
   *
   * \tparam T the storage type of the vector portion of this dataset
   * \see Dataset, hybrid_memory_dataset
   */
  template <typename T = double>
  class hybrid_dataset 
    : public finite_dataset, public vector_dataset<T> {
  public:
    // bring records up from the base classes
    using finite_dataset::records;
    using vector_dataset<T>::records;

    // Types for the Dataset concept
    typedef variable         variable_type;
    typedef domain           domain_type;
    typedef var_vector       var_vector_type;
    typedef assignment       assignment_type;
    typedef hybrid_record<T> record_type;

    struct record_iterator;
    struct const_record_iterator;

    //! Default constructor
    hybrid_dataset() { }

    //! Destructor
    virtual ~hybrid_dataset() { }

    //! Returns the number of datapoints in the dataset.
    virtual size_t size() const = 0;

    //! Returns true if the dataset has no datapoints.
    bool empty() const { return size() == 0; }

    //! Returns the columns of this dataset.
    const domain arguments() const { return make_domain(args); }

    //! Returns the columns of this dataset.
    const var_vector& arg_vector() const { return args; }

    //! Returns a single data point in the dataset's natural ordering.
    hybrid_record<T> record(size_t row) const { return record(row, args); }

    //! Returns a single data point for a subset of the variables.
    virtual hybrid_record<T> record(size_t row, const var_vector& vars) const = 0;

    //! Returns a single data point for a subset of the variables.
    hybrid_record<T> record(size_t row,
                            const finite_var_vector& finite_vars,
                            const vector_var_vector& vector_vars) const {
      return record(row, concat(finite_vars, vector_vars));
    }

    //! Implements finite_dataset::record
    finite_record record(size_t row, const finite_var_vector& vars) const {
      hybrid_record<T> r = record(row, var_vector(vars.begin(), vars.end()));
      return finite_record(r.values.finite, r.weight);
    }

    //! Implements vector_dataset::record
    vector_record<T> record(size_t row, const vector_var_vector& vars) const {
      hybrid_record<T> r = record(row, var_vector(vars.begin(), vars.end()));
      return vector_record<T>(r.values.vector, r.weight);
    }

    //! Returns mutable records for the specified variables.
    std::pair<record_iterator, record_iterator>
    records(const var_vector& vars) {
      return std::make_pair(record_iterator(this, vars),
                            record_iterator(size()));
    }

    //! Returns mutable records for the specified variables.
    std::pair<record_iterator, record_iterator>
    records(const finite_var_vector& finite_vars,
            const vector_var_vector& vector_vars) {
      var_vector vars = concat(finite_vars, vector_vars);
      return std::make_pair(record_iterator(this, vars),
                            record_iterator(size()));
    }

    //! Returns immutable records for the specified variables.
    std::pair<const_record_iterator, const_record_iterator>
    records(const var_vector& vars) const {
      return std::make_pair(const_record_iterator(this, vars),
                            const_record_iterator(size()));
    }

    //! Returns immutable records for the specified variables.
    std::pair<const_record_iterator, const_record_iterator>
    records(const finite_var_vector& finite_vars,
            const vector_var_vector& vector_vars) const {
      var_vector vars = concat(finite_vars, vector_vars);
      return std::make_pair(const_record_iterator(this, vars),
                            const_record_iterator(size()));
    }

    //! Draws a random sample from this dataset.
    template <typename RandomNumberGenerator>
    hybrid_record<T> sample(const var_vector& vars,
                            RandomNumberGenerator& rng) const {
      assert(!empty());
      boost::uniform_int<size_t> uniform(0, size() - 1);
      return record(uniform(rng), vars);
    }

    //! Draws a random sample from this dataset.
    template <typename RandomNumberGenerator>
    hybrid_record<T> sample(const finite_var_vector& finite_vars,
                            const vector_var_vector& vector_vars,
                            RandomNumberGenerator& rng) const {
      assert(!empty());
      boost::uniform_int<size_t> uniform(0, size() - 1);
      return record(uniform(rng), finite_vars, vector_vars);
    }

    // Utility functions, invoked by the iterators and subclasses
    //========================================================================
  protected:
    // the underlying state of finite_dataset and vector_dataset iterators
    typedef raw_record_iterator_state<finite_record>     finite_state_type;
    typedef raw_record_iterator_state<vector_record<T> > vector_state_type;

    // a class that stores the iterator state visible to the dataset
    struct iterator_state_type {
      finite_state_type* finite;
      vector_state_type* vector;
      iterator_state_type(finite_state_type* finite)
        : finite(finite), vector(NULL) { }

      iterator_state_type(vector_state_type* vector)
        : finite(NULL), vector(vector) { }

      iterator_state_type(finite_state_type* finite,
                          vector_state_type* vector)
        : finite(finite), vector(vector) { }
    };
    
    //! initializes the data structures in the record iterator
    virtual aux_data* init(const var_vector& args,
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

    //! prints the dataset
    virtual void print(std::ostream& out) const = 0;

    //! initializes the variables in this dataset
    void initialize(const var_vector& vars) {
      finite_var_vector finite_vars;
      vector_var_vector vector_vars;
      split(vars, finite_vars, vector_vars);
      finite_dataset::initialize(finite_vars);
      vector_dataset<T>::initialize(vector_vars);
      args = vars;
    }

    //! The variables in the dataset's internal ordering of columns.
    var_vector args;

    // friends
    friend class record_iterator;
    friend class const_record_iterator;
    friend class slice_view<hybrid_dataset>;
    
    friend std::ostream& operator<<(std::ostream& out, const hybrid_dataset& ds) {
      ds.print(out);
      return out;
    }

    // Implementation of protected functions from finite_dataset
    //========================================================================
    aux_data* init(const finite_var_vector& args,
                   finite_state_type& state) const {
      var_vector joint_args(args.begin(), args.end());
      iterator_state_type joint_state(&state);
      return init(joint_args, joint_state);
    }

    void advance(ptrdiff_t diff, finite_state_type& state, aux_data* data) const {
      iterator_state_type joint_state(&state);
      advance(diff, joint_state, data);
    }

    size_t load(size_t n, finite_state_type& state, aux_data* data) const {
      iterator_state_type joint_state(&state);
      return load(n, joint_state, data);
    }

    void save(finite_state_type& state, aux_data* data) {
      iterator_state_type joint_state(&state);
      save(joint_state, data);
    }

    // Implementation of protected functions from vector_dataset
    //========================================================================
    aux_data* init(const vector_var_vector& args,
                   vector_state_type& state) const {
      var_vector joint_args(args.begin(), args.end());
      iterator_state_type joint_state(&state);
      return init(joint_args, joint_state);
    }

    void advance(ptrdiff_t diff, vector_state_type& state, aux_data* data) const {
      iterator_state_type joint_state(&state);
      advance(diff, joint_state, data);
    }

    size_t load(size_t n, vector_state_type& state, aux_data* data) const {
      iterator_state_type joint_state(&state);
      return load(n, joint_state, data);
    }

    void save(vector_state_type& state, aux_data* data) {
      iterator_state_type joint_state(&state);
      save(joint_state, data);
    }

    // Iterators
    //========================================================================
  public:
    class record_iterator
      : public std::iterator<std::forward_iterator_tag, record_type> {

    public:
      // singular and past-the-end constructor
      explicit record_iterator(size_t endrow = 0)
        : dataset(NULL), row(endrow) { }
      
      // begin constructor
      record_iterator(hybrid_dataset* dataset, const var_vector& args)
        : dataset(dataset), row(0), rows_left(0) {
        iterator_state_type state(&finite_state, &vector_state);
        aux.reset(dataset->init(args, state));
        record.resize(finite_state.elems.size(), vector_state.elems.size());
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
        save_record();
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
      hybrid_dataset* dataset;         // the underlying dataset
      size_t row;                      // the logical row in the dataset
      size_t rows_left;                // 0 indicates we need to fetch more data
      finite_state_type finite_state;  // the finite state visible to the dataset
      vector_state_type vector_state;  // the vector state visible to the dataset
      boost::shared_ptr<aux_data> aux; // auxiliary data used by the dataset
      record_type record;              // user-facing data

      void load_advance() {
        if (rows_left == 0) {
          iterator_state_type state(&finite_state, &vector_state);
          dataset->save(state, aux.get());
          rows_left = dataset->load(size_t(-1), state, aux.get());
          if (rows_left == 0) return; // no more data left
        }
        for (size_t i = 0; i < finite_state.elems.size(); ++i) {
          record.values.finite[i] = *finite_state.elems[i];
          finite_state.elems[i] += finite_state.e_step[i];
        }
        for (size_t i = 0; i < vector_state.elems.size(); ++i) {
          record.values.vector[i] = *vector_state.elems[i];
          vector_state.elems[i] += vector_state.e_step[i];
        }
        record.weight = *vector_state.weights;
        finite_state.weights += finite_state.w_step;
        vector_state.weights += vector_state.w_step;
        --rows_left;
      }

      // to maintain compatibility with const_record_iterator,
      // we advance in load_advance() and then go back in save_record()
      void save_record() {
        for (size_t i = 0; i < finite_state.elems.size(); ++i) {
          *(finite_state.elems[i] - finite_state.e_step[i]) = record.values.finite[i];
        }
        for (size_t i = 0; i < vector_state.elems.size(); ++i) {
          *(vector_state.elems[i] - vector_state.e_step[i]) = record.values.vector[i];
        }
      }
      
      friend class const_record_iterator;

    }; // class record_iterator

    class const_record_iterator
      : public std::iterator<std::forward_iterator_tag, const record_type> {

    public:
      // singular and past-the-end constructor
      explicit const_record_iterator(size_t endrow = 0)
        : dataset(NULL), row(endrow) { }
      
      // begin constructor
      const_record_iterator(const hybrid_dataset* dataset, const var_vector& args)
        : dataset(dataset), row(0), rows_left(0) {
        iterator_state_type state(&finite_state, &vector_state);
        aux.reset(dataset->init(args, state));
        record.resize(finite_state.elems.size(), vector_state.elems.size());
        load_advance();
      }

      // record iterator conversions are expensive, so we make them explicit
      explicit const_record_iterator(const record_iterator& it)
        : dataset(it.dataset),
          row(it.row),
          rows_left(it.rows_left),
          finite_state(it.finite_state),
          vector_state(it.vector_state),
          aux(it.aux),
          record(it.record) { }

      // mutable iterator can be assigned to a const iterator
      const_record_iterator& operator=(const record_iterator& it){
        dataset = it.dataset;
        row = it.row;
        rows_left = it.rows_left;
        finite_state = it.finite_state;
        vector_state = it.vector_state;
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
      const hybrid_dataset* dataset;   // the underlying dataset
      size_t row;                      // the logical row in the dataset
      size_t rows_left;                // 0 indicates we need to fetch more data
      finite_state_type finite_state;  // the finite state visible to the dataset
      vector_state_type vector_state;  // the vector state visible to the dataset
      boost::shared_ptr<aux_data> aux; // auxiliary data used by the dataset
      record_type record;              // user-facing data

      void load_advance() {
        if (rows_left == 0) {
          iterator_state_type state(&finite_state, &vector_state);
          rows_left = dataset->load(size_t(-1), state, aux.get());
          if (rows_left == 0) return; // no more data left
        }
        for (size_t i = 0; i < finite_state.elems.size(); ++i) {
          record.values.finite[i] = *finite_state.elems[i];
          finite_state.elems[i] += finite_state.e_step[i];
        }
        for (size_t i = 0; i < vector_state.elems.size(); ++i) {
          record.values.vector[i] = *vector_state.elems[i];
          vector_state.elems[i] += vector_state.e_step[i];
        }
        record.weight = *vector_state.weights;
        finite_state.weights += finite_state.w_step;
        vector_state.weights += vector_state.w_step;
        --rows_left;
      }

      friend class record_iterator;

    }; // class const_record_iterator

  }; // class hybrid_dataset
    
} // namespace sill

#endif
