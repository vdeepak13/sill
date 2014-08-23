#ifndef SILL_DATASET_HPP
#define SILL_DATASET_HPP

namespace sill {

  // documentation:
  // initialization
  // accessing the portion of the dataset by passing base class
  // models HybridDataset and InsertableDataset
  // user must not call shuffle on the chilldren directly
  template <typename Tf = size_t, typename Tv = double>
  class dataset 
    : public finite_dataset<Tf>, public vector_dataset<Tv> {

    typedef typename finite_dataset::record_iterator finite_record_old_iterator;
    typedef typename vector_dataset::record_iterator vector_record_old_iterator;
    typedef typename finite_dataset::const_record_iterator finite_const_record_iterator;
    typedef typename vector_dataset::const_record_iterator vector_const_record_iterator;
    
  public:
    typedef variable   variable_type;
    typedef var_vector vector_type;
    typedef domain     domain_type;
    typedef assignment assignment_type;
    typedef record     record_type;

    struct record_iterator;
    struct const_record_iterator;
    
    //! Bring in the implementations from the base classes
    using finite_dataset::records;
    using vector_dataset::records;
 
    //! Creates an uninitialized dataset
    dataset() { }

    //! Returns the finite portion of this dataset
    finite_dataset& finite() {
      return *this;
    }

    //! Returns the finite portion of this dataset
    const finite_dataset& finite() const {
      return *this;
    }

    //! Returns the vector portion of this dataset
    vector_dataset& vector() {
      return *this;
    }

    //! Returns the vector portion of this dataset
    const vector_dataset& vector() const {
      return *this;
    }

    //! Initializes the dataset with the given seqeuence of variables.
    void initialize(const var_vector& variables) {
      finite_var_vector finite_vars;
      vector_var_vector vector_vars;
      split(variables, finite_vars, vector_vars);
      finite().initialize(finite_vars);
      vector().initialize(vector_vars);
    }

    //! Returns the number of datapoints in the dataset
    size_t size() const {
      return finite().size();
    }

    //! Returns the columns of this dataset
    domain arguments() const {
      check_initialized();
      return finite().arguments() + vector().arguments();
    }

    //! Returns mutable records for the specified variables.
    std::pair<record_iterator, record_iterator>
    records(const finite_var_vector& finite_vars,
            const vector_var_vector& vector_vars) {
      check_initialized();
      finite_record_old_iterator fit, fend;
      vector_record_old_iterator vit, vend;
      boost::tie(fit, fend) = finite.records(finite_vars);
      boost::tie(vit, vend) = vector.records(vector_vars);
      return std::make_pair(record_iterator(fit, vit),
                            record_iterator(fend, vend));
    }

    //! Returns mutable records for the specified variables.
    std::pair<record_iterator, record_iterator>
    records(const var_vector& variables) {
      finite_var_vector finite_vars;
      vector_var_vector vector_vars;
      split(variables, finite_vars, vector_vars);
      return records(finite_vars, vector_vars);
    }
    
    //! Returns immutable records for the specified variables.
    std::pair<const_record_iterator, const_record_iterator>
    records(const finite_var_vector& finite_vars,
            const vector_var_vector& vector_vars) const {
      check_initialized();
      finite_const_record_iterator fit, fend;
      vector_const_record_iterator vit, vend;
      boost::tie(fit, fend) = finite.records(finite_vars);
      boost::tie(vit, vend) = vector.records(vector_vars);
      return std::make_pair(const_record_iterator(fit, vit),
                            const_record_iterator(fend, vend));
    }
    
    //! Returns immutable records for the specified variables.
    std::pair<const_record_iterator, const_record_iterator>
    records(const var_vector& variables) const {
      finite_var_vector finite_vars;
      vector_var_vector vector_vars;
      split(variables, finite_vars, vector_vars);
      return records(finite_vars, vector_vars);
    }

    //! Returns a view of the datset for a range of the rows. 
    dataset subset(size_t begin, size_t end) const {
      check_initialized();
      assert(begin <= size());
      assert(end <= size());
      return dataset(*this, begin, end);
    }
    
    dataset restrict(const finite_assignment& a) const {
      check_initialized();
      /// TODO
    }

    //! Inserts finite values in this dataset's ordering.
    //! The dataset must contain only finite variables columns.
    void insert(const finite_record_old& r) {
      check_finite();
      finite_dataset::insert(r);
      vector_dataset::insert(vector_record_old(r.weight));
    }

    //! Inserts finite values from a finite assignment.
    //! The dataset must contain only finite variable columns.
    void insert(const finite_assignment& a, double weight) {
      check_finite();
      finite_dataset::insert(a, weight);
      vector_dataset::insert(vector_assignment(), weight);
    }

    //! Inserts vector values in this dataset's ordering.
    //! The dataset must contain only vector variable columns.
    void insert(const vector_record_old& r) {
      check_vector();
      finite_dataset::insert(finite_record_old(r.weight));
      vector_dataset::insert(r);
    }

    //! Inserts vector values from a vector assignment.
    //! The dataset must contain only vector variable columns.
    void insert(const vector_assignment& a, Tv weight) {
      check_vector();
      finite_dataset::insert(finite_assignment(), weight);
      vector_dataset::insert(a, weight);
    }

    //! Inserts values in this dataset's ordering.
    void insert(const record& r) {
      finite_dataset::insert(r.values.finite, r.weight);
      vector_dataset::insert(r.values.vector, r.weight);
    }

    //! Inserts values from an assignment (all variables must be present).
    void insert(const assignment& a) {
      finite_dataset::insert(a); // a will be cast to finite_assignment
      vector_dataset::insert(a); // a will be cast to vector_assignemnt
    }
    //! Inserts the given number of rows with unit weights and undefined values.
    void insert(size_t nrows) {
      finite_dataset::insert(nrows);
      vector_dataset::insert(nrows);
    }
    
    //! Randomizes the ordering of records in this dataset
    void shuffle() {
      // TODO:
    }


    //! iterator over mutable records
    class record_iterator
      : public std::iterator<std::forward_iterator_tag, record_type> {

      finite_record_old_iterator fit;
      vector_record_old_iterator vit;
      record_type record;

    public:
      record_iterator() : record(*fit, *vit, 0.0) { }
      
      record_iterator(const finite_record_old_iterator& fit,
                      const vector_record_old_iterator& vit)
        : fit(fit), vit(vit), record(*fit, *vit, vit->weight) { }

      // TODO: move constructor

      rectord_type& operator*() const {
        return record;        
      }

      record_iterator& operator++() {
        ++fit;
        ++vit;
        record.weight = vit->weight;
        return *this;
      }

      record_iterator operator++(int) {
        throw std::logic_error("record iterators do not support postincrement");
      }
    
      bool operator==(const record_iterator& other) const {
        return fit == other.fit && vit == other.vit;
      }

      bool operator!=(const record_iterator& other) const {
        return fit != other.fit || vit != other.vit;
      }

    }; // record_iterator


    //! iterator over immutable records
    class const_record_iterator
      : public std::iterator<std::forward_iterator_tag, const record_type> {

      finite_const_record_iterator fit;
      vector_const_record_iterator vit;
      record_type record;

    public:
      record_iterator() : record(*fit, *vit, 0.0) { }
      
      record_iterator(const finite_const_record_iterator& fit,
                      const vector_const_record_iterator& vit)
        : fit(fit), vit(vit), record(*fit, *vit, vit->weight) { }

      // TODO: move constructor

      rectord_type& operator*() const {
        return record;        
      }

      const_record_iterator& operator++() {
        ++fit;
        ++vit;
        record.weight = vit->weight;
        return *this;
      }

      const_record_iterator operator++(int) {
        throw std::logic_error("record iterators do not support postincrement");
      }
    
      bool operator==(const const_record_iterator& other) const {
        return fit == other.fit && vit == other.vit;
      }

      bool operator!=(const const_record_iterator& other) const {
        return fit != other.fit || vit != other.vit;
      }

    }; // const_record_iterator

    // Private member functions
    //========================================================================
  private:
    //! Creates a derived dataset
    dataset(const dataset& other, size_t begin, size_t end)
      : finite_dataset(other, begin, end),
        vector_dataset(other, begin, end)  { }

    //! Throws an exception if the dataset is not initialized
    void check_initialized() const {
      if (!finite().initialized() ||
          !vector().initialized()) {
        throw std::logic_error("The dataset is not initialized!");
      }
      assert(finite().size() == vector().size()); // might as well check this
    }

    //! Throws an exception if the dataset contains non-finite columns
    void check_finite() const {
      // TODO
    }

    //! Throws an exception if the dataset contains non-vector columns
    void check_vector() const {
      // TODO
    }

  }; // class dataset

} // namespace sill

#endif
