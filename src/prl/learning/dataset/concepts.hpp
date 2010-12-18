
#ifndef SILL_LEARNING_DATASET_CONCEPTS_HPP
#define SILL_LEARNING_DATASET_CONCEPTS_HPP

#include <sill/base/assignment.hpp>
#include <sill/stl_concepts.hpp>

#include <sill/learning/dataset/datasource.hpp>
#include <sill/learning/dataset/record.hpp>
#include <sill/range/forward_range.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  class record;

  /**
   * Concept for a data source (online or batch source).
   */
  template <typename D>
  struct DataSource {

    //! Returns true iff the data source is empty or depleted.
    bool empty() const;

    //! Returns the variables in this data source.
    domain variables() const;

    //! Returns the finite variables in this data source.
    const finite_domain& finite_variables() const;

    //! Returns the vector variables in this data source.
    const vector_domain& vector_variables() const;

    //! Order of variable types in dataset's natural order
    const std::vector<variable::variable_typenames>&
    variable_type_order() const;

    //! Returns the finite variables in the natural order
    const finite_var_vector& finite_list() const;

    //! Returns the vector variables in the natural order
    const vector_var_vector& vector_list() const;

    //! Returns the number of finite variables
    std::size_t num_finite() const;

    //! Returns the number of vector variables
    std::size_t num_vector() const;

    //! Returns the finite class variables (if any)
    const finite_var_vector& finite_class_variables() const;

    //! Returns the vector class variables (if any)
    const vector_var_vector& vector_class_variables() const;

    //! Sets the finite class variables to a single variable
    void set_finite_class_variable(finite_variable* class_var);

    //! Sets the finite class variables
    void set_finite_class_variables(const finite_var_vector& class_vars);

    //! Sets the vector class variables to a single variable
    void set_vector_class_variable(vector_variable* class_var);

    //! Sets the vector class variables
    void set_vector_class_variables(const vector_var_vector& class_vars);

    //    concept_assert((Convertible<record, assignment>));

    concept_usage(DataSource) {
      sill::same_type(empty(), b);
      sill::same_type(variables(), d);
      sill::same_type(finite_variables(), fdref);
      sill::same_type(vector_variables(), vdref);
      sill::same_type(variable_type_order(), var_type_vec);
      sill::same_type(finite_list(), fvarvec);
      sill::same_type(vector_list(), vvarvec);
      sill::same_type(num_finite(), i);
      sill::same_type(num_vector(), i);
      sill::same_type(finite_class_variables(), fvarvec);
      sill::same_type(vector_class_variables(), vvarvec);
      (void) set_finite_class_variable(fvar);
      (void) set_vector_class_variable(vvar);
      (void) set_finite_class_variables(fvarvec);
      (void) set_vector_class_variables(vvarvec);
    }

  private:
    bool b;
    domain d;
    static const finite_domain& fdref;
    static const vector_domain& vdref;
    static const std::vector<variable::variable_typenames>& var_type_vec;
    static const finite_var_vector& fvarvec;
    static const vector_var_vector& vvarvec;
    std::size_t i;
    finite_variable* fvar;
    vector_variable* vvar;

  }; // struct DataSource

  /**
   * Concept for a data oracle (online data source).
   * @todo Create a free function for taking a generic oracle and reading
   *       examples from it to create a dataset.
   */
  template <typename O>
  struct Oracle : public DataSource<O> {

    //! Returns the current record.
    record& current() const;

    //! Increments the oracle to the next record.
    //! This returns true iff the operation was successful; false indicates
    //! a depleted oracle.
    //! NOTE: This must be explicitly called after the oracle is constructed.
    bool next();

    concept_usage(Oracle) {
      sill::same_type(current(), r);
      sill::same_type(next(), b);
    }

  private:
    static record& r;
    bool b;

  }; // struct Oracle

  /**
   * Concept for a dataset.
   * \ingroup group_concepts
   */
  template <typename D>
  struct Dataset : public DataSource<D> {

    //! Returns the number of datapoints.
    std::size_t size() const;

    //! Element access without range checking
    record& operator[](std::size_t i) const;

    //! Element access with range checking
    record& at(std::size_t i) const;

    //! Returns a range over the records of this dataset
    //! Eventually, will be able to provide a set of variables
    sill::forward_range<const record&> records() const;

    concept_usage(Dataset) {
      sill::same_type(size(), i);
      sill::same_type(operator[](i), r);
      sill::same_type(at(i), r);
      sill::same_type(records(), record_range); 
    }

  private:
    std::size_t i;
    static record& r;
    sill::forward_range<const record&> record_range;

  }; // struct Dataset

  /**
   * Concept for a mutable dataset.
   * \ingroup group_concepts
   */
  template <typename D>
  struct MutableDataset : public Dataset<D> {

    //! Adds a new record
    void insert(const assignment& a);

    //! Adds a new record with finite variable values fvals and vector variable
    //! values vvals.
    void insert(const std::vector<size_t>& fvals, const vec& vvals);

    concept_usage(MutableDataset) {
      (void) insert(a);
      (void) insert(fvals, vvals);
    }

  private:
    static const assignment& a;
    static const std::vector<size_t>& fvals;
    static const vec& vvals;

  };  // struct MutableDataset

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

