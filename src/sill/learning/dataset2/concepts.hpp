#ifndef SILL_DATASET_CONCEPTS_HPP
#define SILL_DATASET_CONCEPTS_HPP

namespace sill {

  /**
   * Represents a sequence of dense datapoints, where each column
   * is a variable of the specified type. Supports efficient 
   * iteration over all the data or drawing samples from the data
   * for a specified subset of the columns,
   * in the form of the records of the given type. The record type
   * is variable_type-dependent and provides weight and values in the
   * native (flat) representation of the extracted data, so that many 
   * factor operations, such as sampling or log-likelihood can be
   * performed without additional data lookups/transforms or allocs.
   *
   * \ingroup dataset_concepts
   * \see finite_dataset, vector_dataset, dataset
   */
  template <typename DS>
  struct Dataset {

    //! The variable type of each column.
    typedef typename DS::variable_type variable_type;

    //! The set of column variables (e.g., finite_domain).
    typedef typename DS::domain_type domain_type;

    //! A sequence of variables, typically std::vector<variable_type*>.
    typedef typename DS::var_vector_type var_vector_type;

    //! A type that maps variable_type to values (e.g., finite_assignment).
    typedef typename DS::assignment_type assignment_type;

    //! The record that stores the extracted data (e.g., finite_record).
    typedef typename DS::record_type record_type;

    //! An iterator that provides mutable access to the data in this dataset.
    typedef typename DS::record_iterator record_iterator;

    //! An iterator that provides const access to the data in this dataset.
    typedef typename DS::const_record_iterator const_record_iterator;

    //! An iterator that draws samples from the data in this dataset.
    typedef typename DS::sample_iterator sample_iterator;

    //! Returns the number of datapoints in the dataset.
    size_t size() const;

    //! Returns the variables in this dataset.
    domain_type arguments() const;

    //! Provides mutable access to a subset of columns in the specified order.
    std::pair<record_iterator, record_iterator>
    records(const var_vector_type& vars);

    //! Provides const access to a subset of columns in the specified order.
    std::pair<const_record_iterator, const_record_iterator>
    records(const var_vector_type& vars) const;

    //! Draws samples from this dataset for a subset of columns in the given order.
    sample_iterator samples(const var_vector_type& vars, usigned seed = 0) const;

    concept_usage(Dataset) {
      // TODO: finish this up
    }

  };


  /**
   * Dataset, where each column is a finite variable and the member types
   * are the standard types associated with finite variables.
   */
  template <typename DS>
  struct FiniteDataset : public Dataset<DS> {
    typedef finite_variable   variable_type;
    typedef finite_domain     domain_type;
    typedef finite_var_vector var_vector_type;
    typedef finite_assignment assignment_Type;
    typedef finite_record2     record_type;
  };

  
  /**
   * Dataset, where each column is a vector variable and the member types
   * are the standard types associated with vector variables.
   */
  template <typename DS, typename T>
  struct VectorDataset : public Dataset<DS> {
    typedef vector_variable   variable_type;
    typedef vector_domain     domain_type;
    typedef vector_var_vector var_vector_type;
    typedef vector_assignment assignment_type;
    typedef vector_record2<T>  record_type;
  };


  /**
   * Dataset, where each column is a finite or vector variable and the
   * member types are the standard types associated with the variable type.
   */
  template <typename DS>
  struct HybridDataset : public Dataset<DS> {
    typedef variable   variable_type;
    typedef domain     domain_type;
    typedef var_vector var_vector_type;
    typedef assignment assignment_type;
    typedef record2     record_type;
  };


  /**
   * Dataset that permits the insertion of new datapoints.
   */
  template <typename DS>
  struct InsertableDataset : public Dataset<DS> {
    //! Inserts a new datapoint. The values in the record must be in the
    //! internal order of the dataset.
    void insert(const record_type& record);

    //! Inserts a new datapoint. The values in the assignment must include
    //! all the arguments (columns) of this datset.
    void insert(const assignment_type& a, typename Record::weight_type w);

    //! Inserts the given number of datapoints with unit weights and
    //! variable_type-specific special undefined values.
    void insert(size_t nrows);
  };

} // namespace sill

#endif
