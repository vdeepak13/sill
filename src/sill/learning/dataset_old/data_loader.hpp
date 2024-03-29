#ifndef SILL_DATASET_DATA_LOADER_HPP
#define SILL_DATASET_DATA_LOADER_HPP

#include <iostream>

#include <boost/serialization/shared_ptr.hpp>

#include <sill/base/variable_type_group.hpp>
//#include <sill/learning/dataset_old/concepts.hpp>
#include <sill/learning/dataset_old/dataset.hpp>
#include <sill/learning/dataset_old/oracle.hpp>
#include <sill/learning/dataset_old/symbolic_oracle.hpp>
#include <sill/learning/dataset_old/syn_oracle_knorm.hpp>
#include <sill/learning/dataset_old/syn_oracle_majority.hpp>
#include <sill/copy_ptr.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Set of functions for loading datasets and oracles.
   * The purposes of this class are to provide:
   *  - simple functions for loading data which keep the user from
   *    needing to know too much about symbolic formats, etc.
   *  - functions for loading data oracles by their names
   */
  namespace data_loader {

    // Free functions: loading oracles
    //==========================================================================

    /**
     * Load default oracles.  This supports synthetic oracles with 'ds_name':
     *  - knorm
     *  - majority
     * If 'ds_name' is not recognized, this tries to load 'ds_name' as a .sum
     * file.
     */
    template <typename LA>
    boost::shared_ptr<oracle<LA> >
    load_oracle(sill::universe& u, std::string ds_name, double random_seed);

    /**
     * Load default oracles.  This supports synthetic oracles with 'ds_name':
     *  - knorm
     *  - majority
     * If 'ds_name' is not recognized, this tries to load 'ds_name' as a .sum
     * file.
     */
    template <typename LA>
    boost::shared_ptr<oracle<LA> >
    load_oracle(const sill::datasource_info_type& info,
                std::string ds_name, double random_seed);

    /**
     * Reads in a summary file for a symbolic data file, creates new
     * variables in universe u as necessary, and constructs an oracle.
     * @param filename   summary file name
     * @param u          universe in which to create the new variables
     * @param record_limit  See symbolic_oracle::parameters.
     * @param auto_reset    See symbolic_oracle::parameters.
     * @return data oracle
     */
    template <typename LA>
    boost::shared_ptr<symbolic_oracle<LA> >
    load_symbolic_oracle(const std::string& filename, universe& u,
                         size_t record_limit = 0, bool auto_reset = false);

    /**
     * Reads in a summary file for a symbolic data file with data over the
     * given variables and constructs an oracle.
     * @param filename        summary file name
     * @param info            datasource info
     * @param record_limit  See symbolic_oracle::parameters.
     * @param auto_reset    See symbolic_oracle::parameters.
     * @return data oracle
     */
    template <typename LA>
    boost::shared_ptr<symbolic_oracle<LA> >
    load_symbolic_oracle
    (const std::string& filename, const datasource_info_type& info,
     size_t record_limit = 0, bool auto_reset = false);

    // Free functions: loading datasets
    //==========================================================================

    /**
     * Load a dataset from a space-separated file.
     * This uses the default settings for the symbolic file format.
     * @param var_type_order  This allows you to specify a natural ordering
     *                        of variables in the dataset.
     *                        (default = arbitrary)
     * @see symbolic.hpp
     * DEPRECATED: You should use load_symbolic_dataset() instead.
     */
    template <typename Dataset>
    boost::shared_ptr<Dataset>
    load_plain(const std::string& filename,
               const finite_var_vector& finite_vars,
               const vector_var_vector& vector_vars,
               const std::vector<variable::variable_typenames>& var_type_order
               = std::vector<variable::variable_typenames>()) {
      //      concept_assert((sill::Dataset<Dataset>));
      boost::shared_ptr<Dataset>
        data_ptr(new Dataset(finite_vars, vector_vars, var_type_order));
      datasource_info_type ds_info;
      ds_info.finite_seq = finite_vars;
      ds_info.vector_seq = vector_vars;
      ds_info.var_type_order = var_type_order;
      symbolic_oracle<typename Dataset::la_type> o(filename, ds_info);
//      data_ptr->set_finite_class_variables(o.finite_class_variables());
//      data_ptr->set_vector_class_variables(o.vector_class_variables());
      while(o.next())
        data_ptr->insert(o.current(), o.weight());
      return data_ptr;
    }

    /**
     * Load a SYMBOLIC dataset, given the summary file.
     * @param max_records  max number of records which will be loaded
     *                     (default = no limit)
     * @see symbolic.hpp
     */
    template <typename Dataset>
    boost::shared_ptr<Dataset>
    load_symbolic_dataset(const std::string& filename, universe& u,
                          size_t max_records
                          = std::numeric_limits<size_t>::max()) {
      //      concept_assert((sill::Dataset<Dataset>));
      boost::shared_ptr<symbolic_oracle<typename Dataset::la_type> >
        o_ptr(load_symbolic_oracle<typename Dataset::la_type>(filename, u));
      boost::shared_ptr<Dataset>
        data_ptr(new Dataset(o_ptr->datasource_info()));
      for (size_t i = 0; i < max_records; i++) {
        if (o_ptr->next())
          data_ptr->insert(o_ptr->current(), o_ptr->weight());
        else
          return data_ptr;
      }
      return data_ptr;
    }

    /**
     * Load a SYMBOLIC dataset, given the summary file.
     *
     * @param ds           (Return value) Loaded dataset.
     * @param max_records  max number of records which will be loaded
     *                     (default = no limit)
     * @see symbolic.hpp
     */
    template <typename Dataset>
    void
    load_symbolic_dataset(const std::string& filename, universe& u,
                          Dataset& ds,
                          size_t max_records
                          = std::numeric_limits<size_t>::max()) {
      //      concept_assert((sill::Dataset<Dataset>));
      boost::shared_ptr<symbolic_oracle<typename Dataset::la_type> >
        o_ptr(load_symbolic_oracle<typename Dataset::la_type>(filename, u));
      ds = Dataset(o_ptr->datasource_info());
      for (size_t i = 0; i < max_records; i++) {
        if (o_ptr->next())
          ds.insert(o_ptr->current(), o_ptr->weight());
        else
          break;
      }
    }

    /**
     * Load a SYMBOLIC dataset, given the summary file.
     *
     * @param filename     summary file name
     * @param info         datasource info
     * @param max_records  max number of records which will be loaded
     *                     (default = no limit)
     * @see symbolic.hpp
     */
    template <typename Dataset>
    boost::shared_ptr<Dataset>
    load_symbolic_dataset
    (const std::string& filename, const datasource_info_type& info,
     size_t max_records = std::numeric_limits<size_t>::max()) {
      //      concept_assert((sill::Dataset<Dataset>));
      boost::shared_ptr<symbolic_oracle<typename Dataset::la_type> >
        o_ptr(load_symbolic_oracle<typename Dataset::la_type>(filename, info));
      boost::shared_ptr<Dataset>
        data_ptr(new Dataset(o_ptr->datasource_info()));
      for (size_t i = 0; i < max_records; i++) {
        if (o_ptr->next())
          data_ptr->insert(o_ptr->current(), o_ptr->weight());
        else
          return data_ptr;
      }
      return data_ptr;
    }

    /**
     * Load a SYMBOLIC dataset, given the summary file.
     *
     * @param filename     summary file name
     * @param info         datasource info
     * @param ds           (Return value) Loaded dataset.
     * @param max_records  max number of records which will be loaded
     *                     (default = no limit)
     * @see symbolic.hpp
     */
    template <typename Dataset>
    void
    load_symbolic_dataset
    (const std::string& filename, const datasource_info_type& info,
     Dataset& ds,
     size_t max_records = std::numeric_limits<size_t>::max()) {
      //      concept_assert((sill::Dataset<Dataset>));
      boost::shared_ptr<symbolic_oracle<typename Dataset::la_type> >
        o_ptr(load_symbolic_oracle<typename Dataset::la_type>(filename, info));
      ds = Dataset(o_ptr->datasource_info());
      for (size_t i = 0; i < max_records; i++) {
        if (o_ptr->next())
          ds.insert(o_ptr->current(), o_ptr->weight());
        else
          break;
      }
    }

    // Free functions: Utilities
    //==========================================================================

    /**
     * Print a set of variables to an output stream using their datasource
     * indices.
     * @see load_variables
     */
    template <typename VariableType>
    void save_variables
    (std::ostream& out,
     const typename variable_type_group<VariableType>::domain_type& vars,
     const datasource& ds);

    // Template specialization
    template <>
    void save_variables<finite_variable>
    (std::ostream& out, const finite_domain& vars, const datasource& ds);

    // Template specialization
    template <>
    void save_variables<vector_variable>
    (std::ostream& out, const vector_domain& vars, const datasource& ds);

    // Template specialization
    template <>
    void save_variables<variable>
    (std::ostream& out, const domain& vars, const datasource& ds);

    /**
     * Load a set of variables from an output stream using their datasource
     * indices.
     * @param vars  (Return value) Set of variables which are loaded.
     * @see load_variables
     */
    template <typename VariableType>
    void load_variables
    (std::istream& in,
     typename variable_type_group<VariableType>::domain_type& vars,
     const datasource& ds);

    // Template specialization
    template <>
    void load_variables<finite_variable>
    (std::istream& in, finite_domain& vars, const datasource& ds);

    // Template specialization
    template <>
    void load_variables<vector_variable>
    (std::istream& in, vector_domain& vars, const datasource& ds);

    // Template specialization
    template <>
    void load_variables<variable>
    (std::istream& in, domain& vars, const datasource& ds);


    //==========================================================================
    // Implementations of above functions
    //==========================================================================

    // Free functions: loading oracles
    //==========================================================================

    template <typename LA>
    boost::shared_ptr<oracle<LA> >
    load_oracle(sill::universe& u, std::string ds_name, double random_seed) {
      using namespace sill;
      boost::shared_ptr<oracle<LA> > oracle_ptr;
      if (ds_name == "knorm") {
        size_t nfeatures = 20;
        size_t k = 2;
        vector_var_vector var_order;
        std::vector<variable::variable_typenames> var_type_order
          (nfeatures, variable::VECTOR_VARIABLE);
        var_type_order.push_back(variable::FINITE_VARIABLE);
        for (size_t j = 0; j < nfeatures; j++)
          var_order.push_back(u.new_vector_variable(1));
        syn_oracle_knorm::parameters params;
        params.random_seed = random_seed;
        oracle_ptr.reset(new syn_oracle_knorm(var_order, u.new_finite_variable(k),
                                              var_type_order, params));
      } else if (ds_name == "majority") {
        size_t nfeatures = 100;
        finite_var_vector var_order;
        for (size_t j = 0; j <= nfeatures; j++)
          var_order.push_back(u.new_finite_variable(2));
        syn_oracle_majority::parameters params;
        params.random_seed = random_seed;
        oracle_ptr.reset(new syn_oracle_majority(var_order, params));
      } else {
        symbolic::parameters params(load_symbolic_summary(ds_name, u));
        oracle_ptr.reset(new symbolic_oracle<LA>(params));
      }
      return oracle_ptr;
    }

    template <typename LA>
    boost::shared_ptr<oracle<LA> >
    load_oracle(const sill::datasource_info_type& info,
                std::string ds_name, double random_seed) {
      using namespace sill;
      boost::shared_ptr<oracle<LA> > oracle_ptr;
      if (ds_name == "knorm") {
        syn_oracle_knorm::parameters params;
        params.random_seed = random_seed;
        oracle_ptr.reset(new syn_oracle_knorm(info.vector_seq,
                                              info.finite_seq.front(),
                                              info.var_type_order, params));
      } else if (ds_name == "majority") {
        syn_oracle_majority::parameters params;
        params.random_seed = random_seed;
        oracle_ptr.reset(new syn_oracle_majority(info.finite_seq, params));
      } else {
        symbolic::parameters params(load_symbolic_summary(ds_name, info));
        oracle_ptr.reset(new symbolic_oracle<LA>(params));
      }
      return oracle_ptr;
    }

    template <typename LA>
    boost::shared_ptr<symbolic_oracle<LA> >
    load_symbolic_oracle(const std::string& filename, universe& u,
                         size_t record_limit, bool auto_reset) {
      symbolic::parameters sym_params(load_symbolic_summary(filename, u));
      typename symbolic_oracle<LA>::parameters params;
      params.record_limit = record_limit;
      params.auto_reset = auto_reset;
      return boost::shared_ptr<symbolic_oracle<LA> >
        (new symbolic_oracle<LA>(sym_params, params));
    }

    template <typename LA>
    boost::shared_ptr<symbolic_oracle<LA> >
    load_symbolic_oracle
    (const std::string& filename, const datasource_info_type& info,
     size_t record_limit, bool auto_reset) {
      symbolic::parameters sym_params(load_symbolic_summary(filename,info));
      typename symbolic_oracle<LA>::parameters params;
      params.record_limit = record_limit;
      params.auto_reset = auto_reset;
      return boost::shared_ptr<symbolic_oracle<LA> >
        (new symbolic_oracle<LA>(sym_params, params));
    }

  } // namespace data_loader

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_DATASET_DATA_LOADER_HPP
