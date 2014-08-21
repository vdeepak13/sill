#ifndef SILL_LEARNING_DATASET_SYMBOLIC_HPP
#define SILL_LEARNING_DATASET_SYMBOLIC_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <boost/tokenizer.hpp>

#include <sill/base/string_functions.hpp>
#include <sill/base/universe.hpp>
#include <sill/learning/dataset/dataset.hpp>

#include <sill/macros_def.hpp>

/**
 * \file symbolic.hpp File defining the symbolic file format,
 *                    as well as a binary equivalent.
 *
 * Data is stored in file pairs: a summary (.sum) file and a data file.
 *
 * SUMMARY FILES:
 * The summary files holds information about the data necessary for initializing
 * the data source class.  Note that if this information is already available,
 * then the summary file is not actually needed.
 * This summary file has the following format (line by line):
 *  - dataset name
 *  - number of records
 *  - number of variables
 *  - (for each variable, a whitespace-separated line with:)
 *     - arity ('vN' means a vector of size N; 'N' means a finite variable)
 *     - variable name (optional; set to variable index by default)
 *     - (for each variable value for finite variables,) value name (optional)
 *  - data file name
 *  - newline-separated list of options
 *     - These are specified as: OPTION_NAME=option_value
 *     - These have the same defaults as the parameters for symbolic data.
 *     - Options for all data formats:
 *        - FORMAT (0 = text, 1 = binary)
 *        - CLASS_VARIABLES (whitespace-separated list of indices from 0)
 *     - Options for text data files:
 *        - SEPARATOR (string)
 *        - PREFIX (string)
 *        - INDEX_BASE (integer)
 *        - SKIPLINES (integer >= 0)
 *        - SKIPCOLS (integer >= 0)
 *        - WEIGHTED (0 = false; 1 = true)
 *     - Options for binary data files:
 *        (none currently)
 *  - description/comments (All remaining lines beginning with a pipe "|")
 *     - Note these may be interspersed with the options.
 *
 * DATA FILES:
 * For text data files:
 *  - examples are newline-separated
 *  - variable values are separated by any characters in parameter SEPARATOR
 *    (If SEPARATOR does not include whitespace, then any whitespace on
 *     either side of SEPARATOR is not cut out.)
 *     - n-vector variable values are written out as n consecutive values
 *  - variable values may optionally have a prefix PREFIX
 *    (in which case all values must have this prefix)
 *  - finite variable values are indexed from INDEX_BASE
 *  - the first SKIPLINES lines in the file are skipped
 *  - the first SKIPCOLS columns in each line are skipped
 *  - If the dataset is WEIGHTED, then each line should end with the weight for
 *    that example.
 * For binary data files:
 *  - Records are concatenated.
 *  - Records have finite variable values first, and then vector values.
 *    Each type of variable uses the order given in the summary file.
 *  - Finite values are stored as size_t; vector values are stored as doubles.
 *
 * NOTE:
 *  - SEPARATOR should not include any characters in PREFIX.
 *
 * This format should support the format used by the UCI ML Repository,
 * the symbolic format used by Anton, and the old format used by Joseph.
 */

namespace sill {

  /**
   * SYMBOLIC FILE FORMAT METADATA
   */
  namespace symbolic {

    struct parameters {

      // Required parameters
      // --------------------------------------------------------------------

      //! Data file name
      std::string data_filename;

      //! Number of records in data file
      size_t nrecords;

      //! Data format. 0 = text, 1 = binary.
      //!  (default = 0)
      size_t format;

      //! Dataset name
      std::string dataset_name;

      //! Dataset structure
      datasource_info_type datasource_info;

      //! Indices of class variables in var_type_order.
      //! (This is used for loading datasets but not saving them.
      //!  For saving, only datasource_info is used.)
      std::vector<size_t> class_variables;

      // Optional parameters: for text data format
      // --------------------------------------------------------------------

      //! Any characters used to separate variable values
      //!  (default = whitespace)
      std::string separator;

      //! Optional prefix for variable values (which must apply to all values)
      //!  (default = none)
      std::string prefix;

      //! Base value for finite variable values
      //!  (default = 0)
      int index_base;

      //! The first SKIPLINES lines in the file are skipped
      //!  (default = 0)
      size_t skiplines;

      //! The first SKIPCOLS columns in each line are skipped
      //!  (default = 0)
      size_t skipcols;

      //! Indicates if the dataset is weighted
      //!  (default = false)
      bool weighted;

      parameters()
        : data_filename(""), nrecords(0), format(0), dataset_name(""),
          separator(" \t"), prefix(""), index_base(0), skiplines(0),
          skipcols(0), weighted(false) {
      }

      /*
      //! Return the datasource info struct defined by these parameters.
      datasource_info_type datasource_info() const {
        finite_var_vector finite_class_vars;
        vector_var_vector vector_class_vars;
        size_t f_i(0);
        size_t v_i(0);
        std::set<size_t>
          class_var_set(class_variables.begin(), class_variables.end());
        for (size_t j(0); j < variable_type_ordering.size(); ++j) {
          switch(variable_type_ordering[j]) {
          case variable::FINITE_VARIABLE:
            if (class_var_set.count(j) != 0)
              finite_class_vars.push_back(finite_variable_ordering[f_i]);
            ++f_i;
            break;
          case variable::VECTOR_VARIABLE:
            if (class_var_set.count(j) != 0)
              vector_class_vars.push_back(vector_variable_ordering[v_i]);
            ++v_i;
            break;
          default:
            assert(false);
          }
        }
        assert(finite_class_vars.size() + vector_class_vars.size()
               == class_variables.size());
        return datasource_info_type
          (finite_variable_ordering, vector_variable_ordering,
           variable_type_ordering, finite_class_vars, vector_class_vars);
      }
      */

    };  // struct parameters

    //! Read in optional parameters
    parameters load_symbolic_summary_options(std::ifstream& f_in);

    /**
     * Save the given dataset in the binary symbolic format,
     * using the given options + symbolic format defaults.
     *
     * @param ds        Dataset.
     * @param filepath  Directory and filestem (minus the .sum/.data) to
     *                  save to.
     */
    template <typename LA>
    void save_binary_dataset(const dataset<LA>& ds, const std::string& filepath);

    /**
     * Save the given dataset in the text symbolic format,
     * using the given options + symbolic format defaults.
     *
     * @param ds        Dataset.
     * @param filepath  Directory and filestem (minus the .sum/.data) to
     *                  save to.
     */
    template <typename LA>
    void save_text_dataset(const dataset<LA>& ds, const std::string& filepath);

    /**
     * Save the given dataset in the symbolic format, using the given options.
     *
     * @param ds        Dataset.
     * @param filepath  Directory and filestem (minus the .sum/.data) to
     *                  save to.  This overrides params.data_filename.
     * @param params    Symbolic parameters specifying the format to save in.
     *                  Only options not specified by the dataset are used.
     */
    template <typename LA>
    void save_dataset(const dataset<LA>& ds, const std::string& filepath,
                      const parameters& params);

  } // namespace symbolic

  /**
   * Reads in a summary file for a symbolic data file and creates new
   * variables in universe u as necessary, returning the information needed
   * to construct an oracle or dataset.
   * @param filename   summary file name
   * @param u          universe in which to create the new variables
   */
  symbolic::parameters
  load_symbolic_summary(const std::string& filename, universe& u);

  /**
   * Reads in a summary file for a symbolic data file and checks to make
   * sure that the given variable ordering matches the variables in the dataset.
   * This returns the information needed to construct an oracle or dataset.
   * @param filename          summary file name
   * @param info              datasource info
   * @param check_class_vars  If true, this makes sure the info parameter's
   *                          class variables match those in the summary file.
   *                          (default = false)
   */
  symbolic::parameters
  load_symbolic_summary
  (const std::string& filename, const datasource_info_type& info,
   bool check_class_vars = false);

  //============================================================================
  // Implementations of methods in namespace symbolic
  //============================================================================

  namespace symbolic {

    template <typename LA>
    void save_binary_dataset(const dataset<LA>& ds, const std::string& filepath) {
      parameters params;
      params.format = 1;
      params.dataset_name = split_directory_file(filepath).second;
      save_dataset(ds, filepath, params);
    }

    template <typename LA>
    void save_text_dataset(const dataset<LA>& ds, const std::string& filepath) {
      assert(false); // TO BE IMPLEMENTED
    }

    template <typename LA>
    void save_dataset(const dataset<LA>& ds, const std::string& filepath,
                      const parameters& params_) {
      std::map<variable*, size_t> var_order_map(ds.variable_order_map());
      parameters params(params_);
      //      params.data_filename = filepath + ".data";
      const finite_var_vector& finite_seq = ds.finite_list();
      const vector_var_vector& vector_seq = ds.vector_list();
      std::pair<std::string, std::string>
        data_dir_file(split_directory_file(filepath));
      std::vector<size_t> class_var_vec;
      foreach(finite_variable* v, ds.finite_class_variables())
        class_var_vec.push_back(var_order_map[v]);
//        class_var_vec.push_back(ds.var_order_index(v));
      foreach(vector_variable* v, ds.vector_class_variables())
        class_var_vec.push_back(var_order_map[v]);
//        class_var_vec.push_back(ds.var_order_index(v));
      std::sort(class_var_vec.begin(), class_var_vec.end());
      if (ds.is_weighted())
        assert(false); // TO BE IMPLEMENTED
      if (params.format == 0) { // text
        assert(false); // TO BE IMPLEMENTED
      } else if (params.format == 1) { // binary
        // Save the .sum file
        std::ofstream f_out((filepath + ".sum").c_str());
        f_out << params.dataset_name << "\n"
              << ds.size() << "\n"
              << ds.num_variables() << "\n";
        size_t f_i(0);
        size_t v_i(0);
        foreach(variable::variable_typenames v_type, ds.variable_type_order()) {
          switch (v_type) {
          case variable::FINITE_VARIABLE:
            f_out << finite_seq[f_i]->size() << "\t"
                  << finite_seq[f_i]->name() << "\n";
            ++f_i;
            break;
          case variable::VECTOR_VARIABLE:
            f_out << "v" << vector_seq[v_i]->size() << "\t"
                  << vector_seq[v_i]->name() << "\n";
            ++v_i;
            break;
          default:
            assert(false);
          }
        }
        f_out << data_dir_file.second + ".data\n"
              << "FORMAT=1\n";
        if (class_var_vec.size() != 0) {
          f_out << "CLASS_VARIABLES="
                << string_join(" ", class_var_vec) << "\n";
        }
        f_out.flush();
        f_out.close();
        // Save the .data file
        FILE* f = fopen((filepath + ".data").c_str(), "w");
        assert(!ferror(f));
        size_t* finite_buffer = new size_t[finite_seq.size()];
        double* vector_buffer = new double[ds.vector_dim()];
        foreach(const record<LA>& r, ds.records()) {
          for (size_t j(0); j < finite_seq.size(); ++j)
            finite_buffer[j] = r.finite(j);
          for (size_t j(0); j < ds.vector_dim(); ++j)
            vector_buffer[j] = r.vector(j);
          size_t wc =
            fwrite(finite_buffer, sizeof(size_t), finite_seq.size(), f);
          assert(wc == finite_seq.size());
          wc = fwrite(vector_buffer, sizeof(double), ds.vector_dim(), f);
          assert(wc == ds.vector_dim());
        }
        delete(finite_buffer);
        finite_buffer = NULL;
        delete(vector_buffer);
        vector_buffer = NULL;
        fflush(f);
        fclose(f);
        f = NULL;
      } else {
        assert(false);
      }
    } // save_dataset()

  } // namespace symbolic

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DATASET_SYMBOLIC_HPP
