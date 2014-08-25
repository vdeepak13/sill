#ifndef SILL_SYMBOLIC_FORMAT_HPP
#define SILL_SYMBOLIC_FORMAT_HPP

#include <sill/global.hpp>
#include <sill/base/universe.hpp>
#include <sill/parsers/simple_config.hpp>
#include <sill/parsers/string_functions.hpp>

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A utility class for representing the format and the columns of datasets
   * such as those in the UCI Machine Learning Repository. The user needs to
   * specify the attributes, values, and formatting parameters for the dataset;
   * these can be loaded from configuration files using load_config() and 
   * load_summary(). The datasets can be then loaded/saved using the* load()/save()
   * free functions provided in *dataset_io.hpp headers.
   *
   * \see finite_memory_dataset, vector_memory_dataset
   */
  struct symbolic_format {

    //! A class that holds formatting information for a single variable
    class variable_info {
      variable* v;
      std::vector<std::string> labels;

    public:
      //! Default constructor with empty variable.
      variable_info()
        : v(NULL) { }

      //! Constructor for a generic variable
      explicit variable_info(variable* v)
        : v(v) { }

      //! Constructor for a finite variable with named values.
      variable_info(finite_variable* v, const std::vector<std::string>& values)
        : v(v), labels(values) {
        assert(v->size() == labels.size());
      }

      //! Returns the arity / dimensionality of the variable
      size_t size() const {
        return v->size();
      }

      //! Returns the name of the variable
      const std::string& name() const {
        return v->name();
      }

      //! Returns the values vector
      const std::vector<std::string>& values() const {
        return labels;
      }

      //! Returns true if the variable is finite
      bool is_finite() const {
        return v->type() == variable::FINITE_VARIABLE;
      }

      //! Returns true if the variable is finite and uses unnamed values.
      bool is_plain_finite() const {
        return is_finite() && labels.empty();
      }

      //! Returns true if the variable is finite and uses named values.
      bool is_named_finite() const {
        return is_finite() && !labels.empty();
      }

      //! Returns true if the variable is vector
      bool is_vector() const {
        return v->type() == variable::VECTOR_VARIABLE;
      }

      //! Returns the underlying variable
      variable* var() const {
        return v;
      }

      //! Casts the variable to a finite variable. Returns NULL on error.
      finite_variable* as_finite() const { 
        return dynamic_cast<finite_variable*>(v);
      }

      //! Casts the variable to a vector variable. Returns NULL on error.
      vector_variable* as_vector() const {
        return dynamic_cast<vector_variable*>(v);
      }

      //! Returns the finite value corresponding to the given string.
      //! Requires that the variable is finite.
      size_t parse(const char* str) const {
        assert(is_finite());
        if (labels.empty()) {
          return parse_string<size_t>(str);
        }
        std::vector<std::string>::const_iterator it =
          std::find(labels.begin(), labels.end(), str);
        if (it == labels.end()) {
          std::ostringstream os;
          os << "Unknown value " << str << " for variable " << v->name();
          throw new std::invalid_argument(os.str());
        }
        return it - labels.begin();
      }

      //! Prints the finite value to a stream
      void print(std::ostream& out, size_t value) const {
        assert(is_finite());
        if (labels.empty()) {
          out << value;
        } else {
          assert(value < labels.size());
          out << labels[value];
        }
      }
    };

    //! Specifies the separator for the fields (default whitespace)
    std::string separator;

    //! The number of lines at the beginning of the file to skip (default = 0)
    size_t skip_rows;

    //! The number of columns at the beginning of each line to skip (default = 0)
    size_t skip_cols;

    //! Indicates if the dataset is weighted (default = false)
    bool weighted;

    //! The variables in the dataset, one for each column past skip_cols
    std::vector<variable_info> vars;

    /**
     * Constructs the symbolic format with default parameters.
     */
    symbolic_format()
      : skip_rows(0), skip_cols(0), weighted(false) { }

    /**
     * Returns the vector of variables cast to finite_variable.
     * \throw domain_error if some of the variables are not finite.
     */
    finite_var_vector finite_vars() const {
      finite_var_vector result;
      foreach (const variable_info& info, vars) {
        if (info.is_finite()) {
          result.push_back(info.as_finite());
        } else {
          throw std::domain_error("Variable " + info.name() + " is not finite");
        }
      }
      return result;
    }

    /**
     * Returns the vector of variables cast to vector_variable.
     * \throw domain_error if some of the variables are not vector.
     */
    vector_var_vector vector_vars() const {
      vector_var_vector result;
      foreach (const variable_info& info, vars) {
        if (info.is_vector()) {
          result.push_back(info.as_vector());
        } else {
          throw std::domain_error("Variable " + info.name() + " is not vector");
        }
      }
      return result;
    }

    /**
     * Returns the vector of variables in this format.
     */
    var_vector all_vars() const {
      var_vector result;
      foreach (const variable_info& info, vars) {
        result.push_back(info.var());
      }
      return result;
    }

    /**
     * Parses a line according to the format and stores the result in an array
     * of C strings.
     * \param line input (modified)
     * \param current line number (updated)
     * \param tokens output tokens
     * \throw runtime_error if the number of columns in the input data does not
     *        match the format.
     * \return false if the line should be ignored
     */
    bool parse(size_t num_values,
               std::string& line,
               size_t& line_number,
               std::vector<const char*>& tokens) const {
      if (++line_number <= skip_rows) {
        return false;
      }
      string_split(line, separator.empty() ? "\t " : separator, tokens);
      if (tokens.empty()) {
        return false;
      }
      size_t expected_cols = skip_cols + num_values + weighted;
      if (tokens.size() != expected_cols) {
        std::ostringstream os;
        os << "Line " << line_number << ": invalid number of columns "
           << "(expected " << expected_cols << ", found " << tokens.size() << ")";
        throw std::runtime_error(os.str());
      }
      return true;
    }

    /**
     * Loads the symbolic_format from a configuration file with the following
     * format:
     *
     * [variables]
     * variable_name=value0,value1,...,valuek-1 (where k >= 2) OR
     * variable_name=finite(k) (where k >= 2) OR
     * variable_name=vector(k) (where k >= 1) 
     * other_variable_name=...
     *
     * [options]
     * separator="\t" (optional)
     * skip_rows=1   (optional)
     * skip_cols=0    (optional)
     * weighted=1     (optional)
     *
     * Comments can be prepended with #. Whitespace is ignored. Sections order
     * can be swapped.
     */
    void load_config(const std::string& filename, universe& u) {
      simple_config config;
      config.load(filename);
      typedef std::pair<std::string, std::string> config_entry;

      // load the variables
      foreach(const config_entry& entry, config["variables"]) {
        if (entry.second.compare(0, 7, "vector(") == 0) {
          std::string param = entry.second.substr(7, entry.second.size() - 8);
          size_t dim = parse_string<size_t>(param);
          if (dim == 0) {
            throw std::out_of_range(entry.first + ": vector variables must have dim. > 0");
          }
          vars.push_back(variable_info(u.new_vector_variable(entry.first, dim)));
        } else if (entry.second.compare(0, 7, "finite(") == 0) {
          std::string param = entry.second.substr(7, entry.second.size() - 8);
          size_t arity = parse_string<size_t>(param);
          if (arity <= 1) {
            throw std::out_of_range(entry.first + ": finite variables must have arity > 1");
          }
          vars.push_back(variable_info(u.new_finite_variable(entry.first, arity)));
        } else { // finite variable with named values
          std::vector<std::string> values;
          string_split(entry.second, ",", values);
          if (values.size() <= 1) {
            throw std::out_of_range(entry.first + ": finite variables must have arity > 1");
          }
          finite_variable* v = u.new_finite_variable(entry.first, values.size());
          vars.push_back(variable_info(v, values));
        }
      }

      if (vars.empty()) {
        throw std::out_of_range("Please specify at least one variable using [variables]");
      }

      // load the parameters
      foreach(const config_entry& entry, config["options"]) {
        if (entry.first == "separator") {
          separator = parse_escaped(entry.second);
        } else if (entry.first == "skip_rows") {
          skip_rows = parse_string<size_t>(entry.second);
        } else if (entry.first == "skip_cols") {
          skip_cols = parse_string<size_t>(entry.second);
        } else if (entry.first == "weighted") {
          weighted = parse_string<bool>(entry.second);
        } else {
          std::cerr << "Invalid option \"" << entry.first << "\", ignoring" << std::endl;
        }
      }
    }

    /**
     * Saves the symbolic_format to a configuraiton file with the format
     * given in load_config().
     */
    void save_config(const std::string& filename) const {
      simple_config config;
      
      // store the variables
      foreach(const variable_info& info, vars) {
        if (info.is_vector()) {
          std::string dim = to_string(info.size());
          config.add("variables", info.name(), "vector(" + dim + ")");
        } else if (info.is_plain_finite()) {
          std::string arity = to_string(info.size());
          config.add("variables", info.name(), "finite(" + arity + ")");
        } else if (info.is_named_finite()) {
          config.add("variables", info.name(), string_join(",", info.values()));
        } else {
          throw std::logic_error("Unsupported variable type " + info.name());
        }
      }

      // store the options
      config.add("options", "separator", escape_string(separator));
      config.add("options", "skip_rows", skip_rows);
      config.add("options", "skip_cols", skip_cols);
      config.add("options", "weighted", weighted);

      // save the config to the output file
      config.save(filename);
    }

  }; // struct symbolic_format

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
