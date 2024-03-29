#ifndef SILL_VECTOR_DATASET_IO_HPP
#define SILL_VECTOR_DATASET_IO_HPP

#include <sill/learning/dataset/symbolic_format.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/dataset/vector_memory_dataset.hpp>
#include <sill/parsers/string_functions.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>

#include <boost/math/special_functions/fpclassify.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Loads a vector memory dataset using the symbolic format.
   * All the variables in the format must be vector. The dataset
   * must not be initialized.
   * \throw std::domain_error if the format contains variables that are not vector
   * \relates vector_memory_dataset
   */
  template <typename T>
  void load(const std::string& filename,
            const symbolic_format& format,
            vector_memory_dataset<T>& ds) {
    if (!format.is_vector()) {
      throw std::domain_error("The dataset contains variable(s) that are not vector");
    }
    vector_var_vector vars = format.vector_var_vec();
    ds.initialize(vars);

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    std::string line;
    size_t line_number = 0;
    vector_record<T> r(vars);
    while (std::getline(in, line)) {
      std::vector<const char*> tokens;
      if (format.parse(r.values.size(), line, line_number, tokens)) {
        size_t col = format.skip_cols;
        size_t i = 0;
        foreach (vector_variable* v, vars) {
          size_t size = v->size();
          if (std::count(&tokens[col], &tokens[col] + size, format.missing)) {
            // TODO: warning if only a subset of columns missing
            std::fill(&r.values[i], &r.values[i] + size,
                      std::numeric_limits<T>::quiet_NaN());
            col += size;
            i += size;
          } else {
            for (size_t j = 0; j < size; ++j) {
              r.values[i++] = parse_string<T>(tokens[col++]);
            }
          }
        }
        r.weight = format.weighted ? parse_string<T>(tokens.back()) : 1.0;
        ds.insert(r);
      }
    }
  }

  /**
   * Saves a vector dataset using the symbolic format.
   * All the variables in the format must be vector.
   * \throw std::domain_error if the format contains variables that are not vector
   * \relates vector_dataset, vector_memory_dataset
   */
  template <typename T>
  void save(const std::string& filename,
            const symbolic_format& format,
            const vector_dataset<T>& data) {
    if (!format.is_vector()) {
      throw std::domain_error("The dataset contains variable(s) that are not vector");
    }
    vector_var_vector vars = format.vector_var_vec();
    
    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    for (size_t i = 0; i < format.skip_rows; ++i) {
      out << std::endl;
    }
    
    std::string separator = format.separator.empty() ? " " : format.separator;
    foreach(const vector_record<T>& r, data.records(vars)) {
      for (size_t i = 0; i < format.skip_cols; ++i) {
        out << "0" << separator;
      }
      for (size_t i = 0; i < r.values.size(); ++i) {
        if (i > 0) { out << separator; }
        if (boost::math::isnan(r.values[i])) {
          out << format.missing;
        } else {
          out << r.values[i];
        }
      }
      if (format.weighted) {
        out << separator << r.weight;
      }
      out << std::endl;
    }
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
