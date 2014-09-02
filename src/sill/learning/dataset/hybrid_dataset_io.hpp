#ifndef SILL_HYBRID_DATASET_IO_HPP
#define SILL_HYBRID_DATASET_IO_HPP

#include <sill/learning/dataset/hybrid_dataset.hpp>
#include <sill/learning/dataset/hybrid_memory_dataset.hpp>
#include <sill/learning/dataset/symbolic_format.hpp>

#include <fstream>
#include <iostream>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Loads a hybrid memory dataset using the symbolic format.
   * The dataset must not be initialized.
   * \relates hybrid_memory_dataset
   */
  template <typename T>
  void load(const std::string& filename,
            const symbolic_format& format,
            hybrid_memory_dataset<T>& ds) {
    var_vector vars = format.all_var_vec();
    ds.initialize(vars);

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    std::string line;
    size_t line_number = 0;
    hybrid_record<T> r(vars);
    while (std::getline(in, line)) { 
      std::vector<const char*> tokens;
      if (format.parse(r.size(), line, line_number, tokens)) {
        size_t col = format.skip_cols;
        size_t fi = 0;
        size_t vi = 0;
        foreach(const symbolic_format::variable_info& var, format.vars) {
          if (var.is_finite()) {
            r.values.finite[fi++] = var.parse(tokens[col++]);
          } else if (var.is_vector()) {
            for (size_t j = 0; j < var.size(); ++j) {
              r.values.vector[vi++] = parse_string<T>(tokens[col++]);
            }
          } else {
            throw std::logic_error("Unsupported variable type " + var.name());
          }
        }
        assert(r.values.finite.size() == fi);
        assert(r.values.vector.size() == vi);
        r.weight = format.weighted ? parse_string<T>(tokens[col]) : 1.0;
        ds.insert(r);
      }
    }
  }

  /**
   * Saves a hybrid dataset using the symbolic format.
   * \relates hybrid_dataset, hybrid_memory_dataset
   */
  template <typename T>
  void save(const std::string& filename,
            const symbolic_format& format,
            const hybrid_dataset<T>& data) {
    var_vector vars = format.all_var_vec();
    
    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    for (size_t i = 0; i < format.skip_rows; ++i) {
      out << std::endl;
    }
    
    std::string separator = format.separator.empty() ? " " : format.separator;
    foreach(const hybrid_record<T>& r, data.records(vars)) {
      for (size_t i = 0; i < format.skip_cols; ++i) {
        out << "0" << separator;
      }
      size_t fi = 0;
      size_t vi = 0;
      bool first = true;
      foreach(const symbolic_format::variable_info& var, format.vars) {
        if (var.is_finite()) {
          if (first) { first = false; } else { out << separator; }
          var.print(out, r.values.finite[fi++]);
        } else {
          for (size_t j = 0; j < var.size(); ++j) {
            if (first) { first = false; } else { out << separator; }
            out << r.values.vector[vi++];
          }
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
