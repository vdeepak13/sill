#ifndef SILL_FINITE_DATASET_IO_HPP
#define SILL_FINITE_DATASET_IO_HPP

#include <sill/learning/dataset/finite_dataset.hpp>
#include <sill/learning/dataset/finite_memory_dataset.hpp>
#include <sill/learning/dataset/symbolic_format.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Loads a finite memory dataset using the symbolic format.
   * All the variables in the format must be finite. The dataset
   * must not be initialized.
   * \throw std::domain_error if the format contains variables that are not finite
   * \relates finite_memory_dataset
   */
  void load(const std::string& filename,
            const symbolic_format& format,
            finite_memory_dataset& ds) {
    if (!format.is_finite()) {
      throw std::domain_error("The dataset contains variable(s) that are not finite");
    }
    finite_var_vector vars = format.finite_var_vec();
    ds.initialize(vars);

    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    std::string line;
    size_t line_number = 0;
    finite_record r(vars);
    while (std::getline(in, line)) {
      std::vector<const char*> tokens;
      if (format.parse(vars.size(), line, line_number, tokens)) {
        for (size_t i = 0; i < vars.size(); ++i) {
          r.values[i] = format.vars[i].parse(tokens[i + format.skip_cols]);
        }
        r.weight = format.weighted ? parse_string<double>(tokens.back()) : 1.0;
        ds.insert(r);
      }
    }
  }

  /**
   * Saves a finite dataset using the symbolic format.
   * All the variables in the format must be finite.
   * \throw std::domain_error if the format contains variables that are not finite
   * \relates finite_dataset, finite_memory_dataset
   */
  void save(const std::string& filename,
            const symbolic_format& format,
            const finite_dataset& data) {
    if (!format.is_finite()) {
      throw std::domain_error("The dataset contains variable(s) that are not finite");
    }
    finite_var_vector vars = format.finite_var_vec();
    
    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Cannot open the file " + filename);
    }

    for (size_t i = 0; i < format.skip_rows; ++i) {
      out << std::endl;
    }

    std::string separator = format.separator.empty() ? " " : format.separator;
    foreach(const finite_record& r, data.records(vars)) {
      for (size_t i = 0; i < format.skip_cols; ++i) {
        out << "0" << separator;
      }
      for (size_t i = 0; i < vars.size(); ++i) {
        if (i > 0) { out << separator; }
        format.vars[i].print(out, r.values[i]);
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
