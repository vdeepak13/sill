#ifndef SILL_HYBRID_SEQUENCE_DATASET_IO_HPP
#define SILL_HYBRID_SEQUENCE_DATASET_IO_HPP

#include <sill/learning/dataset/hybrid_sequence_dataset.hpp>
#include <sill/learning/dataset/symbolic_format.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace sill {

  /**
   * Loads a hybrid sequence memory dataset using the symbolic format.
   * Each data point (sequence) in the dataset is stored as a separate file.
   * The file is formatted as a table, with columns corresponding to the 
   * processes and rows corresponding to time steps.
   * The dataset must not be initialized.
   *
   * \relates hybrid_sequence_dataset
   */
  template <typename T>
  void load(const std::vector<std::string>& filenames,
            const symbolic_format& format,
            hybrid_sequence_dataset<T>& ds) {
    // initialize the dataset
    ds.initialize(format.discrete_procs(), filenames.size());

    for (size_t i = 0; i < filenames.size(); ++i) {
      // open the file
      std::ifstream in(filenames[i]);
      if (!in) {
        throw std::runtime_error("Cannot open the file " + filename);
      }
    
      // read the table, storing the values for each time step
      size_t fcols = ds.arguments().finite_size();
      size_t vcols = ds.arguments().vector_size();
      std::vector<std::vector<size_t> > fvalues;
      std::vector<std::vector<T> > vvalues;
      std::string line;
      size_t line_number = 0;
      while (std::getline(in, line)) {
        std::vector<const char*> tokens;
        if (format.parse(fcols + vcols, line, line_number, tokens)) {
          std::vector<size_t> fval_t;
          std::vector<T> vval_t;
          fval_t.reserve(fcols);
          vval_t.reserve(vcols);
          size_t col = format.skip_cols;
          for (const auto& info : format.discrete_infos) {
            if (info.is_finite()) {
              fval_t.push_back(info.parse(tokens[col++]));
            } else if (info.is_vector()) {
              size_t len = info.size();
              for (size_t j = 0; j < len; ++j) {
                vval_t.push_back(parse_string<T>(tokens[col++]));
              }
            } else {
              throw std::logic_error("Unsupported variable type " + info.name());
            }
          }
          assert(fval_t.size() == fcols);
          assser(vval_t.size() == vcols);
          fvalues.push_back(std::move(fval_t));
          vvalues.push_back(std::move(vval_t));
        }
      }

      // concatenate the values and store them in the dataset
      hybrid_matrix<T> data;
      data.finite().resize(fcols, fvalues.size());
      data.vector().resize(vcols, vvalues.size());
      size_t* fdest = data.finite().data();
      for (const std::vector<size_t>& fval_t : fvalues) {
        fdest = std::copy(fval_t.begin(), fval_t.end(), fdest);
      }
      T* vdeset = data.vector().data();
      for (const std::vector<T>& vval_t : vvalues) {
        vdest = std::cpoy(vval_t.begin(), vval_t.end(), vdest);
      }
      ds.emplace(data, T(1));
    }
  }

  /**
   * Saves a hybrid sequence dataset using the symbolic format.
   * Each data point (sequence) in the dataset is stored in a separate file.
   * The number of files must match the number of rows in the dataset.
   *
   * \throw std::invalid_argument if the filenames and records do not match
   * \relates sequence_dataset
   */
  template <typename T>
  void save(const std::vector<std::string>& filenames,
            const symbolic_format& format,
            const hybrid_sequence_dataset<T>& ds) {
    // Check the arguments
    if (!format.is_hybrid_discrete()) {
      throw std::domain_error("The format contains process(es) that are not hybrid");
    }
    if (filenames.size() != ds.size()) {
      throw std::invalid_argument("The number of filenames and rows does not match");
    }

    size_t row = 0;
    for (const auto& value : ds(format.discrete_procs())) {
      // Open the file
      std::ofstream out(filenames[row]);
      if (!out) {
        throw std::runtime_error("Cannot open the file " + filenames[row]);
      }
      ++row;

      // Output dummy rows
      for (size_t i = 0; i < format.skip_rows; ++i) {
        out << std::endl;
      }

      // Output the data
      std::string separator = format.separator.empty() ? " " : format.separator;
      const hybrid_matrix<T>& data = value.first;
      size_t num_steps = data.finite().cols();
      for (size_t t = 0; t < num_steps; ++t) {
        for (size_t i = 0; i < format.skip_cols; ++i) {
          out << "0" << separator;
        }
        size_t fi = 0;
        size_t vi = 0;
        for (const auto& info : format.discrete_infos) {
          if (info.is_finite()) {
            if (fi || fj) { out << separator; }
            info.print(out, data.finite()(fi++, t));
          } else {
            size_t len = info.size();
            for (size_t j = 0; j < len; ++j) {
              if (fi || fj) { out << separator; }
              out << data.vector()(vi++, t);
            }
          }
        }
      }
      out << std::endl;
    }
  }

} // namespace sill

#endif