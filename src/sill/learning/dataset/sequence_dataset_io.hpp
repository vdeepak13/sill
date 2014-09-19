#ifndef SILL_SEQUENCE_DATASET_IO_HPP
#define SILL_SEQUENCE_DATASET_IO_HPP

#include <sill/learning/dataset/sequence_dataset.hpp>
#include <sill/learning/dataset/sequence_memory_dataset.hpp>
#include <sill/learning/dataset/symbolic_format.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace sill {

  /**
   * Loads a sequence memory dataset using the symbolic format.
   * Each data point (sequence) in the dataset is stored as a separate file.
   * The file is formatted as a table, with columns corresponding to the 
   * processes and rows corresponding to time steps.
   * The dataset must not be initialized.
   * \throw std::domain_error if the format contains processs that are not
   *        supported by the dataset
   * \relates sequence_memory_dataset
   */
  template <typename BaseDS>
  void load(const std::vector<std::string>& filenames,
            const symbolic_format& format,
            sequence_memory_dataset<BaseDS>& ds) {
    typedef typename BaseDS::argument_type variable_type;
    typedef discrete_process<variable_type> process_type;
    typedef typename BaseDS::sequence_record_type record_type;

    // initialize the dataset
    std::vector<process_type*> procs;
    if (!format.processes(procs)) {
      throw std::domain_error("The format contains unsupported process(es)");
    }
    ds.initialize(procs, filenames.size());

    // load the records
    record_type record(procs);
    for (size_t i = 0; i < filenames.size(); ++i) {
      load_tabular(filenames[i], format, record);
      ds.insert(record);
    }
  }

  /**
   * Saves a sequence dataset (or view) using the symbolic format.
   * Each data point (sequence) in the dataset is stored in a separate file.
   * The number of files must match the number of records in the dataset.
   * \throw std::invalid_argument if the filenames and records do not match
   * \relates sequence_dataset
   */
  template <typename BaseDS>
  void save(const std::vector<std::string>& filenames,
            const symbolic_format& format,
            const sequence_dataset<BaseDS>& ds) {
    typedef typename BaseDS::sequence_record_type record_type;
    if (filenames.size() != ds.size()) {
      throw std::invalid_argument("The number of filenames and rows does not match");
    }
    size_t i = 0;
    foreach(const record_type& record, ds.records()) {
      save_tabular(filenames[i++], format, record);
    }
  }

} // namespace sill

#endif
