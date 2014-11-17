#include <sill/learning/dataset/finite_memory_dataset.hpp>
#include <sill/learning/dataset/finite_dataset_io.hpp>

#include <fstream>
#include <iostream>

#include <sill/macros_def.hpp>

int main(int argc, char** argv) {
  using namespace sill;

  if (argc < 4) {
    std::cout << "Usage: symbolic_to_raw <format> <input> <output>"
              << std::endl;
  }

  universe u;
  symbolic_format format;
  format.load_config(argv[1], u);

  finite_memory_dataset ds;
  load(argv[2], format, ds);

  std::ofstream out(argv[3]);
  if (!out) {
    std::cerr << "Cannot open the output file" << std::endl;
    return 1;
  }

  foreach(const finite_record& r, ds.records(ds.arg_vector())) {
    for (size_t i = 0; i < r.values.size(); ++i) {
      out << r.values[i];
      if (i != r.values.size() - 1) {
        out << ' ';
      }
    }
    if (format.weighted) {
      out << ' ' << r.weight;
    }
    out << std::endl;
  }
}
