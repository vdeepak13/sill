/**
 * \file dataset_converter.cpp  This is a utility for loading a dataset and
 *                              converting it to another format.
 *
 * @todo This should be in a tools folder, not in a tests folder, but
 *       we don't have things organized yet.
 * @todo Make this support more types.
 */

#include <sill/argument/universe.hpp>
#include <sill/learning/dataset_old/data_loader.hpp>
#include <sill/learning/dataset_old/vector_dataset.hpp>

static int usage() {
  std::cerr << "usage: ./dataset_converter [input .sum filepath]"
            << " [output filepath, minus .sum/.data extension]\n"
            << "  This currently converts only to binary--to be extended."
            << std::endl;
  return -1;
}

int main(int argc, char** argv) {

  using namespace sill;

  if (argc != 3)
    return usage();
  std::string input_filepath(argv[1]);
  std::string output_filepath(argv[2]);

  universe u;
  boost::shared_ptr<vector_dataset_old<> > ds_ptr =
    data_loader::load_symbolic_dataset<vector_dataset_old<> >(input_filepath, u);

  symbolic::save_binary_dataset(*ds_ptr, output_filepath);

  return 0;
}
