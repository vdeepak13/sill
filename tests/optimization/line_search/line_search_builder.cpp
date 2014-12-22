#include <sill/optimization/line_search/line_search_builder.hpp>

#include <armadillo>

#include <boost/shared_ptr.hpp>

int main(int argc, char** argv) {
  using namespace sill;

  // Register the options
  namespace po = boost::program_options;
  po::options_description desc("line_search_builder test");
  desc.add_options()("help", "Print command options");
  line_search_builder<arma::vec> builder;
  builder.add_options(desc);

  // Parse the options
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  // Print help if requested
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  
  // Create the line_search object
  boost::shared_ptr<line_search<arma::vec> > ls(builder.get());
  // TODO: change to std::unique_ptr
  std::cout << *ls << std::endl;
  
  return 0;
}
