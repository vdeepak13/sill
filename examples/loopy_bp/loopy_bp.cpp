#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include <boost/progress.hpp>
#include <boost/timer.hpp>
#include <boost/program_options.hpp>

#include <sill/base/variable.hpp>
#include <sill/datastructure/dense_table.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/markov_network.hpp>
#include <sill/model/io.hpp>
#include <sill/inference/loopy/belief_propagation.hpp>

#include <sill/macros_def.hpp>

namespace po = boost::program_options;

int main(int ac, char* av[])
{
  using namespace std;

  // Parse the command line
  int verbose;
  size_t niters;
  double eta, rho;
  string algorithm;
  string output_base;
  bool output_beliefs;

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("iterations,n", po::value<size_t>(&niters)->default_value(10),
     "number of iterations")
    ("algorithm,a",po::value<string>(&algorithm)->default_value("asynchronous"),
     "the algorithm used (synchronous, asynchronous, residual, exponential)")
    ("eta,e", po::value<double>(&eta)->default_value(1), "update factor")
    ("rho,r", po::value<double>(&rho),
     "the exponent used in the randomized algorithms")
    ("verbose,v", po::value<int>(&verbose)->default_value(0)->implicit_value(1),
     "enable verbose output")
    ("output-beliefs,b", po::value<bool>(&output_beliefs)->default_value(false)->implicit_value(true), "output the beliefs")
    ("input-file", po::value<string>(), "input file")
    ("output-base", po::value<string>(&output_base), "output base");

  po::positional_options_description p;
  p.add("input-file", 1).add("output-base", 1).add("iterations", 1);

  po::variables_map vm;
  store(po::command_line_parser(ac, av).options(desc).positional(p).run(), vm);
  notify(vm); // update all the variables

  if (vm.count("help") || !vm.count("input-file") || !vm.count("output-base")) {
    cout << "Usage: " << av[0] << " [options] input_file output_base\n";
    cout << desc;
    return 0;
  }
  assert(niters>0);

  // Load the network and create the inference engine
  using namespace sill;
  typedef sill::pairwise_markov_network<table_factor> network_type;

  universe u;
  cout << "Loading the model" << endl;
  ifstream in(vm["input-file"].as<string>().c_str());
  network_type mn;
  read_model(in, mn, u);
  if(verbose) cout << mn;

  cout << "Running GBP (" << algorithm << ")";
  boost::timer t;
  boost::progress_display progress(niters);

  boost::shared_ptr< loopy_bp_engine<network_type> > p_engine;
  if (algorithm == "synchronous") {
    if (vm.count("rho")) cout << "rho=" << rho << endl; else rho = 0;
    p_engine.reset(new synchronous_loopy_bp<network_type>(mn, rho));
  } else if (algorithm == "asynchronous")
    p_engine.reset(new asynchronous_loopy_bp<network_type>(mn));
  else if (algorithm == "residual")
    p_engine.reset(new residual_loopy_bp<network_type>(mn));
  else if (algorithm == "exponential") {
    if (vm.count("rho")) cout << "rho=" << rho << endl; else rho = 1;
    p_engine.reset(new exponential_loopy_bp<network_type>(mn, rho));
  } else {
    cout << "Invalid algorithm " << algorithm << endl;
    return 1;
  }

  ofstream outb((output_base+"-beliefs.txt").c_str());
  ofstream outs((output_base+"-statistics.txt").c_str());
  double residual = 0, residual2;

  // Output the residual at each iteration
  size_t increment = max(niters/300, size_t(1));
  for(size_t i = 0; i<niters;) {
    increment = min(niters-i, increment);
    residual  = p_engine->iterate(increment, eta);
    residual2 = p_engine->expected_residual();
    i += increment;
    progress += increment;

    if (output_beliefs) { // output the beliefs
      foreach(finite_variable* v, mn.vertices())
        outb << v->name() << ' ';
      outb << -1;
      foreach(finite_variable* v, mn.vertices())
        foreach(double value, p_engine->belief(v).values())
          outb << " " << value;
      outb << endl;
    }

    // output the statistics
    outs << i << ' ' << t.elapsed() << ' '
         << p_engine->num_updates() << ' '
         << residual << ' ' << residual2 << endl;
  }

  if(verbose) {
    foreach(network_type::edge e, mn.edges()) {
      cout << e.source() << "->" << e.target() << ": "
           << p_engine->message(e);
      cout << e.target() << "->" << e.source() << ": "
           << p_engine->message(mn.reverse(e));
    }
    cout << p_engine->node_beliefs() << endl;
  }

  cout << "Maximum residual " << residual << endl;
}
