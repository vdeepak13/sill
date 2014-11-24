#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include <limits>

#include <boost/program_options.hpp>

#include <sill/model/factor_graph_model.hpp>
#include <sill/model/factor_graph_partitioning.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/log_table_factor.hpp>
#include <sill/parsers/alchemy.hpp>

#include <sill/macros_def.hpp>

#include <sill/parallel/timer.hpp>

// Rename the program options namespace
namespace po = boost::program_options;

using namespace std;
using namespace sill;


typedef factor_graph_model<log_table_factor> factor_graph_model_type;
typedef factor_graph_model_type::factor_type factor_type;
typedef factor_graph_model_type::vertex_type vertex_type;
typedef factor_graph_model_type::variable_type variable_type;




int main(int argc, char* argv[]) {
  string input_file;
  size_t numslices;
  bool weighted;
  string cutting_alg;
  // Parse the input
  po::options_description desc("Allowed Options");
  desc.add_options()
    ("infn", po::value<string>(&input_file), 
     "MLN Factor graph")
    ("slices", po::value<size_t>(&numslices)->default_value(8), 
     "Number of slices")
    ("alg", po::value<string>(&cutting_alg)->default_value("kmetis"),
     "cutting algorithm (kmetis, pmetis, random, bfs)")
    ("weighted", po::value<bool>(&weighted)->default_value(false),
     "Used weighted cuts")
    ("help", "produce help message");
  
  po::positional_options_description pos_opts;
  pos_opts.add("infn",1);
  pos_opts.add("slices",1);
  po::variables_map vm;
  store(po::command_line_parser(argc, argv)
        .options(desc).positional(pos_opts).run(), vm);
  notify(vm);
  if((vm.count("help") || !vm.count("infn") || !vm.count("slices"))) {
    cout << "Usage: " << argv[0] << " infn [options] " << endl
         << desc;
    return EXIT_FAILURE;
  }

  cout << "==========================================================" << endl
       << "Program Settings: " << endl
       << "Input File:       " << input_file << endl
       << "Number of slices: " <<  numslices << endl
       << "==========================================================" << endl;

  universe universe;
  factor_graph_model_type fg;

  // Do the actual parsing.
  std::cout << "Loading..." << std::endl; std::cout.flush();
  parse_alchemy(universe, fg, input_file);
  std::cout << "Finished!" << std::endl;

  // Do the cutting
  std::cout << "Cutting..." << std::endl; std::cout.flush();
  typedef factor_graph_model_type::factor_type factor_type;
  typedef factor_graph_partition<factor_type> partition_type;

  sill::timer time;
  time.start();
  partition_type partition(fg, numslices, cutting_alg, weighted); 
  double runtime = time.current_time();
  std::cout << "Finished!" << std::endl;
  std::cout << "Running Time: " << runtime << std::endl;
  


  std::cout << "Unweighted Balance: ";
  partition.print_balance(fg, false, std::cout);
  std::cout << "Weighted Balance: ";
  partition.print_balance(fg, true, std::cout);
  std::cout << "Comm Score: ";
  partition.print_comm_score(fg, std::cout);

  // Output the results
  std::cout << "Saving results..."; std::cout.flush();
  std::stringstream strm;
  strm << input_file << ".part";
  strm.flush();
  std::cout << "Using file: " << strm.str() << std::endl;
  ofstream fout(strm.str().c_str());
  partition.print(fout);
  fout.close();
  std::cout << "Finished!" << std::endl;

} // End of main

#include <sill/macros_undef.hpp>
//End of file



  // cut the graph
//   std::vector<std::set<vertex_type> > owner2vertex;
//   std::map<vertex_type, uint32_t> vertex2owner;  
// //   if (metisin.length() > 0) {
// //     SliceGraphMetis(fg, numslices, owner2vertex,vertex2owner,true, metisin.c_str());
// //   }
// //   else {
// //     SliceGraphMetis(fg, numslices, owner2vertex,vertex2owner,true);
// //   }
  
  // output the graph according to the id order
//   if (metisout.length() > 0) {
//     ofstream fout(metisout.c_str());
//     int numvertices = fg.arguments().size() + fg.size();
//     for (int i = 0;i < numvertices; ++i) {
//       vertex_type v = fg.id2vertex(i);
//       fout << vertex2owner[v] << std::endl;
//     }
//     fout.close();
//   }




