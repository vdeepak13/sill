#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include <limits>

#include <boost/program_options.hpp>

#include <prl/model/factor_graph_model.hpp>
#include <prl/model/factor_graph_partitioning.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/factor/log_table_factor.hpp>
#include <prl/parsers/protein.hpp>

#include <prl/macros_def.hpp>


// Rename the program options namespace
namespace po = boost::program_options;

using namespace std;
using namespace prl;


typedef factor_graph_model<log_table_factor> factor_graph_model_type;
typedef factor_graph_model_type::factor_type factor_type;
typedef factor_graph_model_type::vertex_type vertex_type;
typedef factor_graph_model_type::variable_type variable_type;




int main(int argc, char* argv[]) {
  string input_dir;
  string metisin;
  string metisout;
  int numslices;
  // Parse the input
  po::options_description desc("Allowed Options");
  desc.add_options()
    ("infn", po::value<string>(&input_dir), 
     "protein Factor graph")
    ("metisin", po::value<string>(&metisin), 
     "place to write the metis input file")
    ("metisout", po::value<string>(&metisout), 
     "place to write the metis output file")
    ("slices", po::value<int>(&numslices), 
     "Number of slices")
    ("help", "produce help message");
  
  po::positional_options_description pos_opts;
  pos_opts.add("infn",1);
  po::variables_map vm;
  store(po::command_line_parser(argc, argv)
        .options(desc).positional(pos_opts).run(), vm);
  notify(vm);
  if(((vm.count("help") || !vm.count("infn") || !vm.count("slices"))) || 
      (metisin.length() == 0 && metisout.length() == 0)) {
    cout << "Usage: " << argv[0] << " [options] --metisin= --metisout=" << endl;
    cout << "either metisin or metisout must be specified" << endl;
    cout << desc;
    return EXIT_FAILURE;
  }
  assert(numslices >= 1);
  cout << "==========================================================" << endl
       << "Program Settings: " << endl
       << "Input directory:  " << input_dir << endl
       << "Metis In: " << metisin << endl
       << "Metis Out: " << metisout << endl
       << "==========================================================" << endl;

  universe universe;
  factor_graph_model_type fg;

  // Do the actual parsing.
  std::cout << "Loading..."; std::cout.flush();
  parse_protein(universe, fg, input_dir+"/network.bin");
  
  // cut the graph
  std::vector<std::set<vertex_type> > owner2vertex;
  std::map<vertex_type, uint32_t> vertex2owner;  

  // cut the graph
  typedef factor_graph_partition<factor_type> partition_type;
  partition_type partition(fg,
                           numslices,
                           partition_type::PMETIS,
                           false);

//   if (metisin.length() > 0) { SliceGraphMetis(fg, numslices,
//     owner2vertex,vertex2owner,true, metisin.c_str()); } else {
//     SliceGraphMetis(fg, numslices, owner2vertex,vertex2owner,true);
//     }
  
  // output the graph according to the id order
  if (metisout.length() > 0) {
    ofstream fout(metisout.c_str());
    int numvertices = fg.arguments().size() + fg.size();
    for (int i = 0;i < numvertices; ++i) {
      vertex_type v = fg.id2vertex(i);
      fout << partition.vertex2part(v) << std::endl;
    }
    fout.close();
  }

} // End of main

#include <prl/macros_undef.hpp>
//End of file




