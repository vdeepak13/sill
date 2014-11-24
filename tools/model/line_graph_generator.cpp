#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/program_options.hpp>
namespace po = boost::program_options;
using namespace std;



int main(int argc,char **argv) {
  srand(time(NULL));
  string output_filename;
  size_t length;
  double potential;
  bool maketree = false;
  po::options_description desc("Allowed Options");
  desc.add_options()
    ("outfn", po::value<string>(&output_filename),
     "file to write the beliefs")
    ("help", "produce help message")
    ("length", po::value<size_t>(&length)->default_value(100),
     "length of graph")
    ("pot", po::value<double>(&potential)->default_value(0.5),
     "attractive potential")
    ("maketree", "makes a tree instead of a chain");

  po::positional_options_description pos_opts;
  pos_opts.add("outfn",1);
  po::variables_map vm;
  store(po::command_line_parser(argc, argv)
        .options(desc).positional(pos_opts).run(), vm);
  notify(vm);

  
  if(!vm.count("outfn") || vm.count("help")) {
    cout << desc;
    cout << "Binary potentials are : pot 1 1 pot" << endl;
    cout << "Unary potentials are randomly generated" << endl;
    return 0;
  }
  maketree = (vm.count("maketree") > 0);
  cout << "Binary potential value: " << potential<< " 1 1 " << potential << endl;
  
  ofstream fout;
  fout.open(output_filename.c_str());
  assert(fout.good());
  fout << "variables: " << endl;
  for (size_t i = 0; i < length; ++i) {
    fout << i << endl;
  }
  fout << "factors: " << endl;
  // generate unary potentials
 for (size_t i = 0; i < length; ++i) {
    double r1 = i+1;
    double r2 = length-i+1;
    fout << i << " // " << std::log(r1) << " " << std::log(r2) << endl;
  }
  // generate binary potentials
	for (size_t i = 1; i < length; ++i) {
		int parent;
		if (maketree){
			parent = rand() % i;
		}
		else {
			parent = i-1;
		}
		fout << parent << " / " << i <<  " // " << potential<< " 1 1 " << potential << endl;
	}

  return 0;
}
