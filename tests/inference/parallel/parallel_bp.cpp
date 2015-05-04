#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <map>
#include <limits>


#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/canonical_table.hpp>
#include <boost/program_options.hpp>
#include <sill/parallel/timer.hpp>

#include <sill/parsers/detect_file_format.hpp>
#include <sill/parsers/uai_parser.hpp>
#include <sill/parsers/alchemy.hpp>
#include <sill/parsers/protein.hpp>
#include <sill/inference/interfaces.hpp>
#include <sill/inference/parallel/parallel_roundrobin_bp.hpp>
#include <sill/inference/parallel/parallel_bsplash_bp.hpp>
#include <sill/inference/parallel/parallel_bsplash01_bp.hpp>
#include <sill/inference/parallel/parallel_msplash_bp.hpp>
#include <sill/inference/parallel/parallel_wildfire_bp.hpp>
#include <sill/inference/parallel/parallel_synchronous_bp.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/macros_def.hpp>


// Rename the program options namespace
namespace po = boost::program_options;

using namespace sill;

typedef canonical_table factor_type;
typedef factor_graph_model<factor_type> model_type;
typedef factor_graph_inference<model_type> engine_type;

/**
 A list of all the engines
*/
enum engine_list {
  BLF_SPLASH,
  BLF_SPLASH01,
  MSG_SPLASH,
  ROUND_ROBIN,
  WILDFIRE,
  SYNCHRONOUS
};

/**
  Input parameters to the program
*/
struct input_parameters {
  std::string input_filename;   // input factor graph file
  std::string output_filename;  // output belief file
  double bound;                 // BP termination bound
  double damping;               // amount of damping. 0.0 is no damping
  double timeout;
  size_t splashv;               // splash volume
  size_t numprocs;              // #processors
  std::string evidencefile;     // File containing evidence in UAI format
  std::string uaioutput;        // File to output UAI format style beliefs


  std::string enginename;       // Engine type as a string
  engine_list e_enginename;     // Engine type as an enumeration.
                                // Matches enginename

  std::string truthdir;         // ground truth for MAP assignments
                                // (alchemy directory format)

  std::string belieffile;       // ground truth for belief values
                                // (alchemy format)

  std::string statusfile;       // file to output inference status (such as
                                // runtime, scores, etc)
  std::string proteintruthfile;
};

/*
  A struct which contains the ground truth results for MAP and Marginals
*/
struct ground_truths {
  bool hasmap;          // true if mapassg is set
  finite_assignment mapassg;   // ground truth map assignment

  bool hasbeliefs;     // true if truebeliefs is set
  std::map<variable, canonical_table> truebeliefs; // ground truth marginals
  
  bool hasprotein;
  protein_truth_asg_type proteintruth;
};

/**
  Various statistics of the inference result
*/
struct result_statistics {
  double bethe;         // Bethe Energy
  double inf_maplogprob;// Log likelihood of inference MAP
  double executiontime; // execution time in seconds


  bool hasmapscores;    // The map scores are valid only if this is true
  double mapscore;      // % of MAP variables correct as compared to ground truth


  bool hasblfscores;    // the norm1error scores are valid only if this is true
  double blf_totalnorm1error; // total norm1 error of all variable beliefs
                              // compared ground truth
  double blf_avgnorm1error;   // average norm1 error of all variable beliefs
                              // compared ground truth
  double blf_maxnorm1error;
  bool hasproteinscore;
  double proteinscore;
  std::map<std::string, double> custom;
};

/**
  Saves the MAP assignments
*/
void save_maps(model_type& fg,
               finite_assignment mapassg,
               std::string& output_filename) {
  // Create an ouptut filestream
  std::ofstream fout(output_filename.c_str());
  assert(fout.good());
  foreach(variable v, fg.arguments()) {
    if (mapassg[v]) fout << v->name() << std::endl;
  }
  fout.close();
}




/**
  Converts a name of an engineto the corresponding enum entry
*/
engine_list name_to_engine_list(std::string name) {
  if (name == "blfsplash") {
    return BLF_SPLASH;
  }
  else if (name == "blfsplash01") {
    return BLF_SPLASH01;
  }
  else if (name == "msgsplash") {
    return MSG_SPLASH;
  }
  else if (name == "roundrobin") {
    return ROUND_ROBIN;
  }
  else if (name == "wildfire") {
    return WILDFIRE;
  }
  else if (name == "synchronous") {
    return SYNCHRONOUS;
  }
  else {
    std::cerr << "Invalid Engine Name!" << std::endl;
    exit(0);
  }
}
/**
  Creates an engine of the specified type and initializes it with parameters
  given. This should only be called by the root node
*/
engine_type* create_engine(engine_list elist,
                           model_type* fg,
                           size_t splashv,
                           size_t numprocs,
                           double bound,
                           double damping,
                           double timeout) {
  switch (elist) {
    case BLF_SPLASH: {
        parallel_bsplash_bp<factor_type>* engine =
                                    new parallel_bsplash_bp<factor_type>();
        engine->initialize(fg,
                           splashv,
                           numprocs,
                           bound,
                           damping,
                           timeout);
        return engine;
      }
    case BLF_SPLASH01: {
        parallel_bsplash01_bp<factor_type>* engine =
                                    new parallel_bsplash01_bp<factor_type>();
        engine->initialize(fg,
                           splashv,
                           numprocs,
                           bound,
                           damping,
                           timeout);
        return engine;
      }
    case MSG_SPLASH: {
        parallel_msplash_bp<factor_type>* engine =
                                    new parallel_msplash_bp<factor_type>();
        engine->initialize(fg,
                           splashv,
                           numprocs,
                           bound,
                           damping,
                           timeout);


        return engine;
      }
    case ROUND_ROBIN: {
        parallel_roundrobin_bp<factor_type>* engine =
                                    new parallel_roundrobin_bp<factor_type>();
        engine->initialize(fg,
                           numprocs,
                           bound,
                           damping,
                           timeout);

        return engine;
      } 
    case WILDFIRE: {
        parallel_wildfire_bp<factor_type>* engine =
                                    new parallel_wildfire_bp<factor_type>();
        engine->initialize(fg,
                           numprocs,
                           bound,
                           damping,
                           timeout);

        return engine;
      }
    case SYNCHRONOUS: {
        parallel_synchronous_bp<factor_type>* engine =
                                    new parallel_synchronous_bp<factor_type>();
        engine->initialize(fg,
                           numprocs,
                           bound,
                           damping,
                           timeout);

        return engine;
      }
    default:
      std::cerr << "Engine Type invalid!" << std::endl;
      exit(0);
  }
}



/**
  Prints the inference parameters
*/
void print_input_args(input_parameters &params) {
  std::cout << "==========================================================" << std::endl
      << "Program Settings: " << std::endl
      << "Input filename:  " << params.input_filename << std::endl
      << "Output filename: " << params.output_filename << std::endl
      << "Inference Engine:         " << params.enginename<< std::endl
      << "#procs:           " << params.numprocs << std::endl
      << "bound:           " << params.bound << std::endl
      << "splash volume:   " << params.splashv << std::endl
      << "damping:         " << params.damping << std::endl
      << "==========================================================" << std::endl;

}

/**
  Parses the ground truth information
*/
void parse_ground_truths(universe &u,
                        input_parameters &param, ground_truths &truths) {
  truths.hasbeliefs = false;
  truths.hasmap = false;
  truths.hasprotein = false;
  // if truth directory is set, parse it and store in 'truths'
  if (param.truthdir.length() > 0) {
    assert(alchemy_parse_truthdir(u, truths.mapassg,param.truthdir,true));
    truths.hasmap = true;
    std::cout << "Parsed MAP truth data" << std::endl;
  }

  // if belieffile is set, parse it and store in 'truths'
  if (param.belieffile.length() > 0) {
    assert(alchemy_parse_belief_file(u, truths.truebeliefs,param.belieffile));
    truths.hasbeliefs = true;
    std::cout << "Parsed Marginal truth data" << std::endl;
    // this also means we have map
    // so if map is not set, do it.
    if (truths.hasmap == false) {
      truths.hasmap = true;
      foreach(variable v, keys(truths.truebeliefs)) {
        finite_assignment localmapassg = arg_max(truths.truebeliefs[v]);
        truths.mapassg[v] = localmapassg[v];
      }
    }
  }
  
  if (param.proteintruthfile.length() > 0) {
    protein_load_truth_data_from_file(param.proteintruthfile, truths.proteintruth);
    truths.hasprotein = true;
    std::cout << "Parsed Protein truth data" << std::endl;
  }
}



std::map<variable, canonical_table>
      collect_beliefs(finite_domain vars, engine_type &engine) {

  std::map<variable, canonical_table> ret;
  foreach(variable f, vars) {
    ret[f] = engine.belief(f);
  }

  return ret;
}

/**
  Computes some statistics of the inference results as well as
  compare against ground truth if available
*/
void inference_statistics(model_type &fg,
                          engine_type &engine,
                          ground_truths &truths,
                          result_statistics &stats) {
  // extract the beliefs
  finite_assignment mapassg;
  engine.map_assignment(mapassg);

  std::map<variable, canonical_table> blfs =
                                    collect_beliefs(fg.arguments(), engine);

  // some statistics which do not depend on the truth
  stats.bethe = fg.bethe(engine.belief());
  stats.inf_maplogprob = fg.log_likelihood(mapassg);

 
  stats.hasmapscores = truths.hasmap;
  stats.hasblfscores = truths.hasbeliefs;
  stats.hasproteinscore = truths.hasprotein;
  stats.blf_maxnorm1error = 0.0;
  // if truth data is provided, print some vital statistics for comparison
  if (truths.hasmap) {
    stats.mapscore = double(assignment_agreement(mapassg, truths.mapassg))
                      / truths.mapassg.size();
  }

  if (truths.hasbeliefs) {
    // compute the norm1 error over all the beliefs
    factor_norm_1<canonical_table> norm;
    stats.blf_totalnorm1error = 0.0;
    foreach(variable i, fg.arguments()) {
      double d = norm(truths.truebeliefs[i],engine.belief(i));
      stats.blf_totalnorm1error += d;
      stats.blf_maxnorm1error = std::max(stats.blf_maxnorm1error, d);
    }
    stats.blf_avgnorm1error =  stats.blf_totalnorm1error / truths.truebeliefs.size();
  }

  if (truths.hasprotein) {
    stats.proteinscore = protein_compute_error(fg, engine, truths.proteintruth);
  }
  stats.custom = engine.get_profiling_info();
}


std::string basename(std::string filename) {
  if (filename.length() == 0) return "";
  // cut out a trailing / if one exists (input_filename may be a directory)
  if (filename[filename.length() - 1] == '/') {
    filename= filename.substr(0, filename.length() - 1);
  }
  // search for the last "/"
  std::string ret;
  size_t slashloc = filename.find_last_of('/');
  
  if (slashloc != std::string::npos) {
    // crop out what is after the splash
    ret = filename.substr(slashloc+1);
  }
  else {
    // if not found, the whole filename is the basename
    ret = filename;
  }
  return ret;
}

/** Outputs the statistics in the stats structure.
   This will write to the outputfile if outputfile is not an empty string
*/
void output_statistics(input_parameters &params,
                       result_statistics &stats,
                       std::string outputfile) {
  // write to a stringstream so we can dump the results to 2 places
  std::stringstream output;
  output.precision(10);

  
  output << "Input File: " << params.input_filename << std::endl;
  if (params.evidencefile.length() > 0) {
    output << "Evidence: " << params.evidencefile << std::endl;
  }
  output << "Engine: " << params.enginename << std::endl;
  
  if (params.e_enginename == BLF_SPLASH || params.e_enginename == MSG_SPLASH) {
    output << "Splashsize: " << params.splashv << std::endl;
  }
  output << "Bound: " << params.bound << std::endl;
  output << "Damping: " << params.damping << std::endl;
  output << "Numproc: " << params.numprocs << std::endl;

  output << "Energy: " << stats.bethe << std::endl;
  output << "Inference MAP Log Likelihood: " << stats.inf_maplogprob << std::endl;
  output << "Execution Time: " << stats.executiontime << std::endl;

  // if truth data is provided, print some vital statistics for comparison
  if (stats.hasmapscores) {
    output << "Truth Score: " << stats.mapscore << std::endl;
  }
  if (stats.hasblfscores) {
    output << "Total L1 Error: " << stats.blf_totalnorm1error << std::endl;
    output << "Max L1 Error: " << stats.blf_maxnorm1error << std::endl;
    output << "Average L1 Error: " << stats.blf_avgnorm1error << std::endl;
  }
  if (stats.hasproteinscore) {
    output << "Protein Score: " << stats.proteinscore << std::endl;
  }
  foreach(std::string s, keys(stats.custom)) {
    output << s << ": " << stats.custom[s] << std::endl;
  }
  std::cout << output.str();

  // write to the output file
  if (outputfile.length() > 0){
    std::ofstream fout(outputfile.c_str());
    if (fout.fail()) {
      std::cout << "Unable to open " << outputfile << " for output." << std::endl;
    }
    else {
      fout << output.str();
    }
  }
}

// To be executed by the root MPI node
void exec(input_parameters &params) {
    universe u;
    model_type fg;
    ground_truths truths;
    result_statistics stats;

    // read the input data
    if (parse_factor_graph(params.input_filename, u, fg) == false) {
      std::cout << "Unable to parse " << params.input_filename << std::endl;
      return;
    }
    
    // integrate evidence if there is any evidence
    if (params.evidencefile.length() > 0) {
      fg.integrate_evidence(parse_uai_evidence(u, params.evidencefile));
    }
    
    // simplify and normalize the factor graph
    fg.simplify_stable();
    fg.normalize();
    fg.randomly_shuffle_neighbors();
    fg.simplify_stable();


    // displays the degree distribution
    std::cout << "Degree Distribution " << std::endl;
    fg.print_degree_distribution();
    std::cout << "Finished parsing: " << fg.arguments().size()
        << " variables and " << fg.size() << " factors."
        << std::endl;

    // parse the ground truth data
    parse_ground_truths(u,params,truths);

    std::cout << "Creating Engine:" << std::endl;
    engine_type *engine = create_engine(params.e_enginename,
                                        &fg,
                                        params.splashv,
                                        params.numprocs,
                                        params.bound,
                                        params.damping,
                                        params.timeout);

    // execute BP
    stats.executiontime = engine->loop_to_convergence();

    //collect inference statistics
    inference_statistics(fg, *engine, truths, stats);

    output_statistics(params, stats, params.statusfile);


    std::map<variable, canonical_table> blfs =
                                    collect_beliefs(fg.arguments(), *engine);
    // save results if asked to
    if (params.uaioutput.length() > 0) {
      write_uai_beliefs(u, blfs, params.uaioutput);
    }

    write_alchemy_beliefs(u, blfs, params.output_filename);
    delete engine;
}

int main(int argc, char* argv[]) {
 

  input_parameters params;
  // Parse the input
  po::options_description desc("Allowed Options");
  desc.add_options()
    ("infn", po::value<std::string>(&params.input_filename),
     "MLN Factor graph")
    ("outfn", po::value<std::string>(&params.output_filename),
     "file to write the beliefs")
    ("help", "produce help message")
    ("bound", po::value<double>(&params.bound)->default_value(0.001),
     "accuracy bound")
    ("numprocs", po::value<size_t>(&params.numprocs)->default_value(1),
     "#processors")
    ("splashv", po::value<size_t>(&params.splashv)->default_value(100),
     "volume of splash. Only used for enginetype=blfsplash/msgsplash")
    ("damping", po::value<double>(&params.damping)->default_value(0.3),
     "amount of damping. (0 is no damping)")
    ("evidence", po::value<std::string>(&params.evidencefile)->default_value(""),
     "file containing evidence")
    ("uaioutput", po::value<std::string>(&params.uaioutput)->default_value(""),
     "uai format output")
    ("truthdir", po::value<std::string>(&params.truthdir)->default_value(""),
     "directory containing true assignments in alchemy form")
    ("proteintruth", po::value<std::string>(&params.proteintruthfile)->default_value(""),
     "file containing protein truth data")
    ("belieffile", po::value<std::string>(&params.belieffile)->default_value(""),
     "file containing true belief values in alchemy form")
    ("enginetype", po::value<std::string>(&params.enginename)->default_value("blfsplash"),
     "blfsplash, msgsplash, roundrobin, wildfire,synchronous")
    ("statusfile", po::value<std::string>(&params.statusfile)->default_value(""),
     "File to output inference status results (runtime, scores, etc)")
    ("timeout", po::value<double>(&params.timeout)->default_value(0.0),
     "timeout in seconds");


  po::positional_options_description pos_opts;
  pos_opts.add("infn",1).add("outfn",1);
  po::variables_map vm;
  store(po::command_line_parser(argc, argv)
        .options(desc).positional(pos_opts).run(), vm);
  notify(vm);

  params.e_enginename = name_to_engine_list(params.enginename);

  if(vm.count("help") || !vm.count("infn") || (!vm.count("outfn"))) {
    std::cout << "Usage: " << argv[0] << " [options] infn outfn" << std::endl;
    std::cout << desc;
    return EXIT_FAILURE;
  }
  print_input_args(params);
  exec(params);

  return (EXIT_SUCCESS);
} // End of main

#include <sill/macros_undef.hpp>
//End of file
