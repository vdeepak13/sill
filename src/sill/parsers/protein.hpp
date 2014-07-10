#ifndef SILL_PROTEIN_HPP
#define SILL_PROTEIN_HPP
#include <cmath>
#include <limits>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <ios>
#include <string>

#include <sill/math/logarithmic.hpp>
#include <sill/base/universe.hpp>
#include <sill/model/factor_graph_model.hpp>

#include <sill/macros_def.hpp>
namespace sill{

  class binary_input_stream : public std::ifstream {
    typedef std::ifstream base_type;
    using base_type::bad;
    public:
      binary_input_stream(const char* fname) :
        base_type(fname, std::ios::binary | std::ios::in) {
        assert(bad() == false);
      }

    template<typename T> T read() {
      T t;
      base_type::read(reinterpret_cast<char*>(&t), sizeof(T));
      if(bad()) {
        std::cout << "Error reading file!" << std::endl;
        assert(false);
      }
      return t;
    }
  };
  namespace proteinparsing {
  double bound_infinity(double logvalue) {
    if (logvalue < - std::numeric_limits<double>::max()) {
      logvalue = -std::numeric_limits<double>::max();
    }
    return logvalue;
  }
  }
  /**
   * This function fills in a factor graph with with the protein file
   * provided in filename.  The function returns true if parsing
   * succeeded.
   */
   template<typename F>
  bool parse_protein(universe& universe,
                     factor_graph_model<F>& fg,
                     const std::string& filename) {

    typedef typename factor_graph_model<F>::variable_type variable_type;

    binary_input_stream bis(filename.c_str());
    size_t  numvars = bis.read<int32_t>();
    size_t  numedges = bis.read<int32_t>();
    if (bis.fail()) return false;
    std::vector<variable_type*> idtovar;
    idtovar.resize(numvars);
    // ignore the next 2 * num_edges * int32_t bytes
    // this will contain the list of edges which we do not need
    // we can construct this from the factors later on
    bis.seekg(2 * numedges * sizeof(int32_t), std::ios_base::cur);
    if (bis.fail()) return false;
    // this will now be the node potentials
    for (size_t i = 0;i < numvars; ++i) {
      // will be nice if the node potentials are given to me in numerical order
      // but I certainly can't guarantee that given the file format
      size_t  varid = bis.read<int32_t>();
      size_t  cardinality = bis.read<int32_t>();
      if (bis.fail()) return false;
      if (cardinality <= 0) return false;
      // create the variable
      //char temp[16]; sprintf(temp,"%d", varid);
      std::stringstream varname;
      varname << varid;
      variable_type* var = universe.new_finite_variable(varname.str(), cardinality);
      idtovar[varid] = var;

      // create the factor
      F factor(make_domain(var), 0.0);
      for (size_t asg = 0; asg < cardinality; ++asg) {
        factor.set_logv(asg, proteinparsing::bound_infinity(bis.read<double>()));
      }
      factor.normalize();
      fg.add_factor(factor);
    }

    // this will now be the edge potentials
    size_t table_count = bis.read<int32_t>();
    if (bis.fail()) return false;
    // this may or may not be equal to num_edges depending on whether
    // both edge directions were counted... so lets not check that
    if(! (table_count == numedges || table_count == numedges / 2)) {
      return false;
    }
      // read the binary factors
    for (size_t  i = 0;i < table_count; ++i) {
      // read the src vertex
      size_t srcid = bis.read<int32_t>();
      assert(srcid< numvars);
      variable_type* varsrc = idtovar[srcid];
      size_t srccard = bis.read<int32_t>();
      if (bis.fail()) return false;
      // read the destination vertex
      size_t destid = bis.read<int32_t>();
      assert(destid < numvars);
      variable_type* vardest = idtovar[destid];
      size_t destcard = bis.read<int32_t>();
      if (bis.fail()) return false;
      
      if (srccard != varsrc->size() || destcard != vardest->size()) {
        return false;
      }

      F factor(make_domain(varsrc, vardest), 0.0);
      for (size_t j = 0; j < srccard * destcard; ++j) {
        size_t srcasg = bis.read<int32_t>();
        size_t destasg = bis.read<int32_t>();
        double value = bis.read<double>();
        if (bis.fail()) return false;
        if (!((srcasg < srccard) && (destasg < destcard))) return false;
        //assert(srcasg < srccard);
        //assert(destasg < destcard);
        // slow!
        /*
        finite_assignment fasg;
        fasg[varsrc] = srcasg;
        fasg[vardest] = destasg;
        proteinparsing::valueassign(factor(fasg), value);
        */
        // faster, but a little awkward
        if (factor.arg_list()[0] == varsrc) {
          factor.set_logv(srcasg, destasg, proteinparsing::bound_infinity(value));
        }
        else {
          factor.set_logv(destasg, srcasg, proteinparsing::bound_infinity(value));
        }
      }
      if (std::isinf(-std::log(factor.minimum())) ) {
//        std::cout << factor << "\n";
//        getchar();
      }

      fg.add_factor(factor);
    }
//    std::cout << "done!" << std::endl;
    return true;
  }



  typedef std::set< size_t > protein_asg_set_type;
  typedef std::vector< protein_asg_set_type > protein_truth_asg_type;

  void protein_load_truth_data_from_file(std::string fname,
                      protein_truth_asg_type& truth_asgs) {
    binary_input_stream bis(fname.c_str());
    size_t num_vertices = bis.read<int>();
    truth_asgs.resize(num_vertices);
    foreach(protein_asg_set_type& asgs, truth_asgs) {
      size_t valid_assignments = bis.read<int>();
      assert(valid_assignments > 0);
      for(size_t i = 0; i < valid_assignments; ++i) {
        size_t asg = bis.read<int>();
        asgs.insert(asg);
      }
    }
    bis.close();
  }

  void protein_load_truth_data(std::string network_folder,
                      protein_truth_asg_type& truth_asgs) {
    protein_load_truth_data_from_file(network_folder + "/truth.bin", 
                                      truth_asgs);
  }

  // Save the actual belief image
  template<typename F, typename InferenceEngine>
  double protein_compute_error(factor_graph_model<F>& network,
                      InferenceEngine& engine,
                      const protein_truth_asg_type& truth_asgs) {
    typedef typename factor_graph_model<F>::variable_type variable_type;
    assert(network.arguments().size() == truth_asgs.size());
    size_t correct = 0;
    foreach(variable_type* v, network.arguments()) {
      F blf = engine.belief(v);
      assert(blf.arguments().size() == 1);
      // find the max assignment in the belief
      size_t pred = 0;
      for(size_t asg = 0; asg < v->size(); ++asg) {
        if (blf.v(asg) > blf.v(pred)) {
          pred = asg;
        }
      }

      int varid = atoi(v->name().c_str());
      if( truth_asgs[varid].count(pred) > 0 ) ++correct;
    }
    return static_cast<double>(correct) / network.arguments().size();
  }
}
#include <sill/macros_undef.hpp>
#endif
