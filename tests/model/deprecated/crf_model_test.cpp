#include <iostream>
#include <fstream>

#include <sill/base/universe.hpp>
#include <sill/learning/crf/crf_X_mapping.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

/**
 * \file crf_model_test.cpp CRF model test.
 *
 * Usage:
 *  - ./crf_model_test                  Run test without serialization.
 *  - ./crf_model_test [temp_filepath]  Run test with serialization.
 */
int main(int argc, char** argv) {

  std::string filepath;
  if (argc == 2)
    filepath = argv[1];

  // Create a universe.
  universe u;

  // Create a random chain CRF.
  unsigned random_seed = 912893;
  decomposable<table_factor> Xmodel;
  crf_model<table_crf_factor> YgivenXmodel;
  size_t n = 5;
  boost::tuple<finite_var_vector, finite_var_vector,
               std::map<finite_variable*, copy_ptr<finite_domain> > >
    Y_X_Y2Xmap
    (create_random_chain_crf(Xmodel, YgivenXmodel, n, u, random_seed));
  crf_X_mapping<table_crf_factor> X_mapping(Y_X_Y2Xmap.get<2>());

  std::cout << "Created random chain CRF.\n"
            << "Decomposable model for P(X):\n" << Xmodel << "\n"
            << "CRF for P(Y|X):\n" << YgivenXmodel << "\n"
            << std::endl;

  if (filepath.size() != 0) {
    {
      std::ofstream fout(filepath.c_str());
      oarchive oa(fout);
      oa << u << Xmodel << YgivenXmodel << Y_X_Y2Xmap.get<0>()
         << Y_X_Y2Xmap.get<1>() << X_mapping;
      fout.close();
    }
    std::cout << "Saved model...now reading model and comparing...\n"
              << std::endl;
    universe read_u;
    decomposable<table_factor> read_Xmodel;
    crf_model<table_crf_factor> read_YgivenXmodel;
    finite_var_vector read_Y;
    finite_var_vector read_X;
    crf_X_mapping<table_crf_factor> read_X_mapping;
    {
      std::ifstream fin(filepath.c_str());
      iarchive ia(fin);
      ia >> read_u;
      ia.attach_universe(&read_u);
      ia >> read_Xmodel >> read_YgivenXmodel >> read_Y >> read_X
         >> read_X_mapping;
      fin.close();
    }
    std::cout << "Read decomposable model for P(X):\n" << read_Xmodel << "\n"
              << "Read CRF for P(Y|X):\n" << read_YgivenXmodel << "\n"
              << std::endl;
  } // end of serialization test

} // main

#include <sill/macros_undef.hpp>
