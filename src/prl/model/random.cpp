
#include <prl/math/free_functions.hpp>
#include <prl/model/random.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  // Methods for conditional models over finite variables
  //============================================================================

  boost::tuple<finite_var_vector, finite_var_vector,
               std::map<finite_variable*, copy_ptr<finite_domain> > >
  create_fancy_random_crf(decomposable<table_factor>& Xmodel,
                          crf_model<table_crf_factor>& YgivenXmodel,
                          size_t n, size_t arity, universe& u,
                          const std::string& model_choice, bool tractable,
                          const std::string& factor_choice,
                          double YYstrengthD, double YXstrengthD,
                          double XXstrengthD, double strength_baseD,
                          double YYstrengthS, double YXstrengthS,
                          double XXstrengthS, double strength_baseS,
                          ivec factor_periods, bool add_cross_factors,
                          unsigned random_seed) {

    if ((factor_choice == "random") || (factor_choice == "random_assoc")) {
      assert(YYstrengthD >= 0);
      assert(YXstrengthD >= 0);
      assert(XXstrengthD >= 0);
      assert(YYstrengthS >= 0);
      assert(YXstrengthS >= 0);
      assert(XXstrengthS >= 0);
    }
    assert(strength_baseD >= 0);
    assert(strength_baseS >= 0);
    assert(arity > 0);
    assert((model_choice == "chain") || (model_choice == "tree"));
    assert((factor_choice == "random") || (factor_choice == "associative")
           || (factor_choice == "random_assoc"));
    assert(factor_periods.size() == 3);

    boost::mt11213b rng(random_seed);
    Xmodel.clear();
    YgivenXmodel.clear();
    std::map<finite_variable*, copy_ptr<finite_domain> > Y2X_map;
    if (n == 0)
      return boost::make_tuple(finite_var_vector(), finite_var_vector(),
                               Y2X_map);
    // Create the variables
    finite_var_vector Xvars;
    finite_var_vector Yvars;
    for (size_t i(0); i < n; ++i) {
      Xvars.push_back(u.new_finite_variable("X" + to_string(i), arity));
      Yvars.push_back(u.new_finite_variable("Y" + to_string(i), arity));
    }

    // Add initial vertex pair Y1--X1.
    table_factor f;
    size_t nYYfactors(0); // count number of YY factors added
    size_t nYXfactors(0); // count number of YX factors added
    size_t nXXfactors(0); // count number of XX factors added
    if (tractable) {
      for (size_t i(0); i < n; ++i)
        Y2X_map[Yvars[i]] =
          copy_ptr<finite_domain>
          (new finite_domain(make_domain<finite_variable>(Xvars[i])));
      if ((++nYXfactors) % (factor_periods[1]+1) == 0)
        f = create_random_crf_table_factor(factor_choice, Yvars[0],
                                           Xvars[0], rng, YXstrengthS,
                                           strength_baseS);
      else
        f = create_random_crf_table_factor(factor_choice, Yvars[0],
                                           Xvars[0], rng, YXstrengthD,
                                           strength_baseD);
      YgivenXmodel.add_factor
        (table_crf_factor(f, make_domain<finite_variable>(Yvars[0]), false));
      for (size_t i(1); i < n; ++i) {
        // Choose which existing vertex j to attach to.
        size_t j((model_choice == "chain") ?
                 i-1 : boost::uniform_int<int>(0,i-1)(rng));
        if ((++nXXfactors) % (factor_periods[2]+1) == 0)
          f = create_random_crf_table_factor(factor_choice, Xvars[j], Xvars[i],
                                             rng, XXstrengthS, strength_baseS);
        else
          f = create_random_crf_table_factor(factor_choice, Xvars[j], Xvars[i],
                                             rng, XXstrengthD, strength_baseD);
        Xmodel *= f;
        if ((++nYXfactors) % (factor_periods[1]+1) == 0)
          f = create_random_crf_table_factor(factor_choice, Yvars[i], Xvars[i],
                                             rng, YXstrengthS, strength_baseS);
        else
          f = create_random_crf_table_factor(factor_choice, Yvars[i], Xvars[i],
                                             rng, YXstrengthD, strength_baseD);
        YgivenXmodel.add_factor
          (table_crf_factor(f, make_domain<finite_variable>(Yvars[i]), false));
        if ((++nYYfactors) % (factor_periods[0]+1) == 0)
          f = create_random_crf_table_factor(factor_choice, Yvars[j], Yvars[i],
                                             rng, YYstrengthS, strength_baseS);
        else
          f = create_random_crf_table_factor(factor_choice, Yvars[j], Yvars[i],
                                             rng, YYstrengthD, strength_baseD);
        YgivenXmodel.add_factor
          (table_crf_factor(f, make_domain<finite_variable>(Yvars[j],Yvars[i]),
                            false));
        if (add_cross_factors) {
          if ((++nYXfactors) % (factor_periods[1]+1) == 0)
            f = create_random_crf_table_factor(factor_choice, Yvars[j],Xvars[i],
                                               rng, YXstrengthS,strength_baseS);
          else
            f = create_random_crf_table_factor(factor_choice, Yvars[j],Xvars[i],
                                               rng, YXstrengthD,strength_baseD);
          YgivenXmodel.add_factor
            (table_crf_factor(f, make_domain<finite_variable>(Yvars[j]),false));
          if ((++nYXfactors) % (factor_periods[1]+1) == 0)
            f = create_random_crf_table_factor(factor_choice, Yvars[i],Xvars[j],
                                               rng, YXstrengthS,strength_baseS);
          else
            f = create_random_crf_table_factor(factor_choice, Yvars[i],Xvars[j],
                                               rng, YXstrengthD,strength_baseD);
          YgivenXmodel.add_factor
            (table_crf_factor(f, make_domain<finite_variable>(Yvars[i]),false));
        }
      }
    } else { // intractable
      // Create the model P(X)
      for (size_t i(1); i < n; ++i) {
        // Choose which existing vertex j to attach to.
        size_t j((model_choice == "chain") ?
                 i-1 : boost::uniform_int<int>(0,i-1)(rng));
        if ((++nXXfactors) % (factor_periods[2]+1) == 0)
          f = create_random_crf_table_factor(factor_choice, Xvars[j], Xvars[i],
                                             rng, XXstrengthS, strength_baseS);
        else
          f = create_random_crf_table_factor(factor_choice, Xvars[j], Xvars[i],
                                             rng, XXstrengthD, strength_baseD);
        Xmodel *= f;
      }
      std::vector<size_t> xind(randperm(n, rng)); // Y_i matches to X_{xind[i]}
      for (size_t i(0); i < n; ++i)
        Y2X_map[Yvars[i]] =
          copy_ptr<finite_domain>
          (new finite_domain(make_domain<finite_variable>(Xvars[xind[i]])));
      // Create the model P(Y|X)
      if ((++nYXfactors) % (factor_periods[1]+1) == 0)
        f =create_random_crf_table_factor(factor_choice,Yvars[0],Xvars[xind[0]],
                                          rng, YXstrengthS, strength_baseS);
      else
        f =create_random_crf_table_factor(factor_choice,Yvars[0],Xvars[xind[0]],
                                          rng, YXstrengthD, strength_baseD);
      YgivenXmodel.add_factor
        (table_crf_factor(f, make_domain<finite_variable>(Yvars[0]), false));
      for (size_t i(1); i < n; ++i) {
        // Choose which existing Y_j to attach to.
        size_t j((model_choice == "chain") ?
                 i-1 : boost::uniform_int<int>(0,i-1)(rng));
        if ((++nYXfactors) % (factor_periods[1]+1) == 0)
          f = create_random_crf_table_factor
            (factor_choice, Yvars[i], Xvars[xind[i]],
             rng, YXstrengthS, strength_baseS);
        else
          f = create_random_crf_table_factor
            (factor_choice, Yvars[i], Xvars[xind[i]],
             rng, YXstrengthD, strength_baseD);
        YgivenXmodel.add_factor
          (table_crf_factor(f, make_domain<finite_variable>(Yvars[i]), false));
        if ((++nYYfactors) % (factor_periods[0]+1) == 0)
          f = create_random_crf_table_factor(factor_choice, Yvars[j], Yvars[i],
                                             rng, YYstrengthS, strength_baseS);
        else
          f = create_random_crf_table_factor(factor_choice, Yvars[j], Yvars[i],
                                             rng, YYstrengthD, strength_baseD);
        YgivenXmodel.add_factor
          (table_crf_factor(f, make_domain<finite_variable>(Yvars[j],Yvars[i]),
                            false));
        if (add_cross_factors) {
          if ((++nYXfactors) % (factor_periods[1]+1) == 0)
            f = create_random_crf_table_factor
              (factor_choice, Yvars[j], Xvars[xind[i]],
               rng, YXstrengthS, strength_baseS);
          else
            f = create_random_crf_table_factor
              (factor_choice, Yvars[j], Xvars[xind[i]],
               rng, YXstrengthD, strength_baseD);
          YgivenXmodel.add_factor
            (table_crf_factor(f, make_domain<finite_variable>(Yvars[j]),false));
          if ((++nYXfactors) % (factor_periods[1]+1) == 0)
            f = create_random_crf_table_factor
              (factor_choice, Yvars[i], Xvars[xind[j]],
               rng, YXstrengthS, strength_baseS);
          else
            f = create_random_crf_table_factor
              (factor_choice, Yvars[i], Xvars[xind[j]],
               rng, YXstrengthD, strength_baseD);
          YgivenXmodel.add_factor
            (table_crf_factor(f, make_domain<finite_variable>(Yvars[i]),false));
        }
      }
    }
    return boost::make_tuple(Yvars, Xvars, Y2X_map);
  }  // end of create_fancy_random_crf()

  boost::tuple<finite_var_vector, finite_var_vector,
               std::map<finite_variable*, copy_ptr<finite_domain> > >
  create_random_crf(decomposable<table_factor>& Xmodel,
                    crf_model<table_crf_factor>& YgivenXmodel,
                    size_t n, size_t arity, universe& u,
                    const std::string& model_choice,
                    const std::string& factor_choice,
                    double YYstrength, double YXstrength,
                    double XXstrength, bool add_cross_factors,
                    unsigned random_seed,
                    double strength_base) {
    return create_fancy_random_crf
      (Xmodel, YgivenXmodel, n, arity, u, model_choice, true, factor_choice,
       YYstrength, YXstrength, XXstrength, strength_base, 0, 0, 0, 0,
       ivec(3,std::numeric_limits<int>::max()-10),
       add_cross_factors, random_seed);
  } // end of create_random_crf()

  boost::tuple<finite_var_vector, finite_var_vector,
               std::map<finite_variable*, copy_ptr<finite_domain> > >
  create_random_chain_crf(decomposable<table_factor>& Xmodel,
                          crf_model<table_crf_factor>& YgivenXmodel,
                          size_t n, universe& u,
                          unsigned random_seed) {
    return create_random_crf(Xmodel, YgivenXmodel, n, 2, u, "chain",
                             "random", 2, 2, 2, false, random_seed);
  } // create_random_chain_crf

  // Methods for conditional models over vector variables
  //============================================================================

  boost::tuple<vector_var_vector, vector_var_vector,
               std::map<vector_variable*, copy_ptr<vector_domain> > >
  create_random_gaussian_crf(decomposable<canonical_gaussian>& Xmodel,
                             crf_model<gaussian_crf_factor>& YgivenXmodel,
                             size_t n, universe& u,
                             const std::string& model_choice,
                             double b_max, double c_max, double variance,
                             double YYcorrelation, double YXcorrelation,
                             double XXcorrelation,
                             bool add_cross_factors, unsigned random_seed) {
    assert((model_choice == "chain") || (model_choice == "tree"));

    boost::mt11213b rng(random_seed);
    Xmodel.clear();
    YgivenXmodel.clear();
    std::map<vector_variable*, copy_ptr<vector_domain> > Y2X_map;
    if (n == 0)
      return boost::make_tuple(vector_var_vector(), vector_var_vector(),
                               Y2X_map);
    // Create the variables
    vector_var_vector Xvars;
    vector_var_vector Yvars;
    for (size_t i(0); i < n; ++i) {
      Xvars.push_back(u.new_vector_variable("X" + to_string(i), 1));
      Yvars.push_back(u.new_vector_variable("Y" + to_string(i), 1));
      Y2X_map[Yvars.back()] =
        copy_ptr<vector_domain>
        (new vector_domain(make_domain<vector_variable>(Xvars.back())));
    }
    // Add initial vertex pair Y1--X1.
    moment_gaussian f(make_binary_conditional_gaussian
                      (Yvars[0],Xvars[0], b_max, c_max, rng));
    YgivenXmodel.add_factor(gaussian_crf_factor(f));
    for (size_t i(1); i < n; ++i) {
      // Choose which existing vertex j to attach to.
      size_t j((model_choice == "chain") ?
               i-1 : boost::uniform_int<int>(0,i-1)(rng));
      f = make_binary_marginal_gaussian
        (Xvars[j], Xvars[i], b_max, variance, XXcorrelation, rng);
      Xmodel *= canonical_gaussian(f);
      f = make_binary_conditional_gaussian
        (Yvars[i], Xvars[i], b_max, c_max, rng);
      YgivenXmodel.add_factor(gaussian_crf_factor(f));
      f = make_binary_marginal_gaussian
        (Yvars[j], Yvars[i], b_max, variance, YYcorrelation, rng);
      YgivenXmodel.add_factor(gaussian_crf_factor(f));
      if (add_cross_factors) {
        f = make_binary_conditional_gaussian
          (Yvars[j], Xvars[i], b_max, c_max, rng);
        YgivenXmodel.add_factor(gaussian_crf_factor(f));
        f = make_binary_conditional_gaussian
          (Yvars[i], Xvars[j], b_max, c_max, rng);
        YgivenXmodel.add_factor(gaussian_crf_factor(f));
      }
    }
    return boost::make_tuple(Yvars, Xvars, Y2X_map);
  }  // end of create_random_gaussian_crf()

  boost::tuple<vector_var_vector, vector_var_vector,
               std::map<vector_variable*, copy_ptr<vector_domain> > >
  create_chain_gaussian_crf(decomposable<canonical_gaussian>& YXmodel,
                            crf_model<gaussian_crf_factor>& YgivenXmodel,
                            size_t n, universe& u, unsigned random_seed) {
    return create_random_gaussian_crf(YXmodel, YgivenXmodel, n, u, "chain", 5,
                                      3, 1, .3, .3, .3, false, random_seed);
  }

} // namespace prl

#include <prl/macros_undef.hpp>
