
#include <sill/model/random.hpp>
#include <sill/model/random_crf_builder.hpp>

#include <sill/factor/random/alternating_factor_functor.hpp>
#include <sill/factor/random/alternating_crf_factor_functor.hpp>
#include <sill/factor/random/random_canonical_gaussian_functor.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  void random_crf_builder::add_options
  (boost::program_options::options_description& desc) {

    namespace po = boost::program_options;

    // Model parameters
    po::options_description
      sub_desc1(std::string("Synthetic CRF Options: Model"));
    sub_desc1.add_options()
      ("model_structure",
       po::value<std::string>(&model_structure)->default_value("chain"),
       "Model structure: chain/tree/star")
      ("factor_type",
       po::value<std::string>(&factor_type)->default_value("table"),
       "Factor type: table/gaussian. This implicitly specifies the variable type.")
      ("model_size", po::value<size_t>(&model_size)->default_value(10),
       "Number of Y (and X) variables in the model.")
      ("tractable", po::bool_switch(&tractable),
       "If set, the joint P(Y,X) is tractable. Default = intractable.")
      ("add_cross_factors", po::bool_switch(&add_cross_factors),
       "If set, add cross factors to the model. Default = no cross factors.")
      ("factor_alt_period",
       po::value<size_t>(&factor_alt_period)->default_value(0),
       "If > 1, alternate between default factor parameters and alt_* parameters.  If 0, do not alternate; if 1, only use alt_* parameters.");
    desc.add(sub_desc1);

    // Factor parameters: table factors
    po::options_description
      sub_desc2("Synthetic CRF Options: Table factor parameters");
    YY_rtcff_builder.add_options(sub_desc2, "YY");
    YX_rtcff_builder.add_options(sub_desc2, "YX");
    XX_rtff_builder.add_options(sub_desc2, "XX");
    alt_YY_rtcff_builder.add_options(sub_desc2, "alt_YY");
    alt_YX_rtcff_builder.add_options(sub_desc2, "alt_YX");
    alt_XX_rtff_builder.add_options(sub_desc2, "alt_XX");
    desc.add(sub_desc2);

    // Factor parameters for real variables
    po::options_description
      sub_desc3("Synthetic CRF Options: Gaussian factor parameters");
    YY_rgcff_builder.add_options(sub_desc3, "YY");
    YX_rgcff_builder.add_options(sub_desc3, "YX");
    XX_rmgf_builder.add_options(sub_desc3, "XX");
    alt_YY_rgcff_builder.add_options(sub_desc3, "alt_YY");
    alt_YX_rgcff_builder.add_options(sub_desc3, "alt_YX");
    alt_XX_rmgf_builder.add_options(sub_desc3, "alt_XX");
    desc.add(sub_desc3);

  } // add_options

  void random_crf_builder::check() const {
    assert(factor_type == "table" || factor_type == "gaussian");
    assert(model_structure == "chain" || model_structure == "tree" ||
           model_structure == "star");
    assert(model_size != 0);
    YY_rtcff_builder.check();
    YX_rtcff_builder.check();
    XX_rtff_builder.check();
    alt_YY_rtcff_builder.check();
    alt_YX_rtcff_builder.check();
    alt_XX_rtff_builder.check();
    YY_rgcff_builder.check();
    YX_rgcff_builder.check();
    XX_rmgf_builder.check();
    alt_YY_rgcff_builder.check();
    alt_YX_rgcff_builder.check();
    alt_XX_rmgf_builder.check();
  } // check

  void random_crf_builder::create_model
  (decomposable<table_factor>& Xmodel,
   crf_model<table_crf_factor>& YgivenXmodel,
   finite_var_vector& Y_vec, finite_var_vector& X_vec,
   std::map<finite_variable*, copy_ptr<finite_domain> >& Y2X_map,
   universe& u, unsigned int random_seed) const {

    if (factor_type != "table") {
      std::cerr << "random_crf_builder::create_model() called for discrete"
                << " model, but the factor_type parameter is: "
                << factor_type << std::endl;
      assert(false);
    }
    boost::mt11213b rng(random_seed);
    boost::uniform_int<unsigned> unif_int(0, std::numeric_limits<int>::max());

    random_table_crf_factor_functor
      YY_factor_func(YY_rtcff_builder.create_functor(unif_int(rng)));
    random_table_crf_factor_functor
      YX_factor_func(YX_rtcff_builder.create_functor(unif_int(rng)));
    random_table_factor_functor
      XX_factor_func(XX_rtff_builder.create_functor(unif_int(rng)));

    if (factor_alt_period == 0) {
      boost::tuple<finite_var_vector, finite_var_vector,
        std::map<finite_variable*, copy_ptr<finite_domain> > >
        YX_and_map(create_random_crf
                   (model_structure, model_size, tractable, add_cross_factors,
                    YY_factor_func, YX_factor_func, XX_factor_func,
                    u, unif_int(rng), Xmodel, YgivenXmodel));
      Y_vec = YX_and_map.get<0>();
      X_vec = YX_and_map.get<1>();
      Y2X_map = YX_and_map.get<2>();
    } else {
      random_table_crf_factor_functor
        alt_YY_factor_func(alt_YY_rtcff_builder.create_functor(unif_int(rng)));
      random_table_crf_factor_functor
        alt_YX_factor_func(alt_YX_rtcff_builder.create_functor(unif_int(rng)));
      random_table_factor_functor
        alt_XX_factor_func(alt_XX_rtff_builder.create_functor(unif_int(rng)));
      alternating_crf_factor_functor<random_table_crf_factor_functor> YY_ff;
      YY_ff.params.default_rff = YY_factor_func;
      YY_ff.params.alternate_rff = alt_YY_factor_func;
      YY_ff.params.alternation_period = factor_alt_period;
      alternating_crf_factor_functor<random_table_crf_factor_functor> YX_ff;
      YX_ff.params.default_rff = YX_factor_func;
      YX_ff.params.alternate_rff = alt_YX_factor_func;
      YX_ff.params.alternation_period = factor_alt_period;
      alternating_factor_functor<random_table_factor_functor> XX_ff;
      XX_ff.params.default_rff = XX_factor_func;
      XX_ff.params.alternate_rff = alt_XX_factor_func;
      XX_ff.params.alternation_period = factor_alt_period;

      boost::tuple<finite_var_vector, finite_var_vector,
        std::map<finite_variable*, copy_ptr<finite_domain> > >
        YX_and_map(create_random_crf
                   (model_structure, model_size, tractable, add_cross_factors,
                    YY_ff, YX_ff, XX_ff,
                    u, unif_int(rng), Xmodel, YgivenXmodel));
      Y_vec = YX_and_map.get<0>();
      X_vec = YX_and_map.get<1>();
      Y2X_map = YX_and_map.get<2>();
    }
  } // create_model (table factors)

  void random_crf_builder::create_model
  (decomposable<canonical_gaussian>& Xmodel,
   crf_model<gaussian_crf_factor>& YgivenXmodel,
   vector_var_vector& Y_vec, vector_var_vector& X_vec,
   std::map<vector_variable*, copy_ptr<vector_domain> >& Y2X_map,
   universe& u, unsigned int random_seed) const {

    if (factor_type != "gaussian") {
      std::cerr << "random_crf_builder::create_model() called for real-valued"
                << " model, but the factor_type parameter is: "
                << factor_type << std::endl;
      assert(false);
    }

    boost::mt11213b rng(random_seed);
    boost::uniform_int<unsigned> unif_int(0, std::numeric_limits<int>::max());

    random_gaussian_crf_factor_functor
      YY_factor_func(YY_rgcff_builder.create_functor(unif_int(rng)));
    random_gaussian_crf_factor_functor
      YX_factor_func(YX_rgcff_builder.create_functor(unif_int(rng)));
    random_canonical_gaussian_functor
      XX_factor_func(XX_rmgf_builder.create_functor(unif_int(rng)));

    if (factor_alt_period == 0) {
      boost::tuple<vector_var_vector, vector_var_vector,
        std::map<vector_variable*, copy_ptr<vector_domain> > >
        YX_and_map(create_random_crf
                   (model_structure, model_size, tractable, add_cross_factors,
                    YY_factor_func, YX_factor_func, XX_factor_func,
                    u, unif_int(rng), Xmodel, YgivenXmodel));
      Y_vec = YX_and_map.get<0>();
      X_vec = YX_and_map.get<1>();
      Y2X_map = YX_and_map.get<2>();
    } else {
      random_gaussian_crf_factor_functor
        alt_YY_factor_func(alt_YY_rgcff_builder.create_functor(unif_int(rng)));
      random_gaussian_crf_factor_functor
        alt_YX_factor_func(alt_YX_rgcff_builder.create_functor(unif_int(rng)));
      random_canonical_gaussian_functor
        alt_XX_factor_func(alt_XX_rmgf_builder.create_functor(unif_int(rng)));
      alternating_crf_factor_functor<random_gaussian_crf_factor_functor> YY_ff;
      YY_ff.params.default_rff = YY_factor_func;
      YY_ff.params.alternate_rff = alt_YY_factor_func;
      YY_ff.params.alternation_period = factor_alt_period;
      alternating_crf_factor_functor<random_gaussian_crf_factor_functor> YX_ff;
      YX_ff.params.default_rff = YX_factor_func;
      YX_ff.params.alternate_rff = alt_YX_factor_func;
      YX_ff.params.alternation_period = factor_alt_period;
      alternating_factor_functor<random_canonical_gaussian_functor> XX_ff;
      XX_ff.params.default_rff = XX_factor_func;
      XX_ff.params.alternate_rff = alt_XX_factor_func;
      XX_ff.params.alternation_period = factor_alt_period;

      boost::tuple<vector_var_vector, vector_var_vector,
        std::map<vector_variable*, copy_ptr<vector_domain> > >
        YX_and_map(create_random_crf
                   (model_structure, model_size, tractable, add_cross_factors,
                    YY_ff, YX_ff, XX_ff,
                    u, unif_int(rng), Xmodel, YgivenXmodel));
      Y_vec = YX_and_map.get<0>();
      X_vec = YX_and_map.get<1>();
      Y2X_map = YX_and_map.get<2>();
    }
  } // create_model (Gaussian factors)

  void random_crf_builder::print(std::ostream& out) const {
    out << " random_crf_builder options:\n"
        << "  Model parameters:\n"
        << "   factor_type: " << factor_type << "\n"
        << "   model_structure: " << model_structure << "\n"
        << "   model_size: " << model_size << "\n"
        << "   tractable: " << tractable << "\n"
        << "   add_cross_factors: " << add_cross_factors << "\n"
        << "   factor_alt_period: " << factor_alt_period << "\n"
        << "  Parameters for table factors:\n"
        << "   YY_rtcff_builder: " << YY_rtcff_builder
        << "   YX_rtcff_builder: " << YX_rtcff_builder
        << "   XX_rtff_builder: " << XX_rtff_builder
        << "   alt_YY_rtcff_builder: " << alt_YY_rtcff_builder
        << "   alt_YX_rtcff_builder: " << alt_YX_rtcff_builder
        << "   alt_XX_rtff_builder: " << alt_XX_rtff_builder
        << "  Parameters for Gaussian factors:\n"
        << "   YY_rgcff_builder: " << YY_rgcff_builder
        << "   YX_rgcff_builder: " << YX_rgcff_builder
        << "   XX_rmgf_builder: " << XX_rmgf_builder
        << "   alt_YY_rgcff_builder: " << alt_YY_rgcff_builder
        << "   alt_YX_rgcff_builder: " << alt_YX_rgcff_builder
        << "   alt_XX_rmgf_builder: " << alt_XX_rmgf_builder
        << std::endl;
  } // print

  std::ostream& operator<<(std::ostream& out, const random_crf_builder& rcb) {
    rcb.print(out);
    return out;
  }

}; // namespace sill

#include <sill/macros_undef.hpp>
