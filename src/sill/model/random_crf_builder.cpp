#include <sill/model/random.hpp>
#include <sill/model/random_crf_builder.hpp>

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
      ("var_size", po::value<size_t>(&var_size)->default_value(2),
       "The size of the variables (input and output).")
      ("tractable", po::bool_switch(&tractable),
       "If set, the joint P(Y,X) is tractable. Default = intractable.")
      ("add_cross_factors", po::bool_switch(&add_cross_factors),
       "If set, add cross factors to the model. Default = no cross factors.");
    desc.add(sub_desc1);

    // Factor parameters: table factors
    po::options_description
      sub_desc2("Synthetic CRF Options: Table factor parameters");
    YY_rtcff_builder.add_options(sub_desc2, "YY");
    YX_rtcff_builder.add_options(sub_desc2, "YX");
    XX_rtff_builder.add_options(sub_desc2, "XX");
    desc.add(sub_desc2);

    // Factor parameters for real variables
    po::options_description
      sub_desc3("Synthetic CRF Options: Gaussian factor parameters");
    YY_rgcff_builder.add_options(sub_desc3, "YY");
    YX_rgcff_builder.add_options(sub_desc3, "YX");
    XX_rmgf_builder.add_options(sub_desc3, "XX");
    desc.add(sub_desc3);

  }

  void random_crf_builder::check() const {
    assert(factor_type == "table" || factor_type == "gaussian");
    assert(model_structure == "chain" || model_structure == "tree" ||
           model_structure == "star");
    assert(model_size != 0);
  }

  void random_crf_builder::create_model(
      decomposable<table_factor>& Xmodel,
      crf_model<table_crf_factor>& YgivenXmodel,
      finite_var_vector& Y_vec,
      finite_var_vector& X_vec,
      std::map<finite_variable*, copy_ptr<finite_domain> >& Y2X_map,
      universe& u,
      unsigned int random_seed) const {

    if (factor_type != "table") {
      std::cerr << "random_crf_builder::create_model() called for discrete"
                << " model, but the factor_type parameter is: "
                << factor_type << std::endl;
      assert(false);
    }

    boost::mt19937 rng(random_seed);

    table_crf_factor_fn YY_factor_func = YY_rtcff_builder.factor_fn(rng);
    table_crf_factor_fn YX_factor_func = YX_rtcff_builder.factor_fn(rng);
    marginal_table_factor_fn XX_factor_func = XX_rtff_builder.marginal_fn(rng);

    Y_vec = u.new_finite_variables(model_size, var_size, "Y");
    X_vec = u.new_finite_variables(model_size, var_size, "X");
    Y2X_map = create_random_crf(model_structure,
                                tractable,
                                add_cross_factors,
                                Y_vec,
                                X_vec,
                                YY_factor_func,
                                YX_factor_func,
                                XX_factor_func,
                                Xmodel,
                                YgivenXmodel,
                                rng);
  }

  void random_crf_builder::create_model(
      decomposable<canonical_gaussian>& Xmodel,
      crf_model<gaussian_crf_factor>& YgivenXmodel,
      vector_var_vector& Y_vec,
      vector_var_vector& X_vec,
      std::map<vector_variable*, copy_ptr<vector_domain> >& Y2X_map,
      universe& u,
      unsigned int random_seed) const {

    if (factor_type != "gaussian") {
      std::cerr << "random_crf_builder::create_model() called for real-valued"
                << " model, but the factor_type parameter is: "
                << factor_type << std::endl;
      assert(false);
    }

    boost::mt19937 rng(random_seed);

    gaussian_crf_factor_fn YY_factor_func = YY_rgcff_builder.factor_fn(rng);
    gaussian_crf_factor_fn YX_factor_func = YX_rgcff_builder.factor_fn(rng);
    marginal_moment_gaussian_fn XX_factor_func = XX_rmgf_builder.marginal_fn(rng);

    Y_vec = u.new_vector_variables(model_size, var_size, "Y");
    X_vec = u.new_vector_variables(model_size, var_size, "X");
    Y2X_map = create_random_crf(model_structure,
                                tractable,
                                add_cross_factors,
                                Y_vec,
                                X_vec,
                                YY_factor_func,
                                YX_factor_func,
                                XX_factor_func,
                                Xmodel,
                                YgivenXmodel,
                                rng);
  }

  void random_crf_builder::print(std::ostream& out) const {
    using std::endl;
    out << " random_crf_builder options:" << endl
        << "  Model parameters:" << endl
        << "   factor_type: " << factor_type << endl
        << "   model_structure: " << model_structure << endl
        << "   model_size: " << model_size << endl
        << "   var_size: " << var_size << endl
        << "   tractable: " << tractable << endl
        << "   add_cross_factors: " << add_cross_factors << endl
        << "  Parameters for table factors:" << endl
        << "   YY_rtcff_builder: " << YY_rtcff_builder << endl
        << "   YX_rtcff_builder: " << YX_rtcff_builder << endl
        << "   XX_rtff_builder: " << XX_rtff_builder << endl
        << "  Parameters for Gaussian factors:" << endl
        << "   YY_rgcff_builder: " << YY_rgcff_builder << endl
        << "   YX_rgcff_builder: " << YX_rgcff_builder << endl
        << "   XX_rmgf_builder: " << XX_rmgf_builder << endl;
  }

  std::ostream& operator<<(std::ostream& out, const random_crf_builder& rcb) {
    rcb.print(out);
    return out;
  }

}; // namespace sill

#include <sill/macros_undef.hpp>
