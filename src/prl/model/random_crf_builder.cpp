
#include <prl/model/random.hpp>
#include <prl/model/random_crf_builder.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  void random_crf_builder::add_options
  (boost::program_options::options_description& desc) {

    namespace po = boost::program_options;

    // Model parameters
    po::options_description
      sub_desc1(std::string("Synthetic CRF Options: Model"));
    sub_desc1.add_options()
      ("model_structure",
       po::value<std::string>(&model_structure)->default_value("chain"),
       "Model structure: chain/tree")
      ("variable_type",
       po::value<std::string>(&variable_type)->default_value("discrete"),
       "Variable type: discrete/real")
      ("model_size", po::value<size_t>(&model_size)->default_value(10),
       "Number of Y (and X) variables in the model.")
      ("tractable", po::bool_switch(&tractable),
       "If set, the joint P(Y,X) is tractable. Default = intractable.")
      ("add_cross_factors", po::bool_switch(&add_cross_factors),
       "If set, add cross factors to the model. Default = no cross factors.");
    desc.add(sub_desc1);

    // Factor parameters for discrete variables
    po::options_description
      sub_desc2("Synthetic CRF Options: Factors for Discrete Variables");
    sub_desc2.add_options()
      ("variable_arity",
       po::value<size_t>(&variable_arity)->default_value(2),
       "Variable arity.")
      ("factor_type",
       po::value<std::string>(&factor_type)->default_value("associative"),
       "Factor type for variable_type==discrete: associative/random/random_assoc.")
      ("YYstrength", po::value<double>(&YYstrength)->default_value(1),
       "Strength of Y-Y potentials.")
      ("YXstrength", po::value<double>(&YXstrength)->default_value(1),
       "Strength of Y-X potentials.")
      ("XXstrength", po::value<double>(&XXstrength)->default_value(1),
       "Strength of X-X potentials.")
      ("strength_base", po::value<double>(&strength_base)->default_value(0),
       "")
      ("alternation_period",
       po::value<size_t>(&alternation_period)->default_value(0),
       "Potential strength alternation period; default = 0 (off).")
      ("altYYstrength", po::value<double>(&altYYstrength)->default_value(0),
       "Strength of alternating Y-Y potentials.")
      ("altYXstrength", po::value<double>(&altYXstrength)->default_value(0),
       "Strength of alternating Y-X potentials.")
      ("altXXstrength", po::value<double>(&altXXstrength)->default_value(0),
       "Strength of alternating X-X potentials.")
      ("alt_strength_base",
       po::value<double>(&alt_strength_base)->default_value(0),
       "");
    desc.add(sub_desc2);

    // Factor parameters for real variables
    po::options_description
      sub_desc3("Synthetic CRF Options: Factors for Real Variables");
    sub_desc3.add_options()
      ("b_max", po::value<double>(&b_max)->default_value(1),
       "")
      ("c_max", po::value<double>(&c_max)->default_value(1),
       "")
      ("variance", po::value<double>(&variance)->default_value(1),
       "")
      ("YYcorrelation", po::value<double>(&YYcorrelation)->default_value(1),
       "")
      ("YXcorrelation", po::value<double>(&YXcorrelation)->default_value(1),
       "")
      ("XXcorrelation", po::value<double>(&XXcorrelation)->default_value(1),
       "");
    desc.add(sub_desc3);

  } // add_options

  bool random_crf_builder::valid(bool print_warnings) const {
    // TO DO: FINISH WRITING THESE CHECKS OUT!
    if (!(model_structure == "chain" || model_structure == "tree") ||
        model_size == 0) {
      return false;
    }
    //   Factors:
    if (!(factor_type == "random" || factor_type == "associative" ||
          factor_type == "random_assoc")) {
      return false;
    }
    return true;
  } // valid

  void random_crf_builder::create_model
  (decomposable<table_factor>& Xmodel,
   crf_model<table_crf_factor>& YgivenXmodel,
   finite_var_vector& Y_vec, finite_var_vector& X_vec,
   std::map<finite_variable*, copy_ptr<finite_domain> >& Y2X_map,
   universe& u, unsigned int random_seed) const {
    if (variable_type != "discrete") {
      throw std::invalid_argument("random_crf_builder::create_model() called for discrete model, but the variable_type parameter is: " + variable_type);
    }
    ivec factor_periods(3, alternation_period);
    if (alternation_period == 0)
      factor_periods = ivec(3, std::numeric_limits<int>::max() - 10);
    boost::tuple<finite_var_vector, finite_var_vector,
      std::map<finite_variable*, copy_ptr<finite_domain> > >
      YX_and_map(create_fancy_random_crf
                 (Xmodel, YgivenXmodel, model_size, variable_arity,
                  u, model_structure, tractable, factor_type,
                  YYstrength, YXstrength, XXstrength, strength_base,
                  altYYstrength, altYXstrength, altXXstrength,
                  alt_strength_base,
                  factor_periods, add_cross_factors, random_seed));
    Y_vec = YX_and_map.get<0>();
    X_vec = YX_and_map.get<1>();
    Y2X_map = YX_and_map.get<2>();
  } // create_model (discrete)

  void random_crf_builder::create_model
  (decomposable<canonical_gaussian>& Xmodel,
   crf_model<gaussian_crf_factor>& YgivenXmodel,
   vector_var_vector& Y_vec, vector_var_vector& X_vec,
   std::map<vector_variable*, copy_ptr<vector_domain> >& Y2X_map,
   universe& u, unsigned int random_seed) const {
    if (variable_type != "real") {
      throw std::invalid_argument("random_crf_builder::create_model() called for real-valued model, but the variable_type parameter is: " + variable_type);
    }
    boost::tuple<vector_var_vector, vector_var_vector,
      std::map<vector_variable*, copy_ptr<vector_domain> > >
      YX_and_map(create_random_gaussian_crf
                 (Xmodel, YgivenXmodel, model_size, u,
                  model_structure, b_max, c_max, variance,
                  YYcorrelation, YXcorrelation, XXcorrelation,
                  add_cross_factors, random_seed));
    Y_vec = YX_and_map.get<0>();
    X_vec = YX_and_map.get<1>();
    Y2X_map = YX_and_map.get<2>();
  } // create_model (real)

  void random_crf_builder::print(std::ostream& out) const {
    out << " random_crf_builder options:\n"
        << "  Model parameters:\n"
        << "   model_structure: " << model_structure << "\n"
        << "   variable_type: " << variable_type << "\n"
        << "   model_size: " << model_size << "\n"
        << "   tractable: " << tractable << "\n"
        << "   add_cross_factors: " << add_cross_factors << "\n"
        << "  Factor parameters for discrete variables:\n"
        << "   variable_arity: " << variable_arity << "\n"
        << "   factor_type: " << factor_type << "\n"
        << "   YYstrength: " << YYstrength << "\n"
        << "   YXstrength: " << YXstrength << "\n"
        << "   XXstrength: " << XXstrength << "\n"
        << "   strength_base: " << strength_base << "\n"
        << "   alternation_period: " << alternation_period << "\n"
        << "   altYYstrength: " << altYYstrength << "\n"
        << "   altYXstrength: " << altYXstrength << "\n"
        << "   altXXstrength: " << altXXstrength << "\n"
        << "   alt_strength_base: " << alt_strength_base << "\n"
        << "  Factor parameters for real variables:\n"
        << "   b_max: " << b_max << "\n"
        << "   c_max: " << c_max << "\n"
        << "   variance: " << variance << "\n"
        << "   YYcorrelation: " << YYcorrelation << "\n"
        << "   YXcorrelation: " << YXcorrelation << "\n"
        << "   XXcorrelation: " << XXcorrelation << std::endl;
  } // print

  std::ostream& operator<<(std::ostream& out, const random_crf_builder& rcb) {
    rcb.print(out);
    return out;
  }

}; // namespace prl

#include <prl/macros_undef.hpp>
