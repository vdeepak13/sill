
#include <sill/learning/validation/crossval_builder.hpp>

namespace sill {

  void crossval_builder::add_options
  (boost::program_options::options_description& desc,
   const std::string& desc_prefix) {

    namespace po = boost::program_options;
    po::options_description
      sub_desc1(desc_prefix + "Cross Validation Options");
    sub_desc1.add_options()
      ("nfolds",
       po::value<size_t>(&nfolds)->default_value(10),
       "Number of cross validation folds for choosing regularization. (> 1)")
      ("minvals",
       po::value<vec>(&minvals)->default_value(vec(1,0)),
       "Minimum values for factor regularization parameters. (>= 0) (Specify as, e.g., \"[.5]\")")
      ("maxvals",
       po::value<vec>(&maxvals)->default_value(vec(1,1)),
       "Maximum values for factor regularization parameters. (>= 0)")
      ("nvals",
       po::value<ivec>(&nvals)->default_value(ivec(1,10)),
       "Number of factor regularization values to try in each dimension. (>= 1)")
      ("zoom",
       po::value<size_t>(&zoom)->default_value(0),
       "If true, iteratively try extra factor regularization values around the current best value for this many iterations.")
      ("real_scale",
       po::bool_switch(&real_scale)->default_value(false),
       "If true, try values on a real scale; if false, use a log scale. (default = false)");
    desc.add(sub_desc1);

  } // add_options

} // namespace sill
