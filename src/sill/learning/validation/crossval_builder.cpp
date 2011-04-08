
#include <sill/learning/validation/crossval_builder.hpp>

namespace sill {

  void crossval_builder::add_options
  (boost::program_options::options_description& desc,
   const std::string& desc_prefix) {

    namespace po = boost::program_options;
    po::options_description
      sub_desc1(desc_prefix + "CrossVal Options");
    sub_desc1.add_options()
      ("no_cv",
       po::bool_switch(&no_cv)->default_value(false),
       "If true, do not run CV; use fixed_vals instead.")
      ("fixed_vals",
       po::value<vec>(&fixed_vals)->default_value(vec(1,0)),
       "Values to be used if not running CV.")
      ("nfolds",
       po::value<size_t>(&nfolds)->default_value(10),
       "Number of cross validation folds for choosing regularization. (> 1)")
      ("minvals",
       po::value<vec>(&minvals)->default_value(vec(1,.00001)),
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

  crossval_parameters crossval_builder::get_parameters() {
    size_t N = minvals.size();
    assert(maxvals.size() == N);
    return get_parameters(N);
  } // get_parameters

  crossval_parameters crossval_builder::get_parameters(size_t N) {
    crossval_parameters params(N);
    params.nfolds = nfolds;
    if (minvals.size() == N) {
      params.minvals = minvals;
    } else if (minvals.size() == 1) {
      params.minvals = minvals[0];
    } else {
      throw std::invalid_argument
        ("crossval_builder given minvals of length " +
         to_string(minvals.size()) + " but expected length " + to_string(N));
    }
    if (maxvals.size() == N) {
      params.maxvals = maxvals;
    } else if (maxvals.size() == 1) {
      params.maxvals = maxvals[0];
    } else {
      throw std::invalid_argument
        ("crossval_builder given maxvals of length " +
         to_string(maxvals.size()) + " but expected length " + to_string(N));
    }
    if (nvals.size() == N) {
      params.nvals = nvals;
    } else if (nvals.size() == 1) {
      params.nvals = nvals[0];
    } else {
      throw std::invalid_argument
        ("crossval_builder given nvals of length " +
         to_string(nvals.size()) + " but expected length " + to_string(N));
    }
    params.zoom = zoom;
    params.log_scale = !real_scale;
    return params;
  } // get_parameters

} // namespace sill
