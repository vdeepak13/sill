
#include <sill/factor/random/random_moment_gaussian_functor_builder.hpp>

namespace sill {

  void random_moment_gaussian_functor_builder::
  add_options(boost::program_options::options_description& desc,
              const std::string& opt_prefix) {
    namespace po = boost::program_options;

    po::options_description
      sub_desc("random_moment_gaussian_functor "
               + (opt_prefix=="" ? std::string("") : "(" + opt_prefix + ") ")
               + "options");
    sub_desc.add_options()
      ((opt_prefix + "b").c_str(),
       po::value<double>(&(params.b))->default_value(1),
       "Each element of the mean is chosen from Uniform[-b, b].  (b >= 0)")
      ((opt_prefix + "variance").c_str(),
       po::value<double>(&(params.variance))->default_value(1),
       "Set variances of each variable to this value.  (variance > 0)")
      ((opt_prefix + "correlation").c_str(),
       po::value<double>(&(params.correlation))->default_value(.3),
       "Set covariance of each pair of variables according to this correlation coefficient.  (fabs(correlation) <= 1)")
      ((opt_prefix + "c").c_str(),
       po::value<double>(&(params.c))->default_value(1),
       "Each element of the coefficient matrix C is chosen from c_shift + Uniform[-c, c], where C shifts the mean when conditioning on X=x.  (c >= 0)")
      ((opt_prefix + "c_shift").c_str(),
       po::value<double>(&(params.c_shift))->default_value(0),
       "(See option c)");
    desc.add(sub_desc);
  } // add_options

  void random_moment_gaussian_functor_builder::check() const {
    params.check();
  }

  const random_moment_gaussian_functor::parameters&
  random_moment_gaussian_functor_builder::get_parameters() const {
    return params;
  }

  void
  random_moment_gaussian_functor_builder::print(std::ostream& out) const {
    out << params;
  }

} // namespace sill
