
#include <sill/factor/random/random_table_factor_functor_builder.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  void
  random_table_factor_functor_builder::
  add_options(boost::program_options::options_description& desc) {
    add_options(desc, "");
  }

  void
  random_table_factor_functor_builder::
  add_options(boost::program_options::options_description& desc,
              const std::string& opt_prefix) {

    namespace po = boost::program_options;

    po::options_description
      sub_desc("random_table_factor_functor "
               + (opt_prefix == "" ? std::string("") : "(" + opt_prefix + ") ")
               + "options");
    sub_desc.add_options()
      ((opt_prefix + "factor_choice").c_str(),
      po::value<std::string>(&factor_choice_string)->default_value("random_range"),
       "Type of factor: random_range/associative/random_associative")
      ((opt_prefix + "lower_bound").c_str(),
       po::value<double>(&(params.lower_bound))->default_value(-1),
       "Lower bound for factor parameters (in log space)")
      ((opt_prefix + "upper_bound").c_str(),
       po::value<double>(&(params.upper_bound))->default_value(1),
       "Upper bound for factor parameters (in log space)")
      ((opt_prefix + "base_val").c_str(),
       po::value<double>(&(params.base_val))->default_value(0),
       "Base value used for factor parameters (in log space)")
      ((opt_prefix + "arity").c_str(),
       po::value<size_t>(&(params.arity))->default_value(2),
       "Variable arity used by generate_variable method");
    desc.add(sub_desc);
  } // add_options

  void
  random_table_factor_functor_builder::check() const {
    assert(factor_choice_string == "random_range" ||
           factor_choice_string == "associative" ||
           factor_choice_string == "random_associative");
    params.check();
  } // check

  const random_table_factor_functor::parameters&
  random_table_factor_functor_builder::
  get_parameters() const {
    check();
    if (factor_choice_string == "random_range")
      params.factor_choice =
        random_table_factor_functor::parameters::RANDOM_RANGE;
    if (factor_choice_string == "associative")
      params.factor_choice =
        random_table_factor_functor::parameters::ASSOCIATIVE;
    if (factor_choice_string == "random_associative")
      params.factor_choice =
        random_table_factor_functor::parameters::RANDOM_ASSOCIATIVE;
    return params;
  } // get_parameters

} // namespace sill

#include <sill/macros_undef.hpp>
