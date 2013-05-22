
#include <sill/factor/random/random_table_crf_factor_functor_builder.hpp>

namespace sill {

  void
  random_table_crf_factor_functor_builder::
  add_options(boost::program_options::options_description& desc,
              const std::string& opt_prefix) {

    namespace po = boost::program_options;

    po::options_description
      sub_desc("random_table_crf_factor_functor "
               + (opt_prefix == "" ? std::string("") : "(" + opt_prefix + ") ")
               + "options");
    rtff_builder.add_options(sub_desc, opt_prefix);
    desc.add(sub_desc);
  }

  void
  random_table_crf_factor_functor_builder::check() const {
    rtff_builder.check();
  }

  const random_table_crf_factor_functor_builder::rff_type::parameters&
  random_table_crf_factor_functor_builder::get_parameters() const {
    params.table_factor_func.params = rtff_builder.get_parameters();
    return params;
  }

  void random_table_crf_factor_functor_builder::print(std::ostream& out) const {
    rtff_builder.print(out);
  }

} // namespace sill
