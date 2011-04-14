
#include <sill/factor/random/random_table_crf_factor_functor_builder.hpp>

namespace sill {

  void
  random_table_crf_factor_functor_builder::
  add_options(boost::program_options::options_description& desc) {

    namespace po = boost::program_options;

    po::options_description
      sub_desc("random_table_crf_factor_functor options");
    rtff_builder.add_options(sub_desc);
    desc.add(sub_desc);
  }

  //! Check options.  Assert false if invalid.
  void
  random_table_crf_factor_functor_builder::check() const {
    rtff_builder.check();
  }

  //! Get the parsed options.
  const random_table_crf_factor_functor_builder::rff_type::parameters&
  random_table_crf_factor_functor_builder::get_parameters() const {
    params.table_factor_func.params = rtff_builder.get_parameters();
    return params;
  }

} // namespace sill
