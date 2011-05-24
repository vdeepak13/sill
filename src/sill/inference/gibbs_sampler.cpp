
#include <sill/inference/gibbs_sampler.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <>
  const gibbs_sampler<table_factor>::record_type&
  gibbs_sampler<table_factor>::next_sample() {
    last_v = next_variable();
    // Restrict the factors containing last_v, multiply them together,
    //  and normalize the resulting factor.
    factor_type& f = singleton_factors[last_v];
    factor_type& f_tmp = singleton_factors_tmp[last_v];
    f = 1;
    foreach(const factor_type* fptr, var2factors[last_v]) {
      fptr->restrict_other(r, last_v, f_tmp);
      // Compute f *= f_tmp manually to avoid overhead.
      for (size_t i = 0; i < f.table().size(); ++i) {
        f.table()(i) *= f_tmp.table()(i);
      }
    }
    f.normalize();
    // Sample a new value for last_v.
    r.copy_from_assignment(f.sample(rng));
    return r;
  }

} // namespace sill

#include <sill/macros_undef.hpp>
