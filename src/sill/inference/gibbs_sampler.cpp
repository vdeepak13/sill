
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
    foreach(size_t fptr_i, var2factors[last_v]) {
      const factor_type* fptr = factor_ptrs[fptr_i];
      if (fptr->arguments().size() == 1) {
        // Compute f *= f_tmp manually to avoid overhead.
        for (size_t i = 0; i < f.table().size(); ++i)
          f.table()(i) *= fptr->table()(i);
      } else {
        fptr->restrict_other(r, r_numberings[fptr_i],
                             var_sequence[last_v], f_tmp);
        // Compute f *= f_tmp manually to avoid overhead.
        for (size_t i = 0; i < f.table().size(); ++i)
          f.table()(i) *= f_tmp.table()(i);
      }
    }
    try {
      f.normalize();
    } catch (std::invalid_argument& e) {
      std::cerr << "gibbs_sampler<table_factor>::next_sample error:\n"
                << "  Could not normalize factor:\n" << f << "\n"
                << "  For last_v = " << last_v << "\n"
                << "  Factor inputs:\n";
      foreach(size_t fptr_i, var2factors[last_v]) {
        const factor_type* fptr = factor_ptrs[fptr_i];
        if (fptr->arguments().size() == 1) {
          std::cerr << "Unary factor: " << (*fptr) << "\n";
        } else {
          fptr->restrict_other(r, r_numberings[fptr_i],
                               var_sequence[last_v], f_tmp);
          std::cerr << "Conditioned factor: " << (*fptr) << "\n"
                    << " to get: " << f_tmp << "\n";
        }
      }
      std::cerr << std::endl;
      throw e;
    }
    // Sample a new value for last_v.
    f.sample(rng, r);
    return r;
  }

} // namespace sill

#include <sill/macros_undef.hpp>
