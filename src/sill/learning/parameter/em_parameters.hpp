#ifndef SILL_EM_PARAMETERS_HPP
#defien SILL_EM_PARAMETERS_HPP

namespace sill {

  /**
   * A struct that represents the parameters of an EM algorithm.
   * \tparam Regul a type that represents regularization parameters
   */
  template <typename Regul = double>
  struct em_parameters {
    Regul regul;
    size_t niter;
    real_type tol;
    bool verbose;
    unsigned seed;
    em_parameters(const Regul& regul = Regul(),
                  size_t niter = 100,
                  real_type tol = 1e-6,
                  bool verbose = false,
                  unsigned seed = 0)
      : regul(regul),
        niter(niter),
        tol(tol),
        verbose(verbose),
        seed(seed) { }
  }; // struct em_parameters

} // namespace sill

