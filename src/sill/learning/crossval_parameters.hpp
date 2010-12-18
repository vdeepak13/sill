
#ifndef SILL_CROSSVAL_PARAMETERS_HPP
#define SILL_CROSSVAL_PARAMETERS_HPP

#include <sill/math/vector.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Struct which holds options for a function which chooses a vector
   * of parameters via cross validation.
   *
   * @param N  length of the parameter vector
   */
  template <size_t N>
  struct crossval_parameters {

    //! Length of the parameter vector
    static const size_t dim = N;

    //! Number of cross validation folds for choosing regularization. (> 1)
    //!  (default = 10)
    size_t nfolds;

    //! Minimum values for factor regularization parameters. (>= 0)
    //! NOTE: If you use the create_parameter_grid() methods, then you should
    //!       set this to be > 0.
    //!  (default = 0)
    vec minvals;

    //! Maximum values for factor regularization parameters. (>= 0)
    //!  (default = 1)
    vec maxvals;

    //! Number of factor regularization values to try in each dimension. (>= 1)
    //!  (default = 10)
    ivec nvals;

    //! If true, iteratively try extra factor regularization values around
    //! the current best value for this many iterations.
    //!  (default = 0)
    size_t zoom;

    //! If true, try values on a log scale.
    //!  (default = true)
    bool log_scale;

    crossval_parameters()
      : nfolds(10), minvals(N,0), maxvals(N,1), nvals(N,10),
        zoom(0), log_scale(true) { }

    //! Return true iff the parameters are valid.
    bool valid() const {
      if (nfolds <= 1)
        return false;
      if (minvals.size() != N)
        return false;
      if (maxvals.size() != N)
        return false;
      if (nvals.size() != N)
        return false;
      for (size_t i(0); i < N; ++i) {
        if (minvals(i) < 0)
          return false;
        if (maxvals(i) < minvals(i))
          return false;
        if (nvals(i) <= 0)
          return false;
      }
      return true;
    }

  }; // struct crossval_parameters

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_CROSSVAL_PARAMETERS_HPP
