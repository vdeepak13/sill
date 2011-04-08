
#ifndef SILL_CROSSVAL_PARAMETERS_HPP
#define SILL_CROSSVAL_PARAMETERS_HPP

#include <sill/math/statistics.hpp>
#include <sill/serialization/serialize.hpp>

namespace sill {

  /**
   * Struct which holds options for a function which chooses a vector
   * of parameters via cross validation.
   */
  struct crossval_parameters {

    //! Length of the parameter vector.
    size_t dim;

    //! Number of cross validation folds for choosing regularization. (> 1)
    //!  (default = 10)
    size_t nfolds;

    //! Minimum values for regularization parameters. (>= 0)
    //! NOTE: If you use the create_parameter_grid() methods, then you should
    //!       set this to be > 0.
    //!  (default = 0.000001)
    vec minvals;

    //! Maximum values for regularization parameters. (>= 0)
    //!  (default = 1)
    vec maxvals;

    //! Number of regularization values to try in each dimension. (>= 1)
    //!  (default = 10)
    ivec nvals;

    //! If true, iteratively try extra regularization values around
    //! the current best value for this many iterations.
    //!  (default = 0)
    size_t zoom;

    //! If true, try values on a log scale.
    //!  (default = true)
    bool log_scale;

    //! Specifies how to combine multiple runs/folds of CV to choose
    //! the best result.
    //!  (default = statistics::MEAN)
    statistics::generalized_mean_enum run_combo_type;

    //! Constructor which sets dim = 1.
    crossval_parameters();

    //! Constructor.
    //! @param dim  Length of the parameter vector.
    explicit crossval_parameters(size_t dim);

    //! Return true iff the parameters are valid.
    bool valid() const;

    void save(oarchive& ar) const;

    void load(iarchive& ar);

  }; // struct crossval_parameters

} // namespace sill

#endif // #ifndef SILL_CROSSVAL_PARAMETERS_HPP
