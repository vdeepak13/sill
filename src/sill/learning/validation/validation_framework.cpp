
#include <sill/learning/validation/validation_framework.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  validation_framework::
  validation_framework(const dataset& ds,
                       model_validation_functor& mv_func,
                       unsigned random_seed)
    : build_type_(TEST_CV), rng(random_seed),
      unif_int(0, std::numeric_limits<int>::max()) {

    /*
    - For folds of dataset
       - Run learner
    */

    throw std::runtime_error
      ("validation_framework constructor (test via CV) not yet implemented.");

  } // constructor (test via CV)

  validation_framework::
  validation_framework(const dataset& train_ds,
                       const dataset& test_ds,
                       model_validation_functor& mv_func,
                       unsigned random_seed)
    : build_type_(TEST_VALIDATION), rng(random_seed),
      unif_int(0, std::numeric_limits<int>::max()) {

    /*
    - For iterations
       - Run learner
     */

    throw std::runtime_error("validation_framework constructor (test via validation set) not yet implemented.");

  } // constructor (test via validation set)

} // namespace sill

#include <sill/macros_undef.hpp>
