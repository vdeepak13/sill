#ifndef SILL_FACTOR_MLE_HPP
#define SILL_FACTOR_MLE_HPP

#include <sill/macros_def.hpp>

namespace sill {

  template <typename Factor>
  class factor_mle {
  public:
    BOOST_STATIC_ASSERT_MSG(
      sizeof(Factor) == 0,
      "Missing definition of factor_mle for the given factor type; "
      "you may need to include an appropriate file frome sill/learning/factor_mle/");

//     //! The dataset type used by this class
//     typedef void dataset_type;

//     //! The parameter struct for estimating the factor
//     typedef void param_type;

//     //! Same as Factor::domain_type
//     typedef void domain_type;

//     //! Constructs the maximum likelihood estimator for a dataset and parameters
//     factor_mle(const dataset_type* ds,
//                const param_type& params = param_type());

//     //! Returns the MLE over the given variable set
//     Factor operator()(const domain_type& dom);

//     //! Returns the complete log-likelihood of the factor under the dataset
//     double log_likelihood(const Factor& factor) const;
  };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
