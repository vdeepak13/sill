#ifndef SILL_MLE_TABLE_FACTOR_HPP
#define SILL_MLE_TABLE_FACTOR_HPP

#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset3/finite_dataset.hpp>
#include <sill/learning/dataset3/finite_record.hpp>
#include <sill/learning/mle/mle.hpp>
#include <sill/learning/parameter/factor_estimator.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  // table factor maximum likelihood estimator
  // eventually: add the template argument which is the value type of the factor
  template <>
  class mle<table_factor> : public factor_estimator<table_factor> {
  public:
    typedef finite_dataset dataset_type;

    struct param_type {
      double smoothing;
      param_type() : smoothing(0.0) { }
    };

    mle(const finite_dataset* dataset, const param_type& params = param_type())
      : dataset(dataset), params(params) {
      assert(dataset->size() > 0);
    }

    //! Returns the marginal distribution over a subset of variables
    table_factor operator()(const finite_domain& vars) const {
      return operator()(make_vector(vars));
    }
     
    //! Returns the marginal distribution over a sequence of variables
    table_factor operator()(const finite_var_vector& vars) const {
      table_factor factor(vars, params.smoothing);
      foreach(const finite_record2& r, dataset->records(vars)) {
        factor.table()(r.values) += r.weight;
      }
      factor.normalize();
      return factor;
    }

  private:
    const finite_dataset* dataset;
    param_type params;
  };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
