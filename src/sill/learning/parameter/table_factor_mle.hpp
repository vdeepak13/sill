#ifndef SILL_TABLE_FACTOR_MLE_HPP
#define SILL_TABLE_FACTOR_MLE_HPP

#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset2/finite_record.hpp>
#include <sill/learning/parameter/factor_estimator.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  // table factor maximum likelihood estimator
  // eventually: add the template argument which is the value type of the factor
  template <typename Dataset=finite_dataset<> >
  class table_factor_mle : public factor_estimator<table_factor> {
  public:
    table_factor_mle(const Dataset* dataset, double smoothing = 0.0)
      : dataset(dataset), smoothing(smoothing) {
      assert(dataset->size() > 0);
    }

    //! Returns the marginal distribution over a subset of variables
    table_factor operator()(const finite_domain& vars) const {
      return operator()(make_vector(vars));
    }
     
    //! Returns the marginal distribution over a sequence of variables
    table_factor operator()(const finite_var_vector& vars) const {
      table_factor factor(vars, smoothing);
      foreach(const finite_record2& r, dataset->records(vars)) {
        factor.table()(r.values) += r.weight;
      }
      factor.normalize();
      return factor;
    }

  private:
    const Dataset* dataset;
    double smoothing;
  };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
