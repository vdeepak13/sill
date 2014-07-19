#ifndef SILL_TABLE_FACTOR_LEARNER_HPP
#define SILL_TABLE_FACTOR_LEARNER_HPP

#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset2/finite_record.hpp>
#include <sill/learning/parameter/factor_learner.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  // estimates a table factor
  // eventually: add the template argument which is the value type of the factor
  template <typename Dataset=finite_dataset<> >
  class table_factor_learner : public factor_learner<table_factor> {
  public:
    table_factor_learner(const Dataset& dataset, double smoothing = 0.0)
      : dataset(&dataset), smoothing(smoothing) {
      assert(dataset.size() > 0);
    }
    
    //! Returns the marginal distribution over a subset of variables
    table_factor operator()(const finite_domain& args) const {
      table_factor factor(args, smoothing);
      foreach(const finite_record2& r, dataset->records(factor.arg_list())) {
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
