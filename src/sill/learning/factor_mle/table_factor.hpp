#ifndef SILL_FACTOR_MLE_TABLE_FACTOR_HPP
#define SILL_FACTOR_MLE_TABLE_FACTOR_HPP

#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset/finite_dataset.hpp>
#include <sill/learning/dataset/finite_record.hpp>
#include <sill/learning/factor_mle/factor_mle.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  // table factor maximum likelihood estimator
  // eventually: add the template argument which is the value type of the factor
  template <>
  class factor_mle<table_factor> {
  public:
    typedef finite_dataset dataset_type;
    typedef finite_domain  domain_type;

    struct param_type {
      double smoothing;
      param_type() : smoothing(0.0) { }
    };

    factor_mle(const finite_dataset* dataset,
               const param_type& params = param_type())
      : dataset(dataset), params(params) { }

    //! Returns the marginal distribution over a sequence of variables
    table_factor operator()(const finite_var_vector& vars) const {
      table_factor factor(vars, params.smoothing);
      foreach(const finite_record& r, dataset->records(vars)) {
        factor.table()(r.values) += r.weight;
      }
      factor.normalize();
      return factor;
    }

    table_factor operator()(const finite_var_vector& head,
                            const finite_var_vector& tail) const {
      return operator()(concat(head, tail)).conditional(make_domain(tail));
    }

    //! Returns the marginal distribution over a subset of variables
    table_factor operator()(const finite_domain& vars) const {
      return operator()(make_vector(vars));
    }

    //! Returns the conditional distributino over a subset of variables
    table_factor operator()(const finite_domain& head,
                            const finite_domain& tail) const {
      assert(set_disjoint(head, tail));
      table_factor f = operator()(set_union(head, tail));
      return f /= f.marginal(tail);
    }
     
  private:
    const finite_dataset* dataset;
    param_type params;

  }; // class factor_mle<table_factor>

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
