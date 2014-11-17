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
      param_type(double smoothing) : smoothing(smoothing) { }
    };

    class weighted_estimator;

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

    //! Returns the conditional distribution p(head | tail)
    table_factor operator()(const finite_var_vector& head,
                            const finite_var_vector& tail) const {
      return operator()(concat(head, tail)).conditional(make_domain(tail));
    }

    //! Returns the marginal distribution over a subset of variables
    table_factor operator()(const finite_domain& vars) const {
      return operator()(make_vector(vars));
    }

    //! Returns the conditional distribution over a subset of variables
    table_factor operator()(const finite_domain& head,
                            const finite_domain& tail) const {
      assert(set_disjoint(head, tail));
      table_factor f = operator()(set_union(head, tail));
      return f /= f.marginal(tail);
    }

    table_factor operator()(const finite_domain& vars,
                            const std::vector<table_factor>& weights) const {
      assert(weights.size() == dataset->size());
      finite_var_vector head = make_vector(set_difference(vars, weights[0].arguments()));
      finite_var_vector tail = weights[0].arg_vector();
      table_factor factor(concat(tail, head), params.smoothing);
      dense_table<double>& table = factor.table();
      size_t i = 0;
      //std::cout << head << ": ";
      foreach(const finite_record& r, dataset->records(head)) {
        assert(weights[i].arg_vector() == tail);
        size_t start = table.offset(r.values, tail.size());
        //std::cout << start << ' ';
        dense_table<double>::iterator dest = table.begin() + start;
        foreach(double w, weights[i].table()) {
          *dest++ += w;
        }
        ++i;
      }
      //std::cout << std::endl;
      return factor;
    }

    /**
     * Returns the conditional distribution p(head | tail),
     * where only the head variables are drawn from the data, and the tail variables
     * are distributed according to the given weights.
     */
    table_factor operator()(const finite_var_vector& head,
                            const finite_var_vector& tail,
                            const std::vector<table_factor>& weights) const {
      assert(weights.size() == dataset->size());
      table_factor factor(concat(tail, head), params.smoothing);
      dense_table<double>& table = factor.table();
      size_t i = 0;
      //std::cout << head << ": ";
      foreach(const finite_record& r, dataset->records(head)) {
        assert(weights[i].arg_vector() == tail);
        size_t start = table.offset(r.values, tail.size());
        //std::cout << start << ' ';
        dense_table<double>::iterator dest = table.begin() + start;
        foreach(double w, weights[i].table()) {
          *dest++ += w;
        }
        ++i;
      }
      //std::cout << std::endl;
      return factor.conditional(make_domain(tail));
    }

    /**
     * Returns the weighted estimator.
     */
    weighted_estimator weighted(const finite_var_vector& head,
                                const finite_var_vector& tail) const {
      return weighted_estimator(head, tail, dataset, params);
    }

    // todo: figure out the desired API when the samples are weighted
    class weighted_estimator {
    public:
      weighted_estimator(const finite_var_vector& head,
                         const finite_var_vector& tail,
                         const finite_dataset* dataset,
                         const param_type& params)
        : factor(concat(tail, head), params.smoothing),
          tail(tail),
          records(dataset->records(head)) { }

      void process(const table_factor& ptail) {
        assert(records.first != records.second); // not done yet
        assert(ptail.arg_vector() == tail);
        size_t start = factor.table().offset(records.first->values, tail.size());
        dense_table<double>::iterator dest = factor.table().begin() + start;
        foreach(double w, ptail.table()) {
          *dest++ += w;
        }
        ++records.first;
      }
      
      table_factor estimate() const {
        assert(records.first == records.second); // we are done
        return factor.conditional(make_domain(tail));
      }

    private:
      table_factor factor;
      finite_var_vector tail;
      std::pair<finite_dataset::const_record_iterator,
                finite_dataset::const_record_iterator> records;
    }; // class weighted_type

  private:
    const finite_dataset* dataset;
    param_type params;

  }; // class factor_mle<table_factor>

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
