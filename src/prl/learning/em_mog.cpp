#include <prl/learning/em_mog.hpp>
#include <prl/learning/parameter/gaussian.hpp>
#include <prl/math/linear_algebra.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  em_mog::em_mog(const dataset* data, size_t k)
    : data(data), n(data->size()), k(k), w(zeros(n,k)) {}

  double em_mog::expectation(const mixture_gaussian& mixture) {
    using std::log;
    assert(mixture.size() == k);
    double logl = 0;
    size_t i = 0; // the record
      
    // computes the probability of each sample
    foreach(const record& record, data->records()) {
      for(size_t j = 0; j < k; j++)
        w(i, j) = mixture[j](record.vector());
      double p_i = sum(w.row(i));
      logl += log(p_i);
      w.divide_row(i++, p_i);
    }

    return logl;
  }

  mixture_gaussian em_mog::maximization(double regul) const {
    vector_domain mixturedomain(data->vector_list().begin(),
                                data->vector_list().end());
    mixture_gaussian mixture(k, mixturedomain);

    for(size_t i = 0; i < k; i++)
      // mle also computes the probability of the component, i.e., sum(w(:,i))
      mle(*data, w.column(i), regul, mixture[i]);

    // the standard (non-logarithmic) implementation of normalize works
    // since our probabilities are O(data->size())
    return mixture.normalize();
  }

  mixture_gaussian em_mog::local_maximization() const {
    vector_domain mixturedomain(data->vector_list().begin(),
                                data->vector_list().end());
    mixture_gaussian mixture(k, mixturedomain);

    for(size_t i = 0; i < k; i++) {
      mle(*data, w.column(i), 0, mixture[i]);
      mixture[i].mean() *= mixture[i].norm_constant();
      mixture[i].covariance() *= mixture[i].norm_constant();
    }
    return mixture;
  }

} // namespace prl 

