#include <sill/learning/em_mog.hpp>
#include <sill/learning/parameter/gaussian.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  em_mog::em_mog(const dataset<la_type>* data, size_t k)
    : data(data), n(data->size()), k(k), w(zeros(n,k)) {}

  double em_mog::expectation(const mixture_gaussian& mixture) {
    using std::log;
    assert(mixture.size() == k);
    double logl = 0;
    size_t i = 0; // the record
      
    // computes the probability of each sample
    foreach(const record_type& r, data->records()) {
      for(size_t j = 0; j < k; j++)
        w(i, j) = mixture[j](r.vector());
      double p_i = sum(w.row(i));
      logl += log(p_i);
      w.row(i++) /= p_i;
    }

    return logl;
  }

  mixture_gaussian em_mog::maximization(double regul) const {
    vector_domain mixturedomain(data->vector_list().begin(),
                                data->vector_list().end());
    mixture_gaussian mixture(k, mixturedomain);

    for(size_t i = 0; i < k; i++)
      // mle also computes the probability of the component, i.e., sum(w(:,i))
      mle(*data, w.col(i), regul, mixture[i]);

    // the standard (non-logarithmic) implementation of normalize works
    // since our probabilities are O(data->size())
    return mixture.normalize();
  }

  mixture_gaussian em_mog::local_maximization() const {
    vector_domain mixturedomain(data->vector_list().begin(),
                                data->vector_list().end());
    mixture_gaussian mixture(k, mixturedomain);

    for(size_t i = 0; i < k; i++) {
      mle(*data, w.col(i), 0, mixture[i]);
      mixture[i].mean() *= mixture[i].norm_constant();
      mixture[i].covariance() *= mixture[i].norm_constant();
    }
    return mixture;
  }

} // namespace sill 

