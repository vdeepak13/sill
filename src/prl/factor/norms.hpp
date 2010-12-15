#ifndef PRL_FACTOR_NORM_HPP
#define PRL_FACTOR_NORM_HPP

#include <prl/global.hpp>
#include <prl/factor/concepts.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  //! \addtogroup factor_types
  //! @{

  //! An object that computes a norm between two factors.
  template <typename F>
  struct factor_norm
    : public std::binary_function<F, F, double> {
    concept_assert((Factor<F>));

    //! Evaluates the norm
    virtual double operator()(const F& x, const F& y) const=0;

    //! Returns a copy of the norm
    virtual factor_norm* clone() const = 0;

    //! Deletes the norm
    virtual ~factor_norm() {}
  };

  //! An object that computes the L1-norm between two factors
  template <typename F>
  struct factor_norm_1 : public factor_norm<F> {
    double operator()(const F& x, const F& y) const {
      return norm_1(x, y);
    }
    factor_norm_1* clone() const {
      return new factor_norm_1(*this);
    }
  };

  //! An object that computes the L-infinity norm between two factors
  template <typename F>
  struct factor_norm_inf : public factor_norm<F> {
    double operator()(const F& x, const F& y) const {
      return norm_inf(x, y);
    }
    factor_norm_inf* clone() const {
      return new factor_norm_inf(*this);
    }
  };

  /**
   * An object that computes the L-infinity norm between two factors
   * in log space.
   */
  template <typename F>
  struct factor_norm_inf_log : public factor_norm<F> {
    double operator()(const F& x, const F& y) const {
      return norm_inf_log(x, y);
    }
    factor_norm_inf_log* clone() const {
      return new factor_norm_inf_log(*this);
    }
  }; // end of factor_norm_inf_log

  /**
   * An object that computes the L-infinity norm between two factors
   * in log space.
   */
  template <typename F>
  struct factor_norm_1_log : public factor_norm<F> {
    double operator()(const F& x, const F& y) const {
      return norm_1_log(x, y);
    }
    factor_norm_1_log* clone() const {
      return new factor_norm_1_log(*this);
    }
  }; // end of factor_norm_inf_log

  //! @} group factor
}

#include <prl/macros_undef.hpp>

#endif
