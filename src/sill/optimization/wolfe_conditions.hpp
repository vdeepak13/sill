#ifndef SILL_WOLFE_CONDITIONS_HPP
#define SILL_WOLFE_CONDITIONS_HPP

#include <boost/function.hpp>

namespace sill {

  template <typename RealType>
  class wolfe_conditions {
  public:
    typedef RealType real_type;
    typedef boost::function<real_type(real_type)> real_fn;
    struct param_type {
      real_type c1;
      real_type c2;
      bool strong;
    };

    wolfe_conditions() { }
    
    wolfe_conditions(const real_fn& f,
                     const real_fn& g,
                     const param_type& params = param_type())
      : f_(f), g_(g), params_(params) {
      assert(params.valid());
    }

    void reset() {
      if (f_ && g_) {
        f0_ = f_(0.0);
        g0_ = g_(0.0);
      }
    }
    
    bool operator()(real_type alpha) const {
      if (!f_) {
        return false;
      }
      if (params_.strong) {
        return
          f_(alpha) <= f0_ + params_.c1 * alpha * g0_ &&
          g_(alpha) >= params_.c2 * g0_;
      } else {
        return
          f_(alpha) <= f0_ + params_.c1 * alpha * g0_ &&
          std::fabs(g_(alpha)) <= std::fabs(params_.c2 * g0_);
      }
    }

  private:
    real_fn f_;
    real_fn g_;
    real_type f0_;
    real_type g0_;
    
  }; // class wolfe_conditions

} // namespace sill

#endif
