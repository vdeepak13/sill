#ifndef SILL_DECAYING_STEP_HPP
#define SILL_DECAYING_STEP_HPP

#include <sill/optimization/opt_step.hpp>
#include <sill/serialization/serialize.hpp>

#include <cstdlib>

#include <boost/function.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Parameters for decaying_step.
   * \ingroup optimization_algorithms
   */
  template <typename RealType>
  struct decaying_step_parameters {
    /**
     * Initial step size > 0.
     */
    RealType initial;

    /**
     * Discount factor in (0,1] by which step is shrunk each round.
     */
    RealType discount;

    /**
     * Constructs the parameters.
     */
    decaying_step_parameters(RealType initial = 0.1, RealType discount = 1.0)
      : initial(initial), discount(discount) {
      assert(valid());
    }

    /**
     * Sets the discount, such that the step size will be a given factor smaller
     * after the given number of iterations.
     */
    void set_discount(size_t num_iterations, RealType factor = 0.0001) {
      discount = std::pow(factor, 1.0 / num_iterations);
    }

    /**
     * Returns true if the parameters are valid.
     */
    bool valid() const {
      return initial > 0.0 && discount > 0.0 && discount <= 1.0;
    }

    /** 
     * Serializes the parameters.
     */
    void save(oarchive& ar) const {
      ar << initial << discount;
    }

    /**
     * Deserializes the paramters.
     */
    void load(iarchive& ar) {
      ar >> initial >> discount;
    }

    /**
     * Prints the parameters to the output stream.
     */
    friend std::ostream&
    operator<<(std::ostream& out, const decaying_step_parameters& p) {
      out << p.initial << ' ' << p.discount;
      return out;
    }

  }; // struct decaying_step_parameters

  /**
   * A class that applies a decaying step for gradient-based optimization.
   * \ingroup optimization_algorithms
   */
  template <typename Vec>
  class decaying_step : public opt_step<Vec> {

    // Public types
    //==========================================================================
  public:
    typedef typename Vec::value_type real_type;
    typedef decaying_step_parameters<real_type> param_type;
    typedef boost::function<real_type(const Vec&)> objective_fn;

    // Public functions
    //==========================================================================
  public:
    decaying_step(const objective_fn& objective,
                  const param_type& params = param_type())
      : objective_(objective), params_(params), eta_(params.initial) {
      assert(params.valid());
    }

    real_type apply(Vec& x, const Vec& direction) {
      x += direction * eta_;
      eta_ *= params_.discount;
      return objective_fn(x);
    }

    // Private data
    //==========================================================================
  private:
    objective_fn objective_;
    param_type params_;
    real_type eta_;

  };  // class decaying_step

} // namespace sill

#include <sill/macros_undef.hpp>

#endif

