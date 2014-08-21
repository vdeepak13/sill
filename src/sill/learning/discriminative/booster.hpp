#ifndef SILL_LEARNING_DISCRIMINATIVE_BOOSTER_HPP
#define SILL_LEARNING_DISCRIMINATIVE_BOOSTER_HPP

#include <sill/macros_def.hpp>

namespace sill {

  struct booster_parameters {

    //! Value (>= 0) used for smoothing confidences or weights
    //!  (default = 1 / (2 * number_of_labels * number_of_datapoints),
    //!   or some estimate thereof)
    double smoothing;

    //! Used to make the algorithm deterministic
    //!  (default = random)
    double random_seed;

    //! Number of iterations to run initially
    //!  (default = 0)
    size_t init_iterations;

    //! Value (>=0) which is close enough to zero
    //! to call two numbers differing by this much to be the same number
    //!  (default = .000000001)
    double convergence_zero;

    booster_parameters()
      : smoothing(-1), init_iterations(0), convergence_zero(.000000001) {
      std::time_t time_tmp;
      time(&time_tmp);
      random_seed = time_tmp;
    }

    virtual ~booster_parameters() { }

    virtual bool valid() const {
      if (smoothing < 0)
        return false;
      if (convergence_zero < 0)
        return false;
      return true;
    }

    void set_smoothing(size_t ntrain, size_t nlabels) {
      if (smoothing == -1)
        smoothing = 1. / (2. * ntrain * nlabels);
    }

    void save(std::ofstream& out) const {
      out << smoothing << " " << random_seed
          << " " << init_iterations << " " << convergence_zero << "\n";
    }

    void load(std::ifstream& in) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> smoothing))
        assert(false);
      if (!(is >> random_seed))
        assert(false);
      if (!(is >> init_iterations))
        assert(false);
      if (!(is >> convergence_zero))
        assert(false);
    }

  };  // struct booster_parameters

  /**
   * Boosting algorithm interface.
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo serialization
   */
  class booster {

    // Public methods
    //==========================================================================
  public:

    // Virtual methods from base classes (*means pure virtual):
    //  From iterative_learner:
    //   iteration()*
    //   elapsed_times()
    //   step()*
    //   reset_datasource()*
    //   train_accuracies()
    //   test_accuracies()

    // Constructors and destructors
    //==========================================================================

    virtual ~booster() { }

    // Getters and helpers
    //==========================================================================

    //! Returns the edge from each iteration.
    virtual std::vector<double> edges() const = 0;

    //! Returns an estimate of the norm constant for the distribution on
    //! each iteration.
    //! E.g., p_t for filtering boosters and 1/Z_t for batch boosters.
    //! This is not guaranteed to be implemented.
    virtual std::vector<double> norm_constants() const {
      return std::vector<double>();
    }

  }; // class booster

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_BOOSTER_HPP
