
#ifndef SILL_REGRESSOR_FILTER_ORACLE_HPP
#define SILL_REGRESSOR_FILTER_ORACLE_HPP

#include <sill/assignment.hpp>
#include <sill/learning/dataset/oracle.hpp>
#include <sill/learning/discriminative/binary_regressor.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_smallint.hpp>
#include <boost/random/variate_generator.hpp>

#include <sill/macros_def.hpp>

// uncomment to print debugging information
//#define SILL_REGRESSOR_FILTER_ORACLE_HPP_VERBOSE

namespace sill {

  /**
   * Class for transforming an oracle by rejection sampling or
   * importance weighting using a regressor.
   *  - accept with probability() or 1 - probability()
   *  - accept all, but give them weight
   *
   * This has not been finished yet!
   *
   * \author Joseph Bradley
   * \ingroup learning_dataset
   */
  class regressor_filter_oracle : public filter_oracle {

    typedef oracle base;

    // Public type declarations
    //==========================================================================
  public:

    struct parameters {

      //! Limit on number of examples used to get one filtered example.
      //!  (default = 1000000)
     size_t filter_limit;

      parameters()
        : filter_limit(1000000) { }

    }; // class parameters

    // Protected data members
    //==========================================================================
  protected:

    //! random number generator
    boost::mt11213b rng;

    //! uniform distribution over [0, 1]
    boost::uniform_real<double> uniform_prob;

    //! oracle being filtered
    oracle& o;

    //! filter
    const Filter& filter;

    //! True if using weighting instead of filtering
    bool use_weighting;

    //! Weight of current example
    double weight_;

    //! Limit on number of examples used to get one filtered example
    size_t limit_;

    //! Count of examples used to get one filtered example
    size_t count_;

    // Constructors
    //==========================================================================
  public:

    /**
     * Constructs a filter oracle without a limit.
     * @param o             oracle to be filtered
     *                      This oracle is copied since regressor_filter_oracle assumes
     *                      the oracle will not be touched by anyone else.
     * @param filter        filter functor
     * @param use_weighting use weighting instead of filtering
     * @param random_seed   used to make algorithm deterministic (default=time)
     */
    regressor_filter_oracle(oracle& o, const Filter& filter,
                  bool use_weighting = false,
                  double random_seed = 0)
      : base(o.var_order()), o(o), filter(filter), use_weighting(use_weighting),
        limit_(std::numeric_limits<size_t>::max()), class_vars(o.class_vars) {
      if (random_seed == 0) {
        std::time_t time_tmp;
        time(&time_tmp);
        random_seed = time_tmp;
      }
      rng.seed(static_cast<unsigned>(random_seed));
      uniform_prob = boost::uniform_real<double>(0,1);
      weight_ = (use_weighting ? 0 : 1);
      count_ = (use_weighting ? 1 : 0);
    }

    // Getters and helpers
    //==========================================================================

    size_t limit() const {
      return limit_;
    }

    double weight() const {
      return weight_;
    }

    double count() const {
      return count_;
    }

    // Mutating operations
    //==========================================================================

    bool next() {
      if (use_weighting) {
        if (!(o.next())) {
          weight_ = 0;
          return false;
        }
        weight_ = filter(o.current());
      } else {
        count_ = 0;
        do {
          if (!(o.next()))
            return false;
          count_++;
          if (count_ >= limit_)
            return false;
        } while (uniform_prob(rng) > filter(o.current()));
      }
      // TODO: MAKE THIS MORE EFFICIENT; IT'S COPYING RECORDS TWICE
      //  (WOULD AN ORACLE VIEW SUFFICE?)
      current_record.findata = o.current().finite();
      current_record.vecdata = o.current().vector();
      return true;
    }

    void set_limit(size_t limit) {
      limit_ = limit;
    }

    void set_seed(double random_seed) { 
      rng.seed(static_cast<unsigned>(random_seed));
    }

    void set_weighting(bool weighting) {
      use_weighting = weighting;
      weight_ = (use_weighting ? 0 : 1);
      count_ = (use_weighting ? 1 : 0);
    }

    //! Reset the classifier used by this oracle to the given one.
    void reset_classifier(const binary_classifier<vec>& classifier) {
      this->classifier = classifier;
    }

  }; // class regressor_filter_oracle

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_REGRESSOR_FILTER_ORACLE_HPP
