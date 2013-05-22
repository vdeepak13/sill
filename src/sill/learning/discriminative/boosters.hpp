
#ifndef SILL_LEARNING_DISCRIMINATIVE_BOOSTERS_HPP
#define SILL_LEARNING_DISCRIMINATIVE_BOOSTERS_HPP

#include <sill/datastructure/concepts.hpp>
//#include <sill/learning/discriminative/concepts.hpp>

#include <sill/macros_def.hpp>

/**
 * \file boosters.hpp Boosting Algorithms, as defined by optimization criteria
 */

namespace sill {

  namespace boosting {

    //! Return +1 if label > 0, else -1
    inline double binary_label(double label) { return (label > 0 ? 1 : -1); }

    /////////////////////// OPTIMIZATION OBJECTIVES /////////////////////////
    // These classes define methods for computing weights for weak hypotheses,
    // etc. and contain the optimization objectives for the weak hypotheses.

    /**
     * ADABOOST Objective (exponential loss)
     * Using this with batch_booster yields AdaBoost.  Using this with
     * filtering_booster yields MadaBoost.
     * \ingroup learning_discriminative
     */
    class adaboost {

    public:

      //! Indicates if the predictions are confidence-rated.
      static const bool confidence_rated = false;

      //! Computes the weights for weak hypotheses, given the weak learner's
      //! edge (= 1/2 - error rate).
      static double alpha(double edge, double smoothing = .0000001) {
        return .5 * (std::log(.5 + edge + smoothing)
                     - std::log(.5 - edge + smoothing));
      }

      //! Computes the edge from alpha
      static double inverse_alpha(double alpha, double smoothing = .0000001) {
        return (.5 + smoothing)*(exp(2*alpha) - 1)/(exp(2*alpha) + 1);
      }

      //! Compute the probability that the example has the given label
      static double probability(size_t label, double raw_prediction) {
        return 1. / (1. + exp(-2. * binary_label(label) * raw_prediction));
      }

      //! Compute an example's weight
      static double weight(size_t label, double raw_prediction) {
        return exp(-1. * binary_label(label) * raw_prediction);
      }

      //! Indicates if the example's weights can be updated
      static const bool can_update = true;

      //! Update an example's weight
      static double weight_update
      (double old_weight, double alpha, size_t label, double prediction) {
        return exp(std::log(old_weight)
                   - alpha * binary_label(label) * prediction);
      }

      static std::string name() { return "ADA"; }

    }; // class adaboost

    /**
     * FILTERBOOST Objective (logistic loss)
     * Using this with filtering_booster yields the FilterBoost algorithm,
     * using this with batch_booster yields the sequential method for logistic
     * regression in Collins, Schapire, and Singer (2000).
     * \ingroup learning_discriminative
     */
    class filterboost {

    public:

      //! Indicates if the predictions are confidence-rated.
      static const bool confidence_rated = false;

      //! Computes the weights for weak hypotheses, given the weak learner's
      //! edge (= 1/2 - error rate).
      static double alpha(double edge, double smoothing = .0000001) {
        return .5 * (std::log(.5 + edge + smoothing)
                     - std::log(.5 - edge + smoothing));
      }

      //! Computes the edge from alpha
      static double inverse_alpha(double alpha, double smoothing = .0000001) {
        return (.5 + smoothing)*(exp(2*alpha) - 1)/(exp(2*alpha) + 1);
      }

      //! Compute the probability that the example has the given label
      static double probability(size_t label, double raw_prediction) {
        return 1. / (1. + exp(binary_label(label) * raw_prediction));
      }

      //! Compute an example's weight
      static double weight(size_t label, double raw_prediction) {
        return 1. / (1. + exp(binary_label(label) * raw_prediction));
      }

      //! Indicates if the example's weights can be updated
      static const bool can_update = true;

      //! Update an example's weight
      static double weight_update
      (double old_weight, double alpha, size_t label, double prediction) {
        double s = std::log(1./old_weight - 1.)
          + binary_label(label) * alpha * prediction;
        return 1. / (1. + exp(s));
      }

      static std::string name() { return "FILTER"; }

    }; // class filterboost

  } // namespace boosting

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_BOOSTERS_HPP
