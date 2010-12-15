
#ifndef PRL_LEARNING_DISCRIMINATIVE_HPP
#define PRL_LEARNING_DISCRIMINATIVE_HPP

#include <algorithm>

#include <prl/datastructure/concepts.hpp>
#include <prl/functional.hpp>
#include <prl/learning/discriminative/concepts.hpp>
#include <prl/stl_io.hpp>

#include <prl/macros_def.hpp>

/**
 * \file discriminative.hpp Discriminative learning
 *
 * Some design comments:
 *  - The scores and confidences are given by separate functions.  In many
 *    cases, these might be the same.  However, in boosting decision trees,
 *    one might wish to build the decision tree using, e.g., a mutual
 *    information heuristic but use an exponential loss for confidence-rated
 *    predictions with AdaBoost.
 */

namespace prl {

  /**
   * Currently, this namespace is only used for optimization criteria
   * for domain-partitioning hypotheses.
   */
  namespace discriminative {

    static double BIG_DOUBLE = 100000;

    /////////////////// SPLITTING CRITERIA & CONFIDENCES ///////////////////
    // These classes define methods for computing optimization criteria
    // and confidence ratings for predictions.
    // The criteria are meant to be MAXIMIZED.
    // These may be used as template parameters to any domain-partitioning
    // learners for binary labels (learners which partition a domain and then
    // assign labels to partitions according to their majority classes).
    // (See, e.g., Schapire & Singer, 1999.)
    //
    // Parameters:
    // - rightA is the sum of the weights of examples in partition A
    //   which are predicted correctly.
    // - class0, class1 are the weights of class 0 (or -1) and class +1.

    /**
     * ACCURACY: (exactly) maximize 0-1 accuracy; not confidence-rated
     * \ingroup learning_discriminative
     */
    class objective_accuracy {
    public:

      //! Indicates if the predictions are confidence-rated.
      static bool confidence_rated() { return false; }

      //! normalized 0-1 accuracy
      static double
      objective(double rightA, double wrongA, double rightB, double wrongB) {
        double total = rightA + wrongA + rightB + wrongB;
        assert(total > 0);
        return (rightA + rightB) / total;
      }

      //! unnormalized 0-1 accuracy
      static double unnormalized(double right, double wrong) {
        return right;
      }

      //! -1 if class0 > class1, else +1
      static double confidence(double class0, double class1) {
        if (class0 > class1)
          return -1;
        else
          return 1;
      }

      static std::string name() { return "ACC"; }

    };  // class objective_accuracy

    /**
     * INFORMATION: (exactly) maximize information gain; not confidence-rated
     * \ingroup learning_discriminative
     */
    class objective_information {
    public:

      //! Indicates if the predictions are confidence-rated.
      static bool confidence_rated() { return false; }

      //! negative impurity (See Russell & Norvig on decision trees.)
      static double
      objective(double rightA, double wrongA, double rightB, double wrongB) {
        return unnormalized(rightA, wrongA) + unnormalized(rightB, wrongB);
      }

      //! negative impurity (See Russell & Norvig on decision trees.)
      static double unnormalized(double right, double wrong) {
        double total = right + wrong;
        double v = 0;
        if (right > 0)
          v += right * std::log(right / total);
        if (wrong > 0)
          v += wrong * std::log(wrong / total);
        return v;
      }

      //! -1 if class0 > class1, else +1
      static double confidence(double class0, double class1) {
        if (class0 > class1)
          return -1;
        else
          return 1;
      }

      static std::string name() { return "INFO"; }

    };  // class objective_information

    /**
     * EXPONENTIAL LOSS: (exactly) minimize exponential loss (as in AdaBoost);
     * confidence-rated.
     * \ingroup learning_discriminative
     */
    class objective_exponential {
    public:

      //! Indicates if the predictions are confidence-rated.
      static bool confidence_rated() { return true; }

      //! Exponential loss when using confidence-rated predictions.
      //! This equals Z_t in AdaBoost.
      static double
      objective(double rightA, double wrongA, double rightB, double wrongB) {
        return 2*(unnormalized(rightA, wrongA) + unnormalized(rightB, wrongB));
      }

      //! Exponential loss when using confidence-rated predictions.
      static double unnormalized(double right, double wrong) {
        return sqrt(right * wrong);
      }

      //! Correct alpha for AdaBoost with confidence-rated predictions.
      static double confidence(double class0, double class1) {
        if (class0 > 0) {
          if (class1 > 0)
            return .5 * (std::log(class1) - std::log(class0));
          else
            return - BIG_DOUBLE;
        } else if (class1 > 0)
          return BIG_DOUBLE;
        else // both are 0
          return 0;
      }

      static std::string name() { return "EXP"; }

    };  // class objective_exponential

    // TODO: GINIBOOST_LOSS (Is there a better name?)

  } // namespace discriminative

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_LEARNING_DISCRIMINATIVE_HPP
