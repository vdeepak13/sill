
#ifndef PRL_LEARNING_DISCRIMINATIVE_BINARY_REGRESSOR_HPP
#define PRL_LEARNING_DISCRIMINATIVE_BINARY_REGRESSOR_HPP

#include <prl/assignment.hpp>
#include <prl/datastructure/concepts.hpp>
#include <prl/learning/discriminative/concepts.hpp>
#include <prl/learning/discriminative/binary_classifier.hpp>
#include <prl/stl_io.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * binary_regressor.
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   */
  class binary_regressor : public binary_classifier {

    // Public methods
    //==========================================================================
  public:

    // Constructors and destructors
    //==========================================================================

    virtual ~binary_regressor() { }

    // Prediction methods
    //==========================================================================

    //! Predict the probability of the class variable having value 1.
    double probability(const record& example) const = 0;

    //! Predict the probability of the class variable having value 1.
    double probability(const assignment& example) const = 0;

    //! Predict the label of a new example.
    virtual std::size_t predict(const record& example) const {
      if (probability(example) > .5)
        return 1;
      else
        return 0;
    }

    //! Predict the label of a new example.
    virtual std::size_t predict(const assignment& example) const {
      if (probability(example) > .5)
        return 1;
      else
        return 0;
    }

  }; // class binary_regressor

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_LEARNING_DISCRIMINATIVE_BINARY_REGRESSOR_HPP
