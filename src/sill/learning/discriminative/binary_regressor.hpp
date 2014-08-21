#ifndef SILL_LEARNING_DISCRIMINATIVE_BINARY_REGRESSOR_HPP
#define SILL_LEARNING_DISCRIMINATIVE_BINARY_REGRESSOR_HPP

#include <sill/assignment.hpp>
#include <sill/datastructure/concepts.hpp>
#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/binary_classifier.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

namespace sill {

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

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_BINARY_REGRESSOR_HPP
