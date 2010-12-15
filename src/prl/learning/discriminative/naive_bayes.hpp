
#ifndef PRL_LEARNING_DISCRIMINATIVE_NAIVE_BAYES_HPP
#define PRL_LEARNING_DISCRIMINATIVE_NAIVE_BAYES_HPP

#include <prl/assignment.hpp>
#include <prl/datastructure/concepts.hpp>
#include <prl/learning/discriminative/concepts.hpp>

#include <prl/macros_def.hpp>

/**
 * \file naive_bayes.hpp Naive Bayes
 */

namespace prl {

  /**
   * Naive Bayes
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo serialization
   *
   * THIS IS UNFINISHED!
   */
  class naive_bayes {

  public:

    //////////////// PARAMETERS /////////////////////////////////////////
    /**
     * Parameter names
     * - SMOOTHING: double; pseudocount used for smoothing parameter estimates
     * - RANDOM_SEED: double; used to make the algorithm deterministic
     */
    enum parameters {SMOOTHING, RANDOM_SEED, PARAMETER_ARITY};

    template <typename LabelRange, typename ParamRange>
    naive_bayes(const statistics<vec>& stats,
                const ParamRange& parameters) {
      concept_assert((prl::Dataset<Dataset>));
      // assert that labels are binary or multiclass
      // TODO: Do I need to do concept checking for labels and parameters here
      // since I do it in the Classifier concept class?
    }

    //! Returns the finite variables in their natural order
    //! (including class variable).
    const var_vector& finite_list() const { }
    //! Returns the vector variables in their natural order
    const var_vector& vector_list() const { }

    //! Returns the class variable
    const variable_h label() const {
    }
    //! Predict the label of a new example.
    std::size_t predict(const assignment& example) const {
    }
    //! Value indicating the confidence of the prediction, with
    //!  predict() == (confidence() > 0) ? 1 : 0.
    //! If the classifier does not have actual confidence ratings,
    //!  then this should be any value with the correct sign.
    double confidence(const assignment& example) const {
    }

  }; // class naive_bayes

} // namespace prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_LEARNING_DISCRIMINATIVE_NAIVE_BAYES_HPP
