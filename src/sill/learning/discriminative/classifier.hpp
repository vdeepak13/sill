#ifndef SILL_LEARNING_DISCRIMINATIVE_CLASSIFIER_HPP
#define SILL_LEARNING_DISCRIMINATIVE_CLASSIFIER_HPP

#include <sill/learning/discriminative/learner.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Classifier interface.
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  class classifier : public learner<LA> {

    // Public types
    //==========================================================================
  public:

    typedef learner<LA> base;

    typedef typename base::la_type            la_type;
    typedef typename base::record_type        record_type;
    typedef typename base::value_type         value_type;
    typedef typename base::vector_type        vector_type;
    typedef typename base::matrix_type        matrix_type;
    typedef typename base::dense_vector_type  dense_vector_type;
    typedef typename base::dense_matrix_type  dense_matrix_type;

    // Virtual methods from base classes (*means pure virtual):
    //  From learner:
    //   name()*
    //   fullname()*
    //   is_online()*
    //   training_time()
    //   random_seed()
    //   save(), load()

    // Constructors and destructors
    //==========================================================================

    virtual ~classifier() { }

    // Getters and helpers
    //==========================================================================

    //! Indicates if the predictions are confidence-rated.
    //! Note that confidence rating may be optimized for different objectives.
    //! This is false by default.
    virtual bool is_confidence_rated() const { return false; }

    // Learning and mutating operations
    //==========================================================================

    /**
     * For confidence-rated hypotheses, this sets the confidence-rated
     * outputs according to the sufficient statistics for a test set.
     * For non-confidenced-rated hypotheses, this still may change the
     * sign of the predictions.
     * This asserts false when it is not implemented.
     *
     * @param  ds  dataset used to estimate statistics needed
     * @return estimated error rate for given data and chosen confidences
     */
    virtual value_type
    set_confidences(const dataset<dense_linear_algebra<> >& ds) {
      std::cerr << "classifier::set_confidences() has not been"
                << " implemented for this classifier!" << std::endl;
      assert(false);
      return - std::numeric_limits<value_type>::max();
    }

    /**
     * For confidence-rated hypotheses, this sets the confidence-rated
     * outputs according to the sufficient statistics for a test set.
     * For non-confidenced-rated hypotheses, this still may change the
     * sign of the predictions.
     * This asserts false when it is not implemented.
     *
     * @param o  data oracle used to estimate statistics needed
     * @param n  max number of examples to be drawn from the given oracle
     * @return estimated error rate for given data and chosen confidences
     */
    virtual value_type
    set_confidences(oracle<dense_linear_algebra<> >& o, size_t n) {
      std::cerr << "classifier::set_confidences() has not been"
                << " implemented for this classifier!" << std::endl;
      assert(false);
      return - std::numeric_limits<value_type>::max();
    }

    // Save and load methods
    //==========================================================================

    using base::save;
    using base::load;

  }; // class classifier

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_CLASSIFIER_HPP
