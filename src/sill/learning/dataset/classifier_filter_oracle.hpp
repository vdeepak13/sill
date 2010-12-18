
#ifndef SILL_CLASSIFIER_FILTER_ORACLE_HPP
#define SILL_CLASSIFIER_FILTER_ORACLE_HPP

#include <sill/assignment.hpp>
#include <sill/learning/dataset/oracle.hpp>
#include <sill/learning/discriminative/binary_classifier.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for transforming an oracle by hard thresholding using a classifier.
   *
   * This can filter examples by:
   *  - accepting them only if the class variable matches/does not match
   *    a certain value
   *  - accepting them only if predict_raw() is above/below some threshold
   *  - accepting them only if class variable matches/does not match predict()
   * \author Joseph Bradley
   * \ingroup learning_dataset
   *
   * THIS IS INCOMPLETE!
   */
  class classifier_filter_oracle : public oracle {

    static_assert(std::numeric_limits<double>::has_infinity);

    // Public type declarations
    //==========================================================================
  public:

    //! Base class
    typedef oracle base;

    /**
     * Filter mode
     *  - IS_VALUE = accept if predict() matches parameter LABEL_VALUE
     *  - NOT_VALUE = accept if predict() does match parameter LABEL_VALUE
     *  - ABOVE_THRESHOLD = accept if predict_raw() is > parameter THRESHOLD
     *  - BELOW_THRESHOLD = accept if predict_raw() is < parameter THRESHOLD
     *  - RIGHT_VALUE = accept if class variable matches predict()
     *  - WRONG_VALUE = accept if class variable does not match predict()
     */
    enum filter_mode {IS_VALUE, NOT_VALUE, ABOVE_THRESHOLD, BELOW_THRESHOLD,
                      RIGHT_VALUE, WRONG_VALUE};

    struct parameters {

      //! Limit (> 0) on number of examples used to get one filtered example
      //!  (default = 1000000)
      size_t filter_limit;

      //! See filter_mode
      //!  (required if filter_mode uses LABEL_VALUE)
      size_t label_value;

      //! Threshold above which examples are accepted
      //! (if filtering by thresholding); infinite values are invalid
      //!  (required if filter_mode uses THRESHOLD)
      double threshold;

      //! Used to make the algorithm deterministic.
      //!  (default = time)
      double random_seed;

      parameters()
        : filter_limit_(1000000),
          label_value_(std::numeric_limits<size_t>::max()),
          threshold_(std::numeric_limits<double>::infinity()) {
        std::time_t time_tmp;
        time(&time_tmp);
        random_seed_ = time_tmp;
      }

      bool valid() const {
        if (filter_limit == 0)
          return false;
        return true;
      }

    }; // class parameters

    // Protected data members
    //==========================================================================
  protected:

    parameters params;

    //! Copied from parameters for efficiency
    size_t filter_limit;

    //! Copied from parameters for efficiency
    size_t label_value;

    //! Copied from parameters for efficiency
    double threshold;

    //! filter
    binary_classifier& filter;

    //! mode
    filter_mode mode;

    //! Index of class variable
    size_t class_variable_index;

    // Constructors
    //==========================================================================
  public:

    /**
     * Constructs a filter oracle without a limit.
     * @param o             oracle to be filtered
     *                      This oracle should not be touched by anyone else,
     *                      for doing so will invalidate the filter_oracle's
     *                      current record.
     * @param filter        binary_classifier used to filter
     * @param use_weighting use weighting instead of filtering
     * @param random_seed   used to make algorithm deterministic (default=time)
     */
    classifier_filter_oracle
    (oracle& o, binary_classifier& filter, filter_mode mode,
     parameters params = parameters())
      : base(o), params(params), filter_limit(params.filter_limit),
        label_value(params.label_value), threshold(params.threshold),
        filter(filter), mode(mode), class_variable_index(filter.label_index()) {
      assert(params.valid());
      switch(mode) {
      case IS_VALUE:
      case NOT_VALUE:
        assert(params.label_value < filter.label()->size());
        break;
      case ABOVE_THRESHOLD:
      case BELOW_THRESHOLD:
        assert(params.threshold != std::numeric_limits<double>::infinity());
        break;
      case RIGHT_VALUE:
      case WRONG_VALUE:
        break;
      default:
        assert(false);
      }
    }

    // Mutating operations
    //==========================================================================

    //! Increments the oracle to the next record.
    //! This returns true iff the operation was successful; false indicates
    //! a depleted oracle.
    //! NOTE: This must be explicitly called after the oracle is constructed.
    bool next();

    //! Reset the classifier used by this oracle to the given one.
    void reset_filter(binary_classifier& filter) {
      this->filter = filter;
    }

  }; // class classifier_filter_oracle

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_CLASSIFIER_FILTER_ORACLE_HPP
