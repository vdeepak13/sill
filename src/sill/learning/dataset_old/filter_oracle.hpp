#ifndef SILL_FILTER_ORACLE_HPP
#define SILL_FILTER_ORACLE_HPP

#include <sill/learning/dataset_old/oracle.hpp>

#include <sill/macros_def.hpp>

// uncomment to print debugging information
//#define SILL_FILTER_ORACLE_HPP_VERBOSE

namespace sill {

  /**
   * Class for transforming an oracle by rejection sampling,
   * importance weighting, or hard thresholding using a classifier or regressor.
   *
   * DEPRECATED...or to be fixed up and used later
   *
   * \author Joseph Bradley
   * \ingroup learning_dataset
   * \todo Implement more types of filtering.
   */
  class filter_oracle : public oracle {

    // Protected data members
    //==========================================================================
  protected:

    typedef oracle base;

    //! oracle being filtered
    oracle& o;

    //! Count of examples used to get one filtered example
    size_t filter_count_;

    // Constructors
    //==========================================================================
  public:

    /**
     * Constructor.
     * @param o   oracle to be filtered
     *            This oracle should not be touched by anyone else, for doing
     *            so will invalidate the filter_oracle's current record.
     */
    explicit filter_oracle(oracle& o)
      : base(o), o(o), count_(0) {
      assert(false); // DEPRECATED
    }

    virtual ~filter_oracle() { }

    // Getters and helpers
    //==========================================================================

    //! Returns the current record.
    const record& current() const { return o.current(); }

    //! Returns the number of raw examples used to generate the current
    //! filtered example.
    size_t filter_count() const { return filter_count_; }

  }; // class filter_oracle

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_FILTER_ORACLE_HPP
