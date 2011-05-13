
#ifndef SILL_DS_ORACLE_HPP
#define SILL_DS_ORACLE_HPP

#include <boost/random/mersenne_twister.hpp>

#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/oracle.hpp>
#include <sill/math/permutations.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for creating an oracle from a dataset.  The oracle presents the
   * records in the order given in the dataset.
   * This is useful for pre-loading a dataset for speed, rather than
   * just using, e.g., symbolic_oracle.
   * \author Joseph Bradley
   * \ingroup learning_dataset
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<double,size_t>)
   */
  template <typename LA = dense_linear_algebra<> >
  class ds_oracle : public oracle<LA> {

    // Public type declarations
    //==========================================================================
  public:

    typedef LA la_type;

    //! The base type (oracle)
    typedef oracle<la_type> base;

    typedef record<la_type> record_type;

    struct parameters {

      //! Limit on the number of records which may be
      //! drawn from the oracle; if 0, then no limit.
      //!  (default = 0)
      size_t record_limit;

      //! If true, then automatically resets to the first
      //! record when the end of the dataset is reached.
      //!  (default = true)
      bool auto_reset;

      //! If > 0, randomly permute the order of the samples upon initialization
      //! and every randomization_period times that this oracle resets.
      //! If 0, then do not use randomization.
      //!  (default = 1)
      size_t randomization_period;

      //! Random seed used if randomization_period != 0.
      //!  (default = time)
      unsigned random_seed;

      parameters()
        : record_limit(0), auto_reset(true), randomization_period(1),
          random_seed(time(NULL)) { }

    }; // struct parameters

    // Constructors
    //==========================================================================

    /**
     * Constructs a dataset-based oracle.
     */
    explicit
    ds_oracle(const dataset<la_type>& ds, parameters params = parameters())
      : base(ds.datasource_info()),
        params(params), ds_view(ds), records_used(0), initialized(false),
        num_resets_until_randomization(params.randomization_period),
        rng(params.random_seed) {
      if (params.randomization_period != 0) {
        ds_view.save_record_view();
        randomly_permute();
      }
      ds_it = ds_view.end();
      ds_end = ds_view.end();
    }

    // Getters and helpers
    //==========================================================================

    //! Returns the current record.
    const record_type& current() const {
      return *ds_it;
    }

    //! Returns the weight of the current example.
    double weight() const {
      return ds_it.weight();
    }

    //! Returns the number of records used so far (including current record)
    size_t nrecords_used() const {
      return records_used;
    }

    // Mutating operations
    //==========================================================================

    //! Increments the oracle to the next record.
    //! This returns true iff the operation was successful; false indicates
    //! a depleted oracle.
    //! NOTE: This must be explicitly called after the oracle is constructed.
    bool next() {
      if (params.record_limit != 0 && records_used >= params.record_limit)
        return false;
      if (!initialized) {
        ds_it = ds_view.begin();
        if (ds_it == ds_end)
          return false;
        initialized = true;
      } else {
        ++ds_it;
        if (ds_it == ds_end) {
          if (params.auto_reset) {
            if (num_resets_until_randomization != 0) {
              --num_resets_until_randomization;
              if (num_resets_until_randomization == 0) {
                randomly_permute();
                num_resets_until_randomization = params.randomization_period;
                ds_end = ds_view.end();
              }
            }
            ds_it = ds_view.begin();
          }
          else {
            return false;
          }
        }
      }
      ds_it.load_cur_record();
      ++records_used;
      return true;
    } // next

    // Private data and methods
    //==========================================================================
  private:

    parameters params;

    //! Dataset
    dataset_view<la_type> ds_view;

    //! Iterator into the dataset; this points to the record which will be
    //! loaded when next() is called.
    typename dataset<la_type>::record_iterator ds_it;

    //! Iterator into the dataset (end iterator)
    typename dataset<la_type>::record_iterator ds_end;

    //! Count of number of examples drawn
    size_t records_used;

    //! Indicates if the oracle has been initialized (by calling next() for
    //! the first time).
    //! This is necessary because ds_it has to lag one call to next()
    bool initialized;

    //! Number of resets left until the next random permutation.
    //! If 0, this is ignored.
    size_t num_resets_until_randomization;

    boost::mt11213b rng;

    //! Randomly reorder the samples.
    void randomly_permute() {
      ds_view.restore_record_view();
      ds_view.set_record_indices(randperm(ds_view.size(), rng));
    }

  }; // class ds_oracle

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_DS_ORACLE_HPP
