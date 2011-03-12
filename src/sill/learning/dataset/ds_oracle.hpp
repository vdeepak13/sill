
#ifndef SILL_DS_ORACLE_HPP
#define SILL_DS_ORACLE_HPP

#include <sill/learning/dataset/oracle.hpp>

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

    //! The base type (oracle)
    typedef oracle<LA> base;

    typedef record<LA> record_type;

    struct parameters {

      //! Limit on the number of records which may be
      //! drawn from the oracle; if 0, then no limit.
      //!  (default = 0)
      size_t record_limit;

      //! If true, then automatically resets to the first
      //! record when the end of the dataset is reached.
      //!  (default = true)
      bool auto_reset;

      parameters()
        : record_limit(0), auto_reset(true) { }

    };

    // Private data members
    //==========================================================================
  private:

    parameters params;

    //! Dataset
    const dataset<LA>& ds;

    //! Iterator into the dataset; this points to the record which will be
    //! loaded when next() is called.
    typename dataset<LA>::record_iterator ds_it;

    //! Iterator into the dataset (end iterator)
    typename dataset<LA>::record_iterator ds_end;

    //! Count of number of examples drawn
    size_t records_used;

    //! Indicates if the oracle has been initialized (by calling next() for
    //! the first time).
    //! This is necessary because ds_it has to lag one call to next()
    bool initialized;

    // Constructors
    //==========================================================================
  public:

    /**
     * Constructs a dataset-based oracle.
     */
    explicit ds_oracle(const dataset<LA>& ds, parameters params = parameters())
      : base(ds.datasource_info()),
        params(params), ds(ds), ds_it(ds.end()), ds_end(ds.end()),
        records_used(0), initialized(false) {
//      assert(ds.size() > 0);
    }

    // Getters and helpers
    //==========================================================================

    //! Returns the current record.
    const record_type& current() const {
      return ds_it.r;
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
        ds_it = ds.begin();
        if (ds_it == ds_end)
          return false;
        initialized = true;
      } else {
        ++ds_it;
        if (ds_it == ds_end) {
          if (params.auto_reset) {
            ds_it = ds.begin();
          }
          else {
            return false;
          }
        }
      }
      ds_it.load_cur_record();
//      current_record = ds_it.r;
      //      current_record = ds_it.get_record_ref();
      ++records_used;
      return true;
    } // next

  }; // class ds_oracle

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_DS_ORACLE_HPP
