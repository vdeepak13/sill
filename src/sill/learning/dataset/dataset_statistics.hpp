
#ifndef SILL_DATASET_STATISTICS_HPP
#define SILL_DATASET_STATISTICS_HPP

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <boost/iterator.hpp>
#include <boost/random/uniform_int.hpp>

#include <sill/range/concepts.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/learning/dataset/dataset.hpp>

#include <sill/range/algorithm.hpp>

#include <sill/macros_def.hpp>

/**
 * \file dataset_statistics.hpp This is a class for computing and storing
 *                              statistics about datasets.
 */

namespace sill {

  /**
   * The dataset_statistics class is associated with (and holds a const
   * reference to) a single dataset.  It may be used to compute and
   * (optionally) store statistics about the data.  It currently supports:
   *  - order statistics
   *  - mutual information
   *  - marginals
   *  - min/max of each vector variable
   *
   * \author Joseph Bradley
   * \ingroup learning_dataset
   * @todo Hold data in pointers to save space.
   * @todo Include distribution (or reference to it) in this class.
   */
  class dataset_statistics {

    // Public type declarations
    //==========================================================================
  public:

    // Protected data members
    //==========================================================================
  protected:

    //! The associated dataset.
    const dataset& ds;

    // Protected data members: order statistics
    //==========================================================================

    /**
     * For each vector variable (indexed as in the dataset's record's
     * vector() function), a vector of record indices in order
     * of increasing variable value:
     * order_stats_[i][j] is the index of the record with the j^th smallest
     * value for variable i
     */
    std::vector<std::vector<size_t> > order_stats_;

    //! Functor for computing order statistics
    struct order_stats_functor {

    private:
      const dataset& ds;
      size_t v;

    public:
      //! @param v  index for vector variable
      order_stats_functor(const dataset& ds, size_t v) : ds(ds), v(v) { }

      //! @return true iff record i comes before record j w.r.t. vector value v
      bool operator()(size_t i, size_t j) {
        // TODO: Write a function in dataset to return a view of
        //       a single column, and then use it here.
        // Actually, come back and do this later; do it inefficiently for now.
        // (Test how much of a difference it makes if it's done more efficiently
        // later.)
        return (ds.vector(i,v) < ds.vector(j,v));
      }

    };

    // Protected data members: mutual information
    //==========================================================================

    /**
     * Mutual information between all pairs of variables.
     * mutual_info_[i][j] = mutual information between variables i and j,
     *   in the natural order
     * \todo Generalize this to work for vector variables as well.
     */
    std::vector<std::vector<double> > mutual_info_;

    // Protected data members: marginals
    //==========================================================================

    //! Map: domain --> marginal over that domain
    mutable std::map<finite_domain,table_factor> marginal_map;

    // Constructors, getters, mutating operations
    //==========================================================================
  public:

    explicit dataset_statistics(const dataset& ds)
      : ds(ds) { }

    //! Return associated dataset.
    const dataset& get_dataset() const { return ds; }

    //! Free up space by clearing all saved statistics
    void clear_all() {
      clear_order_stats();
      clear_mutual_info();
      clear_marginals();
    }

    // Public methods: order statistics
    //==========================================================================

    //! Compute order statistics for datapoints, for vector variable v
    void compute_order_stats(size_t v);

    //! Compute order statistics for datapoints, for each vector variable
    void compute_order_stats();

    /**
     * Return all order statistics:
     * For each vector variable value (indexed as in the dataset's record's
     * vector() function), a vector of record indices in order
     * of increasing variable value:
     * order_stats()[i][j] is the index of the record with the j^th smallest
     * value for variable i
     */
    const std::vector<std::vector<size_t> >& order_stats() const {
      return order_stats_;
    }

    /**
     * Return order statistics for vector variable value i
     * (indexed as in the dataset's record's vector() function):
     * a vector of record indices in order of increasing variable value:
     * order_stats(i)[j] is the index of the record with the j^th smallest
     * value for variable i
     */
    const std::vector<size_t>& order_stats(size_t i) const {
      assert(i < order_stats_.size());
      return order_stats_[i];
    }

    /**
     * Return the index in the dataset of the record with
     * the j^th smallest value for vector variable value i.
     */
    const size_t order_stats(size_t i, size_t j) const {
      assert(i < order_stats_.size());
      assert(j < order_stats_[i].size());
      return order_stats_[i][j];
    }

    //! Free up space used for storing order statistics.
    void clear_order_stats() {
      order_stats_.clear();
    }

    // Public methods: mutual information
    //==========================================================================

    //! Compute mutual information between variables i and j, in the natural
    //! order.
    //! @param smoothing   amount of smoothing to use for computing variable
    //!                    marginals
    void compute_mutual_info
    (size_t i, size_t j, double smoothing = -1);

    //! Compute mutual information between all pairs of variables.
    //! @param smoothing   amount of smoothing to use for computing variable
    //!                    marginals
    void compute_mutual_info(double smoothing = -1);

    /**
     * Return mutual information between variables i and j.
     */
    double mutual_info(size_t i, size_t j) const {
      assert(i < mutual_info_.size());
      assert(j - i - 1 < mutual_info_[i].size());
      return mutual_info_[i][j - i - 1];
    }

    //! Free up space used for storing mutual information.
    void clear_mutual_info() {
      mutual_info_.clear();
    }

    // Public methods: marginals
    //==========================================================================

    /**
     * This returns the marginal over the given domain, silently computing
     * it if necessary.
     * This only supports marginals over finite variables now.
     * Note: If the marginal has been pre-computed with the wrong smoothing,
     *       then you must currently clear all marginals and recompute the
     *       one you want.  TO DO: Add the option to force recomputation.
     * @param smoothing  amount of smoothing (amount to add to each count
     *                   before normalizing)
     */
    const table_factor&
    marginal(const finite_domain& d, double smoothing = 1) const;

    //! Free up space used for storing marginals.
    void clear_marginals() const;

    // Public methods: min/max of each vector variable
    //==========================================================================

    //! Return a vector of the minimum values of all vector variables in 'vars'.
    static vec
    vector_var_min(const dataset& data, const vector_var_vector& vars);

    //! Return a vector of the maximum values of all vector variables in 'vars'.
    static vec
    vector_var_max(const dataset& data, const vector_var_vector& vars);

  };  // class dataset_statistics

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_DATASET_STATISTICS_HPP
