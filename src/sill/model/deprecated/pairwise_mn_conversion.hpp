#ifndef SILL_PAIRWISE_MN_CONVERSION_HPP
#define SILL_PAIRWISE_MN_CONVERSION_HPP

#include <cmath>

#include <boost/range/iterator_range.hpp>

#include <sill/factor/table_factor.hpp>
#include <sill/iterator/transform_output_iterator.hpp>
#include <sill/model/bayesian_network.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/markov_network.hpp>
#include <sill/range/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Create a pairwise Markov network from a factorized model.
   * If the factorized model contains factors which are over more than 2
   * variables, then this unrolls the factors to create an equivalent pairwise
   * Markov network as follows:
   *  - Given a factor over ABC (say each are binary),
   *  - Create a factor over a new variable D with 8 values corresponding to
   *    those in the original factor.
   *  - Add an edge potential between A and D which enforces the constraint that
   *    A = 0 iff D has a value which corresponds to A having value 0.  Do the
   *    same for B and C.
   *  - This does not unroll the potential if it is only over one variable.
   *
   * @param fm factorized model to be transformed into a pairwise Markov net
   * @param u  universe used to create new variables for the Markov net
   * @return pair:
   *          - pairwise Markov network
   *          - mapping from the new variables in the Markov network to
   *            the (ordered) sets of old variables the new variables represent
   * @see restore_bp_pairwise_markov_network
   *
   * \todo Extend this so that it works with any types of factor.  This could be
   *       done by creating functions for each type of factor for creating
   *       indicator potentials of the type needed here.
   */
  std::pair
  <pairwise_markov_network<table_factor>,
   std::map<finite_variable*, std::vector<finite_variable*> > >
  fm2pairwise_markov_network
  (const factorized_model<table_factor>& fm, universe& u);

  /**
   * Convert the node beliefs from the given belief propagation engine (over
   * a pairwise Markov net which represents a general Markov net
   * (see fm2pairwise_markov_network())) into a set of factors which are only
   * over the original variables (in the general Markov net).
   * @param engine      BP engine over pairwise Markov net
   * @param orig_vars   variables in original general Markov net
   * @param var_mapping mapping from temp variables in pairwise net to (ordered)
   *                    sets of variables in orig_vars
   * @return set of factors (beliefs) over orig_vars
   * @see fm2pairwise_markov_network
   */
  template <typename F, typename FactorRange>
  std::vector<F> restore_unrolled_markov_network
  (const FactorRange& node_beliefs, const domain& orig_vars,
   std::map<finite_variable*, std::vector<finite_variable*> >& var_mapping) {
    concept_assert((InputRangeConvertible<FactorRange, F>));
    using namespace sill;
    std::vector<F> fvec;
    foreach(const F& f, node_beliefs) {
      finite_variable* v = *(f.arguments().begin());
      if (orig_vars.count(v))
        fvec.push_back(f);
      else
        fvec.push_back(f.roll_up(var_mapping[v]));
    }
    return fvec;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_PAIRWISE_MN_CONVERSION_HPP
