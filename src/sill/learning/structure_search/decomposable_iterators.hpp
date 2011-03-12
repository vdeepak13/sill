
#ifndef SILL_DECOMPOSABLE_ITERATORS_HPP
#define SILL_DECOMPOSABLE_ITERATORS_HPP

#include <boost/tuple/tuple.hpp>

#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/model/learnt_decomposable.hpp>
#include <sill/iterator/subset_iterator.hpp>

#include <sill/macros_def.hpp>

/**
 * \file decomposable_iterators.hpp Functions and classes for methodically
 *                                  generating decomposable models.
 * This is a set of functions and classes for generating all decomposable
 * models of certain types.  These types are currently supported:
 *  - empty: empty model with single-variable cliques
 *  - star: star-shaped junction trees with maximal cliques
 */

namespace sill {

  /**
   * Make the given model into an empty model (with single-variable cliques),
   * estimating marginals using the given statistics class.
   * @param model      This should be an empty model.
   * @param vars       Finite variables to include in the model.
   *                   (Vector variables are not yet supported.)
   * @param smoothing  smoothing for estimating marginals
   */
  template <typename F>
  void create_empty_decomposable(decomposable<F>& model,
                                 typename F::domain_type vars,
                                 const dataset_statistics<>& stats,
                                 double smoothing = 0) {
    std::vector<F> factors;
    foreach(finite_variable* v, vars)
      factors.push_back
      (stats.marginal(make_domain<typename F::variable_type>(v), smoothing));
    model *= factors;
  }

  /**
   * Class for iterating over a set of initial models.
   * These are star-shaped junction trees with maximal cliques.
   * For each set of variables of size max_clique_size, this
   * makes a central clique with those variables.  It then makes one clique
   * for each remaining variable, attached to the central clique
   * and sharing max_clique_size - 1 variables with it.
   *
   * For each set c of variables of size max_clique_size
   *   Make a central node v1 from clique c
   *   For each other variable x
   *     Make a node v2 with variable x and max_clique_size - 1
   *      other variables from c
   *     Attach v2 to v1
   *
   * \todo Choose these variables intelligently if given a decomposable_score
   *  to maximize.
   */
  template <typename F>
  class star_decomposable_iterator {

    concept_assert((Factor<F>));
    //should be:
    //concept_assert((DistributionFactor<F>));
    //concept_assert((Invertible<F>));
    // (Check decomposable.hpp)

  public:
    //! The type of variable associated with a factor
    typedef typename F::variable_type variable_type;

    //! The domain type of the factor
    typedef typename F::domain_type domain_type;

    //! Vertex type
    typedef typename decomposable<F>::vertex vertex;

    /**
     * PARAMETERS
     *  - vars (domain_type): variables to be included in the model
     *     (default = all variables in the data)
     *  - smoothing (double): amount of smoothing >= 0 to use in estimating
     *     marginals
     *     (default = .1)
     *  - max_clique_size (size_t): maximum size of cliques (>= 2)
     *     (default = 2)
     */
    class parameters {
    private:
      domain_type vars_;
      double smoothing_;
      size_t max_clique_size_;
    public:
      parameters() : smoothing_(.1), max_clique_size_(2) { }
      parameters& vars(domain_type value) { vars_ = value; return *this; }
      parameters& smoothing(double value) {
        assert(value >= 0);
        smoothing_ = value; return *this;
      }
      parameters& max_clique_size(size_t value) {
        assert(value >= 2);
        max_clique_size_ = value; return *this;
      }
      const domain_type& vars() const { return vars_; }
      double smoothing() const { return smoothing_; }
      size_t max_clique_size() const { return max_clique_size_; }
    }; // class parameters

  protected:
    // parameters
    domain_type vars;
    double smoothing;
    size_t max_clique_size;

    //! Current model.
    learnt_decomposable<F> model;

    //! Data used for parameter estimation.
    const dataset_statistics<>& stats;

    //! Iterators over subsets of vars of size max_clique_size.
    //! *clique_it holds the next central clique to be loaded by next().
    //! When clique_it == clique_end, the iterator is done.
    subset_iterator<domain_type> clique_it, clique_end;

  public:
    /**
     * Constructor.
     * Function next() must be called to load the first model.
     *
     * @param stats   statistics class for parameter estimation
     */
    explicit star_decomposable_iterator(const dataset_statistics<>& stats,
                                        parameters params = parameters())
      : vars(params.vars()), smoothing(params.smoothing()),
        max_clique_size(params.max_clique_size()), stats(stats) {
      if (vars.size() == 0)
        vars = stats.get_dataset().finite_variables();
      clique_it = subset_iterator<domain_type>(vars, max_clique_size);
    }

    //! Increment to next model.
    //! @return false if iterator is done
    bool next() {
      if (clique_it == clique_end)
        return false;
      model.clear();
      // Make a central node v1 from clique c
      domain_type c(*clique_it);
      vertex c_v(model.add_clique(c, stats.marginal(c, smoothing)));
      // For each other variable x
      domain_type other_vars(set_difference(vars, c));
      // Set c to be one variable smaller, and use it to form
      //  the extra cliques we create.
      // TODO: It'd be better to choose the set more carefully.
      c.erase(c.begin());
      foreach(variable_type* v, other_vars) {
        // Make a factor v2 with variable x and max_clique_size - 1
        //  other variables from c, make a clique for it, and attach it to c_v.
        vertex u(model.add_clique(set_union(c, v),
                                  stats.marginal(set_union(c, v), smoothing)));
        model.add_edge(c_v, u);
      }
      ++clique_it;
      /*
      // Make a central node v1 from clique c
      domain_type c(*clique_it);
      model.clear();
      std::vector<F> factor_vec;
      factor_vec.push_back(stats.marginal(c, smoothing));
      // For each other variable x
      domain_type other_vars(vars.minus(c));
      // Set c to be one variable smaller, and use it to form
      //  the extra cliques we create.
      // TODO: It'd be better to choose the set more carefully.
      c.remove(c.representative());
      foreach(variable_type* v, other_vars)
        // Make a factor v2 with variable x and max_clique_size - 1
        //  other variables from c
        factor_vec.push_back(stats.marginal(c.plus(v), smoothing));
      model *= factor_vec;
      ++clique_it;
      */
      return true;
    }

    //! Returns a const reference to the current model.
    const decomposable<F>& current() const {
      return model;
    }

  }; // class star_decomposable_iterator<F>

} // end of namespace: prl

#endif // #ifndef SILL_DECOMPOSABLE_ITERATORS_HPP
