#ifndef SILL_VARIABLE_ELIMINATION_HPP
#define SILL_VARIABLE_ELIMINATION_HPP

#include <algorithm>
#include <iterator>
#include <vector>
#include <map>
#include <set>

#include <sill/factor/util/commutative_semiring.hpp>
#include <sill/factor/util/operations.hpp>
#include <sill/model/markov_graph.hpp>
#include <sill/graph/elimination.hpp>

#include <sill/stl_concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A visitor used in the elimination algorithm to record the order
   * in which (non-retained) variables should be eliminated.
   *
   * Implements the VertexVisitor concept.
   * \ingroup inference
   */
  template <typename OutIt, typename Node>
  class elimination_order_visitor {
    concept_assert((OutputIterator<OutIt, Node>));

    //! The output iterator to which variables are written.
    OutIt out;

    //! The set of variables that should not be reported.
    //! This set _must_ be stored by value, not by reference
    std::set<Node> retain;

  public:
    elimination_order_visitor(OutIt out, const std::set<Node>& retain)
      : out(out), retain(retain) { }

    //! Reports the variable that corresponds to a node in the Markov graph
    //! (if it is not a retained variable).
    template <typename MarkovGraph>
    void operator()(Node v, const MarkovGraph& g) {
      if (!retain.count(v)) { out = v; ++out; }
    }
  }; // elimination_order_visitor


  //! A convenience function for creating an elimination visitor.
  //! \relates elimination_order_visitor
  template <typename OutIt, typename Node>
  elimination_order_visitor<OutIt, Node>
  make_elimination_order_visitor(OutIt out, const std::set<Node>& retain) {
    return elimination_order_visitor<OutIt, Node>(out, retain);
  }

  /**
   * The variable elimination algorithm.
   * Given a collection of factors and a subset of their arguments,
   * this method efficiently combines the factors and collapses the result
   * to the desired arguments.
   *
   * @tparam F
   *         Factor type.
   * @tparam Strategy
   *         Elimination strategy type which determines the elimination order.
   *         Must model the EliminationStrategy concept.
   *   
   * @param in_factors
   *        The collection of input factors.
   * @param retain
   *        The retained arguments.
   * @param csr
   *        Commutative semiring which determines how factors are combined and
   *        collapsed.
   * @param elim_strategy
   *        The strategy used in choosing the order in which variables
   *        are eliminated.
   * @param factors
   *        The output collection of factors.
   *
   * \ingroup inference
   */
  template <typename F, typename Strategy>
  void variable_elimination(const std::vector<F>& in_factors,
                            const typename F::domain_type& retain,
                            const commutative_semiring<F>& csr,
                            Strategy elim_strategy,
                            std::list<F>& factors) {
    typedef markov_graph<typename F::variable_type*> mg_type;
    concept_assert((Factor<F>));
    concept_assert((EliminationStrategy<Strategy, mg_type>));

    typedef typename F::variable_type variable_type;
    typedef typename F::domain_type domain_type;

    // Copy the factors into a list that will be modified as we eliminate vars
    factors.assign(in_factors.begin(), in_factors.end());

    // Create a Markov graph for the factors.
    markov_graph<variable_type*> mg(arguments(factors));
    
    // Determine the elimination order.
    typename std::vector<variable_type*> elim_order;
    sill::eliminate
      (mg,
       make_elimination_order_visitor(std::back_inserter(elim_order), retain),
       elim_strategy);

    // Eliminate the variables
    foreach(variable_type* elim_var, elim_order) {
      // Combine all factors that have this variable as an argument.
      F combination = csr.combine_init();
      typename std::list<F>::iterator it = factors.begin();
      while(it != factors.end()) {
        if (it->arguments().count(elim_var)) {
          csr.combine_in(combination, *it);
          factors.erase(it++);
        } else ++it;
      }

      // Now we have created the elimination factor.  Collapse out the
      // elimination variable from it, and then add it back to the
      // list of factors.
      domain_type args = combination.arguments();
      args.erase(elim_var);
      factors.push_front(csr.collapse(combination, args));
    }
  }

  /**
   * The variable elimination algorithm.
   * Given a collection of factors and a subset of their arguments,
   * this method efficiently combines the factors and collapses
   * the result to the desired arguments.
   *
   * @tparam F
   *         Factor type.
   * @tparam Strategy
   *         Elimination strategy type which determines the elimination order.
   *         Must model the EliminationStrategy concept.
   *
   * @param in_factors
   *        the collection of input factors
   * @param retain
   *        the retained arguments
   * @param csr
   *        Commutative semiring which determines how factors are combined and
   *        collapsed.
   * @param elim_strategy
   *        The strategy used in choosing the order in which variables
   *        are eliminated.
   *
   * @return a factor representing the result
   *
   * \ingroup inference
   */
  template <typename F, typename Strategy>
  F variable_elimination(const std::vector<F>& in_factors,
                         const typename F::domain_type& retain,
                         const commutative_semiring<F>& csr,
                         Strategy elim_strategy) {
    concept_assert((Factor<F>));
    std::list<F> factors;
    variable_elimination(in_factors, retain, csr, elim_strategy, factors);
    return combine_all(factors, csr);
  }

  //! @}

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_VARIABLE_ELIMINATION_HPP
