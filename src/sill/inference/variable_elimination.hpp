
#ifndef SILL_VARIABLE_ELIMINATION_HPP
#define SILL_VARIABLE_ELIMINATION_HPP

#include <algorithm>
#include <iterator>
#include <vector>
#include <map>
#include <set>

#include <sill/factor/combine_iterator.hpp>
#include <sill/inference/commutative_semiring.hpp>
#include <sill/model/markov_graph.hpp>
#include <sill/graph/elimination.hpp>
#include <sill/copy_ptr.hpp>

#include <sill/stl_concepts.hpp>

#include <sill/range/transformed.hpp>
#include <sill/range/algorithm.hpp>

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
   * \todo Check that the range value is a factor.
   *
   * @tparam F
   *         Factor type.
   * @tparam Strategy
   *         Elimination strategy type which determines the elimination order.
   *         Must model the EliminationStrategy concept.
   * @tparam OutIt
   *         Output iterator type for collecting the output factors.
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
   * @param output
   *        The output iterator to which the result is written.
   *        A collection of factors is written to this
   *        iterator such that their combination implicitly represents
   *        the result of the algorithm.  See the other version of this
   *        algorithm if the explicit combination is desired.
   *
   * \ingroup inference
   */
  template <typename F,
            typename Strategy,
            typename OutIt>
  OutIt variable_elimination(const std::vector<F>& in_factors,
                             const typename F::domain_type& retain,
                             const commutative_semiring<F>& csr,
                             Strategy elim_strategy,
                             OutIt output) {
    typedef markov_graph<typename F::variable_type*> mg_type;
    concept_assert((Factor<F>));
    concept_assert((EliminationStrategy<Strategy, mg_type>));
    concept_assert((OutputIterator<OutIt, F>));

    typedef typename F::variable_type variable_type;
    typedef typename F::domain_type domain_type;

    // Copy the factors into a list that will be modified as we eliminate vars
    std::list<F> factors(in_factors.begin(), in_factors.end());

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
      combine_iterator<F,inplace_combination<F> > combine_it(csr.combine_init(), &csr);
      typename std::list<F>::iterator it = factors.begin();
      //unsigned erased = 0, n = factors.size();
      while(it != factors.end()) {
        // If the factor has elim_var in its arguments, add it to combination
        if (it->arguments().count(elim_var)) {
          combine_it = *it;
          ++combine_it;
          // Remove the factor from the list.
          factors.erase(it++);
        } else ++it;
      }

      // Now we have created the elimination factor.  Collapse out the
      // elimination variable from it, and then add it back to the
      // list of factors.
      domain_type args = combine_it.result().arguments();
      args.erase(elim_var);
      factors.push_front(csr.collapse(combine_it.result(), args));
      //std::cerr << "new result " << combine(factors, product_tag()) << std::endl;
    }
    // Report the remaining factors.
    return sill::copy(factors, output);
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
    combine_iterator<F, inplace_combination<F> > out(csr.combine_init(), &csr);
    return variable_elimination(in_factors, retain, csr, elim_strategy, out)
      .result();
  }

  //! @}

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_VARIABLE_ELIMINATION_HPP
