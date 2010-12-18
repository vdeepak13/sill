// Probabilistic Reasoning Library (PRL)
// Copyright 2005, 2008 (see AUTHORS.txt for a list of contributors)
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

#ifndef SILL_LAZY_FACTOR_HPP
#define SILL_LAZY_FACTOR_HPP

#include <sill/global.hpp>
#include <sill/set.hpp>
#include <sill/variable.hpp>
#include <sill/assignment.hpp>
#include <sill/factor/factor.hpp>
#include <sill/inference/variable_elimination.hpp>
#include <sill/combine_iterator.hpp>
#include <sill/copy_ptr.hpp>

///////////////////////////////////////////////////////////////////
// Needs to be cleaned up & rewritten to use enum-based gdl
///////////////////////////////////////////////////////////////////
namespace sill {

  /**
   * A lazy factor, which uses variable elimination to delay
   * combination operations until collapse operations require them to
   * be carried out.  This type of factor can exploit internal
   * conditional independence structure.  TODO: document this more.
   *
   * \ingroup factor_types
   */
  template <typename factor_t,
            typename csr_tag_t = sill::sum_product_tag,
            typename elim_strategy_t = sill::min_degree_strategy>
  class lazy_factor_t {

  public:

    //! The type of values taken on by this factor.
    typedef typename factor_t::storage_t storage_t;

    /**
     * The \f$+\f$ operation (e.g. addition) of the commutative
     * semiring.
     */
    typedef typename csr_traits_t<csr_tag_t>::cross_op_tag_t cross_op_tag_t;

    /**
     * The \f$\cdot\f$ operation (e.g. multiplication) of the
     * commutative semiring.
     */
    typedef typename csr_traits_t<csr_tag_t>::dot_op_tag_t dot_op_tag_t;

  protected:

    //! The arguments of this factor.
    domain args;

    //! The type of pointer used to hold internal factors.
    typedef sill::copy_ptr<factor_t> factor_ptr_t;

    /**
     * The internal factors; the value of this factor is given by the
     * combination of these factors, using the \f$\cdot\f$ operator of
     * the commutative semiring.  This vector is mutable so that a
     * lazy factor can "flatten" itself within the context of a const
     * method.  This changes the internal representation, but not what
     * is represented.
     *
     * @see #flatten() const
     */
    mutable std::vector<factor_ptr_t> factor_ptr_vec;

    //! The elimination strategy to use.
    elim_strategy_t elim_strategy;

    /**
     * Simplifies the internal representation by combining any
     * internal factors that are subsumed into their subsumers.
     */
    void minimize() {
      typedef typename std::vector<factor_ptr_t>::iterator ptr_it_t;
      for (ptr_it_t ptr_it_a = factor_ptr_vec.begin();
           ptr_it_a != factor_ptr_vec.end(); ++ptr_it_a) {
        ptr_it_t ptr_it_b = ptr_it_a;
        ++ptr_it_b;
        while (ptr_it_b != factor_ptr_vec.end()) {
          // Check for subsumption.
          if ((*ptr_it_a)->arguments().subset_of
              ((*ptr_it_b)->arguments())) {
            // Factor A is subsumed by factor B.  Combine A into B,
            // move B to A's location, and erase B's location.
            (*ptr_it_b)->combine_in(*ptr_it_a, dot_op_tag_t());
            *ptr_it_a = *ptr_it_b;
            ptr_it_b = factor_ptr_vec.erase(ptr_it_b);
          } else if ((*ptr_it_b)->arguments().subset_of
                     ((*ptr_it_a)->arguments())) {
            // Factor B is subsumed by factor A.  Combine B into A,
            // and erase B's location.
            (*ptr_it_a)->combine_in(*ptr_it_b, dot_op_tag_t());
            ptr_it_b = factor_ptr_vec.erase(ptr_it_b);
          } else
            ++ptr_it_b;
        }
      }
    }

  public:

    /**
     * Conversion constructor.  The factor is held by a
     * (copy-on-write) pointer.
     *
     * @param factor_ptr the factor to wrap
     */
    lazy_factor_t(typename sill::copy_ptr<factor_t> factor_ptr,
                  elim_strategy_t elim_strategy = elim_strategy_t())
      : args(sill::const_ptr(factor_ptr)->arguments()),
        elim_strategy(elim_strategy)
    {
      factor_ptr_vec.push_back(factor_ptr);
    }

    //! Copy constructor.
    lazy_factor_t(const lazy_factor_t& f) { *this = f; }

    //! Conversion constructor.
    template <typename storage_t>
    lazy_factor_t(const constant_factor<storage_t>& constant_factor,
                  elim_strategy_t elim_strategy = elim_strategy_t())
      : args(constant_factor.arguments()),
        elim_strategy(elim_strategy)
    {
      factor_ptr_vec.push_back(factor_ptr_t(new factor_t(constant_factor)));
    }

    //! Assignment operator.
    const lazy_factor_t& operator=(const lazy_factor_t& f) {
      this->args = f.args;
      this->factor_ptr_vec = f.factor_ptr_vec;
      this->elim_strategy = f.elim_strategy;
      return *this;
    }

    /**
     * Updates this factor to be a copy of the supplied factor, and
     * updates the supplied factor to be a copy of this factor's
     * original value.
     */
    void swap(lazy_factor_t& f) {
      this->args.swap(f.args);
      this->factor_ptr_vec.swap(f.factor_ptr_vec);
      std::swap(this->elim_strategy, f.elim_strategy);
    }

    //! Returns a const reference to the argument set of this factor.
    const domain& arguments() const { return args; }

    /**
     * Renames the arguments of this factor.
     *
     * @param map
     *        an object such that map[v] maps the variable handle v
     *        a type compatible variable handle; this mapping must be 1:1.
     */
    void subst_args(const var_map& var_map) {
      // Compute the new arguments.
      this->args = subst_vars(this->args, var_map);
      for (typename std::vector<factor_ptr_t>::const_iterator it =
             factor_ptr_vec.begin(); it != factor_ptr_vec.end(); ++it)
        (*it)->subst_args(var_map);
    }

    /**
     * Returns the result of performing any delayed combination
     * operations.
     */
    inline sill::const_ptr_t<factor_t> flatten() const {
      sill::combine_iterator<factor_t, dot_op_tag_t> output_it;
      output_it = std::copy(factor_ptr_vec.begin(),
                factor_ptr_vec.end(),
                output_it);
      sill::const_ptr_t<factor_t> result = output_it.result();
      // Now that the flattening has been computed, store it.  (This
      // avoids repeating the computation.)  Note that it is not
      // always a win; if the flattening is large, then maintaining
      // the factorization can speed up later collapse operations
      // (since variable elimination can be used).
      factor_ptr_vec.clear();
      factor_ptr_vec.push_back(result);
      return result;
    }

    /**
     * Constructs a lazy factor that represents the result of
     * combining two lazy factors using the \f$\cdot\f$ operator of
     * the commutative semiring.  This operation is lazy.
     */
    lazy_factor_t(factor_combine_expr<lazy_factor_t,
                                        lazy_factor_t,
                                        dot_op_tag_t> expr,
                  elim_strategy_t elim_strategy = elim_strategy_t()) {
      this->args = expr.x_ptr->arguments().plus(expr.y_ptr->arguments());
      this->factor_ptr_vec = expr.x_ptr->factor_ptr_vec;
      this->factor_ptr_vec.insert(this->factor_ptr_vec.end(),
                                  expr.y_ptr->factor_ptr_vec.begin(),
                                  expr.y_ptr->factor_ptr_vec.end());
      this->elim_strategy = elim_strategy;
      minimize();
    }

    /**
     * Constructs a lazy factor that represents the result of
     * combining a lazy factors with an internal factor using the
     * \f$\cdot\f$ operator of the commutative semiring.  This
     * operation is lazy.
     */
    lazy_factor_t(factor_combine_expr<lazy_factor_t,
                                        factor_t,
                                        dot_op_tag_t> expr,
                  elim_strategy_t elim_strategy = elim_strategy_t()) {
      this->args = expr.x_ptr->arguments().plus(expr.y_ptr->arguments());
      this->factor_ptr_vec = expr.x_ptr->factor_ptr_vec;
      this->factor_ptr_vec.push_back(expr.y_ptr);
      this->elim_strategy = elim_strategy;
      minimize();
    }

    /**
     * Constructs a lazy factor that represents the result of
     * combining a lazy factors with an internal factor using the
     * \f$\cdot\f$ operator of the commutative semiring.  This
     * operation is lazy.
     */
    lazy_factor_t(factor_combine_expr<factor_t,
                                        lazy_factor_t,
                                        dot_op_tag_t> expr,
                  elim_strategy_t elim_strategy = elim_strategy_t()) {
      this->args = expr.x_ptr->arguments().plus(expr.y_ptr->arguments());
      this->factor_ptr_vec = expr.y_ptr->factor_ptr_vec;
      this->factor_ptr_vec.push_back(expr.x_ptr);
      this->elim_strategy = elim_strategy;
      minimize();
    }

    /**
     * Constructs a lazy factor that represents the result of
     * combining a lazy factors with another factor using the
     * \f$\cdot\f$ operator of the commutative semiring.  This
     * operation is lazy.
     */
    template <typename other_factor_t>
    lazy_factor_t(factor_combine_expr<lazy_factor_t,
                                        other_factor_t,
                                        dot_op_tag_t> expr,
                  elim_strategy_t elim_strategy = elim_strategy_t()) {
      this->args = expr.x_ptr->arguments().plus(expr.y_ptr->arguments());
      this->factor_ptr_vec = expr.x_ptr->factor_ptr_vec;
      this->factor_ptr_vec.push_back(factor_ptr_t(new factor_t(*expr.y_ptr)));
      this->elim_strategy = elim_strategy;
      minimize();
    }

    /**
     * Constructs a lazy factor that represents the result of
     * combining a lazy factors with another factor using the
     * \f$\cdot\f$ operator of the commutative semiring.  This
     * operation is lazy.
     */
    template <typename other_factor_t>
    lazy_factor_t(factor_combine_expr<other_factor_t,
                                        lazy_factor_t,
                                        dot_op_tag_t> expr,
                  elim_strategy_t elim_strategy = elim_strategy_t()) {
      this->args = expr.x_ptr->arguments().plus(expr.y_ptr->arguments());
      this->factor_ptr_vec = expr.y_ptr->factor_ptr_vec;
      this->factor_ptr_vec.push_back(factor_ptr_t(new factor_t(*expr.x_ptr)));
      this->elim_strategy = elim_strategy;
      minimize();
    }

    /**
     * Constructs a lazy factor that represents the result of
     * combining two lazy factors with an arbitrary binary operator
     * (which is not the \f$\cdot\f$ operation of the commutative
     * semiring).  This operation is not lazy.
     */
    template <typename binary_op_tag_t>
    lazy_factor_t(factor_combine_expr<lazy_factor_t,
                                        lazy_factor_t,
                                        binary_op_tag_t> expr,
                  elim_strategy_t elim_strategy = elim_strategy_t()) {
      // Flatten the input factors.
      sill::const_ptr_t<factor_t> x_ptr = expr.x_ptr->flatten();
      sill::const_ptr_t<factor_t> y_ptr = expr.y_ptr->flatten();
      // Initialize this factor with the combination of these two
      // factors.
      sill::const_ptr_t<factor_t> result_ptr
        (new factor_t(sill::combine(x_ptr, y_ptr, binary_op_tag_t())));
      this->factor_ptr_vec.push_back(result_ptr);
      this->args = result_ptr->arguments();
      this->elim_strategy = elim_strategy;
    }

    /**
     * Constructs a lazy factor that represents the result of
     * collapsing a lazy factor with the \f$+\f$ operator of the
     * commutative semiring.  This operation causes all delayed
     * combinations to be performed.  (Variable elimination is used to
     * interleave collapse operations with the delayed combine
     * operations.)
     */
    lazy_factor_t(factor_collapse_expr<lazy_factor_t,
                                         cross_op_tag_t> expr,
                  elim_strategy_t elim_strategy = elim_strategy_t()) {
      // We use variable elimination here.
      variable_elimination(expr.retained,
                           expr.x_ptr->factor_ptr_vec.begin(),
                           expr.x_ptr->factor_ptr_vec.end(),
                           std::back_inserter(this->factor_ptr_vec),
                           csr_tag_t(),
                           elim_strategy);
      this->args = expr.x_ptr->arguments().intersect(expr.retained);
      this->elim_strategy = elim_strategy;
    }

    /**
     * Constructs a lazy factor that represents the result of
     * collapsing a lazy factor with an arbitrary binary operator
     * (which is not the \f$+\f$ operator of the commutative
     * semiring).  This operation is not lazy.
     */
    template <typename binary_op_tag_t>
    lazy_factor_t(factor_collapse_expr<lazy_factor_t,
                                         binary_op_tag_t> expr,
                  elim_strategy_t elim_strategy = elim_strategy_t()) {
      // Flatten the input factor.
      sill::const_ptr_t<factor_t> x_ptr = expr.x_ptr->flatten();
      // Now apply the collapse operation to the result.
      this->factor_ptr_vec.push_back
        (sill::const_ptr_t<factor_t>
         (new factor_t(sill::collapse(x_ptr, expr.retained,
                                     binary_op_tag_t()))));
      this->args = expr.x_ptr->arguments().intersect(expr.retained);
      this->elim_strategy = elim_strategy;
    }

    /**
     * Constructs a lazy factor that represents the result of
     * restricting a lazy factor.
     */
    lazy_factor_t(factor_restriction_expr_t<lazy_factor_t> expr,
                  elim_strategy_t elim_strategy = elim_strategy_t()) {
      // Restrict each of the internal factors.
      for (typename std::vector<factor_ptr_t>::const_iterator it =
             expr.x_ptr->factor_ptr_vec.begin();
           it != expr.x_ptr->factor_ptr_vec.end(); ++it)
        this->factor_ptr_vec.push_back
          (sill::copy_ptr<factor_t>
           (new factor_t(sill::restrict(const_ptr(*it), expr.assignment))));
      this->args = expr.x_ptr->arguments().minus(expr.assignment.keys());
      this->elim_strategy = elim_strategy;
    }

    /**
     * Returns the value associated with the supplied assignment to
     * the factor's variables.
     *
     * @param  assignment
     *         a object such that for any variable_h v,
     *         assignment[v] gives the value of the associated variable
     * @return the value associated with the assignment
     */
    storage_t get(const assignment& assignment) const {
      typename std::vector<factor_ptr_t>::const_iterator it =
        factor_ptr_vec.begin();
      sill::const_ptr_t<factor_t> factor_ptr = *it;
      storage_t result = factor_ptr->get(assignment);
      binary_op<storage_t, dot_op_tag_t> op;
      while (++it != factor_ptr_vec.end()) {
        factor_ptr = *it;
        result = op(result, factor_ptr->get(assignment));
      }
      return result;
    }

    /**
     * Combines the supplied lazy factor into this lazy factor.  The
     * combination is computed using the combine operator of the
     * commutative semiring, and is performed lazily.
     *
     * @param  y
     *         The factor that is combined into this factor using
     *         the supplied binary operator.
     */
    void combine_in(const_ptr_t<lazy_factor_t> y_ptr,
                    dot_op_tag_t) {
      // Append the internal factors of the argument to this factor's
      // list of internal factors.
      this->factor_ptr_vec.insert(this->factor_ptr_vec.end(),
                                  y_ptr->factor_ptr_vec.begin(),
                                  y_ptr->factor_ptr_vec.end());
      // Update the arguments of this factor.
      this->args.insert(y_ptr->arguments());
      minimize();
    }

    /**
     * Combines the supplied factor into this lazy factor.  The
     * combination is computed using the combine operator of the
     * commutative semiring, and is performed lazily.
     *
     * @param  y
     *         The factor that is combined into this factor using
     *         the supplied binary operator.
     */
    void combine_in(const_ptr_t<factor_t> y_ptr,
                    dot_op_tag_t) {
      // Append the new factor to this factor's list of internal
      // factors.
      this->factor_ptr_vec.push_back(y_ptr);
      // Update the arguments of this factor.
      this->args.insert(y_ptr->arguments());
      minimize();
    }

    /**
     * Combines the supplied factor into this lazy factor.  The
     * combination is computed using the operator identified by the
     * supplied tag.  The values in this factor are used as the first
     * argument of the operator, and the values of the supplied factor
     * are used as the second argument.  This operation is not lazy.
     *
     * @param  y
     *         The factor that is combined into this factor using
     *         the supplied binary operator.  The values of y are
     *         used as the second argument of the operator.
     * @param  binary_op_tag
     *         the tag indicating which binary operator to use
     */
    template <typename other_factor_t,
              typename binary_op_tag_t>
    void combine_in(const_ptr_t<other_factor_t> y_ptr,
                    binary_op_tag_t binary_op_tag) {
      sill::copy_ptr<factor_t> flat_ptr = this->flatten();
      // Combine the argument into the flat factor.
      flat_ptr->combine_in(y_ptr, binary_op_tag);
      // Reset this factor to the result.
      factor_ptr_vec.clear();
      factor_ptr_vec.push_back(flat_ptr);
      // Update the arguments of this factor.
      this->args = const_ptr(factor_ptr_vec.front())->arguments();
    }

    /**
     * Applies the supplied functor to all values of the factor.  This
     * method simply calls the apply method of all internal factors.
     * (Note that if this factor has delayed combinations, then the
     * semantics of this method may be unexpected.)
     */
    template <typename functor_t>
    void apply(functor_t f) {
      for (typename std::vector<factor_ptr_t>::const_iterator it =
             factor_ptr_vec.begin(); it != factor_ptr_vec.end(); ++it)
        (*it)->apply(f);
    }

    /**
     * Computes the Kullback-Liebler divergence from this lazy factor
     * to the supplied lazy factor:
     *
     * \f[
     *   \sum_{{\bf x}} p({\bf x}) \log \frac{p({\bf x})}{q({\bf x})}
     * \f]
     *
     * where this factor is \f$p\f$ and the supplied factor is \f$q\f$.
     * Ordinarily this function is used when this factor and the
     * supplied factor are normalized; but this function does not
     * explicitly normalize either factor.
     *
     * @param f the factor to which the KL divergence is computed
     * @return  the KL divergence (in natural logarithmic units)
     */
    storage_t kl_divergence_to(const lazy_factor_t& f) const {
      // Flatten this factor and the argument factor.
      sill::const_ptr_t<factor_t> p_ptr = this->flatten();
      sill::const_ptr_t<factor_t> q_ptr = f.flatten();
      // Compute the KL using these factors.
      return p_ptr->kl_divergence_to(*q_ptr);
    }

    //! Writes a human-readable representation of the lazy factor.
    void write(std::ostream& out) const {
      // Write out the arguments.
      out << "Lazy factor over: " << this->args << std::endl;
      // Write out the internal factors.
      out << "Internal factors:" << std::endl;
      for (typename std::vector<factor_ptr_t>::const_iterator it =
             factor_ptr_vec.begin(); it != factor_ptr_vec.end(); ++it)
        out << *const_ptr(*it) << std::endl;
    }

  }; // end of class: lazy_factor_t

  //! Writes a human-readable representation of the lazy factor.
  template <typename leaf_factor_t>
  std::ostream& operator<<(std::ostream& out,
                           const lazy_factor_t<leaf_factor_t>& lazy_factor) {
    lazy_factor.write(out);
    return out;
  }

} // namespace sill

#ifdef SILL_CONSTANT_FACTOR_HPP
#include <sill/constant_and_lazy_factor.hpp>
#endif // #ifdef SILL_CONSTANT_FACTOR_HPP

#ifdef SILL_TABLE_FACTOR_HPP
#include <sill/table_and_lazy_factor.hpp>
#endif // #ifdef SILL_TABLE_FACTOR_HPP

#endif // SILL_LAZY_FACTOR_HPP



