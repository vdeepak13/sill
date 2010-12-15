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

#ifndef PRL_BSPT_FACTOR_HPP
#define PRL_BSPT_FACTOR_HPP

#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>

#include <prl/global.hpp>
#include <prl/set.hpp>
#include <prl/variable.hpp>
#include <prl/assignment.hpp>
#include <prl/factor/factor.hpp>
#include <prl/factor/constant_factor.hpp>
#include <prl/bsp_tree.hpp>
#include <prl/copy_ptr.hpp>

////////////////////////////////////////////////////////////////////
// Needs clean-up; does not compile at the moment
///////////////////////////////////////////////////////////////////

namespace prl {

  /**
   * A BSPT factor is a factor which uses a binary space partitioning
   * tree (BSPT) to represent context-sensitive independence
   * structure.  At internal nodes of the BSPT there are predicates
   * over assignments, and at leaf nodes of the BSPT there are factors
   * (of another type, such as table factors) which represent the
   * values of the factor for all assignments satisfying all ancestor
   * predicates.
   *
   * \todo Currently this factor implements predicates only for finite
   * variables, but it could be generalized to work with vector
   * variables as well.  This would require a new representation for
   * partition and region predicates, e.g., iso-hyperplanes and box
   * intervals.
   *
   * \ingroup factor_types
   */
  template <typename leaf_factor_t>
  class bspt_factor_t {

  public:

    //! The type of values taken on by this factor.
    typedef typename leaf_factor_t::storage_t storage_t;

  protected:

    /**
     * These are the traits used to make a BSP tree represent a
     * factor.
     */
    struct bsp_traits_t {

      /**
       * The representation of the BSP tree space.  This is the set of
       * variables that can be used for branching (i.e., the domain of
       * the factor minus the (common) domain of the leaf factors).
       */
      typedef domain space_t;

      //! The elements of the space, which are assignments.
      typedef assignment element;

      /**
       * The type of predicate used to partition the space; these are
       * equality predicates for particular variables.
       */
      typedef std::pair<variable_h, finite_value> predicate_t;

      //! Information about the range of a finite variable.
      struct range_info_t {

        //! A flag indicating if this variable has a fixed value.
        bool bound;

        //! If this variable is bound, this is its fixed value.
        finite_value binding;

        /**
         * If this variable is not fixed, this is a list of values
         * that it cannot take on.  If the size of this list were one
         * less than the arity of the corresponding variable, the
         * variable would be bound (to the missing value).  To avoid
         * these two different representations for a bound variable,
         * we require that the size of this list must be at least two
         * (2) less than the arity of the corresponding variable.
         */
        std::vector<finite_value> tabu_vec;

      };

      /**
       * Represents a region of the assignment space.  Each variable
       * of the space is mapped to information about its available
       * range.
       */
      typedef prl::map<variable_h, range_info_t> region_t;

      /**
       * Each leaf is associated with a factor.  This factor is held
       * using a copy-on-write shared-pointer.  Using a shared pointer
       * avoids unnecessary copies in cases like copying BSPT factors.
       * To avoid the copy-on-write penalty when only const access is
       * required, this pointer can be converted to a
       * prl::const_ptr_t<leaf_factor_t> pointer before dereferencing.
       */
      typedef typename prl::copy_ptr<leaf_factor_t> leaf_data_t;

      //! Returns the union space of the two supplied spaces.
      static inline space_t merge_spaces(const space_t& s,
                                         const space_t& t) {
        return s.plus(t);
      }

      /**
       * Returns truth if the supplied region overlaps the portion of
       * the space that satisfies a predicate (or its negation).
       */
      static inline prl::pred_set_rel_t relation(const predicate_t& p,
                                                 const region_t& r) {
        const variable_h& variable = p.first;
        const finite_value& value = p.second;
        const range_info_t& range_info = r.get(variable);
        if (range_info.bound)
          return (range_info.binding == value) ? positive_c : negative_c;
        else if (std::find(range_info.tabu_vec.begin(),
                           range_info.tabu_vec.end(), value) !=
                 range_info.tabu_vec.end())
          return negative_c;
        else
          return both_c;
      }

      /**
       * Returns true iff the supplied predicate is defined over the
       * supplied space.  This method is used when collapsing a BSP
       * tree to a subspace to determine which split predicates can
       * remain in the tree.  A predicate is defined for a space if it
       * can be evaluated for all members of the space.
       */
      static inline bool is_defined(const predicate_t& predicate,
                                    const space_t& space) {
        return space.contains(predicate.first);
      }

      //! Splits node data into two along a supplied partition.
      static inline std::pair<region_t, region_t>
      split_region(const region_t& region,
                   const predicate_t& predicate) {
        const variable_h& variable = predicate.first;
        const finite_value& value = predicate.second;
#if 0
        assert(region[variable].bound == false);
        assert(std::find(region[variable].tabu_vec.begin(),
                         region[variable].tabu_vec.end(), value)
               == region[variable].tabu_vec.end());
        assert(region[variable].tabu_vec.size() <
               variable->as_finite().size() - 1);
#endif
        region_t true_region = region;
        range_info_t& true_range_info = true_region[variable];
        true_range_info.bound = true;
        true_range_info.binding = value;
        true_range_info.tabu_vec.clear();
        region_t false_region = region;
        range_info_t& false_range_info = false_region[variable];
        false_range_info.bound = false;
        false_range_info.tabu_vec.push_back(value);
        // Check for disallowed case where tabu vector contains all
        // values but one (which really is a binding).
        if (false_range_info.tabu_vec.size() ==
            variable->as_finite().size() - 1) {
          // We must find the missing value.  Sort the vector and look
          // for the discontinuity.
          std::sort(false_range_info.tabu_vec.begin(),
                    false_range_info.tabu_vec.end());
          for (finite_value v = 0; v < false_range_info.tabu_vec.size(); ++v)
            if (false_range_info.tabu_vec[v] != v) {
              false_range_info.bound = true;
              false_range_info.binding = v;
              break;
            }
          if (!false_range_info.bound) {
            false_range_info.bound = true;
            false_range_info.binding = variable->as_finite().size() - 1;
          }
          false_range_info.tabu_vec.clear();
        }
        return std::make_pair(true_region, false_region);
      }

      //! Collapses the supplied region to a subspace.
      static inline region_t
      collapse_region(const region_t& region,
                      const space_t& subspace) {
        region_t result;
        space_t::iterator_t it, end;
        for (boost::tie(it, end) = subspace.values(); it != end; ++it) {
          if (region.contains(*it))
            result[*it] = region[*it];
        }
        return result;
      }

      /**
       * Splits leaf data into two along a supplied partition.  This
       * duplicates the factor for both children.
       *
       * @param leaf_data the data associated with a leaf
       * @param partition a partition of the space
       */
      static inline std::pair<leaf_data_t, leaf_data_t>
      split_leaf_data(const leaf_data_t& leaf_data,
                      const predicate_t& /* partition (unused) */) {
        return std::make_pair(leaf_data, leaf_data);
      }

      /**
       * Returns truth if the supplied element satisfies the
       * partitioning predicate.
       */
      static inline bool satisfies(const element& assignment,
                                   const predicate_t& predicate) {
        value_t value = assignment[predicate.first];
        return (as_finite(value) == predicate.second);
      }

      //! Computes the initial node data for a given space.
      static inline region_t init_region(const space_t& space) {
        region_t region;
        space_t::const_iterator_t it, end;
        for (boost::tie(it, end) = space.values(); it != end; ++it)
          region[*it].bound = false;
        return region;
      }

    }; // struct bspt_factor_t::bsp_traits_t

    /**
     * Counts the number of assignments consistent with the predicates
     * satisfied by node.
     */
    static inline finite_value
    num_assignments(const typename bsp_traits_t::region_t& region) {
      finite_value n = 1;
      typename bsp_traits_t::region_t::const_iterator_t it, end;
      for (boost::tie(it, end) = region.values(); it != end; ++it)
        if (!it->second.bound)
          n *= it->first->as_finite().size() - it->second.tabu_vec.size();
      return n;
    }

    //! The arguments of this factor.
    domain args;

    //! The arguments of the leaf factors.
    domain leaf_args;

    //! The type of BSP tree used to store the factors.
    typedef bsp_tree_t<bsp_traits_t> bspt_t;

    //! The BSP tree used to represent the factor.
    copy_ptr<bspt_t> bspt_ptr;

    //! Returns a mutable reference to the underlying BSP tree.
    bspt_t& get_bspt() { return *bspt_ptr; }

    //! Returns a const reference to the underlying BSP tree.
    const bspt_t& get_bspt() const { return *const_ptr(bspt_ptr); }

    //! A simple functor used for combination operations.
    template <typename binary_op_tag_t>
    struct combine_op_t {
      typedef typename bsp_traits_t::leaf_data_t leaf_data_t;
      typedef typename bsp_traits_t::region_t region_t;
      leaf_data_t operator()(const region_t& /* x_measure (unused) */,
                             const leaf_data_t& x_ptr,
                             const region_t& /* y_measure (unused) */,
                             const leaf_data_t& y_ptr) {
        // First convert to const pointers to avoid copy-on-write.
        const_ptr_t<leaf_factor_t> x_const_ptr(x_ptr);
        const_ptr_t<leaf_factor_t> y_const_ptr(y_ptr);
        // Note that the measure is irrelevant to the combination.
        return prl::copy_ptr<leaf_factor_t>
          (new leaf_factor_t(combine(x_const_ptr, y_const_ptr,
                                     binary_op_tag_t())));
      }
    };

    //! A simple functor used for collapse operations.
    template <typename binary_op_tag_t>
    struct unary_collapse_op_t {
      typedef typename bsp_traits_t::region_t region_t;
      typedef typename bsp_traits_t::leaf_data_t leaf_data_t;
      typedef typename bsp_traits_t::space_t space_t;
      const domain& retained;
      unary_collapse_op_t(const domain& retained) : retained(retained) { }
      leaf_data_t operator()(const region_t& /* region (unused) */,
                             const leaf_data_t x_ptr,
                             const space_t& /* subspace (unused) */) const {
        // First convert to a const pointer to avoid copy-on-write.
        const_ptr_t<leaf_factor_t> x_const_ptr(x_ptr);
        return leaf_data_t
          (new leaf_factor_t(collapse(x_const_ptr, retained,
                                      binary_op_tag_t())));
      }
    };

    /**
     * The binary collapse operation used when unrecognized operator
     * tags are supplied to the collapse routines.
     */
    struct invalid_binary_collapse_op_t { };

    /**
     * The binary collapse operation used for collapses implementing
     * marginalizations.
     */
    struct binary_sum_collapse_op_t {
      typedef typename bsp_traits_t::region_t region_t;
      typedef typename bsp_traits_t::leaf_data_t leaf_data_t;
      leaf_data_t operator()(const region_t& x_measure,
                             const leaf_data_t& x_ptr,
                             const region_t& y_measure,
                             const leaf_data_t& y_ptr) {
        // First convert to const pointers to avoid copy-on-write.
        const_ptr_t<leaf_factor_t> x_const_ptr(x_ptr);
        const_ptr_t<leaf_factor_t> y_const_ptr(y_ptr);
        // Count the number of assignments satisfying each leaf.
        finite_value num_x_assignments = num_assignments(x_measure);
        finite_value num_y_assignments = num_assignments(y_measure);
        // Scale each leaf factor by the number of assignments.
        typedef prl::constant_factor<finite_value> constant_t;
        const_ptr_t<constant_t> xc_ptr(new constant_t(num_x_assignments));
        const_ptr_t<constant_t> yc_ptr(new constant_t(num_y_assignments));
        const_ptr_t<leaf_factor_t>
          x_scaled_ptr(new leaf_factor_t(prl::combine(x_const_ptr, xc_ptr,
                                                      product_tag())));
        const_ptr_t<leaf_factor_t>
          y_scaled_ptr(new leaf_factor_t(prl::combine(y_const_ptr, yc_ptr,
                                                      product_tag())));
        // Now add the two scaled factors.
        return prl::copy_ptr<leaf_factor_t>
          (new leaf_factor_t(combine(x_scaled_ptr, y_scaled_ptr,
                                     sum_tag())));
      }
    };

    /**
     * Protected constructor.  Note that the arguments must be the
     * union of the internal arguments and the space of the BSP tree.
     */
    bspt_factor_t(domain args,
                  domain leaf_args,
                  copy_ptr<bspt_t> bspt_ptr)
      : args(args), leaf_args(leaf_args), bspt_ptr(bspt_ptr) { }

    /**
     * Initializes this factor to be a stump.
     *
     * @param var   the variable used in the root predicate
     * @param value the value used in the root predicate
     * @param true_factor
     *        the value of this factor if var = value
     * @param false_factor
     *        the value of this factor if var != value
     */
    inline void stump_initialize(variable_h var,
                                 finite_value value,
                                 const bspt_factor_t& true_factor,
                                 const bspt_factor_t& false_factor) {
      // The value must be valid.
      assert(value < var->as_finite().size());
      // The two leaf factors must have the same leaf arguments.
      assert(true_factor.leaf_args == false_factor.leaf_args);
      // Initialize the arguments and internal arguments of this
      // factor.
      this->leaf_args = true_factor.leaf_args;
      this->args = true_factor.args.plus(false_factor.args);
      this->args.insert(var);
      // Initialize the space of this factor as the arguments minus
      // the internal arguments.
      domain space = this->args.minus(this->leaf_args);
      // Construct the initial node data.
      typename bsp_traits_t::predicate_t predicate(var, value);
      bspt_ptr.reset(new bspt_t(bsp_traits_t(),
                                space,
                                predicate,
                                true_factor.get_bspt(),
                                false_factor.get_bspt()));
    }

  public:

    /**
     * Conversion constructor.  This factor takes ownership of the
     * supplied factor.
     *
     * @param factor_ptr the root factor
     */
    bspt_factor_t(typename prl::copy_ptr<leaf_factor_t> factor_ptr)
      : args(factor_ptr->arguments()),
        leaf_args(factor_ptr->arguments())
    {
      // The initial node data is empty because there are no
      // non-internal arguments.
      bspt_ptr.reset(new bspt_t(bsp_traits_t(),
                                domain(),
                                factor_ptr));
    }

    /**
     * Conversion constructor.  This factor makes a copy of the
     * supplied factor.
     *
     * @param factor_ptr the root factor
     */
    bspt_factor_t(const leaf_factor_t& factor)
      : args(factor.arguments()),
        leaf_args(factor.arguments())
    {
      // The initial node data is empty because there are no
      // non-internal arguments.
      typename bsp_traits_t::leaf_data_t leaf_data(new leaf_factor_t(factor));
      bspt_ptr.reset(new bspt_t(bsp_traits_t(),
                                domain(),
                                leaf_data));
    }

    //! Copy constructor.
    bspt_factor_t(const bspt_factor_t& f) { *this = f; }

    //! Conversion constructor.
    template <typename storage_t>
    bspt_factor_t(const constant_factor<storage_t>& constant_factor)
      : args(domain::empty_set),
        leaf_args(domain::empty_set)
    {
      // The initial node data is empty because there are no
      // non-internal arguments.
      typename bsp_traits_t::leaf_data_t
        leaf_data(new leaf_factor_t(constant_factor));
      bspt_ptr.reset(new bspt_t(bsp_traits_t(),
                                domain(),
                                leaf_data));
    }

    /**
     * BSPT stump constructor.
     *
     * @param var   the variable used in the root predicate
     * @param value the value used in the root predicate
     * @param true_factor
     *        the value of this factor if var = value
     * @param false_factor
     *        the value of this factor if var != value
     */
    bspt_factor_t(variable_h var,
                  finite_value value,
                  const bspt_factor_t& true_factor,
                  const bspt_factor_t& false_factor) {
      stump_initialize(var, value, true_factor, false_factor);
    }

    /**
     * BSPT stump constructor.
     *
     * @param var   the variable used in the root predicate
     * @param value the value used in the root predicate
     * @param true_factor
     *        the value of this factor if var = value.  This
     *        factor must be convertible to this BSPT factor
     *        type via one of the conversion constructors.
     * @param false_factor
     *        the value of this factor if var != value.  This
     *        factor must be convertible to this BSPT factor
     *        type via one of the conversion constructors.
     */
    template <typename true_factor_t,
              typename false_factor_t>
    bspt_factor_t(variable_h var,
                  finite_value value,
                  const true_factor_t& true_factor,
                  const false_factor_t& false_factor) {
      stump_initialize(var, value, bspt_factor_t(true_factor),
                       bspt_factor_t(false_factor));
    }

    //! Assignment operator.
    const bspt_factor_t& operator=(const bspt_factor_t& f) {
      this->args = f.args;
      this->leaf_args = f.leaf_args;
      this->bspt_ptr = f.bspt_ptr;
      return *this;
    }

    /**
     * Updates this factor to be a copy of the supplied factor, and
     * updates the supplied factor to be a copy of this factor's
     * original value.
     */
    void swap(bspt_factor_t& f) {
      this->args.swap(f.args);
      this->leaf_args.swap(f.leaf_args);
      this->bspt_ptr.swap(f.bspt_ptr);
    }

    //! Returns true iff the BSP tree has a single node (the root).
    inline bool is_singleton() const { return get_bspt().is_singleton(); }

    /**
     * Returns a const reference to the only leaf factor associated
     * with this tree; this method requires that the tree is a
     * singleton.
     */
    inline const prl::copy_ptr<leaf_factor_t>& get_singleton_factor() const {
      return get_bspt().get_singleton_data();
    }

    //! Returns a const reference to the argument set of this factor.
    const domain& arguments() const { return args; }

    /**
     * Returns a const reference to the argument set that is common to
     * all leaf factors of this BSPT factor.  This is a subset of
     * this factor's arguments.
     */
    const domain& leaf_arguments() const { return leaf_args; }

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
      // Compute the new internal arguments.
      this->leaf_args = subst_vars(this->leaf_args, var_map);
      // Update the space of the BSP tree.
      bspt_ptr->set_space(this->args.minus(this->leaf_args));
      // Perform the substitution in all predicates and leaf factors.
      typename bspt_t::node_iterator_t n_begin, n_end;
      for (boost::tie(n_begin, n_end) = get_bspt().nodes();
           n_begin != n_end; ++n_begin) {
        typename bspt_t::node_t& node = *n_begin;
        if (node.is_leaf())
          // Substitute arguments in all leaf factors.
          node.as_leaf()->leaf_data->subst_args(var_map);
        else {
          // Substitute variables used in split predicates.
          typename bspt_t::predicate_t& split = node.as_interior()->split;
          split.first = var_map[split.first];
        }
        // Substitute arguments in the measure accumulator.
        typename bspt_t::region_t new_region;
        typename bspt_t::region_t::iterator_t a_it, a_end;
        for (boost::tie(a_it, a_end) = node.region.elts();
             a_it != a_end; ++a_it)
          new_region[var_map[a_it->first]] = a_it->second;
        node.region.swap(new_region);
      }
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
      // Note that a const pointer is used here to avoid copy-on-write.
      typename prl::const_ptr_t<leaf_factor_t> leaf_factor_ptr =
        get_bspt().get_leaf_data(assignment);
      return leaf_factor_ptr->get(assignment);
    }

    /**
     * Changes the value associated with the supplied assignment to
     * the factor's variables.  Note that due to the internal
     * structure of this factor, this may also change the value
     * associated with other assignments.
     *
     * @param  assignment
     *         a object such that for any variable_h v,
     *         assignment[v] gives the value of the associated variable
     * @param  value the value associated with the assignment
     */
    void set(const assignment& assignment, storage_t val) {
      typename prl::copy_ptr<leaf_factor_t> leaf_factor_ptr =
        get_bspt().get_leaf_data(assignment);
      leaf_factor_ptr->set(assignment, val);
    }

    /**
     * Constructs a BSPT factor that represents the result of
     * combining two BSPT factors with a binary operator.
     */
    template <typename binary_op_tag_t>
    bspt_factor_t(factor_combine_expr<bspt_factor_t,
                                        bspt_factor_t,
                                        binary_op_tag_t> expr) {
      // TODO: Currently it must be the case that the internal
      // arguments of x (y) do not overlap the non-internal arguments
      // of y (x, respectively).  In the future, this restriction can
      // be lifted by first collapsing the BSP tree along variables
      // that are present in leaf factors.
      domain x_nonleaf_args = expr.x_ptr->args.minus(expr.x_ptr->leaf_args);
      domain y_nonleaf_args = expr.y_ptr->args.minus(expr.y_ptr->leaf_args);
      assert(expr.x_ptr->leaf_args.intersection_size(y_nonleaf_args) == 0);
      assert(expr.y_ptr->leaf_args.intersection_size(x_nonleaf_args) == 0);
      // The arguments and internal arguments are obtained as unions.
      this->args = expr.x_ptr->arguments().plus(expr.y_ptr->arguments());
      this->leaf_args = expr.x_ptr->leaf_args.plus(expr.y_ptr->leaf_args);
      // Compute the combination.
      this->bspt_ptr.reset(new bspt_t(expr.x_ptr->get_bspt(),
                                      expr.y_ptr->get_bspt(),
                                      combine_op_t<binary_op_tag_t>()));
    }

    /**
     * Constructs a BSPT factor that represents the result of
     * combining a BSPT factor with another (non BSPT) factor with a
     * binary operator.
     */
    template <typename binary_op_tag_t,
              typename other_factor_t>
    bspt_factor_t(factor_combine_expr<bspt_factor_t,
                                        other_factor_t,
                                        binary_op_tag_t> expr) {
      // TODO: Currently it must be the case that the non-leaf
      // arguments of expr.x do not overlap the arguments of expr.y_ptr->
      // In the future, this restriction can be lifted by first
      // collapsing the BSP tree along variables that are present in
      // leaf factors.
      domain x_nonleaf_args = expr.x_ptr->args.minus(expr.x_ptr->leaf_args);
      assert(expr.y_ptr->arguments().intersection_size(x_nonleaf_args) == 0);
      // The structure of the resulting factor is the same as the
      // first input; copy it.
      *this = *(expr.x_ptr);
      // Now update the leaf factors to be the appropriate
      // combinations.
      typename bspt_t::leaf_iterator_t it, end;
      for (boost::tie(it, end) = get_bspt().leaves(); it != end; ++it)
        it->leaf_data = prl::copy_ptr<leaf_factor_t>
          (new leaf_factor_t(prl::combine(it->leaf_data,
                                          expr.y_ptr,
                                          binary_op_tag_t())));
      // The arguments and internal arguments are obtained as unions.
      this->args = expr.y_ptr->arguments().plus(this->args);
      this->leaf_args = expr.y_ptr->arguments().plus(this->leaf_args);
    }

    /**
     * Constructs a BSPT factor that represents the result of
     * collapsing a BSPT factor with a binary operator.
     */
    template <typename binary_op_tag_t>
    bspt_factor_t(factor_collapse_expr<bspt_factor_t,
                                         binary_op_tag_t> expr) {
      // Special case: all arguments are retained.
      if (expr.x_ptr->arguments().subset_of(expr.retained)) {
        *this = *(expr.x_ptr);
        return;
      }
      // The binary collapse operation we hand to the BSP tree depends
      // upon the operation tag in the collapse expression.  However,
      // the operation applied to leaf factors is not just this
      // operation; duplicates must be taken into account.  For
      // example, when collapsing using the sum operator, we must
      // combine leaf factors taking into account their multiplicity.
      // The most natural way to do this is to template the definition
      // of the binary operator handed to the BSP tree by the tag
      // above.  In the generic case (which covers operations like
      // min, max, and, and or), the multiplicity can be ignored.  But
      // in special cases like addition and multiplication, the
      // template would take into account multiplicities.
      // Unfortunately, partial template specialization (the most
      // concise way to represent this) is not supported outside
      // namespace scopes (like class scopes).  So, we have to resort
      // to some template metaprogramming.  Ugh!  TODO: there may be a
      // nice way to use operator iterates to make this reasoning
      // cleaner (e.g., to add two factors that are weighted by the
      // measure, we really want generalized addition ax + by, which
      // is an iterated version of addition).
      using namespace boost;
      using namespace boost::mpl;
      typedef typename if_c<is_same<binary_op_tag_t, sum_tag>::value,
                            binary_sum_collapse_op_t,
                            invalid_binary_collapse_op_t>::type
        binary_collapse_op_t;
      // TODO: the typedef above must be generalized to deal with all
      // possible tags (sum_tag, product_tag, max_tag,
      // min_tag, and_tag, or_tag).
      this->args = expr.x_ptr->arguments().intersect(expr.retained);
      this->leaf_args = expr.x_ptr->leaf_args.intersect(expr.retained);
      this->bspt_ptr.reset
      (new bspt_t(expr.x_ptr->get_bspt(),
                  args.minus(leaf_args),
                  unary_collapse_op_t<binary_op_tag_t>(leaf_args),
                  binary_collapse_op_t()));
    }

    /**
     * Constructs a BSPT factor that represents the result of
     * restricting a BSPT factor.
     */
    bspt_factor_t(factor_restriction_expr_t<bspt_factor_t> expr) {
      domain x_nonleaf_args = expr.x_ptr->args.minus(expr.x_ptr->leaf_args);
      domain bound_vars = expr.assignment.keys();
      // Make this factor a copy of the original factor.
      *this = *(expr.x_ptr);
      // Update the arguments of this factor.
      this->args.remove(bound_vars);
      this->leaf_args.remove(bound_vars);
      // If the non-leaf arguments of x are not disjoint from the
      // bound variables, then we must update the tree structure.
      if (x_nonleaf_args.meets(bound_vars))
        this->bspt_ptr.reset(new bspt_t(*const_ptr(this->bspt_ptr),
                                        x_nonleaf_args.minus(bound_vars),
                                        expr.assignment));
      // If the leaf arguments of x include restricted variables, we
      // must restrict the leaf factors.
      if (expr.x_ptr->leaf_args.meets(bound_vars)) {
        typename bspt_t::leaf_iterator_t it, end;
        for (boost::tie(it, end) = get_bspt().leaves(); it != end; ++it)
          it->leaf_data.reset(new leaf_factor_t
                              (prl::restrict(it->leaf_data,
                                             expr.assignment)));
      }
    }

    /**
     * Combines the supplied BSPT factor into this BSPT factor.  The
     * combination is computed using the operator identified by the
     * supplied tag.  The values in this factor are used as the first
     * argument of the operator, and the values of the supplied factor
     * are used as the second argument.
     *
     * @param  y
     *         The factor that is combined into this factor using
     *         the supplied binary operator.  The values of y are
     *         used as the second argument of the operator.
     * @param  binary_op_tag
     *         the tag indicating which binary operator to use
     */
    template <typename binary_op_tag_t>
    void combine_in(const_ptr_t<bspt_factor_t> y_ptr,
                    binary_op_tag_t binary_op_tag) {
      // TODO: currently, there is no performance advantage to using
      // combine_in rather than combine.
      const_ptr_t<bspt_factor_t> c_ptr(new bspt_factor_t(*this));
      bspt_factor_t d(prl::combine(c_ptr, y_ptr, binary_op_tag));
      this->swap(d);
    }

    /**
     * Combines the supplied BSPT factor into this BSPT factor.  The
     * combination is computed using the operator identified by the
     * supplied tag.  The values in this factor are used as the first
     * argument of the operator, and the values of the supplied factor
     * are used as the second argument.
     *
     * @param  y
     *         The factor that is combined into this factor using
     *         the supplied binary operator.  The values of y are
     *         used as the second argument of the operator.
     * @param  binary_op_tag
     *         the tag indicating which binary operator to use
     */
    template <typename factor_t,
              typename binary_op_tag_t>
    void combine_in(const_ptr_t<factor_t> y_ptr,
                    binary_op_tag_t binary_op_tag) {
      // TODO: Currently it must be the case that the internal
      // arguments of y do not overlap the non-internal arguments of
      // this factor.  In the future, this restriction can be lifted
      // by first collapsing the BSP tree along variables that are
      // present in leaf factors.
      domain x_nonleaf_args = this->args.minus(this->leaf_args);
      assert(y_ptr->arguments().intersection_size(x_nonleaf_args) == 0);
      // The arguments and internal arguments are obtained as unions.
      this->args = y_ptr->arguments().plus(this->args);
      this->leaf_args = y_ptr->arguments().plus(this->leaf_args);
      // Combine x into each leaf's factor.
      typename bspt_t::leaf_iterator_t it, end;
      for (boost::tie(it, end) = get_bspt().leaves(); it != end; ++it)
        it->leaf_data->combine_in(y_ptr, binary_op_tag);
    }

    /**
     * Applies the supplied functor to all values of the factor.  This
     * method simply calls the apply method of all internal factors.
     */
    template <typename functor_t>
    void apply(functor_t f) {
      typename bspt_t::leaf_iterator_t it, end;
      for (boost::tie(it, end) = get_bspt().leaves(); it != end; ++it)
        it->leaf_data->apply(f);
    }

  protected:

    /**
     * A dual-tree visitor which is used to compute Kullback-Liebler
     * divergences between BSPT factors.
     */
    struct kl_divergence_visitor_t {
      typedef typename bsp_traits_t::region_t region_t;
      typedef typename bspt_t::leaf_node_t leaf_node_t;
      storage_t kld;
      kl_divergence_visitor_t() : kld(static_cast<storage_t>(0)) { }
      void visit(const region_t& region,
                 const leaf_node_t& this_leaf,
                 const leaf_node_t& other_leaf) {
        storage_t measure =
          static_cast<storage_t>(num_assignments(region));
        // First convert to const pointers to avoid copy-on-write.
        const_ptr_t<leaf_factor_t> x_const_ptr(this_leaf.leaf_data);
        const_ptr_t<leaf_factor_t> y_const_ptr(other_leaf.leaf_data);

        kld += measure * x_const_ptr->kl_divergence_to(*y_const_ptr);
      }
    };

  public:

    /**
     * Computes the Kullback-Liebler divergence from this BSPT factor
     * to the supplied BSPT factor:
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
    storage_t kl_divergence_to(const bspt_factor_t& f) const {
      kl_divergence_visitor_t visitor;
      this->bspt_ptr->visit_with(*(f.bspt_ptr), visitor);
      return visitor.kld;
    }

    //! Writes a human-readable representation of the BSPT factor.
    void write(std::ostream& out) const {
      // Write out the arguments.
      out << "BSPT factor over: " << this->args << std::endl;
      out << "  Internal arguments: " << this->leaf_args << std::endl;
      out << "  Branching arguments: " << get_bspt().get_space()
          << std::endl;
      typename bspt_t::const_leaf_iterator_t it, end;
      boost::tie(it, end) = get_bspt().leaves();
      std::size_t num_leaves = std::distance(it, end);
      // Write out the leaves.
      out << num_leaves
          << ((num_leaves > 1) ? " leaves:" : " leaf:")
          << std::endl;
      for (int i = 1; it != end; ++i, ++it) {
        const typename bspt_t::leaf_node_t& leaf = *it;
        out << "Leaf " << i << " ("
            << num_assignments(leaf.region)
            << " non-internal assignments):" << std::flush;
        // Write out the predicates satisfied by this leaf.
        const typename bspt_t::interior_node_t* int_node_ptr = leaf.parent;
        bool satisfies_p = leaf.is_true_child();
        while (int_node_ptr != NULL) {
          const typename bsp_traits_t::predicate_t& predicate =
            int_node_ptr->split;
          out << " "
              << predicate.first
              << (satisfies_p ? " = " : " != ")
              << predicate.second << std::flush;
          satisfies_p = int_node_ptr->is_true_child();
          int_node_ptr = int_node_ptr->parent;
        }
        out << std::endl;
        // Write out the factor in this leaf.
        out << "Factor: " << *(leaf.leaf_data) << std::endl;
      }
    }

  }; // end of class: bspt_factor_t

  //! Writes a human-readable representation of the BSPT factor.
  template <typename leaf_factor_t>
  std::ostream& operator<<(std::ostream& out,
                           const bspt_factor_t<leaf_factor_t>& bspt_factor) {
    bspt_factor.write(out);
    return out;
  }

} // namespace prl

#ifdef PRL_CONSTANT_FACTOR_HPP
#include <prl/constant_and_bspt_factor.hpp>
#endif // #ifdef PRL_CONSTANT_FACTOR_HPP

#ifdef PRL_TABLE_FACTOR_HPP
#include <prl/table_and_bspt_factor.hpp>
#endif // #ifdef PRL_TABLE_FACTOR_HPP

#endif // PRL_BSPT_FACTOR_HPP
