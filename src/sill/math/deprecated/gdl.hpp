
#ifndef SILL_GDL_HPP
#define SILL_GDL_HPP
#warning "Deprecated"
#include <sill/global.hpp>
#include <sill/functional.hpp>
#include <sill/range/concepts.hpp>
#include <sill/stl_concepts.hpp>

#include <sill/range/numeric.hpp>

#include <sill/macros_def.hpp>

// This file is deprecated

namespace sill {

  //! The base of all binary operator tags.
  struct binary_op_tag { };

  //! A tag representing the addition operator.
  //! @see BinaryOpTag
  struct sum_tag : public binary_op_tag { };

  //! A tag representing the subtraction operator.
  //! @see BinaryOpTag
  struct minus_tag : public binary_op_tag { };

  //! A tag representing the multiplication operator.
  //! @see BinaryOpTag
  struct product_tag : public binary_op_tag { };

  //! A tag representing the division operator \f$x / y\f$.
  //! @see BinaryOpTag
  struct divides_tag : public binary_op_tag { };

  //! A tag representing the maximization operator.
  //! @see BinaryOpTag
  struct max_tag : public binary_op_tag { };

  //! A tag representing the minimization operator.
  //! @see BinaryOpTag
  struct min_tag : public binary_op_tag { };

  //! A tag representing the conjunction operator.
  //! @see BinaryOpTag
  struct and_tag : public binary_op_tag { };

  //! A tag representing the disjunction operator.
  //! @see BinaryOpTag
  struct or_tag : public binary_op_tag { };

  /**
   * The template that represents a binary operator.
   */
  template <typename T, typename OpTag>
  struct binary_op {
    BOOST_STATIC_ASSERT(sizeof(T)==0);
    // If your compilation fails here, it means that you did not 
    // specialize the op template to the appropriate tag.
  };

  // The specializations below use the PRL's extended functors that
  // contain information about the operator's left&right zeros and identities
  template <typename T>
  struct binary_op<T, sum_tag> : public plus<T> { };

  template <typename T>
  struct binary_op<T, minus_tag> : public minus<T> { };

  template <typename T>
  struct binary_op<T, product_tag> : public multiplies<T> { };

  template <typename T>
  struct binary_op<T, divides_tag> : public divides<T> { };

  template <typename T>
  struct binary_op<T, max_tag> : public maximum<T> { };

  template <typename T>
  struct binary_op<T, min_tag> : public minimum<T> { };

  template <typename T>
  struct binary_op<T, and_tag> : public logical_and<T> { };

  template <typename T>
  struct binary_op<T, or_tag> : public logical_or<T> { };


  //! The base of all commutative semiring tags.
  //! @see op
  template <typename CrossTag, typename DotTag>
  struct csr_tag {
    typedef CrossTag cross_tag;
    typedef DotTag dot_tag;
    static cross_tag cross() { return cross_tag(); }
    static dot_tag dot() { return dot_tag(); }
  };

  /**
   * A tag representing the sum product commutative semiring \f$([0,
   * \infty), +, \times, 0, 1)\f$.
   * @see CsrTag
   */
  struct sum_product_tag : public csr_tag<sum_tag, product_tag> {};

  /**
   * A tag representing the max product commutative semiring \f$([0,
   * \infty), \max, \times, 0, 1)\f$.
   * @see CsrTag
   */
  struct max_product_tag : public csr_tag<max_tag, product_tag> {};

  /**
   * A tag representing the min-sum commutative semiring \f$((-\infty,
   * \infty], \min, +, \infty, 0)\f$.
   * @see CsrTag
   */
  struct min_sum_tag : public csr_tag<min_tag, sum_tag> {};

  /**
   * A tag representing the max-sum commutative semiring \f$([-\infty,
   * \infty), \max, +, -\infty, 0)\f$.
   * @see CsrTag
   */
  struct max_sum_tag : public csr_tag<max_tag, sum_tag> {};

  /**
   * A tag representing the Boolean commutative semiring \f$(\{0, 1\},
   * \lor, \land, 0, 1)\f$.
   * @see CsrTag
   */
  struct boolean_tag : public csr_tag<or_tag, and_tag> {};

  //! Applies the cross operator to a collection of elements.
  template <typename Range, typename CsrTag>
  typename Range::value_type
  cross_all(const Range& range, CsrTag csr_tag) {
    concept_assert((InputRange<Range>));
    binary_op<typename Range::value_type, typename CsrTag::cross_tag> cross_op;
    return sill::accumulate(range, cross_op.left_identity(), cross_op);
  }

  //! Applies the dot operator to a collection of elements.
  template <typename Range, typename CsrTag>
  typename Range::value_type
  dot_all(const Range& range, CsrTag csr_tag) {
    concept_assert((InputRange<Range>));
    binary_op<typename Range::value_type, typename CsrTag::dot_tag> dot_op;
    return sill::accumulate(range, dot_op.left_identity(), dot_op);
  }

#ifndef SWIG
  /**
   * A concept that represents a binary operator tag.
   *
   * @see sum_tag, product_tag, divides_tag, 
   * min_tag, max_tag, and_tag, or_tag, kld_tag, reverse_args
   *
   * @see op
   */
  template <typename Tag>
  struct BinaryOpTag {
    concept_assert((Convertible<Tag, binary_op_tag>));
  };

  /**
   * A concept that represents a commutative semiring tag.
   *
   * A commutative semiring consists of set of elements \f$K\f$, two
   * distinguished elements (called \f$0\f$ and \f$1\f$), and two
   * binary operators (called \f$+\f$ and \f$\cdot\f$).  The nested
   * types #cross_tag and #dot_tag mark the two binary operators
   * \f$+\f$, \f$\cdot\f$ associated with this semiring; use #op to
   * extract information about the operator and invoke the operator
   * for a given number representation.
   *
   * Note that we use an indirect representation of the semiring,
   * specifying first the tags and then the template op,
   * parameterized by the tag.  This is because some factors (e.g.,
   * Guassian), have an implicit representation of the semiring and
   * only need to know which abstract operation to involve.
   *
   * @see Srinivas M. Aji and Robert J. McEliece, "The Generalized
   * Distributive Law", IEEE Transactions on Information Theory,
   * vol. 46, no. 2, March 2000.
   *
   * @see sum_product_tag, max_product_tag, min_sum_tag, boolean_tag
   */
  template <typename Tag>
  struct CsrTag {
    typedef typename Tag::cross_tag cross_tag;
    typedef typename Tag::dot_tag dot_tag;
    concept_assert((BinaryOpTag<cross_tag>));
    concept_assert((BinaryOpTag<dot_tag>));
  };
#endif

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_GDL_HPP
