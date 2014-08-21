#ifndef SILL_DATASTRUCTURE_CONCEPTS_HPP
#define SILL_DATASTRUCTURE_CONCEPTS_HPP

#include <boost/optional.hpp>

#include <sill/global.hpp>
#include <sill/functional.hpp> // for identity_t
#include <sill/stl_concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup datastructure
  //! @{

  /**
   * Concept that represents a table.
   *
   * @see dense_table, sparse_table
   */
  template <typename T>
  struct Table : Mutable_ForwardContainer<T>
  {
    //! The type used to represent the shape and indices.
    typedef typename T::index_type index_type;

    //! Mutable iterator range over the elements of this table
    typedef typename T::mutable_range mutable_range;

    //! Iterator range over the elements of this table
    typedef typename T::const_range const_range;

    //! Iterator range over the indices into the table
    typedef typename T::index_range index_range;

    typedef typename Table::value_type value_type;

    //! Returns the number of dimensions of this table.
    size_t arity() const;

    //! Returns the number of dimensions of this table (Boost convention).
    size_t num_dimensions();

    //! Returns the dimensions of this table.
    const index_type& shape() const;

    //! Returns the number of cells in this table (Boost convention).
    size_t num_elements() const;

    //! Returns the size of the given dimension of this table.
    size_t size(const size_t dim) const;

    //! Const element access to this table.
    template <typename TableIndex>
    const value_type& operator()(const TableIndex& i) const;

    //! Mutable element access to this table.
    template <typename TableIndex>
    T& operator()(const TableIndex& i);

    //! Invokes the supplied functor on the reference to elements of the table.
    template <typename Function>
    void apply(Function f);

    /**
     * Joins two tables and stores the result into this table.
     *
     * Each dimension of an input must correspond to exactly one
     * dimension of the result. Each dimension of the output may
     * correspond to a dimension of x, y, both, or neither.
     *
     * @param x an input table
     * @param y an input table
     * @param x_dim_map
     *        an object such that x_dim_map[i] gives the dimension of
     *        z that corresponds to dimension i of x; these must be unique
     * @param y_dim_map
     *        an object such that y_dim_map[i] gives the dimension of
     *        z that corresponds to dimension i of y; these must be unique
     * @param JoinOp
     *        a type that models the BinaryFunction concept
     */
    template <typename JoinOp>
    void join(const Table& x, const Table& y,
              const index_type& x_dim_map, const index_type& y_dim_map,
              JoinOp op);

    /**
     * Joins one table into this table.  Each element of *this is
     * updated to become the result of applying the supplied binary
     * operator to its previous value (as the first argument) and the
     * corresponding value of y (as the second argument).
     *
     * @param y
     *        an input table; each dimension of y must correspond
     *        to a dimension of x
     * @param y_dim_map
     *        an object such that y_dim_map[i] gives the dimension of
     *        x that corresponds to dimension i of y; these must be unique
     * @param JoinOp
     *        a type that models the BinaryFunction concept
     */
    template <typename JoinOp>
    void join_with(const Table& y, const index_type& y_dim_map, JoinOp op);

    /**
     * Aggregates a table and stores the result into this table. Each
     * element of the output must correspond to a unique dimension of
     * the input.  Each element of x is aggregated into the element of
     * this table with the same indexes in corresponding dimensions.
     * The aggregation is computed using a binary function which
     * incrementally incorporates an element of x into the aggregate
     * element of z as \f$z' \leftarrow f(z, x)\f$; the aggregate is
     * initialized with the current elements of z.
     *
     * @param x
     *        an input table
     * @param dim_map an object such that dimension i in this table
     *        corresponds to dimension dim_map[i] in x; the values of
     *        dim_map must be unique
     * @param AggOpp
     *        a type that models a symmetric binary operator
     */
    template <typename AggOp>
    void aggregate(const Table& x, const index_type& dim_map, AggOp op);


    /**
     * Joins two tables and aggregates the elements of the resulting
     * table into a single element.
     *
     * \todo: This function could be generalized to aggregate the
     * result into a table, not just a scalar.  That would permit
     * fully general combine/collapse operations for table factors,
     * such as the matrix/vector multiply used in the computation of
     * HMM forward messages.
     *
     * @param x
     *        an input table
     * @param y
     *        an input table
     * @param x_dim_map
     *        an object such that x_dim_map[i] gives the dimension of
     *        (the fictitious join table) z that corresponds to
     *        dimension i of x; these must be unique
     * @param y_dim_map
     *        an object such that y_dim_map[i] gives the dimension of
     *        (the fictitious join table) z that corresponds to
     *        dimension i of y; these must be unique
     * @param join_op
     *        an object that models the binary operator concept;
     *        it is used to compute elements of the joined table
     * @param agg_op
     *        an object that models the binary operator concept; it
     *        is used to aggregate the elements of the joined table
     */
    template <typename JoinOp, typename AggOp>
    static T join_aggregate(const Table& x,
                            const Table& y,
                            const index_type& x_dim_map,
                            const index_type& y_dim_map,
                            JoinOp join_op,
                            AggOp agg_op);

    /**
     * Joins two tables and returns a pair of values that satisfy the
     * given predicate. If no pair of values satisfies the predicate,
     * returns boost::none.
     * @see join_aggregate
     */
    template <typename Predicate>
    static boost::optional< std::pair<T,T> >
    join_find(const Table& x, const Table& y,
              const index_type& x_dim_map, const index_type& y_dim_map,
              Predicate p);

    /**
     * Restricts a table and stores the result in this table. Each
     * dimension of this table must correspond to exactly one
     * dimension of x which will not be restricted. The argument
     * restrict_map specifies which dimensions are restricted and
     * which elements in those dimensions will be put in the new
     * table.
     *
     * @param x
     *        an input table
     * @param restrict_map
     *        an object such that restrict_map[i]>=x.size(i) if
     *        dimension i of x is not to be restricted and
     *        restrict_map[i] = j if x is to be restricted to value j
     *        in dimension i.
     * @param dim_map
     *        an object such that dim_map[i] gives the dimension of
     *        x that corresponds to dimension i of this table; the
     *        values of dim_map must be unique
     */
    void restrict(const Table& x,
                   const index_type& restrict_map,
                   const index_type& dim_map);

    // Concept checking
    concept_usage(Table) {
      sill::same_type(t.arity(), a);
      sill::same_type(t.num_dimensions(), a);
      sill::same_type(t.num_elements(), si);
      sill::same_type(t.size(), si);
      sill::same_type(t.size(size_t()), si);
      sill::same_type(t.shape(), sh);
      // sill::same_type(t.elements(), elts); // unsupported by sparse_table
      //sill::same_type(t.indices(), idxs);   // for now
      sill::same_type(t(i), v);
      t.apply(sill::identity_t<value_type>());
      // t.join(t, t, sh, sh, op);
      t.join_with(t, sh, op);
      t.aggregate(t, sh, op);
      t.restrict(t, sh, sh);
      // sill::same_type(T::join_aggregate(t, t, sh, sh, op, op), v);
    }

  private:
    T t;
    size_t a;
    size_t si;
    index_type sh;
    index_type i;
    mutable_range elts;
    index_range idxs;
    value_type v;
    std::plus<value_type> op;

  }; // concept Table

  //! @}

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
