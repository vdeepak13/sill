#ifndef SILL_DENSE_TABLE_HPP
#define SILL_DENSE_TABLE_HPP

#include <sill/global.hpp>
#include <sill/functional.hpp>
#include <sill/range/algorithm.hpp>
#include <sill/range/numeric.hpp>
#include <sill/stl_concepts.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/serialization/vector.hpp>

#include <algorithm>
#include <iterator>
#include <iostream>
#include <numeric>

#include <boost/optional.hpp>

#include <sill/macros_def.hpp>

#ifndef EXPERIMENTAL
#define EXPERIMENTAL
#endif

namespace sill {

  /**
   * A dense table with an arbitrary number of dimensions, each with a
   * finite number of values.
   *
   * @see Table
   *
   * \ingroup datastructure
   */
  template <typename T>
  class dense_table {

    // Public type declarations
    //==========================================================================
  public:
    // Typedefs required to satisfy the Container concept
    typedef typename std::vector<T>::value_type value_type;
    typedef typename std::vector<T>::difference_type difference_type;
    typedef typename std::vector<T>::size_type size_type;
    typedef typename std::vector<T>::reference reference;
    typedef typename std::vector<T>::pointer pointer;
    typedef typename std::vector<T>::iterator iterator;
    typedef typename std::vector<T>::const_reference const_reference;
    typedef typename std::vector<T>::const_pointer const_pointer;
    typedef typename std::vector<T>::const_iterator const_iterator;

    //! The type used to represent the shape and indices.
    typedef std::vector<size_t> index_type;

    // Forward declaration
    class index_iterator;
    
    class offset_functor;
#ifdef EXPERIMENTAL
    class offset_iterator;
#endif

    // Private data members
    //==========================================================================
  private:
    //! The dimensions of this table
    index_type shape_;

    //! Pre-computed number of elements
    size_t size_;

    //! The elements of this table, stored in a linear sequence.
    std::vector<T> elts;

    offset_iterator off_it1;

    offset_iterator off_it2;

  public: // temporary hack by Anton
    //! The offset calculator which maps indices for this table's
    //! geometry into offsets for the #elts sequence.
    offset_functor offset;

  public:
    void save(oarchive & ar) const {
      ar << shape_;
      ar << elts;
    }

    void load(iarchive & ar) {
      ar >> shape_;
      size_ = sill::accumulate(shape_, 1, std::multiplies<size_t>());
      ar >> elts;
      offset = offset_functor(shape_);
    }

    // Constructors
    //==========================================================================
  public:
    //! Constructs a table with the given dimensions and default element
    dense_table(const index_type& extents, T init_elt = T())
      : shape_(extents), 
        size_(sill::accumulate(shape_, 1, std::multiplies<size_t>())),
        elts(size_, init_elt),
        offset(extents) {
      // Check to make sure the size value did not overflow.
      double logsize(0.);
      foreach(size_t s, extents)
        logsize += std::log(s);
      if (logsize > std::log(std::numeric_limits<size_t>::max())) {
        throw std::overflow_error("Argument \"extents\" defined a table size larger than size_t max value.");
      }
    }

    //! Constructs a table with a single element
    explicit dense_table(T init_elt = T()) 
      : size_(1),
        elts(1, init_elt),
        offset(shape_) { }

    // Simple public functions
    //==========================================================================
    //! Swaps the contents of these two tables
    void swap(dense_table& other) {
      shape_.swap(other.shape_);
      std::swap(size_, other.size_);
      std::swap(offset, other.offset);
      elts.swap(other.elts);
    }

    bool operator==(const dense_table& other) const {
      return shape_ == other.shape_ && elts == other.elts;
    }

    bool operator!=(const dense_table& other) const {
      return !(*this == other);
    }

    //! Returns the number of dimensions of this table.
    size_t arity() const {
      return shape_.size();
    }

    //! Returns the dimensions of this table.
    const index_type& shape() const {
      return shape_;
    }

    //! Total number of elements in the table (including the default ones)
    size_t size() const {
      return size_;
    }

    //! Returns the size of the given dimension of this table
    //! @param dim index from 0 to k-1
    size_t size(const size_t dim) const {
      assert(dim < arity());
      return shape_[dim];
    }

    //! The maximum number of elements in the table (requirement of Container)
    size_t max_size() const {
      return size_;
    }

    //! Returns true if one of the dimensions of the table is 0.
    bool empty() const {
      return size_ == 0;
    }

    //! Returns the iterator pointing to the first element.
    iterator begin() {
      return elts.begin();
    }
    
    //! Returns the iterator pointing to the first element.
    const_iterator begin() const {
      return elts.begin();
    }

    //! Returns the iterator pointing to the one past the last element.
    iterator end() {
      return elts.end();
    }

    //! Returns the iterator pointing to the one past the last element.
    const_iterator end() const {
      return elts.end();
    }

    /**
     * Returns a const iterator range over the elements of this table.
     *
     * @param permit_skipping
     *        if this flag is set to true, then the iterator range
     *        is permitted (but not required) to skip all instances of
     *        a designated skip element     * @param skip_elt
     *        if skipping is enabled, then the returned iterator range
     *        is permitted (but not required) to skip all instances of
     *        this element
     */
    std::pair<const_iterator, const_iterator>
    elements() const {
      return std::make_pair(elts.begin(), elts.end());
    }

    /**
     * Returns a mutable iterator range over the elements of this table.
     *
     * @param permit_skipping
     *        if this flag is set to true, then the iterator range
     *        is permitted (but not required) to skip all instances of
     *        a designated skip element
     * @param skip_elt
     *        if skipping is enabled, then the returned iterator range
     *        is permitted (but not required) to skip all instances of
     *        this element
     */
    std::pair<iterator, iterator>
    elements() {
      return std::make_pair(elts.begin(), elts.end());
    }

    //! Returns the index associated with an iterator position
    index_type index(const_iterator it) const {
      assert(it >= begin() && it < end());
      return offset.index(it - begin());
    }

    /**
     * Returns an iterator range over indices into this table.
     * The ordering used is determined by the geometry of the table: the first
     * element in the geometry is the least significant bit of the index.
     */
    std::pair<index_iterator, index_iterator>
    indices() const {
      return std::make_pair(index_iterator(&shape_), index_iterator());
    }

    /**
     * Returns an iterator range over a subspace of indices into this table.
     * The ordering used is determined by the geometry of the table: the first
     * element in the geometry is the least significant bit of the index.
     */
    std::pair<index_iterator, index_iterator>
    indices(const index_type& restrict_map) const {
      return std::make_pair(index_iterator(&shape_, &restrict_map),
                            index_iterator());
    }

    //! implements Table::operator()
    const T& operator()(const index_type& i) const {
      return elts[offset(i)];
    }

    //! implements Table::operator()
    T& operator()(const index_type& i) {
      return elts[offset(i)];
    }

    //! operator with direct indexing
    T& operator()(size_t i) {
      return elts[i];
    }

    //! operator with direct indexing
    const T& operator()(size_t i) const {
      return elts[i];
    }

    //! implements Table::apply
    template <typename Function>
    void apply(Function f) {
      sill::for_each(elts, f);
    }

    //! implements Table::update
    template <typename Function>
    void update(Function f) {
      concept_assert((UnaryFunction<Function, T, T>));
      foreach(T& x, elts) x = f(x);
    }

    template <typename Function>
    void transform(const dense_table& x, Function fn) {
      assert(shape() == x.shape());
      for (size_t i = 0; i < elts.size(); ++i) {
        elts[i] = fn(x.elts[i]);
      }
    }
    
    // Table joins and aggregations
    //==========================================================================
    //! implements Table::join
    template <typename U, typename JoinOp>
    void join(const dense_table& x, const dense_table<U>& y,
              const index_type& x_dim_map, const index_type& y_dim_map,
              JoinOp op) {
      concept_assert((BinaryFunction<JoinOp,T,U,T>));
#ifdef EXPERIMENTAL
            // Get offset calculators for the two input tables.
//      offset_iterator x_offset(x.shape(), this->shape(), x_dim_map);
//      offset_iterator y_offset(y.shape(), this->shape(), y_dim_map);

      offset_iterator& x_offset = off_it1;
      offset_iterator& y_offset = off_it2;
      x_offset.reset(x.shape(), this->shape(), x_dim_map);
      y_offset.reset(y.shape(), this->shape(), y_dim_map);

      // Iterate over the cells of this table, computing the value
      // using the corresponding cells of the input tables.
      foreach(const index_type& index, indices()) {
        (*this)(index) = op(x.elts[x_offset()],
                            y.elts[y_offset()]);
        ++x_offset;
        ++y_offset;
      }
#else
      // Get offset calculators for the two input tables.
      offset_functor x_offset(x.shape(), this->arity(), x_dim_map);
      offset_functor y_offset(y.shape(), this->arity(), y_dim_map);

      // Iterate over the cells of this table, computing the value
      // using the corresponding cells of the input tables.
      foreach(const index_type& index, indices())
        (*this)(index) = op(x.elts[x_offset(index)],
                            y.elts[y_offset(index)]);
#endif
    }

    //! implements Table::join_with
    template <typename U, typename JoinOp>
    void join_with(const dense_table<U>& y, const index_type& y_dim_map,
                   JoinOp op) {
      concept_assert((BinaryFunction<JoinOp,T,U,T>));

#ifdef EXPERIMENTAL
      // Get an offset calculator for y.
      offset_iterator& y_offset = off_it1;
      offset_iterator& this_offset = off_it2;
      y_offset.reset(y.shape(), this->shape(), y_dim_map);
      this_offset.reset(this->shape());
      // Iterate over the cells of this table, computing the value
      // using the corresponding cell of y.
      do {
        elts[this_offset()] = op(elts[this_offset()], y.elts[y_offset()]);
        ++y_offset;
        ++this_offset;
      } while(!y_offset.end() && !this_offset.end());
#else

      // Get an offset calculator for y.
      offset_functor y_offset(y.shape(), this->arity(), y_dim_map);
      // Iterate over the cells of this table, computing the value
      // using the corresponding cell of y.

      foreach(const index_type& index, indices()) {
        (*this)(index) = op((*this)(index), y.elts[y_offset(index)]);
      }
#endif
    }

    //! implements Table::join_with
    template <typename U, typename JoinOp>
    void join_with(const dense_table<U>& y, JoinOp op) {
      concept_assert((BinaryFunction<JoinOp,T,T,T>));
      assert(shape_ == y.shape_);
      iterator it = begin();
      const_iterator y_it = y.begin(), y_end = y.end();
      for (; y_it != y_end; ++it, ++y_it) {
        *it = op(*it, *y_it);
      }
    }


    //! implements Table::aggregate
    //! \todo Do we require that the table is initialized to op.left_identity()?
    template <typename U, typename AggOp>
    void aggregate(const dense_table<U>& x, const index_type& dim_map, AggOp op) {
      concept_assert((BinaryFunction<AggOp,T,U,T>));
#ifdef EXPERIMENTAL
      // Get an offset calculator that maps x indexes to z offsets.
//      offset_iterator z_offset(this->shape(), x.shape(), dim_map);
      offset_iterator& z_offset = off_it1;
      z_offset.reset(this->shape(), x.shape(), dim_map);
      // Iterate over the cells of the input table, computing the
      // aggregate.
      foreach(const index_type& x_index, x.indices()) {
        size_t offset = z_offset();
        elts[offset] = op(elts[offset], x(x_index));
        ++z_offset;
      }

#else
      // Get an offset calculator that maps x indexes to z offsets.
      offset_functor z_offset(this->shape(), x.arity(), dim_map);
      // Iterate over the cells of the input table, computing the
      // aggregate.
      foreach(const index_type& x_index, x.indices()) {
        size_t offset = z_offset(x_index);
        elts[offset] = op(elts[offset], x(x_index));
      }
#endif
    }

    //! Aggregates all dimensions of the table and returns the result
    template <typename AggOp, typename U>
    U aggregate(AggOp op, U initialvalue) const {
      concept_assert((BinaryFunction<AggOp,U,T,U>));
      U result = initialvalue;
      foreach(T value, elts){
        result = op(result, value);
      }
      return result;
    }

    //! Aggregates all dimensions of the table and returns the result
    template <typename AggOp, typename U>
    U foldr(AggOp op, U leftval) const {
      concept_assert((BinaryFunction<AggOp,T,U,T>));
      U result = leftval;
      foreach(T value, elts) result = op(result, value);
      return result;
    }

    //! implements Table::join_aggregate
    template <typename JoinOp, typename AggOp>
    static T
    join_aggregate(const dense_table& x,
                   const dense_table& y,
                   const index_type& x_dim_map,
                   const index_type& y_dim_map,
                   JoinOp join_op, AggOp agg_op, T initialvalue) {
      concept_assert((BinaryFunction<JoinOp,T,T,T>));
      concept_assert((BinaryFunction<AggOp,T,T,T>));

      // Initialize the aggregate with the identity of the aggregation op.
      T aggregate = initialvalue;

      // Compute the shape of the joined table.
      size_t z_arity =
        1 + std::max(sill::accumulate(x_dim_map, 0, maximum<size_t>()),
                     sill::accumulate(y_dim_map, 0, maximum<size_t>()));
      index_type z_shape(z_arity);

      // could simplify the following as:
      // boost::copy(x.shape(), boost::subrange(x, x_dim_map));
      for (size_t d = 0; d < x.arity(); ++d)
        z_shape[x_dim_map[d]] = x.shape()[d];
      for (size_t d = 0; d < y.arity(); ++d)
        z_shape[y_dim_map[d]] = y.shape()[d];

      // Get offset calculators for the two input tables.
      offset_functor x_offset(x.shape(), z_arity, x_dim_map);
      offset_functor y_offset(y.shape(), z_arity, y_dim_map);

      // Iterate over the cells of the result table, computing the value
      // using the corresponding cells of the input tables.
      index_iterator it(&z_shape), end;
      for (; it != end; ++it)
        aggregate = agg_op(aggregate, join_op(x.elts[x_offset(*it)],
                                              y.elts[y_offset(*it)]));
      return aggregate;
    }

    //! implements Table::join_find
    template <typename Pred>
    static boost::optional< std::pair<T,T> >
    join_find(const dense_table& x,
              const dense_table& y,
              const index_type& x_dim_map,
              const index_type& y_dim_map,
              Pred p) {
      concept_assert((BinaryPredicate<Pred, T, T>));

      // Compute the shape of the joined table.
      size_t z_arity =
        1 + std::max(sill::accumulate(x_dim_map, 0, maximum<size_t>()),
                     sill::accumulate(x_dim_map, 0, maximum<size_t>()));
      index_type z_shape(z_arity);

      // could simplify the following as:
      // boost::copy(x.shape(), boost::subrange(x, x_dim_map));
      for (size_t d = 0; d < x.arity(); ++d)
        z_shape[x_dim_map[d]] = x.shape()[d];
      for (size_t d = 0; d < y.arity(); ++d)
        z_shape[y_dim_map[d]] = y.shape()[d];

      // Get offset calculators for the two input tables.
      offset_functor x_offset(x.shape(), z_arity, x_dim_map);
      offset_functor y_offset(y.shape(), z_arity, y_dim_map);

      index_iterator it(&z_shape), end;
      for (; it != end; ++it) {
        T xi = x.elts[x_offset(*it)];
        T yi = y.elts[y_offset(*it)];
        if (p(xi,yi)) return std::make_pair(xi,yi);
      }

      return boost::none;
    }

    //! implements Table::restrict
    void restrict(const dense_table& x,
                  const index_type& restrict_map,
                  const index_type& dim_map) {
      // Get an offset calculator that maps x indexes to offsets of this table
      offset_functor z_offset(this->shape(), x.arity(), dim_map);
      // Iterate over a subspace of the input table, copying to this table.
      foreach(const index_type& x_index, x.indices(restrict_map)) {
        size_t offset = z_offset(x_index);
        elts[offset] = x(x_index);
      }
    }

    template <typename U, typename Op>
    void restrict(const dense_table<U>& x,
                  const index_type& restrict_map,
                  const index_type& dim_map,
                  Op op) {
      offset_functor z_offset(this->shape(), x.arity(), dim_map);
      foreach (const index_type& x_index, x.indices(restrict_map)) {
        size_t offset = z_offset(x_index);
        elts[offset] = op(x(x_index));
      }
    }

    /**
     * More efficient version of restrict which expects this table to be
     * aligned with x as follows:
     *  - If x has dimensions [d1, d2, ..., dk], with d1 being the least
     *    significant digit,
     *  - Then this table must have dimensions [d1, d2, ..., dl] with l <= k.
     *    The value l is determined by this table's current dimensions.
     *
     * @param x
     *        Input table.
     * @param restrict_map  
     *        An object such that restrict_map[i] >= x.size(i) if
     *        dimension i of x is not to be restricted and
     *        restrict_map[i] = j if x is to be restricted to value j
     *        in dimension i.
     *        For this version of restrict, the first l elements of restrict_map
     *        are ignored.
     */ 
    void restrict_aligned(const dense_table& x,
                          const index_type& restrict_map) {
      size_t l = this->arity();
      if (x.arity() < l)
        throw std::invalid_argument
          (std::string("dense_table::restrict_aligned(x, restrict_map)") +
           " was given x with fewer dimensions than this table.");
      if (x.arity() != restrict_map.size())
        throw std::invalid_argument
          (std::string("dense_table::restrict_aligned(x, restrict_map)") +
           " was given x, restrict_map with non-matching dimensions.");
      for (size_t i = 0; i < l; ++i) {
        if (shape_[i] != x.size(i))
          throw std::invalid_argument
            (std::string("dense_table::restrict_aligned(x, restrict_map)") +
             " was called on a table with dimensions not matching x.");
      }

      // Calculate offset for remaining elements.
      size_t off = 0;
      for (size_t i = l; i < x.arity(); ++i) {
        if (restrict_map[i] >= x.size(i))
          throw std::invalid_argument
            (std::string("dense_table::restrict_aligned(x, restrict_map)") +
             " was given restrict_map which did not restrict all required" +
             " dimensions.");
        off += x.offset.get_multiplier(i) * restrict_map[i];
      }

      // Copy elements from x at offset to this table.
      for (size_t i = 0; i < this->size(); ++i)
        elts[i] = x.elts[off + i];

    } // restrict_aligned

    /**
     * More efficient version of restrict which restricts all but one
     * dimension.
     * @param retain_dim  Dimension in x to be retained.
     */
    void restrict_other(const dense_table& x,
                        const index_type& restrict_map,
                        size_t retain_dim) {
      /* We want to copy x[restrict_map, except for retain_dim] to this[],
         along retain_dim.
         So we need to:
          - Check to make sure this has the correct size.
          - Calculate the offset for the first element to be copied from x.
          - Calculate the multiplier for retain_dim in x.
       */
      assert(size() == x.size(retain_dim));
      assert(x.arity() == restrict_map.size());
      size_t x_offset = 0;
      size_t x_retain_dim_multiplier = 0;
      for (size_t d = 0; d < x.arity(); ++d) {
        if (d != retain_dim) {
          assert(restrict_map[d] < x.size(d));
          x_offset += x.offset.get_multiplier(d) * restrict_map[d];
        } else {
          x_retain_dim_multiplier = x.offset.get_multiplier(d);
        }
      }
      for (size_t i = 0; i < size(); ++i) {
        elts[i] = x.elts[x_offset];
        x_offset += x_retain_dim_multiplier;
      }
    } // restrict_other(x, restrict_map, retain_dim)

    /**
     * More efficient version of restrict which restricts all but one
     * dimension.
     * @param retain_dim  Dimension in x to be retained.
     */
    template <typename RestrictMapFunctor>
    void restrict_other(const dense_table& x,
                        const RestrictMapFunctor& restrict_map,
                        size_t retain_dim) {
      /* We want to copy x[restrict_map, except for retain_dim] to this[],
         along retain_dim.
         So we need to:
          - Check to make sure this has the correct size.
          - Calculate the offset for the first element to be copied from x.
          - Calculate the multiplier for retain_dim in x.
       */
      assert(size() == x.size(retain_dim));
      assert(x.arity() == restrict_map.size());
      size_t x_offset = 0;
      {
        size_t d = 0;
        while (d < retain_dim) {
          assert(restrict_map[d] < x.size(d));
          x_offset += x.offset.get_multiplier(d) * restrict_map[d];
          ++d;
        }
        ++d;
        while (d < x.arity()) {
          assert(restrict_map[d] < x.size(d));
          x_offset += x.offset.get_multiplier(d) * restrict_map[d];
          ++d;
        }
      }
      size_t x_retain_dim_multiplier = x.offset.get_multiplier(retain_dim);
      for (size_t i = 0; i < size(); ++i) {
        elts[i] = x.elts[x_offset];
        x_offset += x_retain_dim_multiplier;
      }
    } // restrict_other(x, restrict_map, retain_dim)

    // Iterators
    //==========================================================================
  public:
    /**
     * An iterator over the indices to a table.
     * The ordering used is determined by the geometry of the table: the first
     * element in the geometry is the least significant bit of the index.
     */
    class index_iterator :
      public std::iterator<std::forward_iterator_tag, const index_type> {

      //! The geometry of the table.
      const index_type* geometry;

      //! The current index into the table.
      index_type index;

      //! A flag indicating whether the index has wrapped around.
      bool done;

      //! Restrictions to a certain subspace of the table.
      const index_type* restrict_map;

    public:
      //! End iterator constructor.
      index_iterator()
        : geometry(NULL), index(), done(true), restrict_map(NULL) { }

      //! Begin iterator constructor with no restrictions.
      index_iterator(const index_type* geometry)
        : geometry(geometry),
          index(geometry->size(), 0),
          done(false),
          restrict_map(NULL) {
        // If table is of size 0, then mark iterator as done.
        for (size_t i = 0; i < geometry->size(); i++)
          if ((*geometry)[i] == 0) {
            done = true;
            return;
          }
      }

      //! Begin iterator constructor with restrictions.
      index_iterator(const index_type* geometry,
                     const index_type* restrict_map)
        : geometry(geometry),
          index(geometry->size(), 0),
          done(false), 
          restrict_map(restrict_map) {
        // If table is of size 0, then mark iterator as done.
        for (size_t i = 0; i < geometry->size(); i++)
          if ((*geometry)[i] == 0) {
            done = true;
            return;
          }
        // initialize the dimensions that have been restricted
        // transform(map, geometry, index.begin(), if_else(arg1<arg2, arg1, 0));
        for(size_t i=0; i < restrict_map->size(); i++)
          if ((*restrict_map)[i] < (*geometry)[i])
            index[i] = (*restrict_map)[i];
      }

      /**
       * Advances a table index to the next cell of the table. The ordering
       * used is determined by the geometry of the table: the first element
       * in the geometry is the least significant bit of the index.
       * If the supplied index pointed to the last cell, this function returns
       * true and the index is reset to point to the first cell.
       */
      bool increment(index_type& index, const index_type& geometry) {
        for(size_t i = 0; i<index.size(); i++)
          if (index[i] == geometry[i] - 1)
            index[i] = 0;
          else {
            ++index[i];
            return false;
          }
        return true;
      }

      /**
       * Advances a table index to the next cell of the table. The ordering
       * used is determined by the geometry of the table: the first element
       * in the geometry is the least significant bit of the index.
       * If the supplied index pointed to the last cell, this function returns
       * true and the index is reset to point to the first cell.
       * This version of the increment function keeps the increments restricted
       * to a subspace of the table specified by restrict_map.
       *
       * @param index
       *        current table index
       * @param geometry
       *        geometry of the table with which index is associated
       * @param restrict_map
       *        restriction map of same length as index;
       *        restrict_map[i] = v in [0, geometry[i] - 1] indicates
       *        dimension i is restricted to value v, and larger values
       *        indicate no restriction
       *
       */
      static bool increment(index_type& index,
                            const index_type& geometry,
                            const index_type& restrict_map) {
        assert(index.size() == geometry.size());
        assert(index.size() == restrict_map.size());

        for (size_t i = 0; i < index.size(); i++) {
          // Check to see if we're not skipping this dimension
          if (restrict_map[i] >= geometry[i]) {
            // If we've reached the end of this dimension, reset to zero
            // and continue to the next dimension.  Otherwise, increment
            // the index in this dimension and quit.
            if (index[i] == geometry[i] - 1)
              index[i] = 0;
            else {
              ++index[i];
              return false;
            }
          }
        }
        return true;
      }

      //! Prefix increment.
      index_iterator& operator++() {
        if (restrict_map == NULL)
          done = increment(index, *geometry);
        else
          done = increment(index, *geometry, *restrict_map);
        return *this;
      }

      //! Postfix increment.
      index_iterator operator++(int) {
        index_iterator tmp = *this;
        ++(*this);
        return tmp;
      }

      //! Returns a const reference to the current index.
      const index_type& operator*() {
        return index;
      }

      //! Returns a const pointer to the current index.
      const index_type* operator->() {
        return &index;
      }

      //! Returns truth if the two table indexes are the same.
      bool operator==(const index_iterator& it) const {
        if (done)
          return it.done;
        else
          return !it.done && (index == it.index);
      }

      //! Returns truth if the two table indexes are different.
      bool operator!=(const index_iterator& it) const {
        return !(*this == it);
      }

    }; // class index_iterator

  public:

    /**
     * An offset calculator is an object that translates table indices
     * into linear offsets suitable for storage.  This may be used
     * to map a table's indices to its offsets, to map one table's indices
     * to the offsets of another table of corresponding size, or
     * to map indices from one table to offsets of another of smaller size
     * such that the smaller table represents a subspace of the larger.
     */
    class offset_functor {

      //! The multiplier associated with the index in each dimension.
      index_type multiplier_;

    public:

      /**
       * Constructs an offset calculator for the default indices
       * associated with a table with the supplied geometry.
       */
      offset_functor(const index_type& geometry)
        : multiplier_(geometry.size(), 1) {
        // Calculate the multipliers.
        for (size_t i = 1; i < multiplier_.size(); ++i)
          multiplier_[i] = multiplier_[i - 1] * geometry[i - 1];
      }

      /**
       * Constructs an offset calculator for a table with the supplied
       * geometry and indices for a different table, given a
       * mapping from the dimensions of this table to the
       * dimensions of the other table.  If this mapping is a one-to-one
       * correspondence, then these tables' dimensions must obey the same
       * one-to-one correspondence.  If this mapping is a partial injective
       * mapping, then the table with the supplied geometry must represent
       * a subspace of the other table.
       *
       * Example use: table1 is a subspace (restriction of) of table2
       * table1.var[i] = table2.var[pos[i]]
       *
       * @param  geometry
       *         the geometry of the table for which offsets
       *         are calculated
       * @param  index_dim
       *         the size of the indices that will be
       *         supplied to this object's operator()
       * @param  pos_map
       *         pos_map[i] gives
       *         the tuple position of the table indices
       *         supplied to this object's operator() that
       *         is associated with dimension i of the geometry.
       *         The values in this map must be in the range
       *         [0, index_dim).
       */
      offset_functor(const index_type& geometry,
                     size_t index_dim,
                     const index_type& pos_map)
        : multiplier_(index_dim, 0) {
        // Calculate the multipliers, one per tuple position.
        if (geometry.empty())
          return;
        multiplier_[pos_map[0]] = 1;
        for (size_t i = 1; i < geometry.size(); ++i)
          multiplier_[pos_map[i]] = multiplier_[pos_map[i-1]] * geometry[i-1];
      }

      /**
       * Calculates the offset associated with the supplied table index.
       *
       * \todo This code is often used in the context of repeatedly
       * incrementing an index and computing its offset.  This code does
       * not take into account the fact that when a table index is
       * incremented, most of the positions may not change.  To approach
       * the optimal overhead of native nested loops we should exploit
       * this.
       */
      size_t operator()(const index_type& index) const {
        assert(multiplier_.size() == index.size());
        size_t offset = 0;
        for (size_t d = 0; d < multiplier_.size(); ++d)
          offset += multiplier_[d] * index[d];
        return offset;
      }

      /**
       * Calculates the offset associated with the supplied partial index.
       * The index is assumed to start at pos-th dimension, and may be
       * shorter than the remaining dimensions. The index values in the
       * omitted dimensions are assumed to be 0.
       */
      size_t operator()(const index_type& index, size_t pos) const {
        assert(index.size() + pos <= multiplier_.size());
        size_t offset = 0;
        for (size_t i = 0; i < index.size(); ++i) {
          offset += multiplier_[i + pos] * index[i];
        }
        return offset;
      }

      //! Get the multiplier associated with dimension d.
      size_t get_multiplier(size_t d) const{
        return multiplier_[d];
      }

      //! Calculates the index associated with the supplied offset.
      index_type index(size_t offset) const {
        index_type ind(multiplier_.size());
        // must use int here to avoid wrap-around
        for(int d = multiplier_.size()-1; d >= 0; --d) {
          assert(multiplier_[d] != 0);
          ind[d] = offset / multiplier_[d];
          offset = offset % multiplier_[d];
        }
        return ind;
      }

      /**
       * Computes the index corresponding to the given offset, for the first
       * nlower dimensions.
       * \param offset < multiplier_[nlower]
       */
      void index(size_t offset, size_t nlower, index_type& ind) const {
        ind.resize(nlower);
        for (int d = nlower-1; d >= 0; --d) {
          ind[d] = offset / multiplier_[d];
          offset = offset % multiplier_[d];
        }
      }

    }; // class offset_functor

#ifdef EXPERIMENTAL
    class offset_iterator {
     private:
      //! current 'linear' position of the iterator
      size_t offset;

      //! The multiplier associated with the index in each dimension.
      index_type multiplier;

      //! Geometry of the table for which offsets are calculated.
//      const index_type* geometry_ptr;

      //! Geometry of the table whose indices are being iterated over;
      //! this geometry corresponds to the below indices.
      const index_type* indices_geometry_ptr;

      //! Current 'index' position of the iterator.
      index_type indices;

     public:
      offset_iterator()
        : offset(0), indices_geometry_ptr(NULL) { }

      offset_iterator(const index_type& geometry)
        : offset(0),
          multiplier(geometry.size(), 1),
//          geometry_ptr(&geometry),
          indices_geometry_ptr(&geometry),
          indices(geometry.size(), 0) {
        // Calculate the multipliers.
        for (size_t i = 1; i < multiplier.size(); ++i)
          multiplier[i] = multiplier[i - 1] * geometry[i - 1];
      }

      offset_iterator(const index_type& geometry,
                      const index_type& indices_geometry,
                      const index_type& pos_map)
        : offset(0),
          multiplier(indices_geometry.size(), 0),
//          geometry_ptr(&geometry),
          indices_geometry_ptr(&indices_geometry),
          indices(indices_geometry.size(), 0) {
        // Calculate the multipliers, one per tuple position.
        if (!geometry.empty()) {
          multiplier[pos_map[0]] = 1;
          for (size_t i = 1; i < geometry.size(); ++i)
            multiplier[pos_map[i]] = multiplier[pos_map[i-1]] * geometry[i-1];
        }
      }

      //! Like a constructor, but avoids reallocation when possible.
      void reset(const index_type& geometry) {
        offset = 0;
        if (multiplier.size() != geometry.size())
          multiplier.resize(geometry.size());
        indices_geometry_ptr = &geometry;
        if (indices.size() != geometry.size())
          indices.resize(geometry.size());
        foreach(size_t& i, indices)
          i = 0;
        if (multiplier.size() > 0) {
          multiplier[0] = 1;
          for (size_t i = 1; i < multiplier.size(); ++i)
            multiplier[i] = multiplier[i - 1] * geometry[i - 1];
        }
      }

      //! Like a constructor, but avoids reallocation when possible.
      void reset(const index_type& geometry,
                 const index_type& indices_geometry,
                 const index_type& pos_map) {
        offset = 0;
        if (multiplier.size() != indices_geometry.size())
          multiplier.resize(indices_geometry.size());
        foreach(size_t& i, multiplier)
          i = 0;
        indices_geometry_ptr = &indices_geometry;
        if (indices.size() != indices_geometry.size())
          indices.resize(indices_geometry.size());
        foreach(size_t& i, indices)
          i = 0;
        if (!geometry.empty()) {
          multiplier[pos_map[0]] = 1;
          for (size_t i = 1; i < geometry.size(); ++i)
            multiplier[pos_map[i]] = multiplier[pos_map[i-1]] * geometry[i-1];
        }
      }

      /**
       * Calculates the offset associated with the supplied table index.
       *
       * \todo This code is often used in the context of repeatedly
       * incrementing an index and computing its offset.  This code does
       * not take into account the fact that when a table index is
       * incremented, most of the positions may not change.  To approach
       * the optimal overhead of native nested loops we should exploit
       * this.
       */
      size_t operator()() const {
        return offset;
      }

      /*
      //! Calculates the index associated with the supplied offset.
      index_type index(size_t offset) const {
        index_type ind(multiplier.size());
        // must use int here to avoid wrap-around
        for(int d = multiplier.size()-1; d >= 0; --d) {
          assert(multiplier[d] != 0);
          ind[d] = offset / multiplier[d];
          offset = offset % multiplier[d];
        }
        return ind;
      }
      */

      //! Increment to next index.
      //! If this iterator is at the end of the indices, then this does nothing.
      offset_iterator& operator++() {
        if (this->end())
          return *this;
        ++indices[0];
        offset = offset + multiplier[0];
        for (size_t i = 0; i < indices_geometry_ptr->size() - 1; ++i) {
          if (__builtin_expect
              ((indices[i] >= indices_geometry_ptr->operator[](i)), 0)) {
            offset -= indices[i] * multiplier[i];
            offset += multiplier[i+1];
            indices[i] = 0;
            ++indices[i+1];
          } else {
            break;
          }
        }
        return *this;
      }

      bool end() {
        return (indices.size() == 0 ||
                indices.back() >= indices_geometry_ptr->back());
      }
    }; // class offset_iterator

#endif

    // to allow conversions and multi-type joins
    template <typename U> friend class dense_table;

  }; // class dense_table

  //! Writes a human-readable representation of the table.
  //! \relates dense_table
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const dense_table<T>& table) {
    typedef typename dense_table<T>::index_type index_type;
    foreach(const index_type& index, table.indices()) {
      sill::copy(index, std::ostream_iterator<size_t, char>(out, " "));
      out << table(index) << std::endl;
    }
    return out;
  }

} // namespace sill

#ifdef EXPERIMENTAL
#undef EXPERIMENTAL
#endif

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_DENSE_TABLE_HPP
