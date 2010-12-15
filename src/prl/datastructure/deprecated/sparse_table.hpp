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

#ifndef PRL_SPARSE_TABLE_HPP
#define PRL_SPARSE_TABLE_HPP

#include <utility>
#include <algorithm>
#include <iterator>
#include <vector>
#include <map>
#include <sstream>
#include <boost/tuple/tuple.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/iterator/indirect_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/bind.hpp>
#include <boost/range.hpp>

#include <prl/global.hpp>
#include <prl/datastructure/table.hpp>
#include <prl/functional.hpp>
#include <prl/range/numeric.hpp>

#include <prl/stl_concepts.hpp>

#include <prl/macros_def.hpp>

////////////////////////////////////////////////////////////////////
// Needs clean-up
///////////////////////////////////////////////////////////////////

namespace prl {


  //! The type used to represent table arity (number of dimensions).
  //! This typedef could go directly into the table template definitions
  typedef std::size_t table_arity_t;

  /**
   * The type used to represent the number of elements of a table.
   *
   * \todo This typedef could go directly into the table template definitions
   * Moving the typedef would be especially useful if we are to deal
   * with huge sparse tables. It hould also be renamed to size_type
   * and arity_type.
   */
  typedef std::size_t table_size_t;

  /**
   * A table index describes a position in a table, i.e., the index in
   * each dimensions of the table.  This implementation is based upon
   * a vector, using a custom allocator that statically allocates four
   * indices, regardless of the actual arity of the table.  This is
   * faster than using the standard allocator for tables with four or
   * fewer dimensions, but slower for tables with more dimensions.
   *
   * Should perhaps be a member of dense_table?
   *
   */
  typedef std::vector<table_size_t,
                      prl::pre_allocator<table_size_t, 4>
                     > v_table_index;

  /**
   * The type used to represent a table's geometry; a table geometry
   * represents the number of dimensions and their sizes.
   */
  typedef v_table_index table_geometry;

  /**
   * The type used to represent a restriction to a subset of a table,
   * where a value in [0, dimension size - 1] indicates a restriction
   * of that dimension and a larger value indicates no restriction.
   */
  typedef v_table_index restrict_map;

  //! Computes the number of elements in a table with the supplied geometry.
  //! \todo could refactor dense_table and sparse_table to bring the
  //! common functions including this one into a base class
  inline table_size_t table_size(const table_geometry& geometry) {
    return prl::accumulate(geometry, 1, std::multiplies<size_t>());
  }

  // The following definition uses the standard allocator:
  //   typedef std::vector<table_size_t> v_table_index;

  //! Defined in dense_table.hpp.
  template <typename elt_t>
  class dense_table;

  /**
   * A sparse table with an arbitrary number of dimensions, each with
   * a finite number of values.  This sparse representation uses a
   * red-black tree over table indices to access and update elements
   * in logarithmic time.  There is also an index for each dimension
   * which maps positions in that dimension to elements in that slice;
   * these are used to permit sub-linear time joins between pairs of
   * sparse tables. 
   *
   * This class models a Table concept
   * \ingroup datastructure
   */
  template <typename elt_t>
  class sparse_table_t {
    typedef elt_t T; // helper
  public:

    //! The type of elements stored in this table.
    typedef T element;

    //! The type of elements stored in this table.
    typedef T value_type;

    //! The type used to represent the extents and the total number of
    //! elements in the table.
    typedef std::size_t size_type;

    //! The type used to represent the shape and indices.
    typedef std::vector<size_type, prl::pre_allocator<size_type,4> > shape_type;

    //! The type used to represent the number of dimensions.
    typedef typename shape_type::size_type arity_type;

    // warning: the rest of typedefs are invalid (just to compile)
    //! The storage container type for the elements.
    typedef std::vector<T> array_type;

    //! Iterator over the elements of this table.
    typedef typename array_type::iterator iterator;

    //! Const iterator over the elements of this table.
    typedef typename array_type::const_iterator const_iterator;


    iterator begin() { assert(0); return iterator(); }
    const_iterator begin() const { assert(0); return const_iterator(); }
    iterator end() { assert(0); return iterator(); }
    const_iterator end() const { assert(0); return const_iterator();}

    typedef int difference_type;
    typedef const T& const_reference;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef T* pointer;

    //! Iterator over indices into the table
    //typedef impl::table_index_iterator<shape_type> index_iterator;

    typedef boost::iterator_range<iterator> mutable_range;
    typedef boost::iterator_range<const_iterator> const_range;
    typedef boost::iterator_range<iterator> index_range; // FIXME

    /**
     * The type of an table index paired with its element (ielt =
     * indexed element).
     */
    typedef std::pair<const v_table_index, elt_t> ielt_t;

  protected:

    //! The geometry of this table.
    table_geometry geometry;

    //! The type of storage used to store indexed elements.
    typedef std::map<v_table_index, element> ielt_map_t;

    //! The primary store of this table's elements.
    ielt_map_t ielt_map;

    /**
     * A simple functional that extracts a reference to the element
     * associated with a const indexed element.
     */
    struct get_const_elt_t :
      public std::unary_function<const ielt_t&, const element&> {
      const element& operator()(const ielt_t& ielt) const {
        return ielt.second;
      }
    };

    /**
     * A simple functional that extracts a reference to the element
     * associated with a mutable indexed element.
     */
    struct get_mutable_elt_t :
      public std::unary_function<ielt_t&, element&> {
      element& operator()(ielt_t& ielt) const {
        return ielt.second;
      }
    };

  public:

    //! The type of iterators over indexed elements.
    typedef typename ielt_map_t::iterator ielt_it_t;
    typedef typename ielt_map_t::const_iterator const_ielt_it_t;

    //! The type of iterators over elements (without indexes).
    typedef typename boost::transform_iterator<get_mutable_elt_t,
                                               ielt_it_t> element_it;
    typedef typename boost::transform_iterator<get_const_elt_t,
                                               const_ielt_it_t> const_element_it;

  protected:

    //! A vector of iterators over indexed elements.
    typedef std::vector<ielt_it_t> ielt_it_vec_t;

    /**
     * A type of an index from positions in some (implicit) dimension
     * \f$d\f$ to the elements in that position (in dimension \f$d\f$).
     */
    typedef std::map<table_size_t, ielt_it_vec_t> ielt_index_t;

    //! A vector of ielt indexes, one per dimensions.
    mutable std::vector<ielt_index_t> ielt_index_vec;

    /**
     * The element associated with each cell whose elements are not
     * explicitly represented in #ielt_vec.
     */
    elt_t default_elt;

    /**
     * Returns true if the element with the supplied index is a
     * default element, i.e., it is not represented explicitly.
     *
     * @param i the index into the table
     * @return  true iff the index does not identify an explicit
     *          element
     */
    inline bool is_default(const v_table_index& i) const {
      return (ielt_map.find(i) == ielt_map.end());
    }

  public:

    //! Constructor.
    template <typename InputIterator>
    sparse_table_t(InputIterator begin, InputIterator end,
                   element default_elt = element()) :
      geometry(begin, end),
      ielt_map(),
      ielt_index_vec(std::distance(begin, end)),
      default_elt(default_elt)
    { }

    //! Construct from a range.
    template <typename ForwardRange>
    sparse_table_t(const ForwardRange& extents,
                   element default_lt = element()) :
      geometry(boost::begin(extents), boost::end(extents)),
      ielt_map(),
      ielt_index_vec(boost::size(extents)),
      default_elt(default_elt)
    { }


    /**
     * Scalar constructor.  Builds a table with no dimensions and a
     * single cell.
     */
    sparse_table_t(element init_elt = element()) :
      geometry(),
      ielt_map(),
      ielt_index_vec(),
      default_elt(init_elt)
    { }

    //! Copy constructor.
    sparse_table_t(const sparse_table_t<element>& table) {
      *this = table;
    }

    //! Constructs a sparse table from a dense one.
    template <typename dense_elt_t>
    sparse_table_t(const dense_table<dense_elt_t>& table,
                   element default_elt = element());

    //! Assignment operator.  This creates a deep copy.
    const sparse_table_t& operator=(const sparse_table_t& table) {
      clear();
      this->geometry = table.geometry;
      this->default_elt = table.default_elt;
      this->ielt_index_vec.clear();
      this->ielt_index_vec.resize(geometry.size());
      const_ielt_it_t it, end;
      for (boost::tie(it, end) = table.indexed_elements(); it != end; ++it)
        set(it->first, it->second);
      return *this;
    }

    //! Swaps this sparse table with the supplied one.
    void swap(sparse_table_t& table) {
      this->geometry.swap(table.geometry);
      std::swap(this->default_elt, table.default_elt);
      this->ielt_map.swap(table.ielt_map);
      this->ielt_index_vec.swap(table.ielt_index_vec);
    }

    //! Returns the total number of cells in this sparse table. \deprecated
    table_size_t num_elts() const { return table_size(geometry); }

    //! Returns the total number of cells in this sparse table
    table_size_t num_elements() const { return table_size(geometry); }

    table_size_t size() const { return table_size(geometry); }
    table_size_t max_size() const { return table_size(geometry); }
    bool empty() const { return table_size(geometry)==0; }



    //! Returns the number of cells with non-default values.
    table_size_t num_explicit_elts() const { return ielt_map.size(); }

    //! Returns the number of cells with default values.
    table_size_t num_implicit_elts() const {
      return num_elts() - num_explicit_elts();
    }

    /**
     * Returns true iff this table's default element is used, i.e., if
     * there is an element that is not specified explicitly.
     */
    inline bool uses_default() const {
      return (num_explicit_elts() < num_elts());
    }

    //! Returns the default value of this sparse table.
    const element& get_default_elt() const { return default_elt; }

    //! Updates the default value of this sparse table.
    void set_default_elt(const element& elt) { default_elt = elt; }

    //! Returns the geometry of this table.
    const table_geometry& shape() const { return geometry; }

    //! Returns the given dimension
    table_size_t size(table_arity_t dim) const {
      assert(dim<arity());
      return geometry[dim];
    }

    //! Returns the number of dimensions of this table.
    table_arity_t arity() const { return geometry.size(); }

    table_arity_t num_dimensions() const { return geometry.size(); }

    /**
     * Returns a mutable iterator range over the indexed elements of
     * this table.  The value type of these iterators is
     * indexed_element_t.
     */
    std::pair<ielt_it_t, ielt_it_t> indexed_elements() {
      return std::make_pair(ielt_map.begin(), ielt_map.end());
    }

    /**
     * Returns a const iterator range over the indexed elements of
     * this table.  The value type of these iterators is
     * indexed_element_t.
     */
    std::pair<const_ielt_it_t, const_ielt_it_t> indexed_elements() const {
      return std::make_pair(ielt_map.begin(), ielt_map.end());
    }

    /**
     * Returns a mutable iterator range over the elements of this table.
     *
     * @param permit_skipping
     *        if this flag is set to true, then the iterator range
     *        is permitted (but not required) to skip all instances of
     *        a designated element
     * @param skip_elt
     *        if skipping is enabled, then the returned iterator range
     *        is permitted (but not required) to skip all instances of
     *        this element
     */
    std::pair<element_it, element_it>
    elements(bool permit_skipping = false,
             element skip_elt = element()) {
      if (uses_default()) {
        // For now, only enumeration of explicit elements is
        // supported.
        assert(permit_skipping);
        assert(skip_elt == get_default_elt());
      }
      ielt_it_t begin, end;
      boost::tie(begin, end) = indexed_elements();
      return std::make_pair
        (boost::transform_iterator<get_mutable_elt_t, ielt_it_t>(begin),
         boost::transform_iterator<get_mutable_elt_t, ielt_it_t>(end));
    }

    /**
     * Returns a const iterator range over the elements of this table.
     *
     * @param permit_skipping
     *        if this flag is set to true, then the iterator range
     *        is permitted (but not required) to skip all instances of
     *        a designated element
     * @param skip_elt
     *        if skipping is enabled, then the returned iterator range
     *        is permitted (but not required) to skip all instances of
     *        this element
     */
    std::pair<const_element_it, const_element_it>
    elements(bool permit_skipping = false,
             element skip_elt = element()) const {
      if (uses_default()) {
        // For now, only enumeration of explicit elements is supported.
        assert(permit_skipping);
        assert(skip_elt == get_default_elt());
      }
      const_ielt_it_t begin, end;
      boost::tie(begin, end) = indexed_elements();
      return std::make_pair
        (boost::transform_iterator<get_const_elt_t, const_ielt_it_t>(begin),
         boost::transform_iterator<get_const_elt_t, const_ielt_it_t>(end));
    }

    /**
     * Const element access to this table.
     *
     * @param i the index into the table
     * @return a const reference to the element indexed by i
     */
    inline element get(const v_table_index& i) const {
#ifdef PRL_SANITY_CHECK
      assert(i.size() == arity());
      for (table_arity_t dim = 0; dim < arity(); ++dim) {
        assert(i[dim] >= 0);
        assert(i[dim] < geometry[dim]);
      }
#endif
      const_ielt_it_t ielt_it = ielt_map.find(i);
      if (ielt_it == ielt_map.end())
        return get_default_elt();
      else
        return ielt_it->second;
    }

    /**
     * Const element access to this table.
     *
     * @param table_index_t
     *        any type which uses operator[] to map dimensions to positions
     * @param i the index into the table
     * @return a const reference to the element indexed by i
     */
    template <typename table_index_t>
    inline element get(const table_index_t& i) const {
      return get(v_table_index(i.begin(), i.end()));
    }

    /**
     * Mutable element access to this table.
     *
     * @param i the index into the table
     * @param elt the new value associated with this index
     */
    inline void set(const v_table_index& i, element elt) {
#ifdef PRL_SANITY_CHECK
      assert(i.size() == arity());
      for (table_arity_t dim = 0; dim < arity(); ++dim) {
        assert(i[dim] >= 0);
        assert(i[dim] < geometry[dim]);
      }
#endif
      // Try to find the element with this index.
      ielt_it_t ielt_it = ielt_map.lower_bound(i);
      if ((ielt_it == ielt_map.end()) ||
          (ielt_it->first != i)) {
        // We did not find an element with this index.
        if (elt == get_default_elt())
          // No need to add default elements.
          return;
        // Insert the pair in the primary map and record its position.
        ielt_it = ielt_map.insert(ielt_it, std::make_pair(i, elt));
        // Insert the position into the index for each dimension.
        for (table_arity_t d = 0; d < geometry.size(); ++d) {
          ielt_index_t& ielt_index = ielt_index_vec[d];
          ielt_it_vec_t& ielt_it_vec = ielt_index[i[d]];
          ielt_it_vec.push_back(ielt_it);
        }
      } else {
        ielt_it->second = elt;
      }
    }

    /**
     * Mutable element access to this table.
     *
     * @param table_index_t
     *        any type which uses operator[] to map dimensions to positions
     * @param i the index into the table
     * @param elt the new value associated with this index
     */
    template <typename table_index_t>
    inline void set(const table_index_t& i, element elt) {
      set(v_table_index(i.begin(), i.end()), elt);
    }

    inline element& operator()(const v_table_index& i) {
#ifdef PRL_SANITY_CHECK
      assert(i.size() == arity());
      for (table_arity_t dim = 0; dim < arity(); ++dim) {
        assert(i[dim] >= 0);
        assert(i[dim] < geometry[dim]);
      }
#endif
      // Try to find the element with this index.
      ielt_it_t ielt_it = ielt_map.lower_bound(i);
      if ((ielt_it == ielt_map.end()) ||
          (ielt_it->first != i)) {
        // Insert the pair in the primary map and record its position.
        ielt_it = ielt_map.insert(ielt_it, std::make_pair(i, get_default_elt()));
        // Insert the position into the index for each dimension.
        for (table_arity_t d = 0; d < geometry.size(); ++d) {
          ielt_index_t& ielt_index = ielt_index_vec[d];
          ielt_it_vec_t& ielt_it_vec = ielt_index[i[d]];
          ielt_it_vec.push_back(ielt_it);
        }
      }
      return ielt_it->second;
    }

    template <typename table_index_t>
    inline element& operator()(const table_index_t& i) {
      return operator()(v_table_index(i.begin(), i.end()));
    }

    //! Removes any explicit elements from this sparse table.
    void clear() {
      ielt_map.clear();
      for (table_arity_t d = 0; d < geometry.size(); ++d)
        ielt_index_vec[d].clear();
    }

    /**
     * Invokes the supplied functor on all elements of the table.
     * Note that this functor is called once per explicit element, and
     * once for the default element.
     */
    template <typename functor_t>
    void apply(functor_t f) {
      element_it begin, end;
      boost::tie(begin, end) = elements(true, get_default_elt());
      std::for_each(begin, end, f);
      if (uses_default())
        f(default_elt);
    }

    //! implements Table::update
    template <typename Function>
    void update(Function f) {
      apply(update<Function>(f));
    }

    /**
     * A map from the dimensions of one table to those of another.  We
     * use integers rather than the table_arity_t type to permit the
     * special value -1, which means "no matched dimension".
     */
    typedef std::vector<int> dim_map_t;

  protected:

    /**
     * Represents the equality criteria used when computing a table
     * \f$z\f$ from two input tables \f$x\f$ and \f$y\f$.
     */
    class join_criteria_t {

    protected:

      //! Maps the dimensions of x to those of z.
      dim_map_t x_to_z_dim_map;

      //! Maps the dimensions of y to those of z.
      dim_map_t y_to_z_dim_map;

      /**
       * Maps the dimensions of z to those of x (or -1 if the
       * dimension has no correlate in x).
       */
      dim_map_t z_to_x_dim_map;

      /**
       * Maps the dimensions of z to those of y (or -1 if the
       * dimension has no correlate in y).
       */
      dim_map_t z_to_y_dim_map;

    public:

      //! Constructor.
      join_criteria_t(table_arity_t z_arity,
                      const dim_map_t& x_to_z_dim_map,
                      const dim_map_t& y_to_z_dim_map)
        : x_to_z_dim_map(x_to_z_dim_map),
          y_to_z_dim_map(y_to_z_dim_map),
          z_to_x_dim_map(z_arity, -1),
          z_to_y_dim_map(z_arity, -1)
      {
        // Compute the reverse maps.
        for (table_arity_t d = 0; d < x_to_z_dim_map.size(); ++d)
          z_to_x_dim_map[x_to_z_dim_map[d]] = d;
        for (table_arity_t d = 0; d < y_to_z_dim_map.size(); ++d)
          z_to_y_dim_map[y_to_z_dim_map[d]] = d;
      }

      /**
       * Gets the dimension of x that corresponds to the supplied
       * dimension of z (or -1 if there is no such dimension).
       */
      inline int get_x_for_z(table_arity_t d) const {
        return z_to_x_dim_map[d];
      }

      /**
       * Gets the dimension of x that corresponds to the supplied
       * dimension of z (or -1 if there is no such dimension).
       */
      inline int get_y_for_z(table_arity_t d) const {
        return z_to_y_dim_map[d];
      }

      /**
       * Gets the dimension of z that corresponds to the supplied
       * dimension of x.
       */
      inline table_arity_t get_z_for_x(table_arity_t d) const {
        return x_to_z_dim_map[d];
      }

      /**
       * Gets the dimension of z that corresponds to the supplied
       * dimension of y.
       */
      inline table_arity_t get_z_for_y(table_arity_t d) const {
        return y_to_z_dim_map[d];
      }

      //! Returns the number of dimensions of the result table z.
      inline table_arity_t z_arity() const {
        return z_to_x_dim_map.size();
      }
    };

    /**
     * A join operator which applies a binary operator to the joined
     * elements and inserts them into a result table.
     */
    template <typename binary_op_t>
    class join_inserter_t {

    protected:

      //! The join criteria.
      const join_criteria_t& criteria;

      //! The result table.
      sparse_table_t& z_table;

      //! The binary operator used to compute result elements.
      binary_op_t binary_op;

      /**
       * An index into the result table (z).  This index is stored
       * with the object to avoid the overhead of allocating it once
       * per joined pair.
       */
      v_table_index z_index;

    public:

      //! Constructor.
      join_inserter_t(const join_criteria_t& criteria,
                      binary_op_t binary_op,
                      sparse_table_t& z_table)
        : criteria(criteria), z_table(z_table), binary_op(binary_op),
          z_index(z_table.arity()) { }

      /**
       * The operation called to join two elements.
       */
      inline void operator()(const ielt_t& x_ielt,
                             const ielt_t& y_ielt) {
        // Build up the index into the result table.
        for (table_arity_t z_dim = 0; z_dim < z_table.arity(); ++z_dim) {
          int x_dim = criteria.get_x_for_z(z_dim);
          int y_dim = criteria.get_y_for_z(z_dim);
          if (x_dim != -1)
            z_index[z_dim] = x_ielt.first[x_dim];
          else
            z_index[z_dim] = y_ielt.first[y_dim];
        }
        // Store the result of the binary operation.
        z_table.set(z_index, binary_op(x_ielt.second, y_ielt.second));
      }
    };

    /**
     * A join operator which applies a binary operator to the joined
     * elements and aggregates them using another binary operator.
     *
     * \todo: This could be generalized to aggregate the result into a
     * table, not just a scalar.  That would permit fully general
     * combine/collapse operations for table factors, such as the
     * matrix/vector multiply used in the computation of HMM forwards
     * messages.
     */
    template <typename binary_join_op_t,
              typename binary_agg_op_t>
    class join_aggregator_t {

    protected:

      //! The binary operator used to compute joined elements.
      binary_join_op_t& join_op;

      //! The binary operator used to aggregate joined elements.
      binary_agg_op_t& agg_op;

      //! The aggregate.
      element& aggregate;

    public:

      //! Constructor.
      join_aggregator_t(binary_join_op_t& join_op,
                        binary_agg_op_t& agg_op,
                        element& aggregate)
        : join_op(join_op), agg_op(agg_op), aggregate(aggregate) { }

      /**
       * The operation called to join two elements.
       */
      inline void operator()(const ielt_t& x_ielt,
                             const ielt_t& y_ielt) {
        aggregate = agg_op(aggregate, join_op(x_ielt.second, y_ielt.second));
      }

    };

    /**
     * Checks to see if tuples from two source tables are compatible
     * and if they are, inserts a corresponding tuple in this table.
     */
    template <typename join_op_t>
    static inline bool join_ielts(const ielt_t& x_ielt,
                                  const ielt_t& y_ielt,
                                  const join_criteria_t& criteria,
                                  join_op_t& join_op) {
      // Determine if the two indexes match on the dimensions they
      // share.
      for (table_arity_t z_dim = 0; z_dim < criteria.z_arity(); ++z_dim) {
        int x_dim = criteria.get_x_for_z(z_dim);
        int y_dim = criteria.get_y_for_z(z_dim);
        if ((x_dim != -1) &&
            (y_dim != -1) &&
            (x_ielt.first[x_dim] != y_ielt.first[y_dim]))
          return false;
      }
      // Apply the join operator.
      join_op(x_ielt, y_ielt);
      return true;
    }

    /**
     * Performs an exhaustive join between explicit elements of two
     * tables.  This function takes in two iterator ranges over
     * explicit ielts from two tables x and y.  The dimensions of
     * these tables are mapped into a third table z, via two dimension
     * maps.  Every dimension in z is mapped to a dimension of x or y,
     * or possibly both.  For every ielt pair in these two ranges
     * whose positions match in the dimensions that are mapped to the
     * same z dimension, a binary function is applied to their
     * associated elements, and the resulting value is stored in z.
     */
    template <typename ielt_iterator_t,
              typename join_op_t>
    static inline
    void exhaustive_join_explicit(const ielt_iterator_t x_ielt_begin,
                                  const ielt_iterator_t x_ielt_end,
                                  const ielt_iterator_t y_ielt_begin,
                                  const ielt_iterator_t y_ielt_end,
                                  const join_criteria_t& criteria,
                                  join_op_t& join_op) {
      // Loop over all pairs of ielts from tables x and y.
      for (ielt_iterator_t x_ielt_it = x_ielt_begin;
           x_ielt_it != x_ielt_end; ++x_ielt_it) {
        for (ielt_iterator_t y_ielt_it = y_ielt_begin;
             y_ielt_it != y_ielt_end; ++y_ielt_it)
          join_ielts(*x_ielt_it, *y_ielt_it, criteria, join_op);
      }
    } // exhaustive_join_explicit

    /**
     * An object which orders indexed elements lexicographically on
     * a permuted subset of their index vectors.
     */
    struct ielt_it_comparator_t :
      public std::binary_function<ielt_it_t, ielt_it_t, bool>
    {
      const dim_map_t& a_dims;
      const dim_map_t& b_dims;
      ielt_it_comparator_t(const dim_map_t& dims)
        : a_dims(dims), b_dims(dims) { }
      ielt_it_comparator_t(const dim_map_t& a_dims,
                           const dim_map_t& b_dims)
        : a_dims(a_dims), b_dims(b_dims) {
        assert(a_dims.size() == b_dims.size());
      }
      bool operator()(const ielt_it_t& a, const ielt_it_t& b) const {
        for (table_arity_t d = 0; d < a_dims.size(); ++d)
          if (a->first[a_dims[d]] >= b->first[b_dims[d]])
            return false;
        return true;
      }
    };

    /**
     * Performs a sort-merge join without using an index.  This
     * function takes in two mutable iterator ranges over ielts from
     * two tables x and y.  (The order of these elements are affected
     * by this method.)  The dimensions of these tables are mapped
     * into a third table z, via two dimension maps.  Every dimension
     * in z is mapped to a dimension of x or y, or possibly both.  For
     * every ielt pair in these two ranges whose positions match in
     * the dimensions that are mapped to the same z dimension, a
     * binary function is applied to their associated elements, and
     * the resulting value is stored in z.  This is accomplished by
     * (in-place) sorting the tuples of x and y on the columns present
     * in z, and then using a sort-merge join.
     */
    template <typename join_op_t>
    static
    void unindexed_sm_join_explicit(ielt_it_vec_t& x_ielt_it_vec,
                                    ielt_it_vec_t& y_ielt_it_vec,
                                    const join_criteria_t& criteria,
                                    join_op_t& join_op) {
      // Determine the of columns shared by X and Y.
      dim_map_t x_cdims, y_cdims;
      for (table_arity_t d = 0; d < criteria.z_arity(); ++d)
        if ((criteria.get_x_for_z(d) != -1) &&
            (criteria.get_y_for_z(d) != -1)) {
          x_cdims.push_back(criteria.get_x_for_z(d));
          y_cdims.push_back(criteria.get_y_for_z(d));
        }
      assert(!x_cdims.empty());
      // Sort both sets of tuples lexicographically on their common columns.
      ielt_it_comparator_t x_ielt_it_cmp(x_cdims);
      std::sort(x_ielt_it_vec.begin(), x_ielt_it_vec.end(), x_ielt_it_cmp);
      ielt_it_comparator_t y_ielt_it_cmp(y_cdims);
      std::sort(y_ielt_it_vec.begin(), y_ielt_it_vec.end(), y_ielt_it_cmp);
      // Create an object for comparing tuples from x and y on their
      // common columns.
      ielt_it_comparator_t xy_ielt_it_cmp(x_cdims, y_cdims);
      ielt_it_comparator_t yx_ielt_it_cmp(y_cdims, x_cdims);
      // Initialize the start iterators for both sets.
      typedef typename ielt_it_vec_t::const_iterator ielt_it_it_t;
      ielt_it_it_t x_ielt_it_it = x_ielt_it_vec.begin();
      ielt_it_it_t y_ielt_it_it = y_ielt_it_vec.begin();
      const ielt_it_it_t x_ielt_it_end = x_ielt_it_vec.end();
      const ielt_it_it_t y_ielt_it_end = y_ielt_it_vec.end();
      // Use binary search to advance the iterators efficiently to the
      // first point of intersection.
      x_ielt_it_it = std::lower_bound(x_ielt_it_it, x_ielt_it_end,
                                      *y_ielt_it_it, xy_ielt_it_cmp);
      y_ielt_it_it = std::lower_bound(y_ielt_it_it, y_ielt_it_end,
                                      *x_ielt_it_it, yx_ielt_it_cmp);
      // Begin the merge process.
      v_table_index z_index(criteria.z_arity());
      while ((x_ielt_it_it != x_ielt_it_end) &&
             (y_ielt_it_it != y_ielt_it_end)) {
        if (xy_ielt_it_cmp(*x_ielt_it_it, *y_ielt_it_it))
          ++x_ielt_it_it;
        else if (yx_ielt_it_cmp(*y_ielt_it_it, *x_ielt_it_it))
          ++y_ielt_it_it;
        else {
          // We've hit two iterators whose ielts match on the common
          // columns.  Scan ahead to get ranges.
          const ielt_it_it_t x_ielt_it_r_begin = x_ielt_it_it;
          ielt_it_it_t x_ielt_it_r_end = x_ielt_it_r_begin;
          while ((x_ielt_it_r_end != x_ielt_it_end) &&
                 (!x_ielt_it_cmp(*x_ielt_it_r_begin, *x_ielt_it_r_end)))
            ++x_ielt_it_r_end;
          const ielt_it_it_t y_ielt_it_r_begin = y_ielt_it_it;
          ielt_it_it_t y_ielt_it_r_end = y_ielt_it_r_begin;
          while ((y_ielt_it_r_end != y_ielt_it_end) &&
                 (!y_ielt_it_cmp(*y_ielt_it_r_begin, *y_ielt_it_r_end)))
            ++y_ielt_it_r_end;
          // Now use an exhaustive join on these two ranges.
          exhaustive_join_explicit
            (boost::make_indirect_iterator(x_ielt_it_r_begin),
             boost::make_indirect_iterator(x_ielt_it_r_end),
             boost::make_indirect_iterator(y_ielt_it_r_begin),
             boost::make_indirect_iterator(y_ielt_it_r_end),
             criteria, join_op);
          // Finally, update the iterators past the ranges.
          x_ielt_it_it = x_ielt_it_r_end;
          y_ielt_it_it = y_ielt_it_r_end;
        }
      }
    } // unindexed_sm_join_explicit

    /**
     * Makes this table the join of the two supplied tables.  This
     * method uses sort-merge join to achieve sub-linear time
     * performance.  One of the join dimensions is chosen as the
     * primary equality predicate, and indexes within the two sparse
     * tables are used to access their explicit elements in order of
     * position in this dimension.  (The dimension is chosen to
     * maximize both tables' variation in positions.)  Then a merge
     * process is used to identify pairs of tuples that are further
     * tested for join equality using exhaustive pairwise comparisons.
     *
     * @param element
     *        the type of the elements in all involved tables
     * @param x
     *        a sparse table over elements of type element
     * @param y
     *        a sparse table over elements of type element
     * @param binary_op_t
     *        a type that models the symmetric binary operator concept
     * @param binary_op
     *        a binary operator which when applied to the default
     *        element of tables x and y yields the default element of
     *        this table
     * @param x_dim_map
     *        an object such that x_dim_map[i] gives the dimension of
     *        z that corresponds to dimension i of x; these must be
     *        unique
     * @param y_dim_map
     *        an object such that y_dim_map[i] gives the dimension of
     *        z that corresponds to dimension i of y; these must be
     *        unique
     */
    template <typename join_op_t>
    static void sort_merge_join(const sparse_table_t<element>& x,
                                const sparse_table_t<element>& y,
                                const join_criteria_t& criteria,
                                join_op_t& join_op) {
      // If either table is empty, then z should also be empty.
      if ((x.num_explicit_elts() == 0) || (y.num_explicit_elts() == 0))
        return;
      // Of the dimensions in both, find the one that is best for the
      // sort-merge join equality predicate.  Also count the number of
      // join dimensions.
      table_arity_t z_arity = criteria.z_arity();
      table_arity_t num_join_dims = 0;
      table_arity_t z_join_dim = z_arity; // invalid dimension
      table_size_t best_quality = 0;
      for (table_arity_t d = 0; d < z_arity; ++d) {
        // If this dimension is not present in both x and y, it cannot
        // be the join dimension.
        if ((criteria.get_x_for_z(d) == -1) ||
            (criteria.get_y_for_z(d) == -1))
          continue;
        num_join_dims++;
        // Compute the number of distinct positions in this dimension
        // that have explicit elements in table x.
        table_arity_t x_dim = criteria.get_x_for_z(d);
        table_size_t num_x_distinct = x.ielt_index_vec[x_dim].size();
        // Compute the number of distinct positions in this dimension
        // that have explicit elements in table y.
        table_arity_t y_dim = criteria.get_y_for_z(d);
        table_size_t num_y_distinct = y.ielt_index_vec[y_dim].size();
        // Compute the quality of this join dimension as the product of
        // distinct positions in table x and table y.  If both tables
        // have many distinct positions, most of the join predicate will
        // be taken care of using the primary join predicate.
        table_size_t quality = num_x_distinct * num_y_distinct;
        if (quality > best_quality) {
          best_quality = quality;
          z_join_dim = d;
        }
      }
      // If there is no shared dimension, compute the result by
      // exhaustive enumeration.
      if (z_join_dim == z_arity) {
        exhaustive_join_explicit(x.ielt_map.begin(),
                                 x.ielt_map.end(),
                                 y.ielt_map.begin(),
                                 y.ielt_map.end(),
                                 criteria, join_op);
        return;
      }
      table_arity_t x_join_dim = criteria.get_x_for_z(z_join_dim);
      table_arity_t y_join_dim = criteria.get_y_for_z(z_join_dim);
      // For each table, get the mapping from positions in the join
      // dimension to vectors of ielt iterators.
      typedef typename sparse_table_t<element>::ielt_index_t ielt_index_t;
      ielt_index_t& x_ielt_index = x.ielt_index_vec[x_join_dim];
      ielt_index_t& y_ielt_index = y.ielt_index_vec[y_join_dim];
      // Find the interval of position values where there could be
      // overlap.
      table_size_t start_pos =
        std::max<table_size_t>(x_ielt_index.begin()->first,
                               y_ielt_index.begin()->first);
      table_size_t finish_pos =
        std::min<table_size_t>(x_ielt_index.rbegin()->first,
                               y_ielt_index.rbegin()->first);
      // Compute a [begin, end) iterator range for each table that
      // includes this range of positions.
      typedef typename ielt_index_t::iterator ielt_index_it_t;
      ielt_index_it_t x_it = x_ielt_index.lower_bound(start_pos);
      ielt_index_it_t x_end = x_ielt_index.upper_bound(finish_pos);
      ielt_index_it_t y_it = y_ielt_index.lower_bound(start_pos);
      ielt_index_it_t y_end = y_ielt_index.upper_bound(finish_pos);
      // Perform the merge.
      while ((x_it != x_end) &&
             (y_it != y_end)) {
        table_size_t x_pos = x_it->first;
        table_size_t y_pos = y_it->first;
        if (x_pos == y_pos) {
          // Join the two sets of ielts.
          typedef typename sparse_table_t<element>::ielt_it_vec_t
            ielt_it_vec_t;
          ielt_it_vec_t& x_ielt_it_vec = x_it->second;
          ielt_it_vec_t& y_ielt_it_vec = y_it->second;
          // If there are no more join dimensions, or if there are too
          // few tuples, use an exhaustive join.
          if ((num_join_dims < 2) ||
              (x_ielt_it_vec.size() * y_ielt_it_vec.size() < 100)) {
            exhaustive_join_explicit
              (boost::make_indirect_iterator(x_ielt_it_vec.begin()),
               boost::make_indirect_iterator(x_ielt_it_vec.end()),
               boost::make_indirect_iterator(y_ielt_it_vec.begin()),
               boost::make_indirect_iterator(y_ielt_it_vec.end()),
               criteria, join_op);
          } else {
            // Otherwise, use a nested sort-merge join.
            unindexed_sm_join_explicit(x_ielt_it_vec,
                                       y_ielt_it_vec,
                                       criteria, join_op);
          }
          // Advance both iterators.
          ++x_it;
          ++y_it;
        } else if (x_pos < y_pos)
          ++x_it;
        else
          ++y_it;
      } // while (...)
    }

    /**
     * Makes this table the join of the two supplied tables.  This
     * method uses a linear scan over the explicit elements of the
     * smaller table, and uses an index to find matching tuples in the
     * larger table.
     *
     * @param element
     *        the type of the elements in all involved tables
     * @param x
     *        a sparse table over elements of type element
     * @param y
     *        a sparse table over elements of type element; for best
     *        performance this should be the larger of the two tables
     * @param binary_op_t
     *        a type that models the symmetric binary operator concept
     * @param binary_op
     *        a binary operator which when applied to the default
     *        element of tables x and y yields the default element of
     *        this table
     * @param x_dim_map
     *        an object such that x_dim_map[i] gives the dimension of
     *        z that corresponds to dimension i of x; these must be
     *        unique
     * @param y_dim_map
     *        an object such that y_dim_map[i] gives the dimension of
     *        z that corresponds to dimension i of y; these must be
     *        unique
     */
    template <typename join_op_t>
    static
    void probe_join(const sparse_table_t<element>& x,
                    const sparse_table_t<element>& y,
                    const join_criteria_t& criteria,
                    join_op_t& join_op) {
      // If either table is empty, then z should also be empty.
      if ((x.num_explicit_elts() == 0) || (y.num_explicit_elts() == 0))
        return;
      // Of the dimensions in both, find the one whose index in y is
      // best.  Also count the number of join dimensions.
      table_arity_t z_arity = criteria.z_arity();
      table_arity_t num_join_dims = 0;
      table_arity_t z_join_dim = z_arity; // invalid dimension
      table_size_t best_quality = 0;
      for (table_arity_t z_dim = 0; z_dim < z_arity; ++z_dim) {
        // If this dimension is not present in both x and y, it cannot
        // be the join dimension.
        if ((criteria.get_x_for_z(z_dim) == -1) ||
            (criteria.get_y_for_z(z_dim) == -1))
          continue;
        num_join_dims++;
        // Compute the number of distinct positions in this dimension
        // that have explicit elements in table y.
        table_arity_t y_dim = criteria.get_y_for_z(z_dim);
        table_size_t num_y_distinct = y.ielt_index_vec[y_dim].size();
        // Compute the quality of this index dimension.
        table_size_t quality = num_y_distinct;
        if (quality > best_quality) {
          best_quality = quality;
          z_join_dim = z_dim;
        }
      }
      // If there is no shared dimension, compute the result by
      // exhaustive enumeration.
      if (z_join_dim == z_arity) {
        exhaustive_join_explicit(x.ielt_map.begin(),
                                 x.ielt_map.end(),
                                 y.ielt_map.begin(),
                                 y.ielt_map.end(),
                                 criteria, join_op);
        return;
      }
      table_arity_t x_join_dim = criteria.get_x_for_z(z_join_dim);
      table_arity_t y_join_dim = criteria.get_y_for_z(z_join_dim);
      // Get the mapping from positions in the join dimension to
      // vectors of ielt iterators for table y.
      typedef typename sparse_table_t<element>::ielt_index_t ielt_index_t;
      typedef typename ielt_index_t::const_iterator ielt_it_vec_it_t;
      ielt_index_t& y_ielt_index = y.ielt_index_vec[y_join_dim];
      // Iterate over the tuples of x, and for each tuple, probe the
      // index to find all matches in y.
      const_ielt_it_t x_ielt_it, x_ielt_end;
      v_table_index z_index(criteria.z_arity());
      for (boost::tie(x_ielt_it, x_ielt_end) = x.indexed_elements();
           x_ielt_it != x_ielt_end; ++x_ielt_it) {
        const ielt_t& x_ielt = *x_ielt_it;
        table_size_t join_dim_val = x_ielt.first[x_join_dim];
        ielt_it_vec_it_t ielt_it_vec_it = y_ielt_index.find(join_dim_val);
        if (ielt_it_vec_it == y_ielt_index.end())
          continue;
        // Get the vector of ielt iterators that match the x tuple in
        // the indexed join dimension.
        const ielt_it_vec_t& ielt_it_vec = ielt_it_vec_it->second;
        typedef typename ielt_it_vec_t::const_iterator ielt_it_it_t;
        for (ielt_it_it_t ielt_it_it = ielt_it_vec.begin();
             ielt_it_it != ielt_it_vec.end(); ++ielt_it_it)
          join_ielts(x_ielt, **ielt_it_it, criteria, join_op);
      }
    }

    /**
     * Joins all explicit elements of table x with all matching
     * default elements of table y.
     *
     * @param join_op_t
     *        the type of join operator used (e.g., join_inserter_t)
     * @param x
     *        the left sparse table
     * @param y
     *        the right sparse table
     * @param criteria
     *        the join criteria
     * @param join_op
     *        the join operator
     */
    template <typename join_op_t>
    static
    void join_explicit_to_default(const sparse_table_t<element>& x,
                                  const sparse_table_t<element>& y,
                                  const join_criteria_t& criteria,
                                  join_op_t& join_op) {
      // Compute the dimensions of y that are not in x and construct
      // the geometry of a hypothetical table yp (projected) that
      // consists only of the dimensions of y that are not present in
      // x. Also construct a mapping from the dimensions of z to
      // those of yp.
      dim_map_t z_to_yp_dim_map(criteria.z_arity(), -1);
      table_geometry yp_geometry;
      for (table_arity_t z_dim = 0; z_dim < criteria.z_arity(); ++z_dim) {
        if ((criteria.get_x_for_z(z_dim) == -1) &&
            (criteria.get_y_for_z(z_dim) != -1)) {
          table_arity_t y_dim = criteria.get_y_for_z(z_dim);
          table_size_t y_dim_size = y.shape()[y_dim];
          yp_geometry.push_back(y_dim_size);
          z_to_yp_dim_map[z_dim] = yp_geometry.size() - 1;
        }
      }
      // Iterate over the explicit tuples of x.
      const_ielt_it_t x_ielt_it, x_ielt_end;
      v_table_index y_index(y.arity());
      v_table_index yp_index(yp_geometry.size(), 0);
      for (boost::tie(x_ielt_it, x_ielt_end) = x.indexed_elements();
           x_ielt_it != x_ielt_end; ++x_ielt_it) {
        const ielt_t& x_ielt = *x_ielt_it;
        const v_table_index& x_index = x_ielt.first;
        // For each explicit tuple of x, iterate over all matching
        // default tuples of y.
        bool done = false;
        yp_index.assign(yp_geometry.size(), 0);
        while (!done) {
          // Construct the index into y corresponding to the current
          // indices into x and yp.
          for (table_arity_t z_dim = 0; z_dim < criteria.z_arity(); ++z_dim)
            if (criteria.get_y_for_z(z_dim) != -1) {
              if (criteria.get_x_for_z(z_dim) != -1)
                y_index[criteria.get_y_for_z(z_dim)] =
                  x_index[criteria.get_x_for_z(z_dim)];
              else
                y_index[criteria.get_y_for_z(z_dim)] =
                  yp_index[z_to_yp_dim_map[z_dim]];
            }
          // If the element associated with this index in y is a
          // default element, then add a new tuple to z.
          if (y.is_default(y_index)) {
            // Construct the indexed element with the default value.
            ielt_t y_ielt(y_index, y.get_default_elt());
            // Join the elements.
            join_op(x_ielt, y_ielt);
          }
          // Increment the index into yp.
          done = increment(yp_index, yp_geometry);
        }
      }
    }

  public:

    /**
     * Joins two sparse tables to yield a third sparse table, whose
     * elements are computed using a binary function of the joined
     * elements of the input tables.  It is assumed that when the
     * binary function is applied to the default elements of x or y,
     * the result is the default element of z, i.e., that \f$f(x_0, y)
     * = f(x, y_0) = z_0\f$ for all \f$x\f$ and \f$y\f$, where
     * \f$x_0\f$, \f$y_0\f$, and \f$z_0\f$ are the table defaults.
     * This property holds, e.g., if the binary function is
     * multiplication and the defaults are all zero.
     *
     * This method is named "join" because it is exactly the notion of
     * join used in database computations.  Each sparse table is viewed
     * as a set of tuples (indices), and the result table is computed by
     * joining tuples based upon an equality predicate (which matches up
     * certain dimensions), and applying a binary function to the
     * elements associated with these tuples.
     *
     * This method assumes that each dimension of z is associated with
     * a dimension of x, y, or both.
     *
     * @param binary_op_t
     *        a type that models the symmetric binary operator concept
     * @param element
     *        the type of the elements in all involved tables
     * @param x
     *        a sparse table over elements of type element
     * @param y
     *        a sparse table over elements of type element
     * @param z
     *        a sparse table over elements of type element which is is
     *        used to store the result of the join; each dimension of x
     *        must correspond to exactly one dimension of z; each
     *        dimension of y must correspond to one dimension of z; each
     *        dimension of z must correspond to a dimension of x, y,
     *        or both x and y.
     * @param binary_op
     *        the binary operator applied to the elements of x and y to
     *        obtain the corresponding elements of this table
     * @param x_dim_map
     *        an object such that x_dim_map[i] gives the dimension of
     *        z that corresponds to dimension i of x; these must be
     *        unique
     * @param y_dim_map
     *        an object such that y_dim_map[i] gives the dimension of
     *        z that corresponds to dimension i of y; these must be
     *        unique
     */
    template <typename DimMap, typename JoinOp>
    void join(const sparse_table_t<elt_t>& x,
              const sparse_table_t<elt_t>& y,
              const DimMap& _x_dim_map,
              const DimMap& _y_dim_map,
              JoinOp op) {
      concept_assert((BinaryFunction<JoinOp,T,T,T>));
      dim_map_t x_dim_map(boost::begin(_x_dim_map), boost::end(_x_dim_map));
      dim_map_t y_dim_map(boost::begin(_y_dim_map), boost::end(_y_dim_map));
      // These flags can be used to disable a join technique if it is
      // believed to contain a bug.
      const bool enable_sm_join = true;
      const bool enable_probe_join = true;
      // Call this table z.
      sparse_table_t<element>& z = *this;
      // Remove any existing elements from z.
      z.clear();
      // Create the join criteria and inserter.
      join_criteria_t criteria(z.arity(), x_dim_map, y_dim_map);

      join_inserter_t<JoinOp> inserter(criteria, op, *this);
      // Create a reverse join criteria and inserter.
      join_criteria_t rev_criteria(z.arity(), y_dim_map, x_dim_map);

      // Create a reverse join inserter for this table.
      typedef reverse_args<JoinOp> ReverseOp;
      join_inserter_t<ReverseOp> rev_inserter
        (rev_criteria, ReverseOp(op), *this);
      // Compute the default element of the result table by applying the
      // binary function to the defaults of the two input tables.
      if (x.uses_default() && y.uses_default())
        z.set_default_elt(op(x.get_default_elt(),
                                    y.get_default_elt()));
      // If one table is far smaller than another, use probe join.
      // Otherwise, use sort merge join.
      double ratio = static_cast<double>(x.num_explicit_elts()) /
        static_cast<double>(y.num_explicit_elts());
      if (enable_probe_join &&
          (!enable_sm_join || (ratio > 8.0) || (ratio < 0.125))) {
        /* std::cerr << "using probe join" << std::endl; */
        if (x.num_explicit_elts() < y.num_explicit_elts())
          probe_join(x, y, criteria, inserter);
        else
          probe_join(y, x, rev_criteria, rev_inserter);
      } else if (enable_sm_join) {
        /* std::cerr << "using sort-merge join" << std::endl; */
        sort_merge_join(x, y, criteria, inserter);
      } else
        assert(false);
      // If the default element of both sparse tables is the zero of
      // the binary operator, then a join of the explicit elements is
      // sufficient.  Otherwise, we must also join explicit elements
      // to default elements.
      bool x_def_is_zero =
        op.has_left_zero &&
        (x.get_default_elt() == op.left_zero());
      bool y_def_is_zero =
        op.has_right_zero &&
        (y.get_default_elt() == op.right_zero());
      if (!x_def_is_zero) {
        /* std::cerr << "joining default to explicit" << std::endl; */
        join_explicit_to_default(y, x, rev_criteria, rev_inserter);
      }
      if (!y_def_is_zero){
        /* std::cerr << "joining explicit to default" << std::endl; */
        join_explicit_to_default(x, y, criteria, inserter);
      }
    }

    /**
     * Joins two tables and aggregates the elements of the resulting
     * table into a single element.
     *
     * @param x
     *        an input table
     * @param y
     *        an input table
     * @param join_op
     *        an object that models the binary operator concept;
     *        it is used to compute elements of the joined table
     * @param x_dim_map
     *        an object such that x_dim_map[i] gives the dimension of
     *        (the fictitious join table) z that corresponds to
     *        dimension i of x; these must be unique
     * @param y_dim_map
     *        an object such that y_dim_map[i] gives the dimension of
     *        (the fictitious join table) z that corresponds to
     *        dimension i of y; these must be unique
     * @param agg_op
     *        an object that models the binary operator concept; it
     *        is used to aggregate the elements of the joined table
     *
     * \todo This function and #join_tables have too much common
     * structure; find a way to combine them.
     *
     * \todo: This could be generalized to aggregate the result into a
     * table, not just a scalar.  That would permit fully general
     * combine/collapse operations for table factors, such as the
     * matrix/vector multiply used in the computation of HMM forwards
     * messages.
     */
    template <typename DimMap, typename JoinOp, typename AggOp>
    static
    T join_aggregate(const sparse_table_t& x,
                     const sparse_table_t& y,
                     const DimMap& _x_dim_map,
                     const DimMap& _y_dim_map,
                     JoinOp join_op, AggOp agg_op) {
      concept_assert((BinaryFunction<JoinOp,T,T,T>));
      concept_assert((BinaryFunction<AggOp,T,T,T>));
      dim_map_t x_dim_map(boost::begin(_x_dim_map), boost::end(_x_dim_map));
      dim_map_t y_dim_map(boost::begin(_y_dim_map), boost::end(_y_dim_map));

      // Compute the number of dimensions of the joined table.
      table_arity_t z_arity = 0;
      for (table_arity_t d = 0; d < x.arity(); ++d)
        z_arity = std::max<table_arity_t>(z_arity, 1 + x_dim_map[d]);
      for (table_arity_t d = 0; d < y.arity(); ++d)
        z_arity = std::max<table_arity_t>(z_arity, 1 + y_dim_map[d]);
      // Initialize the aggregate with the identity of the
      // aggregation operator.
      assert(agg_op.has_left_identity);
      element aggregate = agg_op.left_identity();
      // Create the join criteria and inserter.
      join_criteria_t criteria(z_arity, x_dim_map, y_dim_map);
      join_aggregator_t<JoinOp, AggOp> aggregator(join_op, agg_op, aggregate);

      // Create a reverse join criteria and inserter.
      join_criteria_t rev_criteria(z_arity, y_dim_map, x_dim_map);
      // Create a reverse join inserter for this table.
      typedef reverse_args<JoinOp> ReversedJoinOp;
      ReversedJoinOp rev_join_op;
      join_aggregator_t<ReversedJoinOp, AggOp> rev_aggregator
        (rev_join_op, agg_op, aggregate);
      // If one table is far smaller than another, use probe join.
      // Otherwise, use sort merge join.
      double ratio = static_cast<double>(x.num_explicit_elts()) /
        static_cast<double>(y.num_explicit_elts());
      if ((ratio > 8.0) || (ratio < 0.125)) {
        /* std::cerr << "using probe join" << std::endl; */
        if (x.num_explicit_elts() < y.num_explicit_elts())
          probe_join(x, y, criteria, aggregator);
        else
          probe_join(y, x, rev_criteria, rev_aggregator);
      } else {
        /* std::cerr << "using sort-merge join" << std::endl; */
        sort_merge_join(x, y, criteria, aggregator);
      }
      // If the default element of both sparse tables is the zero of
      // the binary operator, then a join of the explicit elements is
      // sufficient.  Otherwise, we must also join explicit elements
      // to default elements.
      bool x_def_is_zero =
        join_op.has_left_zero && (x.get_default_elt() == join_op.left_zero());
      bool y_def_is_zero =
        join_op.has_right_zero && (y.get_default_elt() == join_op.right_zero());
      if (!x_def_is_zero) {
        /* std::cerr << "joining explicit to default" << std::endl; */
        join_explicit_to_default(y, x, rev_criteria, rev_aggregator);
      }
      if (!y_def_is_zero){
        /* std::cerr << "joining explicit to default" << std::endl; */
        join_explicit_to_default(x, y, criteria, aggregator);
      }
      return aggregate;
    }

    /**
     * Joins a sparse table y into this sparse table.  The every
     * dimension of y must be mapped to a unique dimension of this
     * table.
     *
     * @param element
     *        the type of the elements in all involved tables
     * @param y
     *        a sparse table over elements of type element;
     *        every dimension of y must correspond to a dimension
     *        of this table
     * @param y_dim_map
     *        an object such that y_dim_map[i] gives the dimension of
     *        this table that corresponds to dimension i of y; these
     *        must be unique
     */
    template <typename DimMap, typename JoinOp>
    void join_with(const sparse_table_t<elt_t>& y,
                   const DimMap& _y_dim_map,
                   JoinOp op) {
      concept_assert((BinaryFunction<JoinOp,T,T,T>));

      dim_map_t y_dim_map(boost::begin(_y_dim_map), boost::end(_y_dim_map));
      // This flag can be used to disable specialized join techniques
      // if they are believed to contain a bug.
      const bool enable_join_in = true;
      // Call this table x.
      sparse_table_t<element>& x = *this;
      if (!enable_join_in) {
        // Skip the special case code below.  Create a temporary table
        // to hold the result of the join.
        sparse_table_t z(x.shape().begin(),
                         x.shape().end());
        // Create an identity map from the columns of x to those of z.
        dim_map_t x_dim_map(x.arity());
        for (table_arity_t i = 0; i < x.arity(); ++i)
          x_dim_map[i] = i;
        // Run the join.
        z.join(x, y, x_dim_map, y_dim_map, op);
        // Swap this table with the temporary table.
        this->swap(z);
        return;
      }

      // Determine if the default elements of the input tables are the
      // zero or identity elements.
      bool y_def_is_zero =
        op.has_right_zero && (y.get_default_elt() == op.right_zero());
      bool x_def_is_zero =
        op.has_left_zero && (x.get_default_elt() == op.left_zero());
      bool y_def_is_identity =
        op.has_right_identity && (y.get_default_elt() == op.right_identity());
      bool x_def_is_identity =
        op.has_left_identity && (x.get_default_elt() == op.left_identity());
      // Special case.  If one of the tables is all zeros, then the
      // answer has all zeros.
      if (x_def_is_zero && (x.num_explicit_elts() == 0)) {
        /* std::cerr << "Table x is all zeros." << std::endl; */
        return;
      }
      if (y_def_is_zero && (y.num_explicit_elts() == 0)) {
        /* std::cerr << "Table y is all zeros." << std::endl; */
        x.set_default_elt(op.right_zero());
        x.clear();
        return;
      }
      // Special case.  If x and y have the same columns, and all
      // elements of X are identity elements, then x becomes a copy of
      // y (assuming the columns are ordered consistently).
      if (x_def_is_identity &&
          (x.num_explicit_elts() == 0) &&
          (x.arity() == y.arity())) {
        // Check that the columns are ordered consistently.
        bool columns_consistent = true;
        for (table_arity_t i = 0; (i < x.arity()) && columns_consistent; ++i)
          columns_consistent = (static_cast<table_arity_t>(y_dim_map[i]) == i);
        if (columns_consistent) {
          x = y;
          return;
        }
      }
      // Special case.  If the explicit elements of the input tables
      // have no intersection and the default element of y is the
      // identity element and the default element of x is the
      // zero element, then no work is necessary.
      if (y_def_is_identity && x_def_is_zero) {
        // If either table is empty, they are certainly disjoint.
        if ((y.num_explicit_elts() == 0) ||
            (x.num_explicit_elts() == 0)) {
          /* std::cerr << "Using empty table special case." << std::endl; */
          return;
        }
        // Check if y and x have disjoint explicit elements in their
        // shared dimensions.  The test below (which is based upon
        // bounding boxes) is sufficient, but not necessary, for
        // disjointness.
        bool disjoint = false;
        for (table_arity_t y_dim = 0;
             (y_dim < y.arity()) && !disjoint; ++y_dim) {
          ielt_index_t& y_ielt_index = y.ielt_index_vec[y_dim];
          table_size_t y_min = y_ielt_index.begin()->first;
          table_size_t y_max = y_ielt_index.rbegin()->first;
          ielt_index_t& x_ielt_index = x.ielt_index_vec[y_dim_map[y_dim]];
          table_size_t x_min = x_ielt_index.begin()->first;
          table_size_t x_max = x_ielt_index.rbegin()->first;
          // The tables are disjoint if in this (or any dimension)
          // their extents are disjoint.
          disjoint |= (x_max < y_min) || (y_max < x_min);
        }
        if (disjoint) {
          /* std::cerr << "Using no-overlap special case." << std::endl; */
          return;
        }
      }
      // Two other special cases:
      //
      // 1. If the default element of x is the zero element, then we
      // can use a linear scan through its explicit elements and
      // combine into each the corresponding element from y.
      //
      // 2. If the default element of y is the identity element, then
      // we can use a linear scan through its explicit elements and
      // combine each into the (possibly many) matching elements of x.
      //
      // If both options are available, then we choose one based upon
      // the number of set operations required for x.
      //
      // If neither option is available, then we use join_tables()
      // with a temporary table to compute the result.
      double x_to_y_ratio = static_cast<double>(x.num_elts()) /
        static_cast<double>(y.num_elts());
      bool strategy_1_is_available = x_def_is_zero;
      bool strategy_2_is_available = y_def_is_identity;
      bool strategy_1_is_better_than_2 =
        (x.num_explicit_elts() < (y.num_explicit_elts() * x_to_y_ratio));
      bool use_strategy_1 =
        strategy_1_is_available &&
        (!strategy_2_is_available || strategy_1_is_better_than_2);
      bool use_strategy_2 = !use_strategy_1 && strategy_2_is_available;
      if (use_strategy_1) {
        /* std::cerr << "Using strategy 1." << std::endl; */
        // Perform a linear scan through the explicit elements of x
        // and combine into each the corresponding element from y.
        ielt_it_t x_ielt_it, x_ielt_end;
        v_table_index y_index(y.arity());
        for (boost::tie(x_ielt_it, x_ielt_end) = x.indexed_elements();
             x_ielt_it != x_ielt_end; ++x_ielt_it) {
          ielt_t& x_ielt = *x_ielt_it;
          // Compute the corresponding index into y.
          for (table_arity_t d = 0; d < y.arity(); ++d)
            y_index[d] = x_ielt.first[y_dim_map[d]];
          element new_x_elt = op(x_ielt.second, y.get(y_index));
          if (new_x_elt == x.get_default_elt())
            // TODO: remove explicit element and update indices
            x_ielt.second = new_x_elt;
          else
            x_ielt.second = new_x_elt;
        }
      } else if (use_strategy_2) {
        /* std::cerr << "Using strategy 2." << std::endl; */
        // Perform a linear scan through the explicit elements of y
        // and combine each into the (possibly many) matching elements
        // of x.
        //
        // Compute the dimensions of x that are not in y and construct
        // the geometry of a hypothetical table xp (projected) that
        // consists only of the dimensions of x that are not present
        // in y.  Also construct a mapping from the dimensions of xp to
        // those of x.
        table_arity_t y_arity = y.arity();
        table_arity_t x_arity = x.arity();
        std::vector<int> x_to_y_dim_map(x_arity, -1);
        for (table_arity_t d = 0; d < y_arity; ++d)
          x_to_y_dim_map[y_dim_map[d]] = d;
        table_arity_t xp_arity = x_arity - y_arity;
        std::vector<int> xp_to_x_dim_map(xp_arity, -1);
        table_geometry xp_geometry;
        for (table_arity_t d = 0; d < x_arity; ++d)
          if (x_to_y_dim_map[d] == -1) {
            xp_geometry.push_back(x.shape()[d]);
            xp_to_x_dim_map[xp_geometry.size() - 1] = d;
          }
        // Iterate over the explicit tuples of y.
        const_ielt_it_t y_ielt_it, y_ielt_end;
        v_table_index x_index(x_arity);
        v_table_index xp_index(xp_arity, 0);
        for (boost::tie(y_ielt_it, y_ielt_end) = y.indexed_elements();
             y_ielt_it != y_ielt_end; ++y_ielt_it) {
          const ielt_t& y_ielt = *y_ielt_it;
          const v_table_index& y_index = y_ielt.first;
          // Update the index of x to reflect the current index into y.
          for (table_arity_t d = 0; d < y_arity; ++d)
            x_index[y_dim_map[d]] = y_index[d];
          // For each explicit tuple of y, iterate over all matching
          // tuples of x.
          bool done = false;
          xp_index.assign(xp_arity, 0);
          while (!done) {
            // Update the index of x to reflect the current index into
            // xp.
            for (table_arity_t d = 0; d < xp_arity; ++d)
              x_index[xp_to_x_dim_map[d]] = xp_index[d];
            // Now we have a complete index into x.
            x.set(x_index, op(x.get(x_index), y_ielt.second));
            // Increment the index into xp.
            done = increment(xp_index, xp_geometry);
          }
        }
      } else {
        /* std::cerr << "Using join_tables() method." << std::endl; */
        // Create a temporary table to hold the result of the join.
        sparse_table_t z(x.shape().begin(),
                         x.shape().end());
        // Create an identity map from the columns of x to those of z.
        dim_map_t x_dim_map(x.arity());
        for (table_arity_t i = 0; i < x.arity(); ++i)
          x_dim_map[i] = i;
        // Run the join.
        z.join(x, y, x_dim_map, y_dim_map, op);
        // Swap this table with the temporary table.
        this->swap(z);
      }
    }

    template <typename DimMap, typename AggOp>
    void aggregate(const sparse_table_t& x,
                   const DimMap& z_to_x_dim_map,
                   AggOp op) {
      aggregate_table(x, *this, z_to_x_dim_map, op);
    }

    template <typename DimMap>
    void restrict(const sparse_table_t& x,
                  const shape_type& restrict_map,
                  const DimMap& dim_map) {
      assert(0); // not implemented yet
    }

    //! conversion to human-readable representation
    operator std::string() const {
      std::ostringstream out; out << this; return out.str(); 
    }

  }; // class sparse_table_t

  /**
   * Joins two sparse tables to yield a third sparse table, whose
   * elements are computed using a binary function of the joined
   * elements of the input tables.  It is assumed that when the
   * binary function is applied to the default elements of x or y,
   * the result is the default element of z, i.e., that \f$f(x_0, y)
   * = f(x, y_0) = z_0\f$ for all \f$x\f$ and \f$y\f$, where
   * \f$x_0\f$, \f$y_0\f$, and \f$z_0\f$ are the table defaults.
   * This property holds, e.g., if the binary function is
   * multiplication and the defaults are all zero.
   *
   * This method is named "join" because it is exactly the notion of
   * join used in database computations.  Each sparse table is viewed
   * as a set of tuples (indices), and the result table is computed by
   * joining tuples based upon an equality predicate (which matches up
   * certain dimensions), and applying a binary function to the
   * elements associated with these tuples.
   *
   * This method assumes that each dimension of z is associated with
   * a dimension of x, y, or both.
   *
   * @param element
   *        the type of the elements in all involved tables
   * @param x
   *        a sparse table over elements of type element
   * @param y
   *        a sparse table over elements of type element
   * @param z
   *        a sparse table over elements of type element which is is
   *        used to store the result of the join; each dimension of x
   *        must correspond to exactly one dimension of z; each
   *        dimension of y must correspond to one dimension of z; each
   *        dimension of z must correspond to a dimension of x, y,
   *        or both x and y.
   * @param x_dim_map
   *        an object such that x_dim_map[i] gives the dimension of
   *        z that corresponds to dimension i of x; these must be
   *        unique
   * @param y_dim_map
   *        an object such that y_dim_map[i] gives the dimension of
   *        z that corresponds to dimension i of y; these must be
   *        unique
   */
  /*
  template <typename element,
            typename binary_op_tag_t,
            typename dim_map_t>
  void join_tables(const sparse_table_t<element>& x,
                   const sparse_table_t<element>& y,
                   sparse_table_t<element>& z,
                   const dim_map_t& x_dim_map,
                   const dim_map_t& y_dim_map,
                   binary_op_tag_t) {
    typename sparse_table_t<element>::dim_map_t _x_dim_map(x.arity());
    typename sparse_table_t<element>::dim_map_t _y_dim_map(y.arity());
    for (table_arity_t x_dim = 0; x_dim < x.arity(); ++x_dim) {
      assert(z.shape()[x_dim_map[x_dim]] == x.shape()[x_dim]);
      _x_dim_map[x_dim] = x_dim_map[x_dim];
    }
    for (table_arity_t y_dim = 0; y_dim < y.arity(); ++y_dim) {
      assert(z.shape()[y_dim_map[y_dim]] == y.shape()[y_dim]);
      _y_dim_map[y_dim] = y_dim_map[y_dim];
    }
    z.join(x, y, _x_dim_map, _y_dim_map, binary_op_tag_t());
  }
  */

  /**
   * Joins two tables and aggregates the elements of the resulting
   * table into a single element.
   *
   * \todo: This could be generalized to aggregate the result into a
   * table, not just a scalar.  That would permit fully general
   * combine/collapse operations for table factors, such as the
   * matrix/vector multiply used in the computation of HMM forwards
   * messages.
   *
   * @param x
   *        an input table
   * @param y
   *        an input table
   * @param join_op
   *        an object that models the binary operator concept;
   *        it is used to compute elements of the joined table
   * @param x_dim_map
   *        an object such that x_dim_map[i] gives the dimension of
   *        (the fictitious join table) z that corresponds to
   *        dimension i of x; these must be unique
   * @param y_dim_map
   *        an object such that y_dim_map[i] gives the dimension of
   *        (the fictitious join table) z that corresponds to
   *        dimension i of y; these must be unique
   * @param agg_op
   *        an object that models the binary operator concept; it
   *        is used to aggregate the elements of the joined table
   */
  /*
  template <typename binary_join_op_t,
            typename binary_agg_op_t,
            typename dim_map_t,
            typename element>
  element join_aggregate_tables(const sparse_table_t<element>& x,
                                  const sparse_table_t<element>& y,
                                  binary_join_op_t join_op,
                                  const dim_map_t& x_dim_map,
                                  const dim_map_t& y_dim_map,
                                  binary_agg_op_t agg_op) {
    typename sparse_table_t<element>::dim_map_t _x_dim_map(x.arity());
    typename sparse_table_t<element>::dim_map_t _y_dim_map(y.arity());
    for (table_arity_t x_dim = 0; x_dim < x.arity(); ++x_dim)
      _x_dim_map[x_dim] = x_dim_map[x_dim];
    for (table_arity_t y_dim = 0; y_dim < y.arity(); ++y_dim)
      _y_dim_map[y_dim] = y_dim_map[y_dim];
    return sparse_table_t<element>::join_aggregate
      (x, y, _x_dim_map, _y_dim_map, join_op, agg_op);
  }
  */

  /**
   * Joins one table into another.  Each element of x is updated to
   * become the result of applying the supplied binary operator to its
   * previous value (as the first argument) and the corresponding
   * value of y (as the second argument).
   *
   * @param x
   *        an input/output table
   * @param r
   *        an input table; each dimension of y must correspond
   *        to a dimension of x
   * @param binary_op_t
   *        a type that models the symmetric binary operator concept
   * @param y_dim_map
   *        an object such that y_dim_map[i] gives the dimension of
   *        x that corresponds to dimension i of y; these must be
   *        unique
   */
  /*
  template <typename binary_op_tag_t,
            typename dim_map_t,
            typename element>
  void join_into(sparse_table_t<element>& x,
                 const sparse_table_t<element>& y,
                 const dim_map_t& y_dim_map,
                 binary_op_tag_t) {
    typename sparse_table_t<element>::dim_map_t _y_dim_map(y.arity());
    for (table_arity_t y_dim = 0; y_dim < y.arity(); ++y_dim) {
      assert(x.shape()[y_dim_map[y_dim]] == y.shape()[y_dim]);
      _y_dim_map[y_dim] = y_dim_map[y_dim];
    }
    x.join_with(y, _y_dim_map, binary_op_tag_t());
  }
  */

  /**
   * Aggregates a sparse to yield a new table.  Each element of x is
   * aggregated into the element of z with the same indexes in
   * corresponding dimensions.  The aggregation is computed using a
   * binary function which incrementally incorporates an element of
   * x into the aggregate element of z as \f$z' \leftarrow f(z,
   * x)\f$; the aggregate is initialized with the current elements
   * of z.
   *
   * @param x
   *        an input sparse table
   * @param binary_op
   *        a binary function of elements of x
   * @param z
   *        an input/output sparse table; each dimension of z must
   *        correspond to exactly one dimension of x.
   * @param z_to_x_dim_map
   *        an object such that z_dim_map[i] gives the dimension of
   *        x that corresponds to dimension i of z; these must be
   *        unique
   */
  template <typename T, typename DimMap, typename AggOp>
  void aggregate_table(const sparse_table_t<T>& x,
                       sparse_table_t<T>& z,
                       const DimMap& z_to_x_dim_map,
                       AggOp op) {
    concept_assert((BinaryFunction<AggOp,T,T,T>));

    // If the source table has no dimensions, we are done.
    if (x.arity() == 0) {
      assert(z.arity() == 0);
      return;
    }

    // TODO: for now, this is only implemented for the case where the
    // the default element of this sparse table is the identity of the
    // binary operator (the aggregator), or when the table is empty.
    // It is possible cover the other case (when the table is not
    // empty and the default is not the identity of the aggregator),
    // but making it efficient requires some work.  The best way to do
    // it may be to keep a count for each z element of how many
    // explicit x elements map to it, and then use operator iterations
    // to incorporate the contribution from the multiplicity of
    // default elements that map into that z element.
    bool x_def_is_identity =
      op.has_left_identity && (x.get_default_elt() == op.left_identity());
    if (!x_def_is_identity) {
      // Make sure the table has no explicit elements.
      if (x.num_explicit_elts() > 0)
        throw std::runtime_error("aggregation of non-empty sparse tables with non-identity default element not yet implemented");
      // Count how many x elements map to each z element.
      unsigned long int n = x.num_elts() / z.num_elts();
      assert(n > 0);
      // Set the default value of z to be the result of the (n - 1)th
      // left-iteration of the aggregator operator applied to the
      // default element with itself.  For example, if the aggregator
      // is addition, this computes d + d * (n - 1) = d * n.
      z.set_default_elt(op.left_iterate(x.get_default_elt(),
                                        x.get_default_elt(),
                                        n - 1));
      return;
    }
    z.set_default_elt(op.left_identity());
    const table_arity_t z_arity = z.arity();
    typename sparse_table_t<T>::dim_map_t _z_to_x_dim_map(z_arity);
    for (table_arity_t z_dim = 0; z_dim < z_arity; ++z_dim) {
      assert(z.shape()[z_dim] ==
             x.shape()[z_to_x_dim_map[z_dim]]);
      _z_to_x_dim_map[z_dim] = z_to_x_dim_map[z_dim];
    }
    v_table_index z_index(z_arity, 0);
    typename sparse_table_t<T>::const_ielt_it_t it, end;
    for (boost::tie(it, end) = x.indexed_elements(); it != end; ++it) {
      const typename sparse_table_t<T>::ielt_t& x_ielt = *it;
      const v_table_index& x_index = x_ielt.first;
      const T& x_elt = x_ielt.second;
      for (table_arity_t d = 0; d < z_arity; ++d)
        z_index[d] = x_index[_z_to_x_dim_map[d]];
      T z_elt = z.get(z_index);
      z.set(z_index, op(z_elt, x_elt));
    }
  }

  //! Writes a human-readable representation of the sparse table.
  template <typename T>
  std::ostream& operator<<(std::ostream& out,
                           const sparse_table_t<T>& table) {
    typename sparse_table_t<T>::const_ielt_it_t it, end;
    for (boost::tie(it, end) = table.indexed_elements(); it != end; ++it) {
      std::ostream_iterator<table_size_t, char> output_it(out, " ");
      std::copy(it->first.begin(), it->first.end(), output_it);
      out << " " << it->second << std::endl;
    }
    if (table.uses_default())
      out << "<default> " << table.get_default_elt() << std::endl;
    return out;
  }

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#ifdef PRL_DENSE_TABLE_HPP
#include <prl/datastructure/dense_and_sparse_table.hpp>
#endif // #ifdef PRL_DENSE_TABLE_HPP

#endif // #ifndef PRL_SPARSE_TABLE_HPP
