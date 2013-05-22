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

#ifndef SILL_KD_TREE_HPP
#define SILL_KD_TREE_HPP

#include <algorithm>
#include <iterator>
#include <iostream>
#include <map>
#include <climits>
#include <queue>

#include <boost/tuple/tuple.hpp>

#include <sill/set.hpp>
#include <sill/bsp_tree.hpp>

///////////////////////////////////////////////////////////////////
// Needs to be cleaned up
///////////////////////////////////////////////////////////////////

namespace sill {

  /**
   * A kd-tree is a BSP tree that is used to store points.  It
   * supports efficient interval nearest-neighbor queries (though
   * these are not yet implemented).
   *
   * @param dimension_id_t
   *        the type used to identify the dimensions of the points
   * @param coordinate_t
   *        the type of the points' coordinates; this type must
   *        support operator< comparisons, and it must support the
   *        expression std::numeric_limits<coordinate_t>::infinity().
   * @param point_t 
   *        the type of point stored in this kd-tree; this type must
   *        be assignable and it must support the operator[] to access 
   *        its coordinates; examples include coordinate_t* and 
   *        std::vector<coordinate_t>
   *
   * \ingroup datastructure
   */
  template <typename dimension_id_t = std::size_t,
	    typename coordinate_t = double,
	    typename point_t = const coordinate_t*>
  class kd_tree_t {

  protected:

    //! The type of vector used to store the points in this kd tree.
    typedef typename std::vector<point_t> point_vec_t;

    //! The vector used to store the points internally.
    point_vec_t points;

    //! An iterator over the point vector.
    typedef typename point_vec_t::iterator point_it_t;

    /**
     * A range of points, represented by a pair of iterators.  This is
     * the data stored in each leaf of the kd tree.
     */
    struct point_range_t {
      //! An iterator to the first point in the range (inclusive).
      point_it_t begin;
      //! An iterator to the last point in the range (exclusive).
      point_it_t end;
      //! A default constructor.
      point_range_t() { }
      //! An explicit constructor.
      point_range_t(point_it_t begin,
		    point_it_t end)
	: begin(begin),
	  end(end)
      { }
    };

    /**
     * An ordered interval of coordinates; the interval is closed on
     * its minimum boundary, and open on its maximum boundary.
     */
    typedef std::pair<coordinate_t, coordinate_t> interval_t;

    /**
     * These are the traits used to make a BSP tree represent a kd
     * tree.
     */
    struct bsp_traits_t {
      
      /**
       * The representation of the space partitioned by the BSP tree.
       * This is just the number of dimensions.
       */
      typedef std::size_t space_t;

      /**
       * The type of elements of the space partitioned by the BSP
       * tree.
       */
      typedef point_t element;

      /**
       * The type of predicates used as tests in internal nodes of the
       * BSP tree.  In the case of a kd-tree, these are iso-hyperplane
       * partitions of the space, consisting of a dimension identifier
       * and coordinate in that dimension.  When viewed as a
       * predicate, partitions are satisfied by points whose
       * coordinates in the identified dimension are (strictly) less
       * than the specified coordinate value.
       */
      typedef std::pair<dimension_id_t, coordinate_t> predicate_t;

      /**
       * Represents a region of the space partitioned by the BSP tree.
       * This is a vector that maps from the dimension identifiers of
       * the space to intervals.  Each node is associated with the
       * region containing all of its descendant leaves' points.
       */
      typedef std::vector<interval_t> region_t;

      //! Each leaf node is also associated with range of points.
      typedef point_range_t leaf_data_t;

      /**
       * Returns the union space of the two supplied spaces.  This
       * operation is not supported because we don't care about
       * merging kd-trees.
       */
      static inline space_t merge_spaces(const space_t& s,
					 const space_t& t) { 
	assert(false);
	return s;
      }

      /**
       * Returns truth if the supplied region overlaps the portion of
       * the space that satisfies a predicate (or its negation).
       */
      static inline sill::pred_set_rel_t relation(const predicate_t& p,
						 const region_t& r) {
	const interval_t& interval = r[p.first];
	if (interval.second < p.second)
	  return positive_c;
	if (interval.first >= p.second)
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
	return (predicate.first < space);
      }

      //! Computes the initial node data for a given space.
      static inline region_t init_region(const space_t& space) {
	interval_t all(-std::numeric_limits<coordinate_t>::infinity(),
		       std::numeric_limits<coordinate_t>::infinity());
	return region_t(space, all);
      }

      /**
       * Splits a region into two along a supplied partition.
       *
       * @param region a region of the space
       * @param partition a partition of the space
       * @return a pair of regions, the first of which satisfies the
       *         partition predicate, and the second of which does not;
       *         one or both of these may be empty
       */
      static inline std::pair<region_t, region_t> 
      split_region(const region_t& region,
		   const predicate_t& partition) {
	// Compute the region associated with the negative halfspace.
	region_t lt_region = region;
	interval_t& lt_interval = lt_region[partition.first];
	lt_interval.second = partition.second;
	// Compute the region associated with the positive halfspace.
	region_t gte_region = region;
	interval_t& gte_interval = gte_region[partition.first];
	gte_interval.first = partition.second;
	return std::make_pair(lt_region, gte_region);
      }

      /**
       * Collapses the supplied region to a subspace.  This operation
       * is not supported because we don't care about collapsing
       * kd-trees.
       */
      static inline region_t collapse_region(const region_t& region,
					     const space_t& subspace) {
	assert(false);
      }

      /**
       * Splits leaf data into two along a supplied partition.
       *
       * @param leaf_data the data associated with a leaf
       * @param partition a partition of the space
       * @return todo
       */
      static inline std::pair<leaf_data_t, leaf_data_t> 
      split_leaf_data(const leaf_data_t& leaf_data,
		      const predicate_t& partition) {
	assert(false);
      }

      /**
       * Returns truth if the supplied element of the space satisfies
       * the partitioning predicate.  In the case of a kd-tree, the
       * elements are points and the predicates are halfspace
       * partitions; the predicate is satisfied iff the point is in
       * the negative halfspace associated with the partition.
       */
      static inline bool satisfies(const element& elt, 
				   const predicate_t& partition) {
	return elt[partition.first] < partition.second;
      }

    }; // struct kd_tree_t::bsp_traits_t

    //! A local typedef for clarity.
    typedef typename bsp_traits_t::space_t space_t;
    typedef typename bsp_traits_t::region_t region_t;
    typedef typename bsp_traits_t::predicate_t predicate_t;

    //! The type of BSP used to store the points.
    typedef bsp_tree_t<bsp_traits_t> bspt_t;

    //! The BSP tree used to store the points.
    bspt_t* bspt_ptr;

    //! Returns a mutable reference to the underlying BSP tree.
    bspt_t& get_bspt() { return *bspt_ptr; }

    //! Returns a const reference to the underlying BSP tree.
    const bspt_t& get_bspt() const { return *bspt_ptr; }

    //! The number of dimensions of the points.
    std::size_t num_dims;

    /**
     * Defines a leaf splitting strategy for recursively constructing
     * a BSP tree to index a set of points.
     */ 
    class leaf_split_strategy_t {

    protected:

      //! The number of points in each leaf node.
      std::size_t num_pts_per_leaf;

      //! Compares two points along a specified dimension.
      struct coord_comparator_t {
	dimension_id_t dim_id;
	coord_comparator_t(dimension_id_t dim_id) : dim_id(dim_id) { }
	bool operator()(const point_t& a, const point_t& b) { 
	  return a[dim_id] < b[dim_id];
	}
      }; // struct coord_comparator_t

    public:
  
      //! The type of priority assigned to leaves.
      typedef std::size_t priority_type;

      //! Constructor.
      leaf_split_strategy_t(std::size_t num_pts_per_leaf = 1) 
	: num_pts_per_leaf(num_pts_per_leaf) { }

      /**
       * Determines if and how a leaf node should be split.
       *
       * @param domain    the domain of the node
       * @param box       the bounding box of the leaf node
       * @param datum     the datum currently assigned to the leaf
       * @param partition if the leaf should be split, this is updated
       *                  to the partition used for the split
       * @param priority  if the leaf should be split, this is updated
       *                  to the priority of the split
       * @return          true if the leaf should be split, false otherwise
       */
      bool split(const space_t& space,
		 const region_t& /* region (unused) */, 
		 const point_range_t& point_range,
		 predicate_t& partition,
		 priority_type& priority,
		 point_range_t& lt_point_range,
		 point_range_t& gte_point_range) {
	std::size_t size = std::distance(point_range.begin, 
					 point_range.end);
	// If the leaf contains too few points, do not split.
	if (size <= num_pts_per_leaf) return false;
	if (space == 1)
	  partition.first = 0;
	else {
	  // Choose the split dimension as that dimension with greatest
	  // variance.
	  std::vector<coordinate_t> sum(space, 0.0);
	  std::vector<coordinate_t> sum_squared(space, 0.0);
	  for (point_it_t pt_it = point_range.begin; 
	       pt_it != point_range.end; ++pt_it) {
	    for (dimension_id_t dim_id = 0; dim_id < space; ++dim_id) {
	      coordinate_t x = (*pt_it)[dim_id];
	      sum[dim_id] += x;
	      sum_squared[dim_id] += x * x;
	    }
	  }
	  double greatest_var = 0.0; // up to a constant
	  partition.first = 0;
	  for (dimension_id_t d = 0; d < space; ++d) {
	    double var = sum_squared[d] - sum[d] * sum[d]; // up to a constant
	    if ((d == 0) || (var > greatest_var)) {
	      partition.first = d;
	      greatest_var = var;
	    }
	  }
	}
	// Choose the split value as the median value along the split
	// dimension.
	std::size_t median_pos = size / 2;
	point_it_t median_it = point_range.begin + median_pos;
	// Note we must use partial sort here (rather than the faster
	// nth_element) in order to scan backwards to check for
	// duplicates.  If we could ensure there were no duplicates,
	// nth_element would work here too.
	std::partial_sort(point_range.begin, 
			  median_it + 1,
			  point_range.end,
			  coord_comparator_t(partition.first));
	// Record the partition value.
	partition.second = (*median_it)[partition.first];
	priority = size;
	// Scan backwards to catch duplicate values.
	while (median_it != point_range.begin) {
	  point_it_t prev_it = median_it;
	  --prev_it;
	  if ((*prev_it)[partition.first] == partition.second) {
	    median_it = prev_it;
	  } else
	    break;
	}
	if (median_it == point_range.begin)
	  return false;
	// Compute the children's data.
	lt_point_range.begin = point_range.begin;
	lt_point_range.end = gte_point_range.begin = median_it;
	gte_point_range.end = point_range.end;
	return true;
      }

    }; // class kd_tree_t::leaf_split_strategy_t

  public:

    /**
     * Constructor.
     *
     * @param  num_dims
     *         the number of dimensions of the points
     * @param  begin
     * @param  end
     *         a sequence of point_t objects representing the
     *         points to be stored in the kd-tree 
     * @param  max_pts_per_leaf 
     *         the maximum number of points stored in each leaf
     * @param  max_num_leaves
     *         the maximum number of leaves of the tree; this 
     *         constraint is respected even if the maximum number
     *         of points per leaf constraint must be ignored
     */
    template <typename input_iterator_t>
    kd_tree_t(std::size_t num_dims, 
	      input_iterator_t begin,
	      input_iterator_t end,
	      std::size_t max_pts_per_leaf = 1,
	      std::size_t max_num_leaves = UINT_MAX) 
      : points(begin, end), num_dims(num_dims)
    {
      bspt_ptr = new bspt_t(bsp_traits_t(), 
			    num_dims, 
			    point_range_t(points.begin(), points.end()));
      leaf_split_strategy_t lss(max_pts_per_leaf);
      get_bspt().grow(lss, max_num_leaves - 1);
    }

    //! Destructor.
    ~kd_tree_t() { delete bspt_ptr; }

    //! Verifies the integrity of this kd tree.
    void verify() const {
      // Verify that all leaves' regions contain their points.
      typename bspt_t::const_leaf_iterator_t begin, end;
      for (boost::tie(begin, end) = get_bspt().leaves(); 
	   begin != end; ++begin) {
	const typename bspt_t::leaf_node_t& leaf = *begin;
	const point_range_t& point_range = leaf.leaf_data;
	for (point_it_t it = point_range.begin; it != point_range.end; ++it) {
	  point_t point = *it;
	  for (dimension_id_t d = 0; d < num_dims; d++) {
	    const interval_t& interval = leaf.region[d];
	    assert(point[d] >= interval.first);
	    assert(point[d] < interval.second);
	  }
	}
      }
      // Verify that search works properly.
      typedef typename point_vec_t::const_iterator const_point_it_t;
      for (const_point_it_t it = points.begin(); it != points.end(); ++it) {
	const point_t& point = *it;
	const point_range_t& point_range = get_bspt().get_leaf_data(point);
	assert(std::find(point_range.begin, 
			 point_range.end, 
			 point) != point_range.end);
      }
    }

  protected:

    /**
     * A functor which computes the negative square of the minimum
     * Euclidean distance between a supplied region and a fixed point.
     */
    class neg_squared_euclidean_distance_t : 
      public std::unary_function<typename bsp_traits_t::region_t, 
				 coordinate_t> {

    protected:

      //! The fixed point to which minimum distances are computed.
      const point_t point;

    public:

      /**
       * Constructor.
       *
       * @param point the point to which distances are computed
       */
      neg_squared_euclidean_distance_t(const point_t& point) : point(point) { }

      //! Copy constructor.
      neg_squared_euclidean_distance_t
      (const neg_squared_euclidean_distance_t& other) 
	: point(other.point) { }

      /**
       * Returns the negative square of the minimum Euclidean distance
       * between (all points in) the supplied region and the fixed
       * point.
       */
      coordinate_t 
      operator()(const typename bsp_traits_t::region_t& region) const {
	coordinate_t result(0);
	// Iterate through the dimensions.
	for (std::size_t d = 0; d < region.size(); ++d) {
	  const interval_t& interval = region[d];
	  if (point[d] < interval.first) {
	    coordinate_t dist = interval.first - point[d];
	    result += dist * dist;
	  } else if (point[d] > interval.second) {
	    coordinate_t dist = point[d] - interval.second;
	    result += dist * dist;
	  }
	}
	// We have computed the minimum squared Euclidean distance.
	// Return its negation.
	return -result;
      }

    }; // class sill::kd_tree_t::euclidean_dist_priority_t

    /**
     * A visitor used to implement the \f$k\f$ nearest neighbor search
     * algorithm.
     */
    template <typename point_output_it_t>
    class knn_visitor_t {

    protected:

      //! The point whose \f$k\f$ nearest neighbors are sought.
      point_t query;

      //! The number of dimensions of the points.
      std::size_t num_dims;
      
      //! The number of nearest neighbors sought.
      std::size_t k;
      
      /**
       * The output iterator to which the \f$k\f$ nearest neighbors
       * are written.
       */
      point_output_it_t output_it;

      /**
       * A priority queue of at most \f$k\f$ points that are closest
       * to the query point.  The points are prioritized by squared
       * Euclidean distance to the query point, so that the farthest
       * point can be accessed (and perhaps replaced) efficiently.
       */
      std::priority_queue<std::pair<coordinate_t, point_t> > k_best;

      /**
       * Examines a point for possible inclusion into the set of the
       * \f$k\f$ nearest neighbors (seen so far).
       */
      void examine_point(const point_t& point) {
	// Compute the squared Euclidean distance between this point
	// and the query point.
	coordinate_t squared_euclidean_distance(0);
	for (std::size_t d = 0; d < num_dims; ++d) {
	  coordinate_t x = point[d] - query[d];
	  squared_euclidean_distance += (x * x);
	}
	// If we have not yet found k points, enqueue this point in
	// the priority queue and return.
	if (k_best.size() < k) {
	  k_best.push(std::make_pair(squared_euclidean_distance, point));
	  return;
	}
	// If this point is closer than the farthest point found so
	// far, replace the farthest point with this one.
	coordinate_t 
	  squared_euclidean_dist_to_farthest_point = k_best.top().first;
	if (squared_euclidean_distance < 
	    squared_euclidean_dist_to_farthest_point) {
	  k_best.pop();
	  k_best.push(std::make_pair(squared_euclidean_distance, point));
	  return;
	}
	// Otherwise, ignore this point.
	return;
      }

    public:

      /**
       * Constructor.
       *
       * @param query
       *        The point whose \f$k\f$ nearest neighbors are sought.
       * @param num_dims
       *        the number of dimensions of the points
       * @param k
       *        the number of closest neighbors to return
       * @param output_it 
       *        The output iterator to which the \f$k\f$ nearest 
       *        neighbors are written.
       */
      knn_visitor_t(const point_t& query,
		    std::size_t num_dims,
		    std::size_t k,
		    point_output_it_t output_it) 
	: query(query), num_dims(num_dims), k(k), output_it(output_it) { }

      /**
       * Visits a leaf node of the kd tree, possibly recording some of
       * its points as potential nearest neighbors.
       *
       * @param  leaf 
       *         A leaf of the kd tree; these are supplied in order of 
       *         increasing Euclidean distance to the query point
       * @param  neg_squared_euclidean_dist
       *         The negative square of the minimum Euclidean distance
       *         from the supplied leaf to the query point.
       */
      void visit(const typename bspt_t::leaf_node_t& leaf,
		 coordinate_t neg_squared_euclidean_dist) {
	// Examine the points in this leaf.
	const point_range_t& point_range = leaf.leaf_data;
	for (point_it_t it = point_range.begin; it != point_range.end; ++it)
	  examine_point(*it);	
      }

      /**
       *
       */
      bool can_prune(coordinate_t neg_squared_euclidean_dist) const {
	// If we have collected k points and the farthest of these
	// points is closer to the query point than the supplied leaf
	// node, it can be pruned.
	coordinate_t 
	  squared_euclidean_dist_to_leaf = -neg_squared_euclidean_dist;
	coordinate_t 
	  squared_euclidean_dist_to_farthest_point = k_best.top().first;
	if ((k_best.size() == k) &&
	    (squared_euclidean_dist_to_leaf >
	     squared_euclidean_dist_to_farthest_point)) 
	  return true;
	else
	  return false;
      }

      /**
       *
       */
      void finalize() {
	// Report the k nearest neighbors.
	while (!k_best.empty()) {
	  output_it = k_best.top().second;
	  k_best.pop();
	  ++output_it;
	}
      }

    }; // class sill::kd_tree_t::knn_visitor_t

  public:

    /**
     * Finds the \f$k\f$ points in this kd-tree that are closest to
     * the supplied point (in Euclidean distance).  These points are
     * written to an output iterator.
     *
     * @param  point 
     *         the point whose \f$k\f$ nearest neighbors are sought
     * @param  k
     *         the number of closest neighbors to return
     * @param  output
     *         an output iterator to which the \f$k\f$ nearest neighbors
     *         are written
     */
    template <typename point_output_it_t>
    void nearest_neighbors(const point_t& point,
			   std::size_t k,
			   point_output_it_t output) const {
      neg_squared_euclidean_distance_t region_priority_function(point);
      knn_visitor_t<point_output_it_t> visitor(point, num_dims, k, output);
      get_bspt().prioritized_visit(region_priority_function, visitor);
      visitor.finalize();
    }

    /**
     * Writes out the leaves of the index; each leaf is written as a
     * single line containing the bounding box coordinates and the
     * (single) point contained in the leaf.
     */
    template <typename ostream_t>
    void report(ostream_t& out) const {
      typename bspt_t::const_leaf_iterator_t begin, end;
      for (boost::tie(begin, end) = get_bspt().leaves(); 
	   begin != end; ++begin) {
	const typename bspt_t::leaf_node_t& leaf = *begin;
	// Write out the boundaries of this leaf's region.
	for (dimension_id_t d = 0; d < num_dims; d++) {
	  const interval_t& interval = leaf.region[d];
	  out << interval.first << " " << interval.second << " ";
	}
	const point_range_t& point_range = leaf.leaf_data;
	// Write out the number of points in this leaf.
	out << std::distance(point_range.begin, point_range.end) << " ";
	// Write out the points in this leaf.
	for (point_it_t it = point_range.begin; 
	     it != point_range.end; ++it) {
	  const point_t point = *it;
	  out << "(";
	  for (dimension_id_t d = 0; d < num_dims; d++) {
	    const interval_t& interval = leaf.region[d];
	    assert(interval.first <= point[d]);
	    assert(point[d] < interval.second);
	    out << point[d];
	    if (d < num_dims - 1)
	      out << ", ";
	  }
	  out << ")";
	}
	out << std::endl;
      }
    }
	
  }; // end of class: kd_tree_t

} // namespace sill 

#endif // #ifndef SILL_KD_TREE_HPP
