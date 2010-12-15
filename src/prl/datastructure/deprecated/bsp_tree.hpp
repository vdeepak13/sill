
#ifndef PRL_BSP_TREE_HPP
#define PRL_BSP_TREE_HPP

#include <assert.h>
#include <limits>
#include <set>
#include <map>
#include <queue>
#include <iterator>
#include <algorithm>
#include <utility>

#include <boost/tuple/tuple.hpp>

////////////////////////////////////////////////////////////////////
// Needs clean-up
///////////////////////////////////////////////////////////////////

namespace prl {

  /**
   * Describes the relationship between a set and a Boolean predicate
   * over its members.
   */
  enum pred_set_rel_t {
    positive_c, /*!< All members of the set satisfy the predicate. */
    negative_c, /*!< No members of the set satisfy the predicate.  */
    both_c      /*!< Some members of the set satisfy the predicate  
		 *   and some members do not.                      */
  };

  /**
   * A BSP tree is a binary tree that represents a partitioning of
   * some set.  Each interior node is associated with a Boolean
   * predicate on set members and each leaf node is associated with a
   * partition of the set. A kd-tree is an example of a BSP tree where
   * the set is \f$R^n\f$, the Boolean predicates are axis-aligned
   * hyperplanes, and the partitions are axis-aligned
   * hyper-rectangles.  The leaves of BSP trees are often associated
   * with some data; e.g., kd-trees are often used to index a set of
   * points, and in this case each leaf is associated with the points
   * that lie in its partition of the space.
   *
   * For details on this data structure see:
   *
   * @article{NaylorAmanatidesThibault1990a,
   *   Title = {Merging BSP Trees Yields Polyhedral Set Operations},
   *   Author = {Bruce Naylor and John Amanatides and William Thibault},
   *   Journal = {Computer Graphics},
   *   Month = {August},
   *   Number = {4},
   *   Pages = {115--124},
   *   Volume = {24},
   *   Year = {1990}
   * }
   *
   * @param bsp_traits_t
   *        The type of traits used to structure the tree.  This
   *        includes information about the space spanned by the tree,
   *        the predicates used to partition the space, and the
   *        regions that represent the partitioning. 
   *
   * \ingroup datastructure
   */
  template <typename bsp_traits_t>
  class bsp_tree_t {

  public:

    //! A typedef for the type of space in which the BSP tree is defined.
    typedef typename bsp_traits_t::space_t space_t;

    //! A typedef for the type of elements in the space.
    typedef typename bsp_traits_t::element element;

    //! A typedef for the partition type used to split the space.
    typedef typename bsp_traits_t::predicate_t predicate_t;

    /**
     * A typedef for the type of data stored at all nodes of the tree.
     * For example, in a kd-tree, this could be the bounding box of
     * the space enclosed by the node and its descendants.
     */
    typedef typename bsp_traits_t::region_t region_t;

    /**
     * A typedef for the type of data stored at all leaves of the
     * tree.  For example, in a kd-tree, this could be the set of
     * points stored in the leaf.
     */
    typedef typename bsp_traits_t::leaf_data_t leaf_data_t;

    // Forward declarations.
    struct leaf_node_t;
    struct interior_node_t;
  
    //! A BSP tree node.
    struct node_t {
      
      //! The data associated with this node.
      region_t region;
    
      /**
       * The type of this node.  Nodes are either leaves (with no
       * children) or internal nodes (with two children).
       */
      enum type_t {leaf_c, interior_c} type;

      //! The parent of this node.  This is NULL for the root node.
      interior_node_t* parent;

      //! Returns true iff this node is a leaf.
      inline bool is_leaf() const {
	return (type == leaf_c);
      }

      //! Returns true iff this node is an interior node.
      inline bool is_interior() const {
	return (type == interior_c);
      }

      //! Casts this node as a leaf.
      inline leaf_node_t* as_leaf() {
	assert(type == leaf_c);
	return static_cast<leaf_node_t*>(this);
      }

      //! Casts this node as a const leaf.
      inline const leaf_node_t* as_leaf() const {
	assert(type == leaf_c);
	return static_cast<const leaf_node_t*>(this);
      }

      //! Casts this node as an interior node.
      inline interior_node_t* as_interior() {
	assert(type == interior_c);
	return static_cast<interior_node_t*>(this);
      }

      //! Casts this node as a const interior node.
      inline const interior_node_t* as_interior() const {
	assert(type == interior_c);
	return static_cast<const interior_node_t*>(this);
      }
      
      //! Returns true iff this node is the false child of some parent.
      inline bool is_false_child() const; // defined below
      
      //! Returns true iff this node is the true child of some parent.
      inline bool is_true_child() const; // defined below

    }; // struct node_t

    //! An internal (non-leaf) node, which has exactly two children.
    struct interior_node_t : public node_t {
      
      /**
       * The predicate that distinguishes the spaces associated with
       * this node's children.
       */
      predicate_t split;
    
      /**
       * The child associated with the set members for which the split
       * predicate is true.
       */
      node_t* true_child;
    
      /**
       * The child associated with the set members for which the split
       * predicate is true.
       */
      node_t* false_child;

      //! Constructor.
      interior_node_t() : node_t() { }
    
    }; // struct interior_node_t

    //! A leaf node, which has no children.
    struct leaf_node_t : public node_t {
      
      //! The leaf data.
      leaf_data_t leaf_data;
      
    };

  protected:

    //! The traits of this BSP.
    bsp_traits_t traits;
    
    //! The space over which the BSP tree is defined.
    space_t space;
    
    /**
     * The root node of the BSP tree.  This may be a leaf or an
     * interior node.
     */
    node_t* root;

    //! Creates a new leaf node without inserting it in the tree.
    leaf_node_t* new_leaf(const region_t& region,
			  const leaf_data_t& data) const {
      leaf_node_t* leaf = new leaf_node_t();
      assert(leaf != NULL);
      leaf->region = region;
      leaf->type = node_t::leaf_c;
      leaf->leaf_data = data;
      leaf->parent = NULL;
      return leaf;
    }

    //! Deletes a leaf node without removing it from the tree.
    void delete_leaf(leaf_node_t* leaf) const {
      delete leaf;
    }

    /**
     * Creates a new interior node without inserting it in the tree.
     * The regions associated with the nodes are left as-is.
     */
    interior_node_t* new_interior_node(const region_t& region,
				       const predicate_t& split,
				       node_t* true_child,
				       node_t* false_child) const {
      interior_node_t* int_node = new interior_node_t();
      assert(int_node != NULL);
      int_node->parent = NULL;
      int_node->region = region;
      int_node->type = node_t::interior_c;
      int_node->split = split;
      int_node->true_child = true_child;
      int_node->false_child = false_child;
      true_child->parent = false_child->parent = int_node;
      return int_node;
    }

    //! Deletes an interior node without removing it from the tree.
    void delete_interior_node(interior_node_t* int_node) const {
      delete int_node;
    }

    /**
     * Returns a reference to the parent's pointer to the child.  If the
     * supplied node is the root, then this method returns the address
     * of the root pointer.
     */
    node_t** child_ref(node_t* node) {
      if (node->parent == NULL) {
	return &root;
      } else if (node->parent->true_child == node) {
	return &(node->parent->true_child);
      } else {
	return &(node->parent->false_child);
      }
    }
  
    /**
     * Replaces a node in the BSP tree with another node.  No memory
     * (de)allocation is performed.
     */
    void replace(node_t* original_node, node_t* new_node) {
      node_t** child_ref = child_ref(original_node);
      *child_ref = new_node;
      new_node->parent = original_node->parent;
    }

    //! Finds the leaf containing the supplied element.
    leaf_node_t* get_leaf(const element& elt) {
      node_t* cur_node = root;
      while (cur_node->type == node_t::interior_c) {
	interior_node_t* int_node = cur_node->as_interior();
	if (traits.satisfies(elt, int_node->split))
	  cur_node = int_node->true_child;
	else
	  cur_node = int_node->false_child;
      }
      return cur_node->as_leaf();
    }

    //! Finds the leaf containing the supplied element.
    const leaf_node_t* get_leaf(const element& elt) const {
      const node_t* cur_node = root;
      while (cur_node->type == node_t::interior_c) {
	const interior_node_t* int_node = cur_node->as_interior();
	if (traits.satisfies(elt, int_node->split))
	  cur_node = int_node->true_child;
	else
	  cur_node = int_node->false_child;
      }
      return cur_node->as_leaf();
    }

    //! Removes node and all of its descendants.
    void delete_subtree(node_t* node) const {
      if (node->type == node_t::leaf_c)
	delete_leaf(node->as_leaf());
      else {
	interior_node_t* int_node = node->as_interior();
	delete_subtree(int_node->true_child);
	delete_subtree(int_node->false_child);
	delete node;
      }
    }

    //! Clones a subtree.
    node_t* clone(const node_t* node) const {
      if (node->type == node_t::leaf_c) {
	const leaf_node_t* leaf = node->as_leaf();
	return new_leaf(leaf->region, leaf->leaf_data);
      } else {
	const interior_node_t* int_node = node->as_interior();
	return new_interior_node(int_node->region,
				 int_node->split,
				 clone(int_node->true_child),
				 clone(int_node->false_child));
      }
    }

    /**
     * Recomputes the regions associated with a node's descendants
     * using the split_region operation provided by the traits.  If
     * this node has no parent (it is the root), then its region is
     * also initialized using the trait's init_region method.
     */
    static void recompute_region(node_t* node, 
				 const bsp_traits_t& traits,
				 const space_t& space) {
      if (node->parent == NULL)
	// This is the root.
	node->region = traits.init_region(space);
      if (node->is_leaf()) 
	return;
      interior_node_t* int_node = node->as_interior();
      boost::tie(int_node->true_child->region,
		 int_node->false_child->region) =	
	traits.split_region(node->region, int_node->split);
      recompute_region(int_node->true_child, traits, space);
      recompute_region(int_node->false_child, traits, space);
    }

    /**
     * Replaces a leaf node with a new internal node that has two
     * child leaves.  The original leaf node is reused as a child of
     * the new internal node (and is therefore not deleted/allocated).
     * The caller assumes responsiblity for the (two) nodes allocated
     * in this method.
     *
     * @param  leaf      
     *         the leaf node to be split 
     * @param  split     
     *         the predicate defining the partition
     * @param  true_data  
     *         the data associated with the child satisfying split
     * @param  false_data  
     *         the data associated with the child not satisfying split
     * @return the new interior node 
     */
    interior_node_t* split_leaf(leaf_node_t* leaf,
				const predicate_t& split,
				const leaf_data_t& true_data,
				const leaf_data_t& false_data) {
      // Save the leaf's original parent, child reference, and region.
      interior_node_t* const parent = leaf->parent;
      node_t** ptr_to_leaf_ptr = child_ref(leaf);
      const region_t& region = leaf->region;
      // Reuse the input leaf as the less-than leaf.
      leaf_node_t* true_child = leaf;
      true_child->leaf_data = true_data;
      // Make the other leaf.
      leaf_node_t* false_child = new_leaf(region_t(), false_data);
      // Make the new interior node.  It's region is the same as the
      // former leaf.
      interior_node_t* int_node = 
	new_interior_node(region, split, true_child, false_child);
      // Update the regions of the children by splitting the parent
      // region.
      boost::tie(true_child->region, false_child->region) = 
	traits.split_region(region, split);
      // Replace the leaf with the new node.
      int_node->parent = parent;
      *ptr_to_leaf_ptr = int_node;
      // Return the new interior node.
      return int_node;
    }

    /**
     * Partition the subtree rooted at a node into two subtrees.  The
     * caller assumes responsibility for the nodes allocated in this
     * method.  The original subtree is not updated.
     *
     * @param node         the root of the subtree to be partitioned
     * @param split        the predicate on which the tree 
     *                     must be partitioned
     * @return             a pair of nodes that are the roots of the 
     *                     true and false subtrees
     */
    std::pair<node_t*, node_t*> split_tree(const node_t* node,
					   const predicate_t& split) const {
      region_t true_region, false_region;
      boost::tie(true_region, false_region) = 
	traits.split_region(node->region, split);
      if (node->type == node_t::leaf_c) {
	const leaf_node_t* leaf = node->as_leaf();
	// Compute the data associated with the true and false leaves.
	leaf_data_t true_data, false_data;
	boost::tie(true_data, false_data) = 
	  traits.split_leaf_data(leaf->leaf_data, split);
	// Create the two leaves and return them.
	leaf_node_t* true_child = new_leaf(true_region, true_data);
	leaf_node_t* false_child = new_leaf(false_region, false_data);
	return std::make_pair(true_child, false_child);
      } else { // The node to be partitioned is interior.
	const interior_node_t* int_node = node->as_interior();
	// Determine the relationship between the split predicate and
	// the regions associated with the children of this node.
	const pred_set_rel_t true_child_rel = 
	  traits.relation(split, int_node->true_child->region);
	const pred_set_rel_t false_child_rel = 
	  traits.relation(split, int_node->false_child->region);
	// Depending upon these relationships, we may have to split or
	// clone children.  There are seven separate cases.  (There
	// are actually nine cases, but the cases where both relations
	// are positive or negative are impossible; these cases
	// violate the precondition that the split partitions this
	// node's region.)
	if ((true_child_rel == positive_c) && 
	    (false_child_rel == negative_c)) {
	  // The two predicates are equal.  Simply clone the two
	  // subtrees.
	  return std::make_pair(clone(int_node->true_child),
				clone(int_node->false_child));
	} else if ((true_child_rel == negative_c) && 
		   (false_child_rel == positive_c)) {
	  // The two predicates are opposite.  Simply clone the two
	  // subtrees (and return them in the reverse order).
	  return std::make_pair(clone(int_node->false_child),
				clone(int_node->true_child));
	} else if ((true_child_rel == positive_c) && 
		   (false_child_rel == both_c)) {
	  /* The true child of this node satisfies the new split
	   * predicate but the false child of this node does not
	   * satisfy its negation.  Partition the false child and
	   * clone the true child.
	   *
	   * +-------+---+--------+   :: = region satisfying new 
	   * |       ::::|::::::::|        split predicate
	   * |       ::::|::::::::|   |  = current split predicate;
	   * |   A   ::B:|:::C::::|        area to right satisfies 
	   * |       ::::|::::::::|        the predicate; area to
	   * |       ::::|::::::::|        left does not satisfy it
	   * +-------+---+--------+
	   * false (F) <-|-> true (T)
	   *
	   *           new split
	   *             /  \
	   *          F /    \ T
	   *           /      \
	   *          A   current split
	   *                  /  \
	   *               F /    \ T
	   *                /      \
	   *               B        C
	   */ 
	  std::pair<node_t*, node_t*> false_partitioned = 
	    split_tree(int_node->false_child, split);
	  return std::make_pair(new_interior_node(true_region,
						  int_node->split,
						  clone(int_node->true_child),
						  false_partitioned.first),
				false_partitioned.second);
	} else if ((true_child_rel == both_c) && 
		   (false_child_rel == positive_c)) {
	  /* The false child of this node satisfies the new split
	   * predicate but the true child of this node does not
	   * satisfy its negation.  Partition the true child and
	   * clone the false child.
	   *
	   * +-------+------+-----+   :: = region satisfying new 
	   * |:::::::|:::::::     |        split predicate
	   * |:::::::|:::::::     |   |  = current split predicate;
	   * |:: A:::|:::B:::  C  |        area to right satisfies 
	   * |:::::::|:::::::     |        the predicate; area to
	   * |:::::::|:::::::     |        left does not satisfy it
	   * +-------+------+-----+
	   * false <-|-> true 
	   *
	   *           new split
	   *             /  \
	   *          F /    \ T
	   *           /      \
	   *          C   current split
	   *                  /  \
	   *               F /    \ T
	   *                /      \
	   *               A        B
	   */ 
	  std::pair<node_t*, node_t*> true_partitioned = 
	    split_tree(int_node->true_child, split);
	  return 
	    std::make_pair(new_interior_node(true_region,
					     int_node->split,
					     true_partitioned.first,
					     clone(int_node->false_child)),
			   true_partitioned.second);
	} else if ((true_child_rel == negative_c) && 
		   (false_child_rel == both_c)) {
	  /* The true child of this node satisfies the new split
	   * predicate's negation but the false child of this node
	   * does not satisfy it.  Partition the false child and clone
	   * the true child.
	   *
	   * +-------+------+-----+   :: = region satisfying new 
	   * |::::::::      |     |        split predicate
	   * |::::::::      |     |   |  = current split predicate;
	   * |:: A::::  B   |  C  |        area to right satisfies 
	   * |::::::::      |     |        the predicate; area to
	   * |::::::::      |     |        left does not satisfy it
	   * +-------+------+-----+
	   *        false <-|-> true 
	   *
	   *           new split
	   *             /  \
	   *          F /    \ T
	   *           /      \
	   *    current split  A
	   *         /  \
	   *      F /    \ T
	   *       /      \
	   *      B        C
	   */ 
	  std::pair<node_t*, node_t*> false_partitioned = 
	    split_tree(int_node->false_child, split);
	  return std::make_pair(false_partitioned.first,
				new_interior_node(false_region,
						  int_node->split,
						  clone(int_node->true_child),
						  false_partitioned.second));
	} else if ((true_child_rel == both_c) && 
		   (false_child_rel == negative_c)) {
	  /* The false child of this node satisfies the new split
	   * predicate's negation but the true child of this node does
	   * not satisfy it.  Partition the true child and clone the
	   * false child.
	   *
	   * +-------+------+-----+   :: = region satisfying new 
	   * |       |      ::::::|        split predicate
	   * |       |      ::::::|   |  = current split predicate;
	   * |   A   |   B  :::C::|        area to right satisfies 
	   * |       |      ::::::|        the predicate; area to
	   * |       |      ::::::|        left does not satisfy it
	   * +-------+------+-----+
	   * false <-|-> true 
	   *
	   *           new split
	   *             /  \
	   *          F /    \ T
	   *           /      \
	   *    current split  C
	   *         /  \
	   *      F /    \ T
	   *       /      \
	   *      B        A
	   */ 
	  std::pair<node_t*, node_t*> true_partitioned = 
	    split_tree(int_node->true_child, split);
	  return 
	    std::make_pair(true_partitioned.first,
			   new_interior_node(false_region,
					     int_node->split,
					     true_partitioned.second,
					     clone(int_node->false_child)));
	} else {
	  assert(true_child_rel == both_c);
	  assert(false_child_rel == both_c);
	  /* The two predicates do not satisfy any implications.  Both
	   * children must be split.
	   *
	   * +-------+-------+   :: = region satisfying new 
	   * |:::::::|:::::::|        split predicate
	   * |:::A:::|:::B:::|   |  = current split predicate;
	   * |:::::::|:::::::|        area to right satisfies 
	   * |       |       |        the predicate; area to
	   * |   C   |   D   |        left does not satisfy it
	   * |       |       |        
	   * +-------+-------+
	   * false <-|-> true 
	   */
	  std::pair<node_t*, node_t*> true_partitioned = 
	    split_tree(int_node->true_child, split);
	  std::pair<node_t*, node_t*> false_partitioned = 
	    split_tree(int_node->false_child, split);
	  return std::make_pair(new_interior_node(true_region,
						  int_node->split,
						  true_partitioned.first,
						  false_partitioned.first),
				new_interior_node(false_region,
						  int_node->split,
						  true_partitioned.second,
						  false_partitioned.second));
	}
      }
      // For some reason GCC 4.0 complains that this function can
      // exit without returning a value.  This fixes that.
      assert(false);
    }
    
    /**
     * Merges a leaf with a subtree using the supplied merge operator.
     * The caller assumes responsibility for the nodes allocated in
     * this method.
     */
    template <typename merge_leaf_data_op_t>
    node_t* merge_leaf_with_tree(const leaf_node_t* leaf,
				 const node_t* node,
				 merge_leaf_data_op_t merge_op) const {
      // Clone the input tree, and then update its leaves' data.
      node_t* copy = clone(node);
      for (leaf_iterator_t it = leaf_iterator_t(copy); 
	   it != leaf_iterator_t(); ++it) {
	leaf_node_t* cur_leaf = &(*it);
	// Be careful to supply the cloned leaf's data after the
	// original leaf's data to preserve the order of arguments.
	cur_leaf->leaf_data = merge_op(leaf->region,
				       leaf->leaf_data,
				       cur_leaf->region, 
				       cur_leaf->leaf_data);
      }
      return copy;
    }

    /**
     * Merges a subtree with a leaf using the supplied merge operator.
     * The caller assumes responsibility for the nodes allocated in
     * this method.
     */
    template <typename merge_leaf_data_op_t>
    node_t* merge_tree_with_leaf(const node_t* node,
				 const leaf_node_t* leaf,
				 merge_leaf_data_op_t merge_op) const {
      // Clone the input tree, and then update its leaves' data.
      node_t* copy = clone(node);
      for (leaf_iterator_t it = leaf_iterator_t(copy); 
	   it != leaf_iterator_t(); ++it) {
	leaf_node_t* cur_leaf = &(*it);
	// Be careful to supply the cloned leaf's data before the
	// original leaf's data to preserve the order of arguments.
	cur_leaf->leaf_data = merge_op(cur_leaf->region, 
				       cur_leaf->leaf_data, 
				       leaf->region,
				       leaf->leaf_data);
      }
      return copy;
    }

    /**
     * Merges two subtrees using the supplied data merge operator.
     * The caller assumes responsibility for the nodes allocated in
     * this method.  A precondition of this method is that both nodes
     * span the same set of elements.  This method does not update the
     * regions; this is the responsibility of the caller.
     */
    template <typename merge_leaf_data_op_t>
    node_t* merge_trees(const node_t* node_1,
			const node_t* node_2,
			merge_leaf_data_op_t merge_op) const {
      if (node_1->type == node_t::leaf_c)
	return merge_leaf_with_tree(node_1->as_leaf(), node_2, merge_op);
      else if (node_2->type == node_t::leaf_c)
	return merge_tree_with_leaf(node_1, node_2->as_leaf(), merge_op);
      else {
	// Both nodes are interior nodes.
	const interior_node_t* int_node_1 = node_1->as_interior();
	const interior_node_t* int_node_2 = node_2->as_interior();
	// Determine the relationship between the split predicate of
	// the first node and the regions associated with the children
	// of the second node.
	const pred_set_rel_t true_child_rel = 
	  traits.relation(int_node_1->split, int_node_2->true_child->region);
	const pred_set_rel_t false_child_rel = 
	  traits.relation(int_node_1->split, int_node_2->false_child->region);
	// If the nodes' split predicates are the same (or opposite),
	// simply merge their children.
	if ((true_child_rel == positive_c) &&
	    (false_child_rel == negative_c)) {
	  node_t* node = new_interior_node(int_node_1->region,
					   int_node_1->split,
					   merge_trees(int_node_1->true_child,
						       int_node_2->true_child,
						       merge_op),
					   merge_trees(int_node_1->false_child,
						       int_node_2->false_child,
						       merge_op));
	  return node;
	} else if ((true_child_rel == negative_c) &&
		   (false_child_rel == positive_c)) {
	  node_t* node = new_interior_node(int_node_1->region,
					   int_node_1->split,
					   merge_trees(int_node_1->true_child,
						       int_node_2->false_child,
						       merge_op),
					   merge_trees(int_node_1->false_child,
						       int_node_2->true_child,
						       merge_op));
	  return node;
	}
	// Otherwise, use the split of the first tree to partition the
	// second tree.
	std::pair<node_t*, node_t*> partitioned_node_2 = 
	  split_tree(int_node_2, int_node_1->split);
	// Merge each child of the first tree with its matching
	// partition of the second tree.
	node_t* merged_true_child = merge_trees(int_node_1->true_child,
						partitioned_node_2.first,
						merge_op);
	node_t* merged_false_child = merge_trees(int_node_1->false_child,
						 partitioned_node_2.second,
						 merge_op);
	// Deallocate the (intermediate) partitioned subtrees.
	delete_subtree(partitioned_node_2.first);
	delete_subtree(partitioned_node_2.second);
	// Make a parent of these merged trees.
	node_t *node = new_interior_node(int_node_1->region, 
					 int_node_1->split,
					 merged_true_child, 
					 merged_false_child);
	return node;
      }
    }

    /**
     * Collapses this BSP tree over a subset of its dimensions.  The
     * original BSP tree is left unchanged.  The caller is responsible
     * for the deallocation of the returned subtree.  This method does
     * not update the regions; that is the responsibility of the
     * caller.
     *
     * @param  subspace 
     *         the subspace to which the BSP tree is to be collapsed
     * @param  unary_op 
     *         the operator applied to collapse the data at the leaf
     *         nodes
     * @param  binary_op 
     *         the operator applied to merge two child subtrees of a
     *         parent that splits on a predicate that is not defined
     *         in the subspace
     * @param  node 
     *         the subtree to be collapsed to the subspace
     * @return the subtree, collapsed to the supplied subspace
     */
    template <typename collapse_unary_op_t,
	      typename collapse_binary_op_t>
    node_t* collapse(const space_t& subspace,
		     collapse_unary_op_t unary_op,
		     collapse_binary_op_t binary_op,
		     const node_t* node) const {
      if (node->type == node_t::leaf_c) {
	const leaf_node_t* leaf = node->as_leaf();
	leaf_data_t collapsed_leaf_data = 
	  unary_op(leaf->region, leaf->leaf_data, subspace);
	return new_leaf(traits.collapse_region(leaf->region, subspace), 
			collapsed_leaf_data);
      } else {
	const interior_node_t* int_node = node->as_interior();
	// Apply the operation recursively to the children.
	node_t* true_child = 
	  collapse(subspace, unary_op, binary_op, int_node->true_child);
	node_t* false_child = 
	  collapse(subspace, unary_op, binary_op, int_node->false_child);
	// Determine if this node's splitting predicate 
	if (!traits.is_defined(int_node->split, subspace)) {
	  // This node splits on a predicate that is not defined in
	  // the subspace.  Merge the two subtrees using the binary
	  // collapse operator.
	  node_t* merged = merge_trees(true_child, false_child, binary_op);
	  // Deallocate the merged subtrees.
	  delete_subtree(true_child);
	  delete_subtree(false_child);
	  // Return the merged tree.
	  return merged;
	} else {
	  // This node splits on a predicate that is defined in the
	  // subspace.  Return an interior node with the same split
	  // that has the collapsed children.
	  return new_interior_node(traits.collapse_region(int_node->region,
							  subspace),
				   int_node->split,
				   true_child, false_child);
	}
      }
    }

    /**
     * Restricts this BSP tree to a subset of its dimensions.  The
     * original BSP tree is left unchanged.  The caller is responsible
     * for the deallocation of the returned subtree.
     *
     * @param  subspace 
     *         the subspace to which the BSP tree is to be restricted
     * @param  element
     *         An element that specifies how internal nodes whose
     *         predicates are not defined in the subspace are treated.
     *         For all such nodes, the element is tested against the 
     *         predicate and the subtree that agrees with the result
     *         is retained; the other subtree is discarded.
     * @param  node 
     *         the subtree to be collapsed to the subspace
     * @return the subtree, collapsed to the supplied subspace
     */
    node_t* restrict(const space_t& subspace,
		     const element& element,
		     const node_t* node) const {
      if (node->type == node_t::leaf_c) {
	const leaf_node_t* leaf = node->as_leaf();
	return new_leaf(traits.collapse_region(leaf->region, subspace), 
			leaf->leaf_data);
      } else {
	const interior_node_t* int_node = node->as_interior();
	// Determine if this node's splitting predicate is defined in
	// the subspace.
	if (!traits.is_defined(int_node->split, subspace)) {
	  // This node splits on a predicate that is not defined in
	  // the subspace.  Determine which branch is retained.
	  if (traits.satisfies(element, int_node->split)) 
	    return restrict(subspace, element, int_node->true_child);
	  else
	    return restrict(subspace, element, int_node->false_child);
	} else {
	  // This node splits on a predicate that is defined in the
	  // subspace.  Return an interior node with the same split
	  // that has the restricted children.
	  return new_interior_node(traits.collapse_region(int_node->region,
							  subspace),
				   int_node->split,
				   restrict(subspace, element, 
					    int_node->true_child),
				   restrict(subspace, element, 
					    int_node->false_child));
	}
      }
    }

  public:

    /**
     * Default constructor.  The tree consists of a single (leaf) node.
     *
     * @param  space  
     *         the space partitioned by the BSP tree
     * @param  leaf_data
     *         the leaf data associated with the leaf node
     */
    bsp_tree_t(bsp_traits_t traits = bsp_traits_t(),
	       space_t space = space_t(),
	       leaf_data_t leaf_data = leaf_data_t()) 
      : traits(traits), space(space)
    {
      root = new_leaf(traits.init_region(space), leaf_data);
    }

    /**
     * Stump constructor.  The tree consists of a single interior node
     * with two subtrees which are obtained by copying the supplied
     * trees.  (The regions in these copies are recomputed using the
     * split_region method of the traits.)
     *
     * @param  space  
     *         the space partitioned by the BSP tree
     * @param  leaf_data
     *         the leaf data associated with the leaf node
     */
    bsp_tree_t(bsp_traits_t traits,
	       space_t space,
	       predicate_t predicate,
	       const bsp_tree_t& true_tree,
	       const bsp_tree_t& false_tree) 
      : traits(traits), space(space)
    {
      this->root = new_interior_node(region_t(),
				     predicate,
				     clone(true_tree.root),
				     clone(false_tree.root));
      recompute_region(root, traits, space);
    }

    /**
     * Destructor.
     */
    ~bsp_tree_t() {
      delete_subtree(root);
    }

  protected:

    /**
     * A prioritized split represents a potential split of a leaf node,
     * along with a priority that represents how advantageous the split
     * is.  These structures are used to grow BSP trees.
     *
     * @see #grow
     */
    template <typename priority_type>
    struct prioritized_split_t {
      priority_type priority;
      leaf_node_t* leaf;
      predicate_t split;
      leaf_data_t true_data;
      leaf_data_t false_data;
      prioritized_split_t(priority_type priority,
			  leaf_node_t* leaf,
			  predicate_t split,
			  leaf_data_t& true_data,
			  leaf_data_t& false_data) 
	: priority(priority),
	  leaf(leaf),
	  split(split),
	  true_data(true_data),
	  false_data(false_data) 
      { }
      //! Orders prioritized splits by their priority.
      bool operator<(const prioritized_split_t& s) const {
	return priority < s.priority;
      }
    };

  public:

    /**
     * Recursively builds this BSP tree by iteratively splitting leaves
     * according to the supplied strategy.
     */
    template <typename leaf_split_strategy_t>
    void grow(leaf_split_strategy_t& lss,
	      unsigned int max_num_splits = UINT_MAX) {
      // Build a priority queue of the current leaves.
      typedef typename leaf_split_strategy_t::priority_type priority_type;
      typedef prioritized_split_t<priority_type> pq_entry_t;
      typedef typename std::priority_queue<pq_entry_t> split_queue_t;
      split_queue_t pq;
      predicate_t split;
      priority_type priority;
      leaf_data_t true_data, false_data;
      leaf_iterator_t it, end;
      for (boost::tie(it, end) = leaves(); it != end; ++it) {
	leaf_node_t* leaf = &*it;
	if (lss.split(space, leaf->region, leaf->leaf_data, 
		      split, priority, true_data, false_data)) 
	  pq.push(pq_entry_t(priority, leaf, split, true_data, false_data));
      }
      // Repeatedly try to split the leaf with highest priority.
      unsigned int num_splits = 0;
      while (!pq.empty() && (num_splits++ < max_num_splits)) {
	pq_entry_t pq_entry = pq.top();
	pq.pop();
	// Split the leaf to obtain a new interior node.
	interior_node_t* int_node = 
	  split_leaf(pq_entry.leaf, pq_entry.split, 
		     pq_entry.true_data, pq_entry.false_data);
	// Enqueue the new leaves.
	leaf_node_t* leaf = int_node->true_child->as_leaf();
	if (lss.split(space, leaf->region, leaf->leaf_data, 
		      split, priority, true_data, false_data)) 
	  pq.push(pq_entry_t(priority, leaf, split, true_data, false_data));
	leaf = int_node->false_child->as_leaf();
	if (lss.split(space, leaf->region, leaf->leaf_data, 
		      split, priority, true_data, false_data)) 
	  pq.push(pq_entry_t(priority, leaf, split, true_data, false_data));
      }
    }

    //! Assignment operator.
    const bsp_tree_t& operator=(const bsp_tree_t& other) {
      this->space = other.space;
      delete_subtree(this->root);
      this->root = clone(other.root);
      return *this;
    }

    //! Copy constructor.
    bsp_tree_t(const bsp_tree_t& other) 
      : space(other.space),
	root(clone(other.root)) 
    { }

    /**
     * Merging constructor.  This BSP tree is computed by merging the two
     * supplied BSP trees.
     */
    template <typename merge_leaf_data_op_t>
    bsp_tree_t(const bsp_tree_t& t,
	       const bsp_tree_t& u,
	       merge_leaf_data_op_t merge_op) 
      : traits(),
	space(traits.merge_spaces(t.space, u.space))
    {
      // First we must extend both input trees to the joint spaces.
      // TODO: It's unfortunate that we must clone the trees to do
      // this.  Adding some methods to the traits concept would allow
      // use to merge trees from different spaces.
      node_t* t_root = clone(t.root);
      node_t* u_root = clone(u.root);
      recompute_region(t_root, traits, space);
      recompute_region(u_root, traits, space);
      this->root = merge_trees(t_root, u_root, merge_op);
      delete_subtree(t_root);
      delete_subtree(u_root);
      recompute_region(this->root, traits, space);
    }

    /**
     * Collapsing constructor.  This BSP tree is computed by
     * collapsing the supplied BSP tree to a subspace.
     */
    template <typename collapse_unary_op_t,
	      typename collapse_binary_op_t>
    bsp_tree_t(const bsp_tree_t& t,
	       const space_t& subspace,
	       collapse_unary_op_t unary_op,
	       collapse_binary_op_t binary_op) 
      : space(subspace)
    {
      this->root = collapse(subspace, unary_op, binary_op, t.root);
      recompute_region(root, traits, space);
    }

    /**
     * Restricting constructor.  This BSP tree is computed by
     * restricting the supplied BSP tree to a subspace.  The supplied
     * element is used to determine which branch is used for internal
     * nodes that split on predicates that are not defined in the
     * subspace.
     */
    bsp_tree_t(const bsp_tree_t& t,
	       const space_t& subspace,
	       const element& element)
      : space(subspace)
    {
      this->root = restrict(subspace, element, t.root);
    }

    //! Returns true iff the BSP tree has a single node (the root).
    inline bool is_singleton() const { return root->is_leaf(); }

    /**
     * Returns the leaf data associated with the root; this method
     * requires that the tree is a singleton.
     */
    inline const leaf_data_t& get_singleton_data() const { 
      return root->as_leaf()->leaf_data; 
    }

    /**
     * Updates this BSP tree so that all points satisfying the supplied
     * predicate are associated with the supplied data.
     *
     * @param partition the hyperplane defining the boundary of the halfspace
     * @param halfspace the side of the boundary included in the halfspace
     * @param data     the data associated with the points in this halfspace
     *
     */
    void set_data(const predicate_t& predicate,
		  const leaf_data_t& leaf_data) {
      // \todo right now this uses partitioning, which does too much
      // work; we really don't need the true child.
      node_t* true_child, false_child;
      boost::tie(true_child, false_child) = split_tree(root, predicate);
      leaf_node_t* leaf = new_leaf(true_child->region, leaf_data);
      node_t* new_root = new_interior_node(root->region, predicate, 
					   leaf, false_child);
      delete_subtree(true_child);
      delete_subtree(this->root);
      this->root = new_root;
      recompute_region(this->root, traits, space);
    }

    /**
     * Gets a mutable reference to the leaf data associated with an
     * element.
     */
    leaf_data_t& get_leaf_data(const element& element) {
      return get_leaf(element)->leaf_data;
    }

    /**
     * Gets a const reference to the leaf data associated with an
     * element.
     */
    const leaf_data_t& get_leaf_data(const element& element) const {
      return get_leaf(element)->leaf_data;
    }

    //! Gets the space associated with this BSP tree.
    const space_t& get_space() const {
      return space;
    }

    //! Updates the space associated with this BSP tree.
    void set_space(const space_t& new_space) {
      this->space = new_space;
    }

    /**
     * Visits the entire space as a sequence of partitions obtained by
     * intersecting the partitions of this tree with those of the
     * supplied tree.  For each such partition \f$P\f$, the visitor
     * has access to the region that would be associated with \f$P\f$
     * (computed using the split function of the traits), the leaf of
     * this tree whose partition \f$Q\f$ contains \f$P\f$, and the
     * leaf of the supplied tree whose partition \f$R\f$ contains
     * \f$P\f$.  The visitor object must support the expression
     *
     *   visitor.visit(const region_t& region,
     *                 const leaf_node_t& this_leaf,
     *                 const leaf_node_t& other_leaf);
     */
    template <typename visitor_t>
    void visit_with(const bsp_tree_t& other,
		    visitor_t& visitor) const {
      assert(this->space == other.space);
      region_t region = traits.init_region(this->space);
      // A state in the visit search consists of two nodes (one from
      // each tree) whose regions overlap, and a region_t object
      // that corresponds to the intersection of their regions.
      typedef typename boost::tuple<region_t, const node_t*, const node_t*> 
	state_t;
      std::queue<state_t> queue;
      queue.push(state_t(region, this->root, other.root));
      while (!queue.empty()) {
	// Get the front state from the queue.  (It is popped below.)
	const state_t& state = queue.front();
	const region_t& region = boost::get<0>(state);
	const node_t* this_node = boost::get<1>(state);
	const node_t* other_node = boost::get<2>(state);
	if ((this_node->type == node_t::leaf_c) && 
	    (other_node->type == node_t::leaf_c)) 
	  visitor.visit(region, 
			*(this_node->as_leaf()), 
			*(other_node->as_leaf()));
	else if ((this_node->type == node_t::leaf_c) && 
		 (other_node->type == node_t::interior_c)) {
	  const predicate_t& predicate = other_node->as_interior()->split;
	  const pred_set_rel_t relation = traits.relation(predicate, region);
	  if (relation == positive_c) 
	    queue.push(state_t(region, this_node, 
			       other_node->as_interior()->true_child));
	  else if (relation == negative_c) 
	    queue.push(state_t(region, this_node, 
			       other_node->as_interior()->false_child));
	  else /* relation == both_c */ {
	    region_t true_region, false_region;
	    boost::tie(true_region, false_region) = 
	      traits.split_region(region, predicate);
	    queue.push(state_t(true_region, this_node, 
			       other_node->as_interior()->true_child));
	    queue.push(state_t(false_region, this_node, 
			       other_node->as_interior()->false_child));
	  }
	} else if (this_node->type == node_t::interior_c) {
	  // Note that we do not care in this case if the other node
	  // is a leaf or is internal.  The game plan is to first
	  // recurse to the leaves in this tree, and then when we hit
	  // leaves in this tree, the case above will descend in the
	  // other tree.
	  const predicate_t& predicate = this_node->as_interior()->split;
	  const pred_set_rel_t relation = traits.relation(predicate, region);
	  if (relation == positive_c) 
	    queue.push(state_t(region, this_node->as_interior()->true_child,
			       other_node));
	  else if (relation == negative_c) 
	    queue.push(state_t(region, this_node->as_interior()->false_child,
			       other_node));
	  else /* relation == both_c */ {
	    region_t true_region, false_region;
	    boost::tie(true_region, false_region) = 
	      traits.split_region(region, predicate);
	    queue.push(state_t(true_region, 
			       this_node->as_interior()->true_child,
			       other_node));
	    queue.push(state_t(false_region,
			       this_node->as_interior()->false_child,
			       other_node));
	  } 
	}
	// We have finished processing this state; pop it.
	queue.pop();
      }
    }

    /**
     * Visits the leaves of this BSP tree in order of decreasing
     * priority.  The priority of a node is computed by a user defined
     * functor of its region; this function must satisfy the property
     * that the priority of any node is no greater than its parent
     * node.
     *
     * @param rpf
     *        A unary functor which accepts an argument of type
     *        const bsp_traits_t::region_t (preferably by reference) 
     *        and returns a priority value.  This priority value
     *        must be of type region_priority_function_t::result_type,
     *        a type which supports the less-than (<) operator.
     * @param visitor
     *        An object with a visit method that accepts an argument of 
     *        type const leaf_node_t (preferably by reference) and an 
     *        argument of type region_priority_function_t::result_type; 
     *        each visited leaf node and its associated priority are passed
     *        to the visitor.  The object must also have a can_prune
     *        method which accepts a priority and returns true if the
     *        node with that priority can be pruned (i.e., not visited at a 
     *        later time).
     */
    template <typename region_priority_function_t,
	      typename leaf_visitor_t>
    void prioritized_visit(region_priority_function_t& rpf,
			   leaf_visitor_t& visitor) const {
      // Get the type of the priorities.
      typedef typename region_priority_function_t::result_type priority_type;
      // Create a priority queue.
      typename std::priority_queue<std::pair<priority_type, const node_t*> > 
	queue;
      // Enqueue the root of the tree.
      queue.push(std::make_pair(rpf(root->region), root));
      // Enter the visitation loop.
      while (!queue.empty()) {
	// Dequeue the next node to visit.
	priority_type priority;
	const node_t* node_ptr;
	boost::tie(priority, node_ptr) = queue.top();
	queue.pop();
	if (node_ptr->is_leaf()) {
	  visitor.visit(*node_ptr->as_leaf(), priority);
	} else {
	  // Compute the priority of the children and enqueue them.
	  const interior_node_t* int_node_ptr = node_ptr->as_interior();
	  const node_t* true_child_ptr = int_node_ptr->true_child;
	  const node_t* false_child_ptr = int_node_ptr->false_child;
	  priority_type true_child_priority = rpf(true_child_ptr->region);
	  if (!visitor.can_prune(true_child_priority)) 
	    queue.push(std::make_pair(true_child_priority, true_child_ptr));
	  priority_type false_child_priority = rpf(false_child_ptr->region);
	  if (!visitor.can_prune(false_child_priority)) 
	    queue.push(std::make_pair(false_child_priority, false_child_ptr));
	}
      }
    }

    /**
     * A forward iterator over the nodes of the BSP tree.  The type of
     * node is left as a template parameter to implement mutable and
     * const iterators.
     *
     * @see node_iterator_t
     * @see const_node_iterator_t
     */
    template <typename my_node_t,
	      typename my_bsp_tree_t>
    class base_node_iterator_t : 
      public std::iterator<std::forward_iterator_tag, my_node_t*> {
    protected:
      std::queue<my_node_t*> nodes;
    public:
      base_node_iterator_t() { }
      base_node_iterator_t(my_node_t* node) { nodes.push(node); }
      base_node_iterator_t(my_bsp_tree_t& bsp_tree) {
	nodes.push(bsp_tree.root);
      }
      base_node_iterator_t(const base_node_iterator_t& other) {
	this->nodes = other.nodes;
      }
      const base_node_iterator_t& operator=(const base_node_iterator_t& other) {
	this->nodes = other.nodes;
	return *this;
      }
      base_node_iterator_t& operator++() {
	my_node_t* n = nodes.front();
	nodes.pop();
	if (n->type == node_t::interior_c) {
	  nodes.push(n->as_interior()->true_child);
	  nodes.push(n->as_interior()->false_child);
	}
	return *this;
      }
      base_node_iterator_t operator++(int) {
	base_node_iterator_t tmp = *this;
	++(*this);
	return tmp;
      }
      my_node_t& operator*() const { return *(nodes.front()); }
      my_node_t* operator->() const { return nodes.front(); }
      bool operator==(const base_node_iterator_t& other) const {
	if (nodes.empty())
	  return other.nodes.empty();
	else if (other.nodes.empty())
	  return false;
	else
	  return (this->nodes.front() == other.nodes.front());
      }
      bool operator!=(const base_node_iterator_t& other) const {
	return !(*this == other);
      }
    };

    /**
     * A mutable iterator over the nodes of a kd tree.
     */
    typedef base_node_iterator_t<node_t, bsp_tree_t> node_iterator_t;

    /**
     * A const iterator over the nodes of a kd tree.
     */
    typedef base_node_iterator_t<const node_t, const bsp_tree_t> 
    const_node_iterator_t;

    /**
     * A forward iterator over the leaves of the kd tree.  The type of
     * node is left as a template parameter to implement mutable and
     * const iterators.
     *
     * @see leaf_iterator_t
     * @see const_leaf_iterator_t
     */
    template <typename my_leaf_node_t,
	      typename my_node_t,
	      typename my_bsp_tree_t>
    class base_leaf_iterator_t 
      : public std::iterator<std::forward_iterator_tag, my_node_t*> {
    protected:
      base_node_iterator_t<my_node_t, my_bsp_tree_t> it;
      inline void advance_to_leaf() {
	base_node_iterator_t<my_node_t, my_bsp_tree_t> end;
	while ((it != end) && (it->type == node_t::interior_c))
	  ++it;
      }
    public:
      base_leaf_iterator_t() { }
      base_leaf_iterator_t(my_bsp_tree_t& bsp_tree) : it(bsp_tree) {
	advance_to_leaf();
      }
      base_leaf_iterator_t(my_node_t* node_ptr) : it(node_ptr) {
	advance_to_leaf();
      }
      base_leaf_iterator_t(const base_leaf_iterator_t& other) : it(other.it) { }
      const base_leaf_iterator_t& operator=(const base_leaf_iterator_t& other) {
	this->it = other.it;
	return *this;
      }
      base_leaf_iterator_t& operator++() {
	++it;
	advance_to_leaf();
	return *this;
      }
      base_leaf_iterator_t operator++(int) {
	base_leaf_iterator_t tmp = *this;
	++(*this);
	return tmp;
      }
      my_leaf_node_t& operator*() const { 
	assert(it->type == node_t::leaf_c);
	return *(it->as_leaf());
      }
      my_leaf_node_t* operator->() const { 
	assert(it->type == node_t::leaf_c);
	return it->as_leaf();
      }
      bool operator==(const base_leaf_iterator_t& other) const {
	return (it == other.it);
      }
      bool operator!=(const base_leaf_iterator_t& other) const {
	return !(*this == other);
      }
    };

    /**
     * A mutable iterator over the leaves of a kd tree.
     */
    typedef base_leaf_iterator_t<leaf_node_t, 
				 node_t, 
				 bsp_tree_t> leaf_iterator_t;

    /**
     * A const iterator over the leaves of a kd tree.
     */
    typedef base_leaf_iterator_t<const leaf_node_t, 
				 const node_t, 
				 const bsp_tree_t> const_leaf_iterator_t;

    //! Returns a mutable iterator range over the nodes of this BSP tree.
    std::pair<node_iterator_t, node_iterator_t> nodes() { 
      return std::make_pair(node_iterator_t(*this), 
			    node_iterator_t()); 
    }

    //! Returns a const iterator range over the nodes of this BSP tree.
    std::pair<const_node_iterator_t, const_node_iterator_t> nodes() const { 
      return std::make_pair(const_node_iterator_t(*this), 
			    const_node_iterator_t()); 
    }

    //! Returns a mutable iterator range over the leaves of this BSP tree.
    std::pair<leaf_iterator_t, leaf_iterator_t> leaves() { 
      return std::make_pair(leaf_iterator_t(*this),
			    leaf_iterator_t()); 
    }

    //! Returns a mutable iterator range over the leaves of this BSP tree.
    std::pair<const_leaf_iterator_t, const_leaf_iterator_t> leaves() const { 
      return std::make_pair(const_leaf_iterator_t(*this),
			    const_leaf_iterator_t()); 
    }

  }; // class bsp_tree_t

  //! Returns true iff this node is the false child of some parent.
  template <typename bsp_traits_t>
  inline bool bsp_tree_t<bsp_traits_t>::node_t::is_false_child() const {
    return (this->parent != NULL) && (this == this->parent->false_child);
  }
    
  //! Returns true iff this node is the true child of some parent.
  template <typename bsp_traits_t>
  inline bool bsp_tree_t<bsp_traits_t>::node_t::is_true_child() const {
    return (this->parent != NULL) && (this == this->parent->true_child);
  }

} // namespace prl 


#endif // PRL_BSP_TREE_HPP
