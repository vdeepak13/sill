#ifndef SILL_SUBSET_ITERATOR_HPP
#define SILL_SUBSET_ITERATOR_HPP

#include <limits>
#include <list>
#include <vector>
#include <iterator>
#include <cassert>

#include <set>

#include <sill/base/stl_util.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for iterating over subsets of a set.
   * The iterator may be restricted to different types of subsets.
   *
   * --------------------------------------------------------------------
   *
   * This class has 2 methods for iterating over subsets:
   * 1) For iterating over all subsets
   * 2) For iterating over all subsets of size k << n
   *
   * 1) For iterating over all subsets:
   * This uses Gray codes so that
   * it takes time O(n) (but generally less than n)
   * on each increment but copies at most one element.
   *
   * flip lowest bit every 2 times:  01100110011001100 ...
   * flip next lowest every 4 times: 001111000011110000 ...
   * flip next lowest every 8 times: 00001111111100000000 ...
   * ...
   *
   * So if we have all 0's on iterator i = 0, then
   * on iteration i = 1,2,...,
   * flip bit 0,1,0,2,0,1,0,3
   * So flip bit 0 if i % 2 = 1    - flip 0 to 1 if i % 4 = 1, else 1 to 0
   *             1 if i % 4 = 2    - flip 0 to 1 if i % 8 = 2, else 1 to 0
   *             2 if i % 8 = 4    - flip 0 to 1 if i % 16 = 4, else 1 to 0
   *  flip j if i % 2^(j+1) = 2^j  - flip 0 to 1 if i % 2^(j+2) = 2^j
   *
   * \todo This could be done using a different representation for
   *  the iteration number or n_subsets, in which case much bigger sets
   *  could be handled.
   *
   * --------------------------------------------------------------------
   *
   * 2) For iterating over all subsets of size k << n:
   * This takes time O(k) on each increment and copies at most k elements
   *  (but much less on average).
   * 
   * Order items.
   * This imposes an ordering on subsets, where the leftmost item is
   *  the most significant bit.
   * Initially, add k left-most items.
   * Iterate until the set contains the k right-most items:
   *   Given a current set,
   *   Find right-most item i in current set.
   *   If item i is the right-most item in set_vec,
   *     Find next item (set bit j) to the left which has at least one empty
   *      spot to the right (unset bit j+1) of it.
   *     Unset bit j, and set bit j+1.
   *     If r bits to the right of j+1 are set, unset them, and
   *      set the r bits directly to the right of j+1.
   *   Else
   *     Unset bit i, and set bit i-1.
   *
   * This can be done efficiently by storing a list of indices for
   *  items which are currently in the set.
   *
   * \todo Use boost::iterator_facade for this?
   * \todo Does the set need to be ordered?
   *
   * \ingroup iterator
   */
  template <typename Set>
  class subset_iterator
    : public std::iterator<std::forward_iterator_tag, const Set> {

  protected:

    //! Set to true to print out extra warnings.
    static const bool VERBOSE = false;

  public:

    //! The type of object stored in the sets.
    typedef typename Set::value_type value_type;

  protected:

    /**
     * Method for iterating over subsets.  Note iterators may be further
     * restricted.
     */
    enum iterator_method_enum {
      // End iterator, or iterator which is finished.
      END_ITERATOR,
      // Iterator over all subsets.
      ALL_SUBSET_ITERATOR,
      // Iterator over subsets of a fixed size (or multiple fixed sizes).
      FIXED_SIZE_SUBSET_ITERATOR,
    };

    iterator_method_enum iterator_method;

    //! Current set.
    Set set;

    //! Vector over set elements.
    std::vector<value_type> set_vec;

    //! Min subset size allowed.
    std::size_t min_subset_size;

    //! Max subset size allowed.
    std::size_t max_subset_size;

    /*---------------STUFF FOR ITERATORS OVER ALL SUBSETS--------------*/

    //! Size of the power set of init_set.
    unsigned long n_subsets;

    //! Iteration, starting with 0.  At end if >= n_subsets
    unsigned long iteration;

    /*---------------STUFF FOR ITERATORS OVER FIXED-SIZE SUBSETS-------*/

    //! Size of fixed-size subsets currently begin iterated over.
    std::size_t fixed_subset_size;

    //! Type for indices of items in the vector of set elements.
    typedef typename std::vector<value_type>::size_type v_size_type;

    //! List of indices of items in current set.
    std::list<v_size_type> set_list;

  public:

    //! End iterator constructor.
    subset_iterator()
      : iterator_method(END_ITERATOR)
    { }

    /**
     * Constructor for iterator over all subsets.
     *
     * @param init_set        Set of items.
     */
    subset_iterator(const Set& init_set)
      : min_subset_size(0), max_subset_size(init_set.size()) {
      initialize_iterator(init_set);
    }

    /**
     * Constructor for iterator over fixed-size subsets.
     *
     * @param init_set        Set of items.
     * @param subset_size     Fixed size for subsets.
     */
    subset_iterator(const Set& init_set,
		    std::size_t subset_size)
      : min_subset_size(std::min(init_set.size(), subset_size)),
        max_subset_size(std::min(init_set.size(), subset_size)) {
      initialize_iterator(init_set);
    }

    /**
     * Constructor for iterator over subsets of a range of sizes.
     *
     * @param init_set        Set of items.
     * @param min_subset_size Min size of subsets allowed
     * @param max_subset_size Max size of subsets allowed
     */
    subset_iterator(const Set& init_set,
		    std::size_t min_subset_size,
		    std::size_t max_subset_size)
      : min_subset_size(min_subset_size),
        max_subset_size(std::min(init_set.size(), max_subset_size)) {
      initialize_iterator(init_set);
    }

    void initialize_iterator(const Set& init_set) {
      if (min_subset_size > init_set.size()
          || max_subset_size < min_subset_size) {
        if (VERBOSE)
          std::cerr << "WARNING: possible error: "
                    << "subset_iterator set to end iterator:" << std::endl
                    << "  init_set: " << init_set << std::endl
                    << "  min_subset_size = " << min_subset_size
                    << ", max_subset_size = " << max_subset_size << std::endl;
        iterator_method = END_ITERATOR;
        return;
      }

      foreach(const value_type& val, init_set)
        set_vec.push_back(val);
      /*
      typename Set::const_iterator it, end;
      for (boost::tie(it, end) = init_set.values(); it != end; ++it)
        set_vec.push_back(*it);
      */
      // Choose what method to use to iterate over subsets.
      // TODO: The below choice of the iterator method is based on 2
      //       things:
      //   1) Make sure that 'iteration' and 'n_subsets' won't overflow.
      //      Later, we should add an alternate representation to avoid
      //      this issue.
      //   2) Iterate only over fixed-size subsets if it's faster.
      //      Later, we should check this experimentally and make a
      //      smarter decision rule.
      if (init_set.size() < 8 * sizeof(unsigned long)
          && max_subset_size - min_subset_size > 2) {
        iterator_method = ALL_SUBSET_ITERATOR;
        iteration = 0;
        n_subsets = 1;
        for (std::size_t i = 0; i < init_set.size(); ++i)
          n_subsets *= 2;
        if (min_subset_size > 0)
          ++(*this);
      } else {
        iterator_method = FIXED_SIZE_SUBSET_ITERATOR;
        fixed_subset_size = min_subset_size;
        for (v_size_type i = 0;
             i < fixed_subset_size; ++i) {
          set_list.push_back(i);
          set.insert(set_vec[i]);
        }
      }
    }

    //! Prefix increment.
    subset_iterator& operator++() {
      switch(iterator_method) {
      case END_ITERATOR:
        assert(false);
      case ALL_SUBSET_ITERATOR:
        do {
          iteration++;
          if (iteration >= n_subsets) {
            set = Set();
            iterator_method = END_ITERATOR;
            return *this;
          }
          // Enclose in braces to avoid compiler error from
          //  initializing pow_j.
          {
            unsigned long pow_j = 1;
            for (std::size_t j = 0; j < set_vec.size(); j++) {
              unsigned long pow_j2 = 2 * pow_j;
              if (iteration % pow_j2 == pow_j) {
                if (iteration % (2 * pow_j2) == pow_j) {
                  set.insert(set_vec[j]);
                } else {
                  set.erase(set_vec[j]);
                }
                break;
              }
              pow_j = pow_j2;
            }
          }
        } while (set.size() > max_subset_size
                 || set.size() < min_subset_size);
        break;
      case FIXED_SIZE_SUBSET_ITERATOR:
        // If no more subsets of current fixed_subset_size to iterate over
        if (fixed_subset_size == 0
            || set_list.front() == set_vec.size() - fixed_subset_size) {
          ++fixed_subset_size;
          if (fixed_subset_size > max_subset_size) {
            set = Set();
            iterator_method = END_ITERATOR;
            return *this;
          }
          set_list.clear();
          set.clear();
          for (v_size_type i = 0; i < fixed_subset_size; ++i) {
            set_list.push_back(i);
            set.insert(set_vec[i]);
          }
          return *this;
        }
        // Enclose in braces to avoid compiler error from
        //  initializing i.
        {
          // Set i to be index of right-most item in current set.
          v_size_type i = set_list.back();
          // If i is right-most item in set_vec,
          if (i == set_vec.size() - 1) {
            // Find next item (index *it) to left of i s.t. item (*it)+1
            //  is not in current set.
            typedef typename std::list<v_size_type>::reverse_iterator
              reverse_iterator;
            reverse_iterator it = set_list.rbegin();
            reverse_iterator end = set_list.rend();
            ++it;
            // Counter j keeps track of how many items to the right of
            //  *it are in the current set.
            v_size_type j = 1;
            do {
              // If there is an item between *it and i = *(--it),
              if (*it + 1 < i) {
                // Remove *it from current set, and add (*it)+1.
                set.erase(set_vec[*it]);
                ++(*it);
                set.insert(set_vec[*it]);
                // Set elements *(--it) and on to be i = (*it)+1, (*it)+2, ...
                i = *it + 1;
                while (j > 0) {
                  --it;
                  set.erase(set_vec[*it]);
                  *it = i;
                  set.insert(set_vec[*it]);
                  ++i;
                  --j;
                }
                break;
              }
              i = *it;
              ++j;
              ++it;
            } while (it != end);
          } else {
            set.erase(set_vec[i]);
            i++;
            set_list.back() = i;
            set.insert(set_vec[i]);
          }
        }
        break;
      default:
        assert(false);
      } // switch(iterator_method)
      return *this;
    }

    //! Postfix increment.
    subset_iterator operator++(int) {
      subset_iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    //! Returns a const reference to the current set.
    const Set& operator*() const { return set; }

    //! Returns a const pointer to the current set.
    const Set* const operator->() const { return &set; }

    //! Returns truth if the two subset iterators are the same.
    bool operator==(const subset_iterator& it) const {
      if (iterator_method != it.iterator_method)
	return false;
      switch (iterator_method) {
      case END_ITERATOR:
	return true;
      case ALL_SUBSET_ITERATOR:
	if (iteration == it.iteration
	    && max_subset_size == it.max_subset_size
	    && min_subset_size == it.min_subset_size
	    && set_vec == it.set_vec)
	  return true;
      case FIXED_SIZE_SUBSET_ITERATOR:
	if (max_subset_size == it.max_subset_size
	    && min_subset_size == it.min_subset_size
	    && fixed_subset_size == it.fixed_subset_size
	    && set_list == it.set_list)
	  return true;
      default:
	assert(false);
      }
      return false;
    }

    //! Returns truth if the two subset iterators are different.
    bool operator!=(const subset_iterator& it) const {
      return !operator==(it);
    }

  }; // class subset_iterator

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_SUBSET_ITERATOR_HPP
