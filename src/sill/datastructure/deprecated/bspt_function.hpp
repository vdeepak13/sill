
#ifndef SILL_BSPT_FUNCTION_HPP
#define SILL_BSPT_FUNCTION_HPP

#include <algorithm>
#include <iostream>
#include <map>
#include <climits>
#include <list>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <boost/shared_ptr.hpp>
#include "kd_tree.hpp"

////////////////////////////////////////////////////////////////////
// Needs clean-up
///////////////////////////////////////////////////////////////////

namespace sill {

  /**
   * Represents a piecewise constant approximation of a scalar-valued
   * function with vector inputs.  The regions of constant value are
   * indexed using a BSP tree.
   *
   * \ingroup datastructure
   */
  template <typename dimension_id_t,
	    typename input_value_t = double,
	    typename output_value_t = double>
  class bspt_function_t {

  protected:
  
    /**
     * These are the traits used to make a BSP tree represent a
     * piecewise constant function.
     */
    struct bsp_traits_t {
      
      /**
       * The representation of a space.  This is a set of dimension
       * identifiers.
       */
      typedef sill::set<dimension_id_t> space_t;

      /**
       * An iso-hyperplane partition of the input space; it consists
       * of the dimension identifier and value in that dimension.
       * When viewed as a predicate, partitions are satisfied by
       * inputs whose coordinates in the identified dimension are
       * (strictly) less than the specified input value.
       */
      typedef std::pair<dimension_id_t, input_value_t> partition_t;

      /**
       * A region of the input space.  This is a map from dimension
       * identifiers to intervals.
       */
      typedef std::map<dimension_id_t, interval_t> region_t;

      //! Returns the region containing the entire space.
      static inline region_t universe(const space_t& space) { 
	// \todo these limits are dimension specific
	std::pair<input_value_t, input_value_t> all = 
	  std::make_pair(-std::numeric_limits<input_value_t>::infinity(),
			 std::numeric_limits<input_value_t>::infinity());
	region_t region;
	for (typename space_t::const_iterator_t it = space.begin();
	     it != space.end(); ++it) 
	  region[*it] = all;
	return region;
      }

      //! Returns truth if two partitions are identical.
      static inline bool are_equal(const partition_t& p, 
				   const partition_t& q) { return (p == q); }

      //! Returns truth if two partitions are opposites.
      static inline bool are_opposite(const partition_t& p, 
				      const partition_t& q) { return false; }

      //! Returns truth if the supplied region overlaps the halfspace
      //! associated with the supplied partition.
      static inline bool do_overlap(const region_t& region,
				    const partition_t& partition) {
	dimension_id_t dim_id = partition.first;
	const interval_t& interval = region.find(dim_id)->second;
	return (!(partition.second < interval.first) &&
		(partition.second < interval.second));
      }
      
      //! Splits a region into two along a supplied partition.
      static inline std::pair<region_t, region_t> 
      split_region(const region_t& region,
		   const partition_t& partition) {
	// Compute the region associated with the negative halfspace.
	region_t lt_region = region;
	interval_t& lt_interval = lt_region->find(partition.first)->second;
	lt_interval.second = partition.second;
	// Compute the region associated with the positive halfspace.
	region_t gte_region = region;
	interval_t& gte_interval = gte_region->find(partition.first)->second;
	gte_interval.first = partition.second;
	return std::make_pair(lt_region, gte_region);
      }

      /**
       * Returns truth if the supplied point satisfies the
       * partitioning predicate, i.e., if the point is in the negative
       * halfspace associated with the partition.
       */
      template <typename input_t>
      static inline bool satisfies(const input_t& input, 
				   const partition_t& partition) {
	return point[partition.first] < partition.second;
      }

    }; // struct kd_tree_t::bsp_traits_t

    /**
     * The traits used for a kd tree used to approximate a function.
     * Each leaf of the kd tree is associated with a double precision
     * function value.
     */
    struct kdt_traits_t {
      typedef double coordinate_t;
      typedef int dimension_id_t;
      typedef std::less<coordinate_t> compare_coord_t;
      typedef double data_t;
      typedef std::set<dimension_id_t> domain;
      typedef std::pair<coordinate_t, coordinate_t> interval_t;
      typedef std::map<dimension_id_t, interval_t> box_t;
      typedef std::map<dimension_id_t, coordinate_t> point_t;

      static inline interval_t& interval(box_t& box,
					 dimension_id_t dim_id) {
	return box.find(dim_id)->second;
      }

      static inline const interval_t& interval(const box_t& box,
					       dimension_id_t dim_id) {
	return box.find(dim_id)->second;
      }

      static box_t collapse_box(const box_t& box, const domain& domain) {
	box_t collapsed;
	typedef domain::const_iterator dim_id_iterator_t;
	typedef box_t::const_iterator box_elt_iterator_t;
	for (dim_id_iterator_t it = domain.begin(); it != domain.end(); ++it) {
	  box_elt_iterator_t b_it = box.find(*it);
	  if (b_it != box.end())
	    collapsed[*it] = b_it->second;
	}
	return collapsed;
      } 

      static coordinate_t min_coord(const dimension_id_t& dim_id) {
	return -std::numeric_limits<double>::infinity();
      }
      static coordinate_t max_coord(const dimension_id_t& dim_id) {
	return std::numeric_limits<double>::infinity();
      }

      inline static coordinate_t coordinate(const point_t& point,
					    const dimension_id_t& dim_id) {
	point_t::const_iterator it = point.find(dim_id);
	assert(it != point.end());
	return it->second;
      }

      static inline bool 
      split_datum(const box_t& box,
		  const data_t& datum,
		  const iso_partition_t<kdt_traits_t>& partition,
		  data_t& lt_datum,
		  data_t& gte_datum) {
	lt_datum = gte_datum = datum;
	return true;
      }
    }; // end of struct: bspt_function_t::kdt_traits_t
  
    /**
     * The type of kd tree used to store the points.
     */
    typedef kd_tree_t<kdt_traits_t> kdt_t;

    /**
     * The kd tree used to represent the function approximation.  This
     * is handled using a shared pointer to permit efficient copying and
     * assignment of bspt_function_t objects.
     */
    boost::shared_ptr<kdt_t> kdt;

  public:

    /**
     * A dimension descriptor.
     */
    struct dim_t {
      int id;
      double l_bound;
      double u_bound;
      dim_t(int id,
	    double l_bound,
	    double u_bound) 
	: id(id),
	  l_bound(l_bound),
	  u_bound(u_bound)
      { }
      bool operator==(const dim_t& d) const { return (id == d.id); }
      bool operator!=(const dim_t& d) const { return !(*this == d); }
      bool operator<(const dim_t& d) const { return (id < d.id); }
    };

    /**
     * Constant constructor.
     *
     * @param value the value of the constant
     */
    bspt_function_t(double value) {
      kdt = boost::shared_ptr<kdt_t>(new kdt_t(kdt_t::domain(), value));
    }

    /**
     * Approximation constructor.
     *
     * @param begin       a starting iterator over dim_t structures
     *                    representing the argument list of the function
     * @param end         a past-the-end iterator over the arguments
     * @param function    the function to be approximated; it must accept
     *                    its arguments as an array of coordinate values
     * @param default_value the value of the function outside the bounds 
     *                      of its arguments (as specified by the dim_t
     *                      structures)
     * @param min_gain    the minimum error reduction required to justify
     *                    splitting a leaf of the kd tree
     * @param max_leaves  the maximum number of leaves in the kd tree
     * @param num_samples the number of samples used to search for the
     *                    best split of each leaf
     */
    template <typename dim_iterator_t,
	      typename function_t>
    bspt_function_t(dim_iterator_t begin, 
		      dim_iterator_t end, 
		      const function_t& function,
		      double default_value = 0.0,
		      double min_gain = 0.0,
		      unsigned int max_leaves = UINT_MAX,
		      unsigned int num_samples = 100) 
    {
      typename kdt_t::domain domain;
      typename kdt_t::box_t box;
      std::list<dim_t> dim_list(begin, end);
      while (begin != end) {
	domain.insert(begin->id);
	box[begin->id] = std::pair<double, double>(begin->l_bound,
						   begin->u_bound);
	++begin;
      }
      // TODO: initialize root_data
      kdt = boost::shared_ptr<kdt_t>(new kdt_t(domain, 0.0));
      kdt->install_default(box, 0.0);

      kdt_leaf_split_strategy_t<function_t> lss(dim_list, function, default_value,
						num_samples, min_gain);
      kdt->grow(lss, max_leaves - 1);
    }

    /**
     * Destructor.
     */
    ~bspt_function_t() {
      // Note that the kd tree destructor is not invoked here.  This
      // will happen automatically when all boost::shared_ptr objects
      // referencing the kd tree go out of scope.
    }

    /**
     * Combination constructor.  This function approximator is
     * initialized to be the combination of the two supplied function
     * approximators, where the combination is defined by the supplied
     * operator.  Possibilities for combine_op include std::plus<double>
     * and std::multiplies<double> objects.
     */
    template <typename combine_op_t>
    bspt_function_t(const bspt_function_t& f,
		      const bspt_function_t& g,
		      combine_op_t combine_op) {
      kdt = boost::shared_ptr<kdt_t>(new kdt_t(*(f.kdt), *(g.kdt), combine_op));
    }

    /**
     * An operator which can be used to integrate a function over a
     * subset of its arguments.
     */
    struct integration_op_t {
      typedef kdt_t::box_t box_t;
      typedef kdt_t::data_t data_t;
      typedef kdt_t::interval_t interval_t;
      typedef kdt_t::domain domain;

      /**
       * The operator used to collapse the datum associated with a leaf.
       * This operator simply multiplies the constant function value by
       * the area associated with the collapsed dimensions.
       */
      struct collapse_datum_op_t {
	void operator()(const box_t& box, 
			const data_t& datum,
			const domain& domain,
			data_t& collapsed_datum) {
	  if (datum == 0) {
	    collapsed_datum = 0.0;
	    return;
	  }
	  double area = 1.0;
	  typedef box_t::const_iterator box_elt_it_t;
	  for (box_elt_it_t it = box.begin(); it != box.end(); ++it) 
	    if (domain.find(it->first) == domain.end()) {
	      const interval_t& interval = it->second;
	      const double length = interval.second - interval.first;
	      area *= length;
	    }
	  collapsed_datum = datum * area;
	}
      } collapse_datum;

      /**
       * The operator used to merge two data associated with collapsed
       * leaves.
       */
      std::plus<double> merge_data;
    };
  
    /**
     * Collapse constructor.  This function approximator is computed by
     * collapsing the supplied function approximator over a subset of
     * its dimensions, where the collapse operator is defined by the
     * supplied object.  
     */
    template <typename dim_id_iterator_t,
	      typename collapse_op_t>
    bspt_function_t(const bspt_function_t& f,
		      dim_id_iterator_t begin,
		      dim_id_iterator_t end,
		      collapse_op_t collapse_op) {
      typedef kdt_t::domain domain;
      domain domain(begin, end);
      this->kdt = boost::shared_ptr<kdt_t>(new kdt_t(*(f.kdt), domain, 
						     collapse_op.collapse_datum, 
						     collapse_op.merge_data));
    }

    /**
     * Writes out the leaves of the index; each leaf is written as a
     * single line containing the bounding box coordinates and the
     * constant function value in the leaf.
     */
    template <typename ostream_t>
    void report(ostream_t& out) {
      for (typename kdt_t::leaf_iterator_t it = kdt->leaves_begin();
	   it != kdt->leaves_end(); ++it) {
	typename kdt_t::box_t& box = it->box;
	typedef typename kdt_t::domain::const_iterator iterator;
	for (iterator d_it = kdt->domain().begin();
	     d_it != kdt->domain().end(); ++d_it) 
	  out << box[*d_it].first << " " << box[*d_it].second << " ";
	typename kdt_t::data_t& leaf_data = it->datum;
	out << leaf_data << std::endl;
      }
    }

    /**
     * Evaluates this function.
     *
     * @param args an array of num_dims input arguments
     */
    double operator()(kdt_traits_t::point_t& args) const {
      return kdt->get_datum(args);
    }

  protected:

    /**
     * Defines a leaf splitting strategy for recursively constructing a kd
     * tree to represent a piecewise-constant approximation of a function.
     */ 
    template <typename function_t>
    class kdt_leaf_split_strategy_t {

    protected:

      typedef kdt_t::coordinate_t coordinate_t;
      typedef kdt_t::dimension_id_t dimension_id_t;
      typedef kdt_t::compare_coord_t compare_coord_t;
      typedef kdt_t::data_t data_t;
      typedef kdt_t::domain domain;
      typedef kdt_t::box_t box_t;

      /**
       * The argument list of the function being approximated.
       */
      const std::list<dim_t>& arg_list;

      /**
       * The function being approximated.
       */
      const function_t& function;

      /**
       * The default value used outside the bounds of the arguments.
       */
      double default_value;

      /**
       * A source of pseudorandomness.
       */
      gsl_rng* random;

      /**
       * The number of samples used in each leaf to determine its
       * optimal split.
       */
      unsigned int num_samples;

      /**
       * The minimum error reduction required to justify a leaf split.
       */
      double min_gain;

      /**
       * Returns a bound on the error associated with a constant
       * approximation to a probability distribution in a
       * iso-rectangular region.  See (Kozlov & Koller, 1997).
       */
      static inline double error_bound(double min, double max, 
				       double mean, double volume) {
	return (((max - mean) / (max - min)) * min * log(min / mean) +
		((mean - min) / (max - min)) * max * log(max / mean)) * volume;
      }

      /**
       * Computes (an upper bound on) the reduction in error associated
       * with splitting a leaf in two.
       */
      static inline double gain(double min, 
				double max, 
				double mean,
				double volume,
				double lt_min,
				double lt_max,
				double lt_mean,
				double lt_volume_frac,
				double gte_min,
				double gte_max,
				double gte_mean) {
	return error_bound(min, max, mean, volume)
	  - error_bound(lt_min, lt_max, lt_mean, lt_volume_frac * volume) 
	  - error_bound(gte_min, gte_max, gte_mean, 
			(1.0 - lt_volume_frac) * volume);
      }

      /**
       * A sampled location and associated function value.
       */
      struct function_sample_t {
	coordinate_t* point;
	double val;
	function_sample_t(unsigned int d) { point = new coordinate_t[d]; }
	~function_sample_t() { delete [] point; }
      };
    
      /**
       * Places an ordering on iterators over weighted function samples.
       */
      template <typename iterator_t>
      struct fs_ptr_comparator {
	unsigned int dim;
	fs_ptr_comparator(unsigned int dim) : dim(dim) { }
	bool operator()(const iterator_t& a, const iterator_t& b) const {
	  return (a->point[dim] < b->point[dim]);
	}
      };

    public:
  
      /**
       * The priority type of leaves.
       */
      typedef double priority_type;

      /**
       * Constructor.
       */
      kdt_leaf_split_strategy_t(const std::list<dim_t>& arg_list,
				const function_t& function,
				double default_value = 0.0,
				unsigned int num_samples = 100,
				double min_gain = 0.0) 
	: arg_list(arg_list), function(function), default_value(default_value),
	  num_samples(num_samples), min_gain(min_gain)
      { 
	random = gsl_rng_alloc(gsl_rng_taus);
      }

      /**
       * Destructor.
       */
      ~kdt_leaf_split_strategy_t() {
	gsl_rng_free(random);
      }

      /**
       * Determines if and how a leaf node should be split.
       *
       * @param domain    the domain of the node
       * @param box       the bounding box of the leaf node
       * @param datum     the datum currently assigned to the leaf
       * @param partition if the leaf should be split, this is updated
       *                  to the partition used for the split
       * @param priority  if the leaf should be split, this is the priority
       *                  of the split
       * @return          true if the leaf should be split, false otherwise
       */
      bool split(const domain& domain,
		 const box_t& box, 
		 data_t& datum,
		 iso_partition_t<kdt_traits_t>& partition,
		 priority_type& priority,
		 data_t& lt_datum,
		 data_t& gte_datum) {
	// If the leaf has infinite area, do not try to split it.
	for (domain::const_iterator it = domain.begin();
	     it != domain.end(); ++it) {
	  const double l_bound = box.find(*it)->second.first;
	  const double u_bound = box.find(*it)->second.second;
	  if ((l_bound == -std::numeric_limits<double>::infinity()) ||
	      (u_bound == std::numeric_limits<double>::infinity()))
	    return false;
	}
	// Sample points uniformly in the box and compute the function
	// values.  Also compute the min, sum, and max values.
	function_sample_t** fs = new function_sample_t*[num_samples];
	double min = std::numeric_limits<double>::infinity();
	double max = -std::numeric_limits<double>::infinity();
	double sum = 0.0;
	const unsigned int num_dims = arg_list.size();
	for (unsigned int i = 0; i < num_samples; i++) {
	  fs[i] = new function_sample_t(num_dims);
	  unsigned int dim_index = 0;
	  for (std::list<dim_t>::const_iterator it = arg_list.begin();
	       it != arg_list.end(); ++it, ++dim_index) {
	    const double l_bound = box.find(it->id)->second.first;
	    const double u_bound = box.find(it->id)->second.second;
	    fs[i]->point[dim_index] = 
	      gsl_ran_flat(random, l_bound, u_bound);
	  }
	  fs[i]->val = function(fs[i]->point);
	  if (fs[i]->val < min) min = fs[i]->val;
	  if (fs[i]->val > max) max = fs[i]->val;
	  sum += fs[i]->val;
	}
	// Record the mean.  (This is the function estimate in the leaf.)
	datum = sum / double(num_samples);
	// Compute the volume of the leaf.
	double volume = 1.0;
	for (std::list<dim_t>::const_iterator it = arg_list.begin();
	     it != arg_list.end(); ++it) {
	  const double l_bound = box.find(it->id)->second.first;
	  const double u_bound = box.find(it->id)->second.second;
	  volume *= u_bound - l_bound;
	}
	// Search in each dimension for the best split.
	priority = -std::numeric_limits<double>::infinity();
	unsigned int dim_index = 0;
	for (std::list<dim_t>::const_iterator it = arg_list.begin();
	     it != arg_list.end(); ++it, ++dim_index) {
	  // Sort the points in this dimension.
	  fs_ptr_comparator<function_sample_t*> comp(dim_index);
	  std::sort(fs, fs + num_samples, comp);

	  /*
	    std::cerr << "Samples sorted by dimension " 
	    << it->id << ":" << std::endl;
	    for (unsigned int i = 0; i < num_samples; ++i) {
	    std::cerr << i << ":";
	    for (unsigned int a = 0; a < num_dims; a++)
	    std::cerr << " " << fs[i]->point[a];
	    std::cerr << std::endl;
	    }
	  */

	  // Compute the cumulative min, sum, and max in the forward
	  // and backward directions.
	  std::vector<double> f_min(num_samples), 
	    f_max(num_samples), f_sum(num_samples);
	  f_min[0] = f_sum[0] = f_max[0] = fs[0]->val;
	  for (unsigned int i = 1; i < num_samples; ++i) {
	    f_min[i] = std::min(f_min[i - 1], fs[i]->val);
	    f_max[i] = std::max(f_max[i - 1], fs[i]->val);
	    f_sum[i] = f_sum[i - 1] + fs[i]->val;
	  }
	  std::vector<double> b_min(num_samples), 
	    b_max(num_samples), b_sum(num_samples);
	  b_min[num_samples - 1] = b_sum[num_samples - 1] = 
	    b_max[num_samples - 1] = fs[num_samples - 1]->val;
	  for (int i = num_samples - 2; i >= 0; --i) {
	    b_min[i] = std::min(b_min[i + 1], fs[i]->val);
	    b_max[i] = std::max(b_max[i + 1], fs[i]->val);
	    b_sum[i] = b_sum[i + 1] + fs[i]->val;
	  }
	  // Look for improved splits.
	  const double l_bound = box.find(it->id)->second.first;
	  const double u_bound = box.find(it->id)->second.second;
	  for (unsigned int i = 0; i < num_samples - 1; ++i) {
	    double lt_volume_frac = 
	      (fs[i]->point[dim_index] - l_bound) / (u_bound - l_bound);
	    double g = 
	      gain(min, max, sum / double(num_samples), volume, 
		   f_min[i], f_max[i], f_sum[i] / double(i + 1), lt_volume_frac,
		   b_min[i + 1], b_max[i + 1], 
		   b_sum[i + 1] / double(num_samples - i + 1));
	    /*
	      std::cerr << "splitting at " << d << " = "
	      << fs[i]->point[d] << " yields gain: "
	      << g << std::endl;
	    */
	    if (g > priority) {
	      priority = g;
	      partition.dim_id = it->id;
	      partition.val = fs[i]->point[dim_index];
	    }
	  }
	}
      
	/*
	  std::cerr << box.find(partition.dim_id)->second.first << " " 
	  << box.find(partition.dim_id)->second.second << " " 
	  << "BEST SPLIT is " << partition.dim_id
	  << " = " << partition.val
	  << " with gain: " << priority
	  << std::endl;
	*/

	return (priority > min_gain);
      }

    }; // end of class: bspt_function_t::leaf_split_strategy_t

  }; // end of class: bspt_function_t

  /**
   * Multiplies two functions.
   * \relates bspt_function_t
   */
  bspt_function_t operator*(const bspt_function_t& f,
                            const bspt_function_t& g) {
    return bspt_function_t(f, g, std::multiplies<double>());
  }

  /**
   * Adds two functions.
   * \relates bspt_function_t
   */
  bspt_function_t operator+(const bspt_function_t& f,
                            const bspt_function_t& g) {
    return bspt_function_t(f, g, std::plus<double>());
  }

} // namespace sill

#endif // SILL_BSPT_FUNCTION_HPP
