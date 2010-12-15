
#ifndef PRL_LEARNING_DISCRIMINATIVE_PARAMETER_GRID_HPP
#define PRL_LEARNING_DISCRIMINATIVE_PARAMETER_GRID_HPP

#include <prl/math/vector.hpp>

/**
 * \file parameter_grid.hpp  This contains helper functions for dealing with
 *                           grids of parameters, especially when testing large
 *                           sets of regularization parameters.
 */

namespace prl {

  /**
   * Create a grid of parameters to test.
   * @param minvals    Minimum values for each dimension in the grid.
   * @param maxvals    Maximum values for each dimension in the grid.
   * @param k          Number of values in each dimension (>= 1).
   *                   If k = 1, then minvals must equal maxvals.
   * @param log_scale  If true, make values evenly spaced on a log scale;
   *                   if true, the values must be positive.
   * @param inclusive  If true, include the min and max values in the grid.
   *
   * @return  constructed grid, in the form of a list of grid points
   */
  std::vector<vec>
  create_parameter_grid(const vec& minvals, const vec& maxvals, size_t k,
                        bool log_scale, bool inclusive = true);

  /**
   * Create a grid of parameters to test.
   * @param minvals    Minimum values for each dimension in the grid.
   * @param maxvals    Maximum values for each dimension in the grid.
   * @param k          Number of values in each dimension (>= 1).
   *                   If k(i) = 1, then minvals(i) must equal maxvals(i).
   * @param log_scale  If true, make values evenly spaced on a log scale;
   *                   if true, the values must be positive.
   * @param inclusive  If true, include the min and max values in the grid.
   *
   * @return  constructed grid, in the form of a list of grid points
   */
  std::vector<vec>
  create_parameter_grid(const vec& minvals, const vec& maxvals, const ivec& k,
                        bool log_scale, bool inclusive = true);

  /**
   * Create a grid of parameters to test.
   * This variant of the above method is for single-parameter grids.
   * @param minval     Minimum value.
   * @param maxval     Maximum value.
   * @param k          Number of values to try (> 1).
   * @param log_scale  If true, make values evenly spaced on a log scale;
   *                   if true, the values must be positive.
   * @param inclusive  If true, include the min and max values.
   *
   * @return  constructed grid, in the form of a list of grid points
   */
  vec create_parameter_grid(double minval, double maxval, size_t k,
                            bool log_scale, bool inclusive = true);

  /**
   * Create a grid of parameters to test, but return it via lists of grid
   * coordinates for each dimension (k*d values) instead of grid points
   * (k^d values).
   * @param minvals    Minimum values for each dimension in the grid.
   * @param maxvals    Maximum values for each dimension in the grid.
   * @param k          Number of values in each dimension (>= 1).
   *                   If k = 1, then minvals must equal maxvals.
   * @param log_scale  If true, make values evenly spaced on a log scale;
   *                   if true, the values must be positive.
   * @param inclusive  If true, include the min and max values in the grid.
   *
   * @return  <values for dim 1, values for dim 2, etc.>
   */
  std::vector<vec>
  create_parameter_grid_alt(const vec& minvals, const vec& maxvals, size_t k,
                            bool log_scale, bool inclusive = true);

  /**
   * Create a grid of parameters to test, but return it via lists of grid
   * coordinates for each dimension (k*d values) instead of grid points
   * (k^d values).
   * @param minvals    Minimum values for each dimension in the grid.
   * @param maxvals    Maximum values for each dimension in the grid.
   * @param k          Number of values in each dimension (>= 1).
   *                   If k = 1, then minvals must equal maxvals.
   * @param log_scale  If true, make values evenly spaced on a log scale;
   *                   if true, the values must be positive.
   * @param inclusive  If true, include the min and max values in the grid.
   *
   * @return  <values for dim 1, values for dim 2, etc.>
   */
  std::vector<vec>
  create_parameter_grid_alt(const vec& minvals, const vec& maxvals,
                            const ivec& k,
                            bool log_scale, bool inclusive = true);

  /**
   * Change a grid of parameters created in the normal way (e.g., via
   * create_parameter_grid()) into a grid in the alternative form (as from
   * create_parameter_grid_alt()).
   * @param vals        Values in original representation: List of grid points.
   * @return  Values in alternative representation: For each dimension, list of
   *          values.
   */
  std::vector<vec>
  convert_parameter_grid_to_alt(const std::vector<vec>& vals);

  /**
   * Given an old grid of parameters and a value in it,
   * create a new grid of parameters to test s.t. the new grid is bounded by
   * the points in the old grid surrounding the given value.
   * The new grid will not contain any points from the old grid.
   * @param oldgrid  The old grid.
   * @param val      Value of interest.
   * @param k        Number of values in each dimension in the new grid (> 1).
   * @param log_scale  If true, make values evenly spaced on a log scale;
   *                   if true, the values must be positive.
   *
   * @return  constructed grid, in the form of a list of grid points
   */
  std::vector<vec>
  zoom_parameter_grid(const std::vector<vec>& oldgrid, const vec& val,
                      size_t k, bool log_scale);

  /**
   * Given an old grid of parameters and a value in it,
   * create a new grid of parameters to test s.t. the new grid is bounded by
   * the points in the old grid surrounding the given value.
   * The new grid will not contain any points from the old grid.
   * @param oldgrid  The old grid.
   * @param val      Value of interest.
   * @param k          Number of values in each dimension (>= 1).
   *                   If k(i) = 1, then minvals(i) must equal maxvals(i).
   * @param log_scale  If true, make values evenly spaced on a log scale;
   *                   if true, the values must be positive.
   *
   * @return  constructed grid, in the form of a list of grid points
   */
  std::vector<vec>
  zoom_parameter_grid(const std::vector<vec>& oldgrid, const vec& val,
                      const ivec& k, bool log_scale);

  /**
   * Given an old grid of parameters and a value in it,
   * create a new grid of parameters to test s.t. the new grid is bounded by
   * the points in the old grid surrounding the given value.
   * The new grid will not contain any points from the old grid.
   * This variant of the above method is for single-parameter grids.
   * @param oldgrid  The old grid (represented as a vec).
   * @param val      Value of interest.
   * @param k        Number of values in each dimension in the new grid (> 1).
   * @param log_scale  If true, make values evenly spaced on a log scale;
   *                   if true, the values must be positive.
   *
   * @return  constructed grid, in the form of a list of grid points
   */
  vec zoom_parameter_grid(const vec& oldgrid, double val, size_t k,
                          bool log_scale);

  /**
   * This is just like zoom_parameter_grid(), but it is for use with
   * create_parameter_grid_alt().
   * One difference: It may contain the value of interest from oldgrid.
   */
  std::vector<vec>
  zoom_parameter_grid_alt(const std::vector<vec>& oldgrid, const vec& val,
                          size_t k, bool log_scale);

  /**
   * This is just like zoom_parameter_grid(), but it is for use with
   * create_parameter_grid_alt().
   * One difference: It may contain the value of interest from oldgrid.
   */
  std::vector<vec>
  zoom_parameter_grid_alt(const std::vector<vec>& oldgrid, const vec& val,
                          const ivec& k, bool log_scale);

} // namespace prl

#include <prl/macros_undef.hpp>

#endif // PRL_LEARNING_DISCRIMINATIVE_PARAMETER_GRID_HPP
