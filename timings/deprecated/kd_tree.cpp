#include <iostream>
#include <vector>
#include <algorithm>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/progress.hpp>
#include <prl/datastructure/kd_tree.hpp>
#include <prl/set.hpp>
#include <prl/set_insert_iterator.hpp>

/////////////////////////////////////////////////////////////////
// Needs to be cleaned up; doesn't compile.
/////////////////////////////////////////////////////////////////

/**
 * A comparator that orders points by their distance from a fixed
 * query point.
 */
struct euclidean_dist_comparator_t {

  //! The number of dimensions.
  std::size_t d;

  //! The fixed point's coordinates.
  const double *query;

  //! Constructor.
  euclidean_dist_comparator_t(const double* query, std::size_t d)
    : query(query), d(d) { }

  //! The comparison function.
  bool operator()(const double* a, const double* b) const {
    double dist_a = 0.0;
    double dist_b = 0.0;
    for (std::size_t i = 0; i < d; ++i) {
      dist_a += (a[i] - query[i]) * (a[i] - query[i]);
      dist_b += (b[i] - query[i]) * (b[i] - query[i]);
    }
    return (dist_a < dist_b);
  }

};

int main(int argc, char** argv) {

  // Choose the number of points and dimensions.
  std::size_t n = 1000000;
  std::size_t d = 3;
  if (argc > 1)
    n = static_cast<std::size_t>(atoi(argv[1]));
  if (argc > 2)
    d = static_cast<std::size_t>(atoi(argv[2]));

  // Make a data set of random points in the d-dimensional hypercube.
  std::cout << "Generating " << n << " " <<
    d << "-dimensional points..." << std::flush;
  double time;
  boost::mt19937 rng;
  boost::uniform_01<boost::mt19937, double> unif(rng);
  typedef double* point_t;
  std::vector<double*> pts(n);
  for (std::size_t i = 0; i < n; i++) {
    pts[i] = new double[d];
    for (std::size_t j = 0; j < d; j++)
      pts[i][j] = unif();
  }

  // Make the kd tree.
  std::cout << "done." << std::endl << "Building kd-tree..." << std::flush;
  typedef prl::kd_tree_t<std::size_t, double, const double*> kdt_t;
  kdt_t* kd_tree_ptr;
  const std::size_t max_pts_per_leaf_c = 10;
  {
    boost::timer t;
    kd_tree_ptr = new kdt_t(d, pts.begin(), pts.end(), max_pts_per_leaf_c);
    time = t.elapsed();
  }
  std::cout << "done." << std::endl << "Built kd-tree for "
            << n << " " << d << "-dimensional points in "
            << time << "s." << std::endl;

  // Verify the kd tree is correct.
  {
    boost::timer t;
    kd_tree_ptr->verify();
    time = t.elapsed();
  }
  std::cout << "Verified-kd tree in "
            << time << "s." << std::endl;

  // Print out the leaves.
  // kd_tree_ptr->report(std::cout);

  // Perform some k nearest neighbors searches.
  const std::size_t k = 10;
  const std::size_t num_trials_c = 10;
  double kd_tree_time = 0.0;
  double exhaustive_time = 0.0;
  typedef prl::set<const double*> point_set_t;
  for (std::size_t trial = 0; trial < num_trials_c; ++trial) {

    // Use the kd-tree.
    point_set_t nearest_neighbors;
    {
      boost::timer t;
      kd_tree_ptr->nearest_neighbors(pts[0], k,
                                     prl::set_inserter(nearest_neighbors));
      kd_tree_time += t.elapsed();
    }

    // Compute the correct answer without the index.
    std::vector<double*> pts_copy(pts);
    euclidean_dist_comparator_t cmp(pts[0], d);
    point_set_t true_nearest_neighbors;
    {
      boost::timer t;
      std::nth_element(pts_copy.begin(), pts_copy.begin() + k,
                       pts_copy.end(), cmp);
      std::copy(pts_copy.begin(), pts_copy.begin() + k,
                prl::set_inserter(true_nearest_neighbors));
      exhaustive_time += t.elapsed();
    }

    // Check that the kd-tree algorithm works.
    if (nearest_neighbors != true_nearest_neighbors)
      std::cout << "ERROR: kd-tree " << k
                << " nearest neighbors incorrect!" << std::endl;
  }
  // Report the time difference.
  std::cout << "Performed " << num_trials_c << " " << k
            << "-nearest-neighbor searches.  Time required:" << std::endl
            << "  kd-tree: " << kd_tree_time << std::endl
            << "  exhaustive: " << exhaustive_time << std::endl;

  // Deallocate the tree.
  delete kd_tree_ptr;

  return EXIT_SUCCESS;
}
