#include <sill/base/universe.hpp>
#include <sill/learning/dataset/syn_oracle_knorm.hpp>

#include <sill/macros_def.hpp>

namespace sill {

    // Private methods
    //==========================================================================

    void syn_oracle_knorm::init() {
      assert(params.valid());
      // Initialize the random number generator
      rng.seed(static_cast<unsigned>(params.random_seed));
//      normal_dist = boost::normal_distribution<double>(0, params.std_dev);
      discrete_uniform_dist = boost::uniform_smallint<>(0, nmeans_ - 1);

      // Choose the centers:
      // first, create a random number generator for the centers
      boost::uniform_real<> uniform_dist(-1,1);
      // now choose some centers
      centers.resize(nmeans_);
      for (size_t k = 0; k < nmeans_; k++) {
        centers[k].set_size(nfeatures_);
        for (size_t j = 0; j < nfeatures_; j++)
          centers[k][j] = uniform_dist(rng);
      }
      #ifdef SILL_SYN_ORACLE_KNORM_HPP_VERBOSE
      std::cerr << " Centers:\n";
      for (size_t k = 0; k < nmeans_; k++) {
        std::cerr << "\t";
        for (size_t j = 0; j < nfeatures_; j++)
          std::cerr << centers[k][j] << " ";
        std::cerr << std::endl;
      }
      #endif
      // find the distances between all centers
      std::vector<std::vector<double> > nearest_neighbors(nmeans_ - 1);
      for (size_t k = 0; k < nmeans_ - 1; k++) {
        // (k indexes centers and nearest_neighbors)
        nearest_neighbors[k].resize(nmeans_ - k - 1);
        for (size_t i = k+1; i < nmeans_; i++) {
          size_t k2 = i - (k+1);
          // (i indexes centers; k2 indexes nearest_neighbors)
          nearest_neighbors[k][k2] = 0;
          for (size_t j = 0; j < nfeatures_; j++)
            nearest_neighbors[k][k2] += (centers[k][j] - centers[i][j])
              * (centers[k][j] - centers[i][j]);
          nearest_neighbors[k][k2] = sqrt(nearest_neighbors[k][k2]);
        }
      }
      #ifdef SILL_SYN_ORACLE_KNORM_HPP_VERBOSE
      std::cerr << " Distances:\n";
      for (size_t k = 0; k < nmeans_ - 1; k++) {
        std::cerr << "\t";
        for (size_t k2 = 0; k2 < nmeans_ - k - 1; k2++)
          std::cerr << nearest_neighbors[k][k2] << " ";
        std::cerr << std::endl;
      }
      #endif
      // find the nearest neighbor for each center, and average the distances
      double avg_distance = 0;
      for (size_t k = 0; k < nmeans_ - 1; k++) {
        double min_distance = std::numeric_limits<double>::max();
        for (size_t i = 0; i < k; i++) {
          // (i indexes centers and nearest_neighbors)
          if (min_distance > nearest_neighbors[i][k])
            min_distance = nearest_neighbors[i][k];
        }
        for (size_t i = k+1; i < nmeans_; i++) {
          size_t k2 = i - (k+1);
          // (i indexes centers; k2 indexes nearest_neighbors)
          if (min_distance > nearest_neighbors[k][k2])
            min_distance = nearest_neighbors[k][k2];
        }
        avg_distance += min_distance;
      }
      avg_distance /= (nmeans_ - 1);
      // scale all feature values by dividing by avg_distance and multiplying
      // by 2 * RADIUS
      double scale = 2. * params.radius / avg_distance;
      for (size_t k = 0; k < nmeans_; k++) {
        for (size_t j = 0; j < nfeatures_; j++)
          centers[k][j] *= scale;
      }
      #ifdef SILL_SYN_ORACLE_KNORM_HPP_VERBOSE
      std::cerr << " Average min distance = " << avg_distance << std::endl;
      std::cerr << " Rescaled centers:\n";
      for (size_t k = 0; k < nmeans_; k++) {
        std::cerr << "\t";
        for (size_t j = 0; j < nfeatures_; j++)
          std::cerr << centers[k][j] << " ";
        std::cerr << std::endl;
      }
      #endif
    }

    double syn_oracle_knorm::normal_dist() {
      double s = 0;
      for (size_t i = 0; i < NDRAWS; i++)
        s += (boost::uniform_int<>(0, BIG_INT)(rng)
              / static_cast<double>(BIG_INT));
      return (params.std_dev * (s - NDRAWS * .5)
              / sqrt(static_cast<double>(NDRAWS)));
    }

    // Mutating operations
    //==========================================================================

    bool syn_oracle_knorm::next() {
      // choose a center
      size_t k = discrete_uniform_dist(rng);
      current_rec.fin_ptr->operator[](0) = k;
      // choose feature values
      for (size_t j = 0; j < nfeatures_; j++) {
        current_rec.vec_ptr->operator[](j) = centers[k][j] + normal_dist();
      }
      return true;
    }

  // Free functions
  //==========================================================================

  syn_oracle_knorm
  create_syn_oracle_knorm
  (size_t k, size_t nfeatures, universe& u,
   syn_oracle_knorm::parameters params) {
    assert(k >= 2);
    assert(nfeatures > 0);
    vector_var_vector vector_var_order;
    std::vector<variable::variable_typenames> var_type_order
      (nfeatures, variable::VECTOR_VARIABLE);
    for (size_t j = 0; j < nfeatures; j++)
      vector_var_order.push_back(u.new_vector_variable(1));
    var_type_order.push_back(variable::FINITE_VARIABLE);
    return syn_oracle_knorm(vector_var_order, u.new_finite_variable(k),
                            var_type_order, params);
  }

} // namespace sill

#include <sill/macros_undef.hpp>
