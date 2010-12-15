#include <prl/learning/discriminative/tree_sampler.hpp>

#include <prl/macros_def.hpp>

namespace prl {

    // Private methods
    //==========================================================================

    void tree_sampler::init() {
      if (params.allow_sampling) {
        using std::pow;
        depth = static_cast<size_t>(ceil(std::log(static_cast<double>(n))
                                         /std::log(2.)));
        tree.clear();
        tree.resize((size_t)(pow(2.,static_cast<double>(depth))));
        depth_offsets.resize(depth);
        depth_offsets[0] = 0;
        for (size_t i = 1; i < depth; ++i)
          depth_offsets[i] = 2 * (depth_offsets[i-1] + 1) - 1;
      }
      rng.seed(static_cast<unsigned>(params.random_seed));
      uniform_prob = boost::uniform_real<double>(0,1);
    }

    size_t tree_sampler::sample(double r, size_t d, size_t i) const {
      if (d == depth) {
        if (r <= distrib[i])
          return i;
        else
          return i+1;
      }
      if (r <= tree[depth_offsets[d] + i])
        return sample(r, d+1, 2*i);
      else
        return sample(r - tree[depth_offsets[d]+i], d+1, 2*(i+1));
    } 

    size_t tree_sampler::sample2(double r, size_t d, size_t i) const {
      std::cout << "sample2(" << r << "," << d << "," << i << "):\n"
                << "\t distrib[i] = " << distrib[i] << "\n"
                << "\t depth_offsets[d] = " << depth_offsets[d] << "\n"
                << "\t tree[depth_offsets[d] + i] = "
                << tree[depth_offsets[d] + i] << "\n" << std::endl;
      if (d == depth) {
        if (r <= distrib[i])
          return i;
        else
          return i+1;
      }
      if (r <= tree[depth_offsets[d] + i])
        return sample2(r, d+1, 2*i);
      else
        return sample2(r - tree[depth_offsets[d]+i], d+1, 2*(i+1));
    }

    // Getters and helpers
    //==========================================================================

    size_t tree_sampler::sample() const {
      assert(params.allow_sampling);
      double r = uniform_prob(rng);
      if (depth == 0)
        return 0;
      size_t i = sample(r, 1, 0);
      if (i >= n) {
        // This should never happen, save for numerical reasons.
        if (debug)
          std::cerr << "tree_sampler::sample(): sampled i = " << i
                    << " but n = " << n << std::endl;
        i = sample2(r,1,0);
        if (debug)
          std::cerr << "tree_sampler::sample2(): sampled i = " << i
                    << " but n = " << n << std::endl;
        return n-1;
      }
      return i;
    }

    void tree_sampler::check_validity() const {
      double total = 0;
      for (size_t i = 0; i < n; ++i)
        total += distrib[i];
      if (fabs(total - 1) > .0000001) {
        std::cerr << "tree_sampler::check_validity(): ERROR: distribution "
                  << "sums to " << total << std::endl;
        assert(false);
      }
      if (params.allow_sampling) {
        for (size_t d = 0; d < depth; ++d) {
          total = 0;
          for (size_t i = 0; i < depth_offsets[d] + 1; ++i)
            total += tree[depth_offsets[d] + i];
          if (fabs(total - 1) > .0000001) {
            std::cerr << "tree_sampler::check_validity(): ERROR: tree at depth "
                      << d << " sums to " << total << std::endl;
            assert(false);
          }
        }
      }
    }

    // Mutating operations
    //==========================================================================

    void tree_sampler::set(size_t i, double val) {
      assert(i <= n);
      assert(val >= 0);
      distrib[i] = val;
    }

    void tree_sampler::commit_update() {
      // Normalize distribution
      double dist_norm = 0;
      for (size_t i = 0; i < n; ++i)
        dist_norm += distrib[i];
      if (dist_norm > 0)
        for (size_t i = 0; i < n; ++i)
          distrib[i] /= dist_norm;
      else {
        std::cerr << "error: tree_sampler::commit_update() was called when the"
                  << " distribution was 0" << std::endl;
        assert(false);
      }
      // Update tree
      if (params.allow_sampling) {
        for (size_t i = 0; i < n; ++i) {
          size_t parent = i / 2 + depth_offsets[depth-1];
          tree[parent] = distrib[i];
          ++i;
          if (i < n)
            tree[parent] += distrib[i];
        }
        for (size_t d = 0; d < depth - 1; ++d) {
          size_t d2 = depth - 2 - d;
          for (size_t i = 0; i < depth_offsets[d2] + 1; ++i) {
            size_t childL = depth_offsets[d2 + 1] + 2 * i;
            size_t childR = childL + 1;
            tree[depth_offsets[d2] + i] = tree[childR] + tree[childL];
          }
        }
      }
    }

    // Save and load methods
    //==========================================================================

    void tree_sampler::save(std::ofstream& out) const {
      params.save(out);
      out << " " << distrib << " " << n << " " << tree << " " << depth << " "
          << depth_offsets << "\n";
    }

    void tree_sampler::load(std::ifstream& in) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      params.load(is);
      read_vec(is, distrib);
      if (!(is >> n))
        assert(false);
      read_vec(is, tree);
      if (!(is >> depth))
        assert(false);
      read_vec(is, depth_offsets);
      rng.seed(static_cast<unsigned>(params.random_seed));
      uniform_prob = boost::uniform_real<double>(0,1);
    }

} // namespace prl

#include <prl/macros_undef.hpp>
