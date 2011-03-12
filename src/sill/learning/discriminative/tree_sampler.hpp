
#ifndef SILL_LEARNING_DISCRIMINATIVE_TREE_SAMPLER_HPP
#define SILL_LEARNING_DISCRIMINATIVE_TREE_SAMPLER_HPP

#include <fstream>
#include <sstream>
#include <string>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

//#include <sill/learning/discriminative/concepts.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for efficiently sampling from a large finite distribution
   * (over a finite set of indices) by using a tree to sample in log(n) time.
   * This expects the distribution to be updated and sampled from alternately;
   * after it has been updated, commit_update() must be called before it
   * is sampled from again.
   *
   * \author Joseph Bradley
   * @todo Use multinomial_distribution instead.
   */
  class tree_sampler {

    static const bool debug = true;

    // Public type declarations
    //==========================================================================
  public:

    struct parameters {

      //! Used to make the algorithm deterministic
      //!  (default = time)
      double random_seed;

      //! Set to false to turn off sampling (to make this faster)
      //!  (default = true)
      bool allow_sampling;

      parameters() : random_seed(time(NULL)), allow_sampling(true) {
      }

      void save(std::ofstream& out) const {
        out << random_seed << " " << allow_sampling;
      }

      void load(std::istringstream& is) {
        if (!(is >> random_seed))
          assert(false);
        if (!(is >> allow_sampling))
          assert(false);
      }

    }; // struct parameters

    // Private data members
    //==========================================================================
  private:

    parameters params;

    //! random number generator
    mutable boost::mt11213b rng;

    //! uniform distribution over [0, 1]
    mutable boost::uniform_real<double> uniform_prob;

    //! Distribution
    std::vector<double> distrib;

    //! Size of distribution
    size_t n;

    //! Balanced binary tree over indices
    //! tree[i] = weight of indices in subtree at node i
    std::vector<double> tree;

    //! Depth of tree (not including distrib, which isn't stored in tree)
    size_t depth;

    //! depth_offsets[i] = offset in tree for left-most node at depth i
    std::vector<size_t> depth_offsets;

    // Private methods
    //==========================================================================

    void init();

    //! Recursive sampling function:
    //!  test node i (from left) versus node i+1 at depth d
    size_t sample(double r, size_t d, size_t i) const;

    //! Recursive sampling function:
    //!  test node i (from left) versus node i+1 at depth d
    size_t sample2(double r, size_t d, size_t i) const;

    //! Read in a vector of values [val1,val2,...], ignoring an initial space
    //! if necessary.
    //! \todo Make this more general, and move it to stl_io.hpp
    template <typename Char, typename Traits, typename U>
    void read_vec(std::basic_istream<Char,Traits>& in,
                  std::vector<U>& vec) {
      char c;
      U val;
      vec.clear();
      in.get(c);
      if (c == ' ')
        in.get(c);
      assert(c == '[');
      if (in.peek() != ']') {
        do {
          if (!(in >> val))
            assert(false);
          vec.push_back(val);
          if (in.peek() == ',')
            in.ignore(1);
        } while (in.peek() != ']');
      }
      in.ignore(1);
    }

    // Constructors and destructors
    //==========================================================================
  public:

    /**
     * Construct an empty tree_sampler.
     */
    tree_sampler() : n(0) {
      params.allow_sampling = false;
    }

    /**
     * Construct a tree_sampler.
     * @param Distribution  must implement: double operator[](size_t);
     *                      does not need to be normalized
     * @param n             number of indices
     */
    template <typename Distribution>
    tree_sampler(const Distribution& distrib, size_t n,
                 parameters params = parameters())
      : params(params), n(n) {
      assert(n > 0);
      this->distrib.resize(n);
      for (size_t i = 0; i < n; ++i) {
        assert(distrib[i] >= 0);
        this->distrib[i] = distrib[i];
      }
      init();
      commit_update();
    }

    /**
     * Construct a tree_sampler with a distribution which must be filled in
     * (after which commit_update() must be called before sampling).
     * @param n             number of indices
     */
    explicit tree_sampler(size_t n, parameters params = parameters())
      : params(params), n(n) {
      assert(n > 0);
      this->distrib.resize(n);
      for (size_t i = 0; i < n; ++i)
        this->distrib[i] = 0;
      init();
    }

    // Getters and helpers
    //==========================================================================

    //! Returns distribution
    const std::vector<double>& distribution() const {
      return distrib;
    }

    //! Sample from distribution
    size_t sample() const;

    //! Prints out the tree_sampler (for debugging)
    template <typename Char, typename Traits>
    std::basic_ostream<Char,Traits>&
    write(std::basic_ostream<Char,Traits>& out) const {
      out << "tree_sampler: n = " << n << ", depth = " << depth
          << "\n  tree = " << tree
          << "\n  distribution = " << distrib;
      return out;
    }   

    //! This should only be called after commit_update() has been called.
    void check_validity() const;

    // Mutating operations
    //==========================================================================

    //! Set value of distribution element i.
    void set(size_t i, double val);

    //! Commit updates made via set()
    void commit_update();

    // Save and load methods
    //==========================================================================

    //! Output the tree_sampler to a human-readable file which can be reloaded.
    //! NOTE: This does not save the state of the random number generator.
    void save(std::ofstream& out) const;

    /**
     * Input the tree_sampler from a human-readable file.
     * @param is    input filestream for file holding the saved tree_sampler
     */
    void load(std::ifstream& in);

  }; // class tree_sampler

  // Free functions
  //==========================================================================

  //! Writes a human-readable representation of the tree_sampler.
  template <typename Char, typename Traits>
  std::basic_ostream<Char, Traits>&
  operator<<(std::basic_ostream<Char, Traits>& out,
             const tree_sampler& r) {
    r.write(out);
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_TREE_SAMPLER_HPP
