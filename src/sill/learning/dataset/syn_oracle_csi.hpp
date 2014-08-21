#ifndef SILL_SYN_ORACLE_CSI_HPP
#define SILL_SYN_ORACLE_CSI_HPP

#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

#include <sill/assignment.hpp>
#include <sill/learning/dataset/oracle.hpp>
#include <sill/math/free_functions.hpp>
#include <sill/model/bayesian_network.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for generating discrete data from a Bayesian network
   * which exhibits Context-Specific Independence (CSI).
   * This creates a random Bayesian network as follows:
   *  - Create a graph by attaching each new variable to a random set of
   *    parents.  (Once 2 * NUM_PARENTS variables have been added, each new
   *    variables is given NUM_PARENTS parents.)
   *  - For each variable with > 1 parent,
   *    make it exhibit CSI with probability FRACTION_CSI.
   *     - If it is supposed to exhibit CSI, choose a random parent to be the
   *       context.  For each value, make that value cut the links with the
   *       other parents with probability FRACTION_CSI.  Add CSI_NOISE
   *       to the CPT values (which are sampled from (0,1] and then normalized).
   *
   * For more info on CSI, see, e.g.,
   * C. Boutilier, N. Friedman, M. Goldszmidt, and D. Koller (1996).
   * "Context-specific independence in Bayesian networks."
   * Proc. Twelfth Annual Conference on Uncertainty in Artificial Intelligence
   * (UAI '96) (pp. 115-123).
   *
   * \ingroup learning_dataset
   * \author Joseph Bradley
   *
   * THIS IS INCOMPLETE!
   */
  class syn_oracle_csi : public oracle {

    // Public type declarations
    //==========================================================================
  public:

    //! The base type (oracle)
    typedef oracle base;

    struct parameters {

      //! Number of parents for all but the first 2 * NUM_PARENTS variables.
      //!  (default = 3)
      size_t num_parents;

      //! Fraction in [0,1] of variables whose CPTs are designed to exhibit CSI.
      //!  (default = .4)
      double fraction_csi;

      //! Noise (>= 0) added to CPT values (to make CSI not hold strictly).
      //!  (default = .05)
      double csi_noise;

      //! Used to make this algorithm deterministic.
      //!  (default = time)
      double random_seed;

      parameters() : num_parents(3), fraction_csi(.4), csi_noise(.05) {
        std::time_t time_tmp;
        time(&time_tmp);
        random_seed = time_tmp;
      }

      bool valid() const {
        if (fraction_csi < 0 || fraction_csi > 1)
          return false;
        if (csi_noise < 0)
          return false;
        return true;
      }

    }; // class parameters

    // Private data members
    //==========================================================================
  private:

    //! parameters
    parameters params;

    //! random number generator
    boost::mt11213b rng;

    //! uniform [0,1] distribution
    boost::uniform_real<double> uniform_prob;

    //! Bayes net used to generate samples
    bayesian_network<tablef> bn;

    //! Current record
    record current_rec;

    // Private methods
    //==========================================================================

    //! Initialize the oracle
    void init() {
      assert(params.valid());

      // Initialize the random number generator
      rng.seed(static_cast<unsigned>(params.random_seed));
      uniform_prob = boost::uniform_real<double>(0,1);

      // Construct the graph and make CPTs
      if (num_finite() > 2 * params.num_parents) {
        for (size_t j = 2 * params.num_parents; j < num_finite(); ++j) {
          // Choose NUM_PARENTS random parents
          finite_domain parents;
          std::vector<size_t> r(randperm(j, rng));
          for (size_t k = 0; k < params.num_parents; ++k)
            parents.insert(finite_seq[r[k]]);
          parents.insert(finite_seq[j]);
          tablef f(parents.plus(finite_seq[j]));
          // RIGHT HERE NOW: MAKE FACTOR
          bn.add_factor(parents, finite_seq[j]);
        }
      }


    }

    // Constructors
    //==========================================================================
  public:

    /**
     * Constructs a synthetic oracle for data exhibiting CSI.
     * @param var_order  variables to be used for this dataset
     */
    explicit syn_oracle_csi(const finite_var_vector& var_order,
                            const parameters& params = parameters())
      : base(var_order, vector_var_vector(), std::vector<variable::variable_typenames>()),
        params(params), bn(finite_domain(var_order)),
        current_rec(finite_numbering_, vector_numbering_, dvector) {
      init();
    }

    // Getters and helpers
    //==========================================================================

    //! Returns the current record.
    const record& current() const {
      return current_rec;
    }

    template <typename Char, typename Traits>
    void write(std::basic_ostream<Char, Traits>& out) const {
      out << "csi oracle using Bayes net:\n" << bn << "\n";
    }

    // Mutating operations
    //==========================================================================

    //! Increments the oracle to the next record.
    //! This returns true iff the operation was successful; false indicates
    //! a depleted oracle.
    //! NOTE: This must be explicitly called after the oracle is constructed.
    bool next() {
      assert(false);
      return true;
    }

  }; // class syn_oracle_csi

  // Free functions
  //==========================================================================

  template <typename Char, typename Traits>
  std::basic_ostream<Char,Traits>&
  operator<<(std::basic_ostream<Char, Traits>& out,
             const syn_oracle_csi& csi) {
    csi.write(out);
    return out;
  }

  /**
   * Constructs a synthetic oracle for csi data, creating new variables
   * in universe u.
   * @param nfeatures number of features (non-class variables)
   * @param u         universe
   * @param params    other parameters for syn_oracle_csi
   */
  syn_oracle_csi
  create_syn_oracle_csi
  (size_t nfeatures, universe& u,
   const syn_oracle_csi::parameters& params
   = syn_oracle_csi::parameters()) {
    assert(false);
    assert(nfeatures > 0);
    finite_var_vector var_order;
    for (size_t j = 0; j < nfeatures+1; j++)
      var_order.push_back(u.new_finite_variable(2));
    return syn_oracle_csi(var_order, params);
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_SYN_ORACLE_CSI_HPP
