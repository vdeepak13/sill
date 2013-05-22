
#ifndef SILL_SYN_ORACLE_MAJORITY_HPP
#define SILL_SYN_ORACLE_MAJORITY_HPP

#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

#include <sill/learning/dataset/oracle.hpp>

#include <sill/macros_def.hpp>

// uncomment to print debugging information
//#define SILL_SYN_ORACLE_MAJORITY_HPP_VERBOSE

namespace sill {

  /**
   * Class for generating synthetic data from a majority vote rule.
   * This has N binary features, with R voting ones, whose values are chosen
   * independently and uniformly at random.  The value of the class variable
   * is determined by a majority vote of the R voting features.
   * Noise may be added to the feature values or the class variable value.
   * \ingroup learning_dataset
   * @author Joseph Bradley
   */
  class syn_oracle_majority : public oracle<dense_linear_algebra<> > {

    // Public type declarations
    //==========================================================================
  public:

    typedef dense_linear_algebra<> la_type;

    //! The base type (oracle)
    typedef oracle<la_type> base;

    typedef record<la_type> record_type;

    struct parameters {

      //! Fraction of voting binary features.
      //!  (default = .4)
      double r_vars;

      //! Probability with which feature values are flipped
      //!  (default = 0)
      double feature_noise;

      //! Probability with which the class variable value is flipped
      //!  (default = 0)
      double label_noise;

      //! Used to make this algorithm deterministic.
      //!  (default = time)
      double random_seed;

      parameters()
        : r_vars(.4), feature_noise(0), label_noise(0) {
        std::time_t time_tmp;
        time(&time_tmp);
        random_seed = time_tmp;
      }

      bool valid() const {
        if (feature_noise < 0 || feature_noise > 1)
          return false;
        if (label_noise < 0 || label_noise > 1)
          return false;
        return true;
      }

      template <typename Char, typename Traits>
      void write(std::basic_ostream<Char, Traits>& out) const {
        out << "parameters: r_vars=" << r_vars
            << ", feature_noise=" << feature_noise
            << ", label_noise=" << label_noise
            << ", random_seed=" << random_seed;
      }

    }; // struct parameters

    // Private data members
    //==========================================================================
  private:

    //! parameters
    parameters params;

    //! random number generator
    boost::mt11213b rng;

    //! uniform 0/1 distribution
    boost::bernoulli_distribution<double> bernoulli_dist;

    //! uniform [0,1] distribution
    boost::uniform_real<double> uniform_prob;

    //! Indices of the r relevant/voting features
    std::vector<size_t> voting;

    //! Current record
    record_type current_rec;

    // Private methods
    //==========================================================================

    //! Initialize the oracle
    void init();

    // Constructors
    //==========================================================================
  public:

    /**
     * Constructs a synthetic oracle for majority data.
     * @param var_order  variables to be used for this dataset
     *                   If var_order has size n, then the first n-1 variables
     *                   will be used as features and the last as the class
     *                   variable.  All must be binary variables.
     */
    explicit syn_oracle_majority(const finite_var_vector& var_order,
                                 const parameters& params = parameters())
      : base(var_order, vector_var_vector(), std::vector<variable::variable_typenames>()),
        params(params),
        current_rec(finite_numbering_ptr_, vector_numbering_ptr_, dvector) {
      finite_class_vars.push_back(finite_seq[finite_seq.size() - 1]);
      init();
    }

    // Getters and helpers
    //==========================================================================

    //! Returns the current record.
    const record_type& current() const {
      return current_rec;
    }

    template <typename Char, typename Traits>
    void write(std::basic_ostream<Char, Traits>& out) const {
      out << "majority oracle\n";
      params.write(out);
      out << " Voting features:";
      foreach(size_t j, voting)
        out << " " << j;
      out << "\n";
    }

    // Mutating operations
    //==========================================================================

    //! Increments the oracle to the next record.
    //! This returns true iff the operation was successful; false indicates
    //! a depleted oracle.
    //! NOTE: This must be explicitly called after the oracle is constructed.
    bool next();

  }; // class syn_oracle_majority

  // Free functions
  //==========================================================================

  template <typename Char, typename Traits>
  std::basic_ostream<Char,Traits>&
  operator<<(std::basic_ostream<Char, Traits>& out,
             const syn_oracle_majority& majority) {
    majority.write(out);
    return out;
  }

  /**
   * Constructs a synthetic oracle for majority data, creating new variables
   * in universe u.
   * @param nfeatures number of features (non-class variables)
   * @param u         universe
   * @param params    other parameters for syn_oracle_majority
   */
  syn_oracle_majority
  create_syn_oracle_majority
  (size_t nfeatures, universe& u,
   const syn_oracle_majority::parameters& params
   = syn_oracle_majority::parameters());

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_SYN_ORACLE_MAJORITY_HPP
