#ifndef SILL_SYN_ORACLE_BAYES_NET_HPP
#define SILL_SYN_ORACLE_BAYES_NET_HPP

#include <boost/random/mersenne_twister.hpp>

#include <sill/model/bayesian_network.hpp>
#include <sill/learning/dataset_old/oracle.hpp>

#include <sill/macros_def.hpp>


namespace sill {

  /**
   * Class for generating synthetic data from a Bayesian network.
   *
   * @param Factor  type of factor used in Bayesian network
   *
   * \author Joseph Bradley
   * \ingroup learning_dataset
   */
  template <typename Factor>
  class syn_oracle_bayes_net : public oracle<> {

    // Public type declarations
    //==========================================================================
  public:

    //! The base type (oracle)
    typedef oracle<> base;

    struct parameters {

      //! Used to make this algorithm deterministic.
      //!  (default = time)
      double random_seed;

      parameters() {
        std::time_t time_tmp;
        time(&time_tmp);
        random_seed = time_tmp;
      }

    };  // struct parameters

    // Private data members
    //==========================================================================
  private:

    parameters params;

    //! Bayesian network from which to sample
    const bayesian_network<Factor>& bn;

    //! random number generator
    boost::mt11213b rng;

    //! Current record
    record<> current_rec;

    // Constructors
    //==========================================================================
  public:

    /**
     * Constructs a synthetic oracle for data generated from a Bayesian network.
     * @param vector_var_order  vector variables for dataset features
     * @param class_variable    finite variable for class
     * @param var_type_order    order of variable types
     */
    explicit syn_oracle_bayes_net(const bayesian_network<Factor>& bn,
                                  parameters params = parameters())
      : base(bn.arguments(), vector_var_vector(),
             std::vector<variable::variable_typenames>
             (bn.arguments().size(), variable::FINITE_VARIABLE)),
        params(params), bn(bn),
        current_rec(finite_numbering_ptr_, vector_numbering_ptr_, dvector) {
      rng.seed(static_cast<unsigned>(params.random_seed));
    }

    // Getters and helpers
    //==========================================================================

    //! Returns the current record.
    const record<>& current() const {
      return current_rec;
    }

    void write(std::ostream& out) const {
      out << "Bayesian network oracle using model:\n"
          << bn << "\n";
    }

    // Mutating operations
    //==========================================================================

    //! Increments the oracle to the next record.
    //! This returns true iff the operation was successful; false indicates
    //! a depleted oracle.
    //! NOTE: This must be explicitly called after the oracle is constructed.
    bool next() {
      finite_assignment a(bn.sample(rng));
      current_rec = a;
      return true;
    }

  }; // class syn_oracle_bayes_net

  // Free functions
  //==========================================================================

  template <typename Factor>
  std::ostream&
  operator<<(std::ostream& out,
             const syn_oracle_bayes_net<Factor>& model) {
    model.write(out);
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_SYN_ORACLE_BAYES_NET_HPP
