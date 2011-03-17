#ifndef SILL_MULTINOMIAL_DISTRIBUTION_HPP
#define SILL_MULTINOMIAL_DISTRIBUTION_HPP

#include <sill/math/operations.hpp>
#include <sill/math/vector.hpp>

namespace sill {
  
  /**
   * A class that represents a multinomial distribution:
   * \f$p(x = i) = p_i\f$ for \f$i = 0, \ldots, n-1\f$.
   * 
   * \todo if this distribution is used repeatedly, it may be more
   * efficient to pre-compute the cumulative sum and perform a binary
   * search foreach sample generation.
   */
  class multinomial_distribution {
    
    //! The probability vector
    vec p_;
    
  public:
    //! The type of outcomes
    typedef size_t result_type;

    //! The number representation of the random number generator
    typedef int input_type;

  public:
    //! Standard constructor
    explicit multinomial_distribution(const vec& p);

    //! Returns the probability
    const vec& p() const {
      return p_;
    }

    //! Returns the expected value of the distribution
    double mean() const;

    //! What is this supposed to do?
    void reset() { }

    //! Draws a sample from the distribution using the specified generator
    template <typename Engine>
    size_t operator()(Engine& eng) const {
      // concept_assert((RandomNumberGenerator<Engine>));
      double toss = double(eng() - eng.min())/(eng.max() - eng.min());
      for(size_t i = 0; i < p_.size(); i++) {
        toss -= p_[i];
        if (toss <= 0) return i;
      }
      return p_.size() - 1;
    }

    //! Prints the distribution to a stream
    friend std::ostream& 
    operator<<(std::ostream& out, const multinomial_distribution& d);
    
    //! Loads the distribution parameter from a stream
    friend std::istream&
    operator>>(std::istream& in, multinomial_distribution& d);

  }; // class multinomial_distribution

} // namespace sill

#endif
