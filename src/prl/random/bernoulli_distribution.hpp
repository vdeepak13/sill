#ifndef SILL_BERNOULLI_DISTRIBUTION_HPP
#define SILL_BERNOULLI_DISTRIBUTION_HPP

#include <cassert>
#include <iosfwd>

namespace sill {

  /**
   * Bernoulli distribution: p(true) = p, p(false) = 1-p;
   */
  class bernoulli_distribution {
  public:
    //! The type of outcomes
    typedef bool result_type;

    //! The required number representation of the random number generator
    typedef int input_type;

    //! Standard constructor
    explicit bernoulli_distribution(double p) : p_(p) { 
      assert(p >=0 && p <=1);
    }

    //! Returns the probability
    double p() const {
      return p_;
    }

    //! What is this supposed to do?
    void reset() { }

    //! Draws a sample from the distribution using the specified r.n.g.
    template <typename Engine>
    bool operator()(Engine& eng) const {
      // concept_assert((RandomNumberGenerator<Engine>));
      if (p_ == 0)
        return false;
      else
        return double(eng() - eng.min()) <= p_ * (eng.max()-eng.min());
    }

    //! Prints the distribution to a stream
    friend std::ostream& 
    operator<<(std::ostream& out, const bernoulli_distribution& d);
    
    //! Loads the distribution parameter from a stream
    friend std::istream&
    operator>>(std::istream& in, bernoulli_distribution& d);

  private:
    double p_;
  
  }; // class bernoulli_distribution

} // namespace sill

#endif
