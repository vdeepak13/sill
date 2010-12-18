#include <iostream>
#include <sstream>

#include <sill/factor/soft_max.hpp>
#include <sill/factor/constant_factor.hpp>
#include <sill/factor/gaussian_factors.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  // Constructors and conversion operators
  //============================================================================
  soft_max::soft_max(finite_variable* head, const var_vector& tail) 
    : head(head), tail(tail) {
    // initialize the matrix sizes
    // check the tail variables 
  }

  soft_max::soft_max(finite_variable* head, const var_vector& tail, 
                     const mat& w, const vec& tail)
    : head(head), tail(tail) {
    // do the rest
  }
    
  soft_max::soft_max(const constant_factor& other) : head() {
    
  }
  
  soft_max::operator std::string() const {
    std::ostringstream out; out << *this; return out.str(); 
  }

  // Factor operations
  //==========================================================================
  //! Evaluates the probabilities for each value
  vec soft_max::operator()(const vec& values) {
    assert(values.size() == size_tail());
    return exp(b + w*tail);
  }
  
  canonical_gaussian combine(const soft_max& f, const canonical_gaussian& cg) {
    
  }
  
  // Free functions
  //==========================================================================
  std::ostream& operator<<(std::ostream& out, const soft_max& factor) {
    out << "#F(SM|" << head << "," << tail << "|" << b << "|" << w << ")";
    return out;
  }

} // namespace sill

