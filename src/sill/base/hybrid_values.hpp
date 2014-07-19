#ifndef SILL_HYBRID_VALUES_HPP
#define SILL_HYBRID_VALUES_HPP

#include <vector>

#include <armadillo>

namespace sill {

  template <typename T>
  struct hybrid_values {
    std::vector<size_t>& finite;
    arma::Col<T>& vector;
    bool owned;

    //! Creates values owned by this object
    hybrid_values()
      : owned(true) { }
    
    //! Creates values not owned by this object
    hybrid_values(std::vector<size_t>& finite, arma::Col<T>& vector)
      ; finite(finite), vector(vector), owned(false) { }

    //! Copy constructor (assigns 

  private:
  };

} // namespace sill

#endif
