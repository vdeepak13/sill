#ifndef SILL_HYBRID_VALUES_HPP
#define SILL_HYBRID_VALUES_HPP

#include <sill/global.hpp>

#include <vector>

#include <armadillo>

namespace sill {

  template <typename T>
  struct hybrid_values {
    std::vector<size_t> finite;
    arma::Col<T> vector;

    //! Constructs empty (zero-length) values
    hybrid_values() { }

    //! Constructs values of given lengths
    hybrid_values(size_t nfinite, size_t nvector)
      : finite(nfinite), vector(nvector, arma::fill::zeros) { }

    //! Constructs values using the given finite and vector components
    hybrid_values(const std::vector<size_t>& finite,
                  const arma::Col<T>& vector)
      : finite(finite), vector(vector) { }

    //! Resizes the values
    void resize(size_t nfinite, size_t nvector) {
      finite.resize(nfinite);
      vector.resize(nvector);
    }

    //! Returns true if the two values are equal
    bool operator==(const hybrid_values& other) const {
      return finite == other.finite 
        && vector.size() == other.vector.size()
        && all(vector == other.vector);
    }

  }; // struct hybrid_values

} // namespace sill

#endif
