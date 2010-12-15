#ifndef PRL_MATH_CONCEPTS_HPP
#define PRL_MATH_CONCEPTS_HPP

#include <prl/global.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /** 
   * A concept that represents a linear algebra.
   * Note that the functions and binary operations may not necessarily 
   * return objects of type value_type, vector_type, or matrix_type,
   * but they are guaranteed to return objects that are convertible to
   * the given expression type.
   * In addition, the following fucntions should be supported
   * svd, det, ...
   *
   * \ingroup math_concepts
   */
  template <typename LA>
  struct LinearAlgebra {
    
    //! The type that represents the field
    typedef typename LA::value_type value_type;
    
    //! The type that represents a vector
    typedef typename LA::vector_type vector_type;
    //concept_assert((Vector<vector_type>));

    //! The type that represents a matrix
    typedef typename LA::matrix_type matrix_type;
    //concept_assert((Matrix<matrix_type>));

    //! An open range of indices
    typedef typename LA::index_range index_range;

    //! An array that can be used to extract a subvector/submatrix
    typedef typename LA::index_array index_array;

    //! The type that represents a mutable vector reference
    typedef typename LA::vector_reference vector_reference;

    //! The type that represents a mutable matrix reference
    typedef typename LA::matrix_reference matrix_reference;
    
    //! The type that represents a mutable vector reference
    typedef typename LA::vector_const_reference vector_const_reference;

    //! The type that represents a mutable matrix reference
    typedef typename LA::matrix_const_reference matrix_const_reference;

    //! Returns an open range of indices
    static index_range range(size_t start, size_t stop);

    //! Returns the identity matrix of the given dimension
    static matrix_type identity(size_t size);

    //! Returns a unit vector with the specified index set to 1
    static vector_type unit_vector(size_t size, size_t index);

    //! Returns the zero vector of the specified length
    static vector_type zeros(size_t size);

    //! Returns the all-zero matrix of the specified dimensions
    static matrix_type zeros(size_t size1, size_t size2);

    //! Returns the all-ones vector of the specified length
    static vector_type ones(size_t size);

    //! Returns the all-one matrix of the specified dimensions
    static matrix_type ones(size_t size1, size_t size2);

    //! Returns the all-scalar vector of the specified length
    static vector_type scalars(size_t size, value_type value);

    //! Returns the all-scalar matrix of the specified dimensions
    static matrix_type scalars(size_t size1, size_t size2, value_type value);

    //! Generates n equally spaced points between start and stop.
    //! If n == 1, returns stop.
    static vector_type linspace(value_type start, value_type stop, size_t n);

    concept_usage(LinearAlgebra) {
      
    }
      
  }; // concept LinearAlgebra
   
} // namespace prl

#include <prl/macros_undef.hpp>

#endif
