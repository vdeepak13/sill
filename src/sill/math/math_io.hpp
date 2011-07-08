#ifndef SILL_MATH_IO_HPP
#define SILL_MATH_IO_HPP

#include <iostream>

#include <sill/math/linear_algebra/armadillo.hpp>

/**
 * \file math_io.hpp  Functions for reading and writing matrices and vectors
 *                    from files (not including native SILL serialization).
 */

namespace sill {

  /**
   * Read a matrix from an input stream:
   *  - one matrix row per line
   *  - whitespace-separated elements in each line
   * This stops reading once it finds the first line with a number of elements
   * which does not match the number in the first line (unless the first line
   * has 0 elements, in which case this method returns immediately).
   */
  template <typename T>
  arma::Mat<T> load_matrix(std::istream& in) {
    std::vector<T> elements;
    size_t ncols;
    size_t nrows = 0;
    std::string line;
    getline(in, line);
    arma::Vec<T> v(line);
    ncols = v.size();
    if (ncols == 0)
      return arma::Mat<T>();
    elements.insert(elements.end(), v.begin(), v.end());
    ++nrows;
    while (in.good()) {
      std::streampos where = in.tellg();
      getline(in, line);
      v = arma::Vec<T>(line);
      if (v.size() != ncols) {
        in.seekg(where);
        arma::Mat<T> m(nrows, ncols);
        for (size_t i = 0; i < nrows; ++i)
          for (size_t j = 0; j < ncols; ++j)
            m(i,j) = elements[ncols * i + j];
        return m;
      }
      elements.insert(elements.end(), v.begin(), v.end());
      ++nrows;
    }
    assert(false); // Bad input stream
    return arma::Mat<T>();
  } // load_matrix

} // namespace sill

#endif // #ifndef SILL_MATH_IO_HPP
