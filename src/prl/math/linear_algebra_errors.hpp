#ifndef PRL_LINEAR_ALGEBRA_ERRORS_HPP
#define PRL_LINEAR_ALGEBRA_ERRORS_HPP

#include <stdexcept>

namespace prl {

  /**
   * An exception thrown when a matrix inverse inv() fails.
   *
   * \ingroup linear_algebra_exceptions
   */
  class inv_error : public std::runtime_error {
  public:
    explicit inv_error(const std::string& what)
      : std::runtime_error(what) { }
  };

  /**
   * An exception thrown when a Cholesky decomposition chol() fails.
   *
   * \ingroup linear_algebra_exceptions
   */
  class chol_error : public std::runtime_error {
  public:
    explicit chol_error(const std::string& what)
      : std::runtime_error(what) { }
  };

  /**
   * An exception thrown when a solve via Cholesky decomposition
   * ls_solve_chol() fails.
   *
   * \ingroup linear_algebra_exceptions
   */
  class ls_solve_chol_error : public std::runtime_error {
  public:
    explicit ls_solve_chol_error(const std::string& what)
      : std::runtime_error(what) { }
  };

} // namespace prl

#endif  // PRL_LINEAR_ALGEBRA_ERRORS_HPP
