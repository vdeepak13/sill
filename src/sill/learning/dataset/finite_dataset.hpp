#ifndef SILL_FINITE_DATASET_HPP
#define SILL_FINITE_DATASET_HPP

#include <sill/argument/finite_assignment.hpp>
#include <sill/argument/domain.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/datastructure/finite_index.hpp>
#include <sill/learning/dataset/basic_dataset.hpp>

#include <limits>

namespace sill {

  /**
   * The traits for a dataset that stores observations for finite variables.
   * \tparam T the type representing the weights
   */
  template <typename T>
  struct finite_data_traits {
    typedef finite_variable          variable_type;
    typedef domain<finite_variable*> domain_type;
    typedef finite_assignment        assignment_type;
    typedef finite_index             data_type;
    typedef size_t                   element_type;
    typedef T                        weight_type;

    //! Returns the number of columns occupied by a variable (always 1).
    static constexpr size_t ncols(finite_variable* v) {
      return 1;
    }

    //! Returns the number of columns occupied by a domain.
    static size_t ncols(const domain_type& dom) {
      return dom.size();
    }

    //! Returns the special "missing" value.
    static constexpr size_t missing() {
      return std::numeric_limits<size_t>::max();
    }

    //! Extracts the value for a single variable.
    static void copy(size_t* const* ptrs, size_t ncols, size_t row, size_t& val) {
      val = ptrs[0][row];
    }

    //! Copies the value for a single variable.
    static void copy(size_t val, size_t ncols, size_t* dest) {
      *dest = val;
    }

  }; // struct finite_data_traits
  
  /**
   * A dense dataset that stores observations for finite variables
   * in memory. The observations are stored in the column-major
   * format, i.e., we first store the values for the first variable
   * for all time steps, then for the second variable etc.
   *
   * \tparam T the type representing the weights
   * \see Dataset
   */
  template <typename T = double>
  using finite_dataset = basic_dataset<finite_data_traits<T> >;

} // namespace sill

#endif
