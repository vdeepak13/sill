#ifndef SILL_EUCLIDEAN_PROCESS_HPP
#define SILL_EUCLIDEAN_PROCESS_HPP

#include <sill/base/process.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A process over Euclidean space. Useful for GPs etc.
   * \ingroup base
   */
  template <typename V>
  class euclidean_process : public process {
    concept_assert((Variable<V>));
    
    // Public type declarations
    //==========================================================================
  public:
    //! The variable type associated with this process
    typedef V variable_type;

    //! The type that represents an index
    typedef std::vector<double> index_type;

    // Private data members
    //==========================================================================
  private:
    //! The instances of the process at different time steps
    mutable std::map<index_type, variable_type*> vars; 
    
  public:
    // etc.

  }; // class euclidean_process
  
} // namespace sill

#include <sill/macros_undef.hpp>

#endif
