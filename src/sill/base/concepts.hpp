#ifndef SILL_BASE_CONCEPTS_HPP
#define SILL_BASE_CONCEPTS_HPP

#include <string>

#include <boost/type_traits/is_base_of.hpp>

#include <sill/global.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // forward declarations
  class variable;
  class process;

  /**
   * The concept that represents a variable.
   *
   * A concrete variable must define a value_t type that holds the values of
   * that variable.
   * 
   * \todo add parse_value(string) function
   *
   * \ingroup base_concepts
   */
  template <class V>
  struct Variable {
    static_assert((boost::is_base_of<variable, V>::value));

    //! The type of values assigned to this variable
    typedef typename V::value_type value_type;

    //! Returns true iff the variables have equivalent domains
    bool type_compatible(V* other) const;

    //! Parses the value from a string and checks if it matches the 
    //! variable's domain
    value_type value(const std::string& str) const;

    concept_usage(Variable) {
      sill::same_type(var1->type_compatible(var2), bool_value);
    }

    static std::string __unused_str;
    
    Variable():str(__unused_str) {
    }
  private:
    const V* var1;
    V* var2;
    const std::string& str;
    bool bool_value;
  };
  
  template<class V> std::string Variable<V>::__unused_str;
  /** 
   * The concept that represents a process.
   * Conceptually, a process is a collection of variables, indexed by some type.
   * Typically, distinct processes have disjoint collections of variables.
   * The variables are created as needed, and are deleted when the 
   * process is destroyed.
   * 
   * \ingroup base_concepts
   */
  template <typename P>
  struct Process {
    static_assert((boost::is_base_of<process, P>::value));
    
    //! The type of variables used by this process
    typedef typename P::variable_type variable_type;

    //! The type used to index the variables of this process
    typedef typename P::index_type index_type;

    //! Returns the variable for the given index
    variable_type* at(const index_type& index) const;

    concept_usage(Process) {
      sill::same_type(proc->at(index_type()), var);
    }

  private:
    variable_type* var;
    const P* proc;
  };

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
