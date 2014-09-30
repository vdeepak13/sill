#ifndef SILL_FINITE_VARIABLE_HPP
#define SILL_FINITE_VARIABLE_HPP

#include <vector>
#include <set>
#include <map>

#include <sill/base/variable.hpp>
#include <sill/serialization/serialize.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup base_types
  //! @{

  /**
   * A variable that can take on finitely many values.
   */
  class finite_variable : public variable {

    // Private data members
    //==========================================================================
  private:  
    //! The number of values this variable can take on.
    size_t size_;

    // Public functions
    //==========================================================================
  public:
    typedef std::size_t value_type;
    finite_variable() { }
  
    //! Constructs a variable with the given name and domain size
    finite_variable(const std::string& name, size_t size)
      : variable(name), size_(size) { }

    //! Constructs a variable with the given name, domain size, and process
    finite_variable(const std::string& name, size_t size,
                    sill::process* process, const boost::any& index)
      : variable(name, process, index), size_(size) { }

    //! Conversion to human-readable format
    operator std::string() const;

    //! Returns the number of values this variable can take on.
    size_t size() const {
      return size_;
    }

    //! Returns true iff the variables have the same domain size
    bool type_compatible(finite_variable* v) const;

    //! Returns true if the supplied variable is finite and has the same size
    bool type_compatible(variable* v) const;

    /**
     * Parses an integral value from a string and checks if it matches the 
     * variable's domain size.
     * @param offset 
     *        the integral value that corresponds to the smallest element
     * @throw boost::bad_lexical_cast
     *        if the string cannot be converted to an unsigned integer.
     **/
    size_t value(const std::string& str, size_t offset = 0) const;

    //! Serializes this variable and all attached information. 
    //! This performs a deep serialization of this variable as opposed to 
    //! just storing an ID.   
    void save(oarchive& ar) const;
    
    //! deserializes this variable and all attached information. 
    //! This performs a deep deserialization. If this variable is part of a 
    //! process, the archive must be provided with the universe. 
    //! Otherwise the deserialization can be performed an attached universe.
    void load(iarchive& ar);
    //! Default constructor (only used by serialization)
  }; // class finite_variable

  /**
   * A set of finite variables.
   */
  typedef std::set<finite_variable*> finite_domain;

  /**
   * A vector of finite variables.
   * This type is used primarily to refer to variables in some specific order.
   */
  typedef std::vector<finite_variable*> finite_var_vector;

  /**
   * A map from variables to variables of the same type.  
   * This kind of map is used to perform variable substitutions.
   */
  typedef std::map<finite_variable*, finite_variable*> finite_var_map; 

  // Free functions
  //============================================================================

  /**
   * Counts the number of assignments to the supplied set of finite variables.  
   * @param  domain a set of finite variables; 
   * @return the product of the sizes of the variables
   *
   * \relates finite_variable
   */
  size_t num_assignments(const finite_domain& vars);

  /**
   * Returns the number of assignments to the supplied collection of
   * finite variables. Assumes the variables are distinct.
   */
  size_t num_assignments(const finite_var_vector& vars);

  //! @} group base_types

  //! Serializes the variable* pointer. This only serializes an id.
  //! The deserializer will look for the id in the universe
  oarchive& operator<<(oarchive& ar, finite_variable* const& v);
  
  //! Deserializes a variable* pointer by reading an id from the archive.
  //! The archive must have an attached universe
  iarchive& operator>>(iarchive& ar, finite_variable*& v);


} // namespace sill

#include <sill/macros_undef.hpp>

#endif
