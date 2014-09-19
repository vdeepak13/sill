#ifndef SILL_VECTOR_VARIABLE_HPP
#define SILL_VECTOR_VARIABLE_HPP

#include <vector>
#include <set>
#include <map>

#include <sill/base/variable.hpp>
#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/serialization/serialize.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A variable that can take on values in \f$\Re^d\f$
   * for some number of dimensions \f$d\f$.
   * \ingroup base_types
   */
  class vector_variable : public variable {

  public:
    //! The type of values taken on by a vector variable
    typedef vec value_type;

    // Public functions
    //==========================================================================
  public:

    vector_variable() { }

    //! Creates a vector variable with the given name and dimension
    vector_variable(const std::string& name, size_t size)
      : variable(name), size_(size) {
      assert(size > 0);
    }

    //! Constructs a variable with the given name, dimensionality, and process
    vector_variable(const std::string& name, size_t size,
                    sill::process* process, const boost::any& index)
      : variable(name, process, index), size_(size) {
      assert(size > 0);
    }

    //! Conversion to human-readable format
    operator std::string() const;

    //! Returns the number of dimensions of this vector variable.
    size_t size() const {
      return size_;
    }

    //! Returns true iff the variables have the same dimension
    bool type_compatible(vector_variable* x) const;
    
    //! Returns true if the supplied variable is vector and has the same size
    bool type_compatible(variable* v) const;

    /** 
     * Parses the vector from a string token and verifies that its 
     * length matches the the number of dimensions of this variable.
     * \throw boost::bad_lexical_cast
     *        if the vector values cannot be parsed as a double
     **/
    value_type value(const std::string& str) const;
    
    //! Serializes this variable and all attached information. 
    //! This performs a deep serialization of this variable as opposed to 
    //! just storing an ID. 
    void save(oarchive& ar) const;
    
    //! deserializes this variable and all attached information. 
    //! This performs a deep deserialization. If this variable is part of a 
    //! process, the archive must be provided with the universe. 
    //! Otherwise the deserialization can be performed an attached universe.
    void load(iarchive& ar);

    // Private data members
    //==========================================================================
  private:

    //! The number of dimensions of this vector variable.
    size_t size_;

  }; // class vector_variable

  /**
   * A set of vector variables.
   */
  typedef std::set<vector_variable*> vector_domain;

  /**  
   * A vector of vector variables.
   * This type is used primarily to refer to variables in some specific order.
   */
  typedef std::vector<vector_variable*> vector_var_vector;

  /**
   * A map from variables to variables of the same type.  
   * This kind of map is used to perform variable substitutions.
   */
  typedef std::map<vector_variable*, vector_variable*> vector_var_map;

  // Free functions
  //============================================================================

  //! Returns the size of a collection of vector variables
  //! \relates vector_variable
  template <typename Range>
  size_t vector_size(const Range& args) {
    size_t size = 0;
    foreach(typename Range::value_type arg, args) size += arg->size();
    return size;
  }

  //! Serializes the variable* pointer. This only serializes an id.
  //! The deserializer will look for the id in the universe
  oarchive& operator<<(oarchive& ar, vector_variable* const& v);
  
  //! Deserializes a variable* pointer by reading an id from the archive.
  //! The archive must have an attached universe
  iarchive& operator>>(iarchive& ar, vector_variable*& v);

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
