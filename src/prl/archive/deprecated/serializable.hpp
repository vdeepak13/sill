#ifndef SILL_SERIALIZABLE_HPP
#define SILL_SERIALIZABLE_HPP

#include <string>
#include <sstream>
#include <cstring>

#include <sill/serialization/serialize.hpp>
#warning "Deprecated"
namespace sill {


  /**
   * An interface that represents an object that can be serialized through
   * base pointers. 
   */
  class serializable {

  public:
    //! Serialization
    virtual void serialize(oarchive& ar) const { assert(false); }
    virtual void deserialize(iarchive& ar) { assert(false); }
  
  protected:
    //! serializable class can only be constructed by the descendants
    serializable() {}

  public:
    //! Destructor
    virtual ~serializable() {}
  };

  //char serializable::cstr[10000];

} // namespace sill

#endif
