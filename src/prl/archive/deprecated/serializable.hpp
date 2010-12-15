#ifndef PRL_SERIALIZABLE_HPP
#define PRL_SERIALIZABLE_HPP

#include <string>
#include <sstream>
#include <cstring>

#include <prl/serialization/serialize.hpp>
#warning "Deprecated"
namespace prl {


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

} // namespace prl

#endif
