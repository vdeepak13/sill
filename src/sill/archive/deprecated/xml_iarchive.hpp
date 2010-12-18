#ifndef SILL_XML_IARCHIVE_HPP
#define SILL_XML_IARCHIVE_HPP

#include <string>
#include <vector>

#include <boost/type_traits/is_base_of.hpp>

#include <sill/base/variable.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/base/named_universe.hpp>
#include <sill/serializable.hpp> // for serializable
#include <sill/map.hpp>
#include <sill/stl_concepts.hpp>

#include <sill/macros_def.hpp>

// forward declaration
namespace xmlpp {
  class TextReader;
}

namespace sill {

  class universe;  

  /**
   * An archive that can load PRL models from an XML file.
   * It uses libxml2's TextReader pull parser. 
   * As a convention, we skip past text fields and other whitespace
   * right _before_ an element is loaded (e.g., at the beginning of 
   * load_begin(). This allows us to interleave load operations that
   * parse text nodes as well as those that ignore them.
   *
   * \todo Fix the serialization / deserialization of variable ids 
   *       to allow 32/64 bit portability.
   *
   * \ingroup serialization
   */
  class xml_iarchive {

    struct deserializer;

    // Private data members
    //==========================================================================
  private:

    //! A vector that, optionally, holds a copy of the parsed data
    std::vector<unsigned char> data;
    
    //! The XML parser
    xmlpp::TextReader* reader;

    //! Universe that is used to create new variables
    named_universe& u;

    //! A map from variable ids to variable objects
    map<size_t, variable*> var_map;
    
    //! A map from variable ids to process objects
    map<size_t, process*> proc_map;

    //! A map that stores the registered types that can be read using read()
    static map<std::string, deserializer*> deserializers;

    //! A name that overrides the name of the next block (empty => none)
    std::string custom_name;

    // Disable copying and the assignment operator
    xml_iarchive(const xml_iarchive& other);
    xml_iarchive& operator=(const xml_iarchive& other);

    // Public functions
    //==========================================================================
  public:
    //! Creates an archive that loads the data from a file
    xml_iarchive(const char* filename, named_universe& u);

  #ifndef SWIG
    //! Creates an archive that loads data from memory
    //! The data must remain valid while xml_iarchive is being used
    xml_iarchive(const unsigned char* data, size_t size, named_universe& u);

    //! Creates an archive that loads data from memory
    //! The data must remain valid while xml_iarchive is being used
    xml_iarchive(const char* data, size_t size, named_universe& u); 
  #endif

    //! Creates an archive that loads data from memory
    //! Makes a defensive copy of the data
    xml_iarchive(const std::vector<unsigned char>& data_, named_universe& u);

    //! Destructor. Deletes the XML reader
    ~xml_iarchive();

    //! Registers a type for use in the load() function
    template <typename Type>
    static void register_type(const std::string& tag) {
      assert(!deserializers.contains(tag));
      deserializers[tag] = new concrete_deserializer<Type>();
    }

    //! Returns the name of the current element
    std::string name() const;

    //! Returns the value of an attribute with the specified name
    std::string attribute(const char* name) const;

    //! Returns text in the current text element
    std::string text() const;

    //! Returns true if the stream has more siblings at the current level
    //! This function may skip over whitespace (including text)
    bool has_next();
    
    //! Sets a custom name for the next element. Only the first call applies; 
    //! subsequent calls are ignored until start_element() is called.
    void override_name(const char* name);

    //! Reads the header and checks the tag name
    void start_element(const char* name);
    
    //! Reads past the end of the element
    void end_element();

    //! Reads a sequence of variables from a text node
    xml_iarchive& operator>>(var_vector& vars);

    //! Reads a sequence of finite variables from a test node
    xml_iarchive& operator>>(finite_var_vector& finite_vars);

    //! Reads a sequence of finite variables from a test node
    xml_iarchive& operator>>(vector_var_vector& vector_vars);

    //! Reads a set of variables from a text node
    xml_iarchive& operator>>(domain& vars);

    //! Reads a set of variables from a text node
    xml_iarchive& operator>>(finite_domain& vars);

    //! Reads a set of variables from a text node
    xml_iarchive& operator>>(vector_domain& vars);

    //! Writes a size_t
    xml_iarchive& operator>>(size_t& x);

    //! Stores the object with a custom element name
    template <typename T>
    void load(const char* name, T& object) {
      override_name(name);
      *this >> object;
    }

    //! Loads a string element with the specified name
    void load(const char* name, std::string& str);

    //! Loads a string element with the specified name
    std::string load_string(const char* name);

    //! Loads an object from the archive
    //! The tag must match one of the registered types
    serializable* load();

    // Private functions
    //==========================================================================
  private:
    //! Moves past the initial header
    void initialize();

    //! Parses the process definitions at the current location
    //! Does not skip white space
    void register_process();

    //! Parses process / variable definitions at the current location
    //! Does not skip over white space initially
    void register_variables();

    //! Skip over the whitespace and text nodes
    void skip_whitespace();

    /**
     * An interface that represents an object that can load other objects
     * from xml_iarchive.
     */
    struct deserializer {
      virtual serializable* deserialize(xml_iarchive& in) = 0;
      virtual ~deserializer() {} 
    };

  #ifndef SWIG // SWIG gets confused by this declaration
    /**
     * A basic implementation of the deserializer interface
     * \param Type a type that is a descendant of serializable.
     *        Type must be DefaultConstructible.
     */
    template <typename Type>
    struct concrete_deserializer : public deserializer {
      concept_assert((DefaultConstructible<Type>));
      static_assert((boost::is_base_of<serializable, Type>::value));

      serializable* deserialize(xml_iarchive& in) {
        Type* obj = new Type();
        in >> *obj;
        return obj;
      }
    };
  #endif
    
  }; // class xml_iarchive

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
