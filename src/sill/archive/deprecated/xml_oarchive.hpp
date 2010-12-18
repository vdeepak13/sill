#ifndef SILL_XML_OARCHIVE_HPP
#define SILL_XML_OARCHIVE_HPP

#include <iosfwd>
#include <stack>
#include <string>

#include <sill/map.hpp>
#include <sill/base/variable.hpp>
#include <sill/base/process.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>

namespace sill {

  /**
   * An archive that can store PRL models in an XML format.
   * TODO: check that no more writing after the archive is finalized / closed
   * \ingroup serialization
   */
  class xml_oarchive {

    // Private data members
    //==========================================================================
  private:

    //! The variables that have already been registered
    domain registered_vars;

    //! The processes that have already been registered
    set<process*> registered_procs;

    //! The underlying text stream
    std::ostream& out;

    //! The indent level
    int level;

    //! True if we are currently writing the XML header
    bool inside_header;

    //! True if the stream has already been finalized
    bool finalized;

    //! Was the current element a leaf node?
    bool leaf;

    //! A name that overrides the name of the next block (empty => none)
    std::string custom_name;

    //! The stack of all element names
    std::stack<std::string> elements;

    // Public functions
    //==========================================================================
  public:

    //! Creates an archive for the given stream
    xml_oarchive(std::ostream& out);

    //! Closes the stream
    ~xml_oarchive();

    //! Registers a process in the archive
    void register_process(process* p);

    //! Registers a variable in the archive
    void register_variable(variable* v);

    //! Registers a collection of variables in the archive
    void register_variables(const forward_range<variable*>& vars);

    //! Sets a custom name for the next element
    void override_name(const char* name);
    
    //! Begins a new element with the given tag
    void start_element(const char* name);

    //! Adds a string attribute 
    void add_attribute(const char* name, const char* value);

    //! Adds an integral attribute
    void add_attribute(const char* name, size_t value);

    //! Adds an integral attribute
    void add_attribute(const char* name, int value);

    //! Ends the most recent element
    void end_element();

    //! Stores the ids of a set of variables
    xml_oarchive& operator<<(const forward_range<variable*>& vars);

    //! Stores an unsigned number
    xml_oarchive& operator<<(size_t x);

    //! Stores the object with a custom element name
    template <typename T>
    void save(const char* name, const T& object) {
      override_name(name);
      *this << object;
    }

    //! Finalizes any open element headers, and returns the underlying stream
    std::ostream& stream() {
      finalize_header();
      return out;
    }

    // Protected member functions
    //==========================================================================
  protected:
    
    //! Closes an open element header with ">", if any
    void finalize_header() {
      if (inside_header) out << ">";
      inside_header = false;
    }

    //! Closes the entire XML document (if not done yet)
    void finalize_document();

  }; // class xml_oarchive
  
} // namespace sill

#endif
