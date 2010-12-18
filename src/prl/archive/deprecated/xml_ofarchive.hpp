#ifndef SILL_XML_OFARCHIVE_HPP
#define SILL_XML_OFARCHIVE_HPP

#include <fstream>

#include <boost/scoped_ptr.hpp>

#include <sill/archive/xml_oarchive.hpp>

namespace sill {

  /**
   * An archive that stores PRL models in an XML file.
   * \ingroup serialization
   */
  class xml_ofarchive : public xml_oarchive {

    boost::scoped_ptr<std::ofstream> out_ptr;

    // Public functions
    //==========================================================================
  public:
    //! Opens a file for writing
    xml_ofarchive(const char* name)
      : xml_oarchive(*new std::ofstream(name)),
        out_ptr(static_cast<std::ofstream*>(&xml_oarchive::stream())) { }

    //! Destructor
    ~xml_ofarchive() {
      close();
    }

    //! Finalizes the archive and closes the underlying file
    //! Can be called multiple times (only the first call has any effect)
    void close() {
      finalize_document();
      out_ptr.reset();
    }

  }; // class xml_ofarchive

} // namespace sill


#endif
