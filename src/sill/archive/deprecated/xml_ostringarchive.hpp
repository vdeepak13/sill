#ifndef SILL_XML_OSTRINGARCHIVE_HPP
#define SILL_XML_OSTRINGARCHIVE_HPP

#include <sstream>

#include <boost/scoped_ptr.hpp>

#include <sill/archive/xml_oarchive.hpp>

namespace sill {

  /**
   * An archive that stores PRL models in a string buffer.
   * \ingroup serialization
   */
  class xml_ostringarchive : public xml_oarchive {

    boost::scoped_ptr<std::ostringstream> out_ptr;

    // Public functions
    //==========================================================================
  public:
    //! Opens a string buffer for writing
    xml_ostringarchive() 
      : xml_oarchive(*new std::ostringstream()), 
        out_ptr(static_cast<std::ostringstream*>(&xml_oarchive::stream())) { }

    //! Finalizes the archive and returns the underlying string
    std::string str() {
      xml_oarchive::finalize_document();
      return out_ptr->str();
    }

  }; // class xml_ostringarchive

} // namespace sill 


#endif
