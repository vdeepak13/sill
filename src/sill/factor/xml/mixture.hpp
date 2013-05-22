#ifndef SILL_MIXTURE_XML_HPP
#define SILL_MIXTURE_XML_HPP

#include <iosfwd>
#include <list>

#include <sill/factor/mixture.hpp>
#include <sill/archive/xml_iarchive.hpp>
#include <sill/archive/xml_oarchive.hpp>
#include <sill/archive/xml_tag.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  template <typename F>
  const char* xml_tag(mixture<F>*) {
    return "mixture";
  }

  template <typename F>
  xml_oarchive& operator<<(xml_oarchive& out, const mixture<F>& f) {
    out.register_variables(f.arguments());

    out.save_begin("mixture");
    out.add_attribute("factor_type", xml_tag((F*)NULL));

    out << make_nvp("arguments", f.arguments());
    foreach(const F& component, f.components()) 
      out << component;
    
    out.save_end();
    return out;
  }

  template <typename F>
  xml_iarchive& operator>>(xml_iarchive& in, mixture<F>& f) {
    assert(false); // TODO: finish

    /*
    in.load_begin("mixture");

    // Load the arguments
    domain args;
    in >> make_nvp("arguments", args);
    // Bug: in >> args does not yield desired results

    // Load the components
    std::list<F> components;
    while(in.has_next()) {
      pls.push_back(F());
      in >> pls.back();
    }
    
    // Create the fragment and return
    foreach(f = mixure(compoents.size(), args);
    in.load_end();
    */

    return in;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
