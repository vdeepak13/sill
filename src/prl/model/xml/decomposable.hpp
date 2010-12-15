#ifndef PRL_DECOMPOSABLE_XML_HPP
#define PRL_DECOMPOSABLE_XML_HPP

#include <list>

#include <prl/model/decomposable.hpp>
#include <prl/archive/xml_iarchive.hpp>
#include <prl/archive/xml_oarchive.hpp>
#include <prl/archive/xml_tag.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  template <typename F>
  const char* xml_tag(decomposable<F>*) { 
    return "decomposable"; 
  }

  template <typename F>
  xml_oarchive& operator<<(xml_oarchive& out, const decomposable<F>& dm) {
    out.register_variables(dm.arguments());
    out.save_begin("decomposable");
    out.add_attribute("factor_type", xml_tag((F*)NULL));
    foreach(const F& f, dm.clique_marginals()) 
      out << f;
    out.save_end();
    return out;
  }

  template <typename F>
  xml_iarchive& operator>>(xml_iarchive& in, decomposable<F>& dm) {
    in.load_begin("decomposable");
    
    // load the clique marginals
    std::list<F> factors;
    while(in.has_next()) {
      factors.push_back(F());
      in >> factors.back();
    }
    
    dm.initialize(factors);
    in.load_end();
    return in;
  }

}

#include <prl/macros_undef.hpp>

#endif
