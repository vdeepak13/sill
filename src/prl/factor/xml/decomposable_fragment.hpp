#ifndef PRL_DECOMPOSABLE_FRAGMENT_XML_HPP
#define PRL_DECOMPOSABLE_FRAGMENT_XML_HPP

#include <iosfwd>

#include <prl/factor/decomposable_fragment.hpp>
#include <prl/factor/xml/prior_likelihood.hpp>
#include <prl/archive/xml_iarchive.hpp>
#include <prl/archive/xml_oarchive.hpp>
#include <prl/archive/xml_tag.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  template <typename F, typename G>
  const char* xml_tag(decomposable_fragment<F,G>*) {
    return "decomposable_fragment";
  }

  template <typename F, typename G>
  xml_oarchive& operator<<(xml_oarchive& out, 
                           const decomposable_fragment<F,G>& f) {
    foreach(typename F::domain_type clique, f.cliques()) 
      out.register_variables(clique);

    out.save_begin("decomposable_fragment");
    out.add_attribute("prior", xml_tag((F*)NULL));
    out.add_attribute("likelihood", xml_tag((G*)NULL));

    out << make_nvp("arguments", f.arguments());
    typedef prior_likelihood<F,G> pl_type;
    foreach(const pl_type& pl, f.factors()) out << pl;
    
    out.save_end();
    return out;
  }

  template <typename F, typename G>
  xml_iarchive& operator>>(xml_iarchive& in, decomposable_fragment<F,G>& f) {
    in.load_begin("decomposable_fragment");

    // Load the arguments
    typename F::domain_type args;
    in >> make_nvp("arguments", args);
    // Bug: in >> args does not yield desired results

    // Load the components
    std::list< prior_likelihood<F,G> > pls;
    while(in.has_next()) {
      pls.push_back(prior_likelihood<F,G>());
      in >> pls.back();
    }
    
    // Create the fragment and return
    f = decomposable_fragment<F,G>(pls, args);
    in.load_end();
    return in;
  }

} // namespace prl

#include <prl/macros_undef.hpp>

#endif
