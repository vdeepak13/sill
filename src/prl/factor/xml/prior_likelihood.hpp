#ifndef SILL_PRIOR_LIKELIHOOD_XML_HPP
#define SILL_PRIOR_LIKELIHOOD_XML_HPP

#include <iosfwd>

#include <sill/factor/prior_likelihood.hpp>
#include <sill/archive/xml_iarchive.hpp>
#include <sill/archive/xml_oarchive.hpp>
#include <sill/archive/xml_tag.hpp>

namespace sill {

  template <typename F, typename G>
  const char* xml_tag(prior_likelihood<F,G>*) {
    return "prior_likelihood";
  }

  template <typename F, typename G>
  xml_oarchive& operator<<(xml_oarchive& out, const prior_likelihood<F,G>& pl) {
    out.register_variables(pl.prior().arguments());
    out.save_begin("prior_likelihood");
    out.add_attribute("prior", xml_tag((F*)NULL));
    out.add_attribute("likelihood", xml_tag((G*)NULL));
    out << pl.prior();
    out << pl.likelihood();
    out.save_end();
    return out;
  }

  template <typename F, typename G>
  xml_iarchive& operator>>(xml_iarchive& in, prior_likelihood<F,G>& pl) {
    in.load_begin("prior_likelihood");
    F f;
    G g;
    in >> f;
    in >> g;
    in.load_end();
    pl = prior_likelihood<F,G>(f, g);
    return in;
  }

} // namespace sill

#endif
