#ifndef PRL_NAMED_VALUE_PAIR_HPP
#define PRL_NAMED_VALUE_PAIR_HPP

#include <boost/serialization/nvp.hpp> // for now

#include <prl/archive/xml_iarchive.hpp>
#include <prl/archive/xml_oarchive.hpp>

namespace prl {

  using ::boost::serialization::nvp;
  using ::boost::serialization::make_nvp;
  
  /** Saves the value under a different name other than the standard. */
  template <typename T>
  xml_oarchive& operator<<(xml_oarchive& out, const nvp<T>& p) {
    out.override_name(p.name());
    out << p.const_value();
    return out;
  }

  template <typename T>
  xml_iarchive& operator>>(xml_iarchive& in, const nvp<T>& p) {
    in.override_name(p.name());
    in >> p.value();
    return in;
  }
                           
} // namespace prl

#endif
