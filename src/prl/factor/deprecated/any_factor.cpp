#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include <prl/factor/any_factor.hpp>
#include <prl/factor/constant_factor.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/factor/moment_gaussian.hpp>
#include <prl/factor/canonical_gaussian.hpp>

#include <prl/macros_def.hpp>

BOOST_CLASS_EXPORT(prl::any_factor);

namespace prl {
  
  // Serialization
  //============================================================================
  template <typename Archive>
  void any_factor::serialize(Archive& ar, const unsigned int /*file_version*/) {
    ar & base_object_nvp(base);
    if (typename Archive::is_loading()) {
      factor* fp;
      ar & serialization_nvp(fp);
      operator=(*fp);
      delete fp;
    } else {
      const factor& fr = wrapper->get();
      factor* fp = const_cast<factor*>(&fr);
      ar & serialization_nvp(fp);
    }
    // ar & serialization_nvp(wrapper); // a little ugly
  }

  // Type registration
  //============================================================================
  any_factor::factor_map any_factor::factor_registry;
  any_factor::binary_map any_factor::binary_registry;

  // Static initialization block
  namespace {
    int register_factors() {
      any_factor::register_factor<constant_factor>();
      any_factor::register_factor<tablef>();
      any_factor::register_factor<moment_gaussian>();
      any_factor::register_factor<canonical_gaussian>();

      any_factor::register_binary<constant_factor, tablef>();
      any_factor::register_binary<constant_factor, moment_gaussian>();
      any_factor::register_binary<constant_factor, canonical_gaussian>();
      any_factor::register_binary<moment_gaussian, canonical_gaussian>();
      return 0; // dummy
    }
    static int registered = register_factors();
  }

  // Other functions
  //============================================================================
  factor_placeholder* any_factor::new_wrapper(const factor& f) {
    factor_map::iterator it = factor_registry.find(&typeid(f));
    if (it != factor_registry.end())
      return it->second->make(f);
    else
      throw std::out_of_range
        (std::string("any_factor: unregisted class ") + 
         typeid(f).name());
  }
  
  const factor_binary& any_factor::binary(const factor& x, const factor& y) {
    info_ptr_pair key(&typeid(x), &typeid(y));
    binary_map::iterator it = binary_registry.find(key);
    if (it != binary_registry.end())
      return *it->second;
    else
      throw std::out_of_range
        (std::string("any_factor: unregistered operation of ") +
         typeid(x).name() + " and " + typeid(y).name());
    }
  
  any_factor& any_factor::operator=(const factor& f) {
    wrapper.reset(new_wrapper(f));
    args = wrapper->arguments();
    return *this;
  }

  any_factor::operator std::string() const {
    return wrapper->get();
  }
  
  bool any_factor::operator==(const any_factor& g) const {
    if (typeid(*wrapper) == typeid(*g.wrapper)) {
      return (*wrapper == *g.wrapper);
    } else return false;
  }
  
  bool any_factor::operator<(const any_factor& g) const {
    if (typeid(*wrapper).before(typeid(*g.wrapper)))
      return true;
    else if(typeid(*wrapper)==typeid(*g.wrapper))
      return *wrapper < *g.wrapper;
    else 
      return false;
  }

  any_factor& any_factor::subst_args(const var_map& var_map) {
    ensure_unique();
    args = subst_vars(args, var_map);
    wrapper->subst_args(var_map);
    return *this;
  }
  
  any_factor& any_factor::normalize() { 
    ensure_unique();
    wrapper->normalize();
    return *this;
  }

} // namespace prl



