#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/string.hpp>
#include <boost/lexical_cast.hpp>

#include <iostream>

#include <prl/base/variable.hpp>
#include <prl/base/variables.hpp>
#include <prl/base/process.hpp>
#include <prl/base/finite_variable.hpp>
#include <prl/base/vector_variable.hpp>
#include <prl/base/finite_assignment_iterator.hpp>

#include <prl/macros_def.hpp>

// All variables are implemented here (in a single file) to cut down 
// on the compilation time

namespace prl {

  // Variable
  //============================================================================
  template <class Archive>
  void variable::serialize(Archive& ar, const unsigned int /* file_version */) {
    ar & name_;
    // std::cerr << "About to serialize/deserialize the process" << std::endl;
    ar & process_;
    // std::cerr << "Serialized/deserialized the process " << std::endl;
    if (process_ == NULL) 
      return;
    std::string index_str;
    if (Archive::is_saving::value) {
      index_str = process_->index_str(index_);
      ar & index_str;
    } else { 
      ar & index_str;
      index_ = process_->index(index_str);
    }
  }

  std::ostream& operator<<(std::ostream& out, variable* v) {
    out << std::string(*v) << (void*)(v);
    return out;
  }

  finite_variable::operator std::string() const {
    return "#V(" + name() + "|F|" 
      + boost::lexical_cast<std::string>(size()) + ")";
  }

  bool finite_variable::type_compatible(finite_variable* v) const {
    assert(v);
    return size() == v->size();
  }
  
  bool finite_variable::type_compatible(variable* v) const {
    finite_variable* fv = dynamic_cast<finite_variable*>(v);
    return fv && size() == fv->size();
  }

  size_t finite_variable::value(const std::string& str, size_t offset) const {
    size_t val = boost::lexical_cast<size_t>(str);
    val -= offset;
    assert(val < size());
    return val;
  }

  size_t num_assignments(const finite_domain& vars) {
    size_t n = 1;
    foreach(finite_variable* v, vars)
      n *= v->size();
    return n;
  }

  finite_assignment_range assignments(const finite_domain& vars) {
    return std::make_pair(finite_assignment_iterator(vars),
                          finite_assignment_iterator());
  }

  // Vector variable
  //============================================================================
  template <class Archive>
  void vector_variable::serialize(Archive& ar, 
                                  const unsigned int /* file_version */) {
    using namespace boost::serialization;
    static_assert((tracking_level<vector_variable>::value != track_never));
    ar & boost::serialization::base_object<variable>(*this);
    ar & size_;
  }

  vector_variable::operator std::string() const {
    return "V(" + name() + "|V|"
      + boost::lexical_cast<std::string>(size()) + ")";
  }
  
  bool vector_variable::type_compatible(vector_variable* x) const {
    assert(x);
    return size() == x->size();
  }

  bool vector_variable::type_compatible(variable* v) const {
    vector_variable* vv = dynamic_cast<vector_variable*>(v);
    return vv && size() == vv->size();
  }

  /*
  vec vector_variable::value(const std::string& str) const {
    //value_type vec = boost::lexical_cast<value_type>(str);
    //assert(vec.size() == size());
    //return vec;
    assert(false);
    return value_type();
  }
  */

  // Template instantiations
  //============================================================================
  template
  void finite_variable::serialize(boost::archive::text_iarchive&,
                                  const unsigned int);
  template
  void finite_variable::serialize(boost::archive::text_oarchive&,
                                  const unsigned int);
  template
  void vector_variable::serialize(boost::archive::text_iarchive&,
                                  const unsigned int);
  template
  void vector_variable::serialize(boost::archive::text_oarchive&,
                                  const unsigned int);
  template
  void variable::serialize(boost::archive::text_iarchive&, const unsigned int);
  template
  void variable::serialize(boost::archive::text_oarchive&, const unsigned int);
  template
  void variable::serialize(boost::archive::binary_iarchive&,
                           const unsigned int);
  template
  void variable::serialize(boost::archive::binary_oarchive&,
                           const unsigned int);

} // namespace prl

BOOST_CLASS_EXPORT(prl::variable);
BOOST_CLASS_EXPORT(prl::finite_variable);
BOOST_CLASS_EXPORT(prl::vector_variable);

