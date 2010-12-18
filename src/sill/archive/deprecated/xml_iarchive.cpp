#include <stdexcept>

#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/range/functions.hpp>

#include <sill/archive/xml_iarchive.hpp>
#include <sill/base/variable.hpp>
#include <sill/process.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/base/universe.hpp>

#ifdef HAVE_LIBXMLPP

#include <libxml++/parsers/textreader.h>

namespace sill {
  
  // Implementation
  map<std::string, xml_iarchive::deserializer*> xml_iarchive::deserializers;

  // Constructors
  //============================================================================
  xml_iarchive::xml_iarchive(const char* filename, 
                             named_universe& u)
    : reader(new xmlpp::TextReader(filename)), u(u) { 
    initialize();
  }
  
  xml_iarchive::xml_iarchive(const unsigned char* dat, size_t size, 
                             named_universe& u)
    : reader(new xmlpp::TextReader(dat, size)), u(u) {
    initialize();
  } 

  xml_iarchive::xml_iarchive(const char* dat, size_t size, 
                             named_universe& u) 
    : reader(new xmlpp::TextReader((unsigned char*)(dat), size)), u(u) {
    initialize();
  }

  xml_iarchive::xml_iarchive(const std::vector<unsigned char>& dat, 
                             named_universe& u)
    : data(dat), reader(new xmlpp::TextReader(&data[0], data.size())), u(u) {
    initialize();
  }

  xml_iarchive::~xml_iarchive() {
    if (reader) delete reader;
  }

  // Public functions 
  //============================================================================
  std::string xml_iarchive::name() const {
    return reader->get_name().c_str();
  }

  std::string xml_iarchive::attribute(const char* name) const {
    return reader->get_attribute(name).c_str();
  }
  
  std::string xml_iarchive::text() const {
    if (reader->get_node_type() != reader->EndElement) {
      assert(reader->get_node_type() == reader->Text);
      return reader->get_value().c_str();
    } else {
      return std::string();
    }
  }

  bool xml_iarchive::has_next() {
    skip_whitespace();
    return reader->get_node_type() == reader->Element;
  }
    
  void xml_iarchive::override_name(const char* name) {
    assert(name && *name); // !name.empty()
    if (custom_name.empty()) custom_name = name;
  }

  void xml_iarchive::start_element(const char* name) {
    skip_whitespace();
    register_variables();
    assert(reader->get_node_type() == reader->Element);
    if (custom_name.empty()) {
      assert(reader->get_name() == name);
    } else {
      assert(reader->get_name() == custom_name.c_str());
      custom_name.clear();
    }
    assert(!reader->is_empty_element());
    reader->read();
  }
    
  void xml_iarchive::end_element() {
    skip_whitespace();
    assert(reader->get_node_type() == reader->EndElement);
    reader->read();
  }

  xml_iarchive& xml_iarchive::operator>>(var_vector& vars) {
    std::istringstream in(load_string("var_vector"));
    size_t id;
    vars.clear();
    while(in >> id)
      vars.push_back(var_map[id]);
    return *this;
  }

  xml_iarchive& xml_iarchive::operator>>(finite_var_vector& vars) {
    std::istringstream in(load_string("var_vector"));
    size_t id;
    vars.clear();
    while(in >> id) {
      vars.push_back(dynamic_cast<finite_variable*>(var_map[id]));
      assert(vars.back());
    }
    return *this;
  }

  xml_iarchive& xml_iarchive::operator>>(vector_var_vector& vars) {
    std::istringstream in(load_string("var_vector"));
    size_t id;
    vars.clear();
    while(in >> id) {
      vars.push_back(dynamic_cast<vector_variable*>(var_map[id]));
      assert(vars.back());
    }
    return *this;
  }

  xml_iarchive& xml_iarchive::operator>>(domain& vars) {
    var_vector vec;
    load("domain", vec);
    vars.clear();
    vars.insert(vec);
    return *this;
  }

  xml_iarchive& xml_iarchive::operator>>(finite_domain& vars) {
    finite_var_vector vec;
    load("finite_domain", vec);
    vars.clear();
    vars.insert(vec);
    return *this;
  }

  xml_iarchive& xml_iarchive::operator>>(vector_domain& vars) {
    vector_var_vector vec;
    load("vector_domain", vec);
    vars.clear();
    vars.insert(vec);
    return *this;
  }

  xml_iarchive& xml_iarchive::operator>>(size_t& x) {
    std::string str;
    load("size_t", str);
    x = boost::lexical_cast<size_t>(x);
    return *this;
  }

  serializable* xml_iarchive::load() {
    skip_whitespace();
    register_variables();
    if (reader->get_node_type() != reader->EndElement) {
      std::string tag = reader->get_name().c_str();

      // look up the available types
      if (deserializers.contains(tag)) {
        return deserializers[tag]->deserialize(*this);
      } else 
        throw std::invalid_argument("Unknown type " + tag);
    } else return NULL;
  }

  void xml_iarchive::load(const char* name, std::string& str) {
    start_element(name);
    if (reader->get_node_type() != reader->EndElement) {
      assert(reader->get_node_type() == reader->Text);
      str = reader->get_value().c_str();
    } else {
      str.clear();
    }
    end_element();
  }

  std::string xml_iarchive::load_string(const char* name) {
    std::string str;
    load(name, str);
    return str;
  }

  // Private functions 
  //============================================================================
  void xml_iarchive::initialize() {
    reader->read();
    skip_whitespace();
    assert(reader->get_name() == "prl");
    assert(!reader->is_empty_element());
    reader->read();
  }

  void xml_iarchive::register_process() {
    process* p;
    size_t id = boost::lexical_cast<size_t>(reader->get_attribute("id"));
    size_t sz = boost::lexical_cast<size_t>(reader->get_attribute("size"));
    std::string type = reader->get_attribute("type").c_str();
    std::string name = reader->get_attribute("name").c_str();
    if (type == "FT")
      p = u.add_process(new finite_timed_process(name, sz), true);
    else if(type == "VT")
      p = u.add_process(new vector_timed_process(name, sz), true);
    else
      assert(false);
    proc_map[id] = p;
  }

  void xml_iarchive::register_variables() {
    assert(reader->get_node_type() == reader->Element);
    while(reader->get_name() == "variable" || reader->get_name() == "process") {
      assert(reader->is_empty_element());
      if (reader->get_name() == "process")
        register_process();
      else if (reader->get_attribute("process") != "") {
        using boost::lexical_cast;
        size_t var_id  = lexical_cast<size_t>(reader->get_attribute("id"));
        size_t proc_id = lexical_cast<size_t>(reader->get_attribute("process"));
        std::string index_str = reader->get_attribute("index");
        process* p = proc_map.get(proc_id);
        boost::any index = p->index(index_str);
        variable* v = p->at_any(index);
        var_map[var_id] = v;
      } else {
        variable* v;
        size_t id = boost::lexical_cast<size_t>(reader->get_attribute("id"));
        size_t sz = boost::lexical_cast<size_t>(reader->get_attribute("size"));
        std::string type = reader->get_attribute("type").c_str();
        std::string name = reader->get_attribute("name").c_str();
        if (type == "F")
          v = u.new_finite_variable(name, sz);
        else if(type == "V")
          v = u.new_vector_variable(name, sz);
        else 
          assert(false);
        var_map[id] = v;
      }
      reader->read();
      skip_whitespace();
    }
  }

  void xml_iarchive::skip_whitespace() {
    while(reader->get_node_type() != reader->Element &&
          reader->get_node_type() != reader->EndElement)
      if(!reader->read()) return;
  }

} // namespace sill
  
#else // ifdef HAVE_LIBXMLPP

namespace xmlpp {
  struct TextReader { }; 
}

#define SILL_LIBXMLPP_ERROR                                              \
  throw std::runtime_error                                              \
    ("PRL needs to be compiled with libxml++ to use xml_iarchive");

namespace sill {

  // Constructors
  //============================================================================
  xml_iarchive::xml_iarchive(const char* filename, named_universe& u)
    : u(u) {
    SILL_LIBXMLPP_ERROR;
  }
  
  xml_iarchive::xml_iarchive(const unsigned char* dat, size_t size, named_universe& u)
    : u(u) {
    SILL_LIBXMLPP_ERROR;
  } 

  xml_iarchive::xml_iarchive(const char* dat, size_t size, named_universe& u)
    : u(u) {
    SILL_LIBXMLPP_ERROR;
  }

  xml_iarchive::xml_iarchive(const std::vector<unsigned char>& dat, named_universe& u)
    : u(u) {
    SILL_LIBXMLPP_ERROR;
  }

  xml_iarchive::~xml_iarchive() {
    SILL_LIBXMLPP_ERROR;
  }

  // Public functions 
  //============================================================================
  std::string xml_iarchive::name() const {
    SILL_LIBXMLPP_ERROR;
  }

  std::string xml_iarchive::attribute(const char* name) const {
    SILL_LIBXMLPP_ERROR;
  }
  
  std::string xml_iarchive::text() const {
    SILL_LIBXMLPP_ERROR;
  }

  bool xml_iarchive::has_next() {
    SILL_LIBXMLPP_ERROR;
  }
    
  void xml_iarchive::override_name(const char* name) {
    SILL_LIBXMLPP_ERROR;
  }

  void xml_iarchive::start_element(const char* name) {
    SILL_LIBXMLPP_ERROR;
  }
    
  void xml_iarchive::end_element() {
    SILL_LIBXMLPP_ERROR;
  }

  xml_iarchive& xml_iarchive::operator>>(var_vector& vars) {
    SILL_LIBXMLPP_ERROR;
  }

  xml_iarchive& xml_iarchive::operator>>(finite_var_vector& vars) {
    SILL_LIBXMLPP_ERROR;
  }

  xml_iarchive& xml_iarchive::operator>>(vector_var_vector& vars) {
    SILL_LIBXMLPP_ERROR;
  }

  xml_iarchive& xml_iarchive::operator>>(domain& vars) {
    SILL_LIBXMLPP_ERROR;
  }

  xml_iarchive& xml_iarchive::operator>>(finite_domain& vars) {
    SILL_LIBXMLPP_ERROR;
  }

  xml_iarchive& xml_iarchive::operator>>(vector_domain& vars) {
    SILL_LIBXMLPP_ERROR;
  }

  serializable* xml_iarchive::load() {
    SILL_LIBXMLPP_ERROR;
  }

  void xml_iarchive::load(const char* name, std::string& str) {
    SILL_LIBXMLPP_ERROR;
  }

  std::string xml_iarchive::load_string(const char* name) {
    SILL_LIBXMLPP_ERROR;
  }

  // Private functions 
  //============================================================================
  void xml_iarchive::initialize() {
    SILL_LIBXMLPP_ERROR;
  }

  void xml_iarchive::register_variables() {
    SILL_LIBXMLPP_ERROR;
  }

  void xml_iarchive::skip_whitespace() {
    SILL_LIBXMLPP_ERROR;
  }

} // namespace sill

#endif 



