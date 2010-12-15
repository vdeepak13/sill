#include <prl/archive/xml_oarchive.hpp>

#include <prl/process.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  namespace {

    //! A structure that represents an indentation level
    struct indent {
      int level_;
      indent() : level_() { }
      indent(int level): level_(level) { }
      operator int() { return level_; }
      indent operator+(int i) { return indent(level_+i); }
      indent& operator++() { level_++; return *this; }
      indent& operator--() { level_--; return *this; }
    }; // struct indent
    
    
    inline std::ostream& operator<<(std::ostream& out, const indent& indent) {
      assert(indent.level_ > 0);
      for(int i = 0; i < indent.level_; ++i)
        out << "  ";
      return out;
    }

  } // unnamed namespace

  //! Creates an archive for the given stream
  xml_oarchive::xml_oarchive(std::ostream& out) 
    : out(out), inside_header(false), finalized(false), leaf(false) { 
    out << "<?xml version=\"1.0\"?>" << std::endl;
    out << std::endl;
    out << "<prl>";
    level = 1;
  }

  xml_oarchive::~xml_oarchive() {
    finalize_document();
  }

  void xml_oarchive::register_process(process* p) {
    if (!registered_procs.contains(p)) {
      registered_procs.insert(p);
      const char* type;
      size_t size;
      if (dynamic_cast<finite_timed_process*>(p) != NULL) {
        type = "FT";
        size = static_cast<finite_timed_process*>(p)->size();
      } else if(dynamic_cast<vector_timed_process*>(p) != NULL) {
        type = "VT"; 
        size = static_cast<vector_timed_process*>(p)->size();
      } else {
        assert(false);
        size = 0;
      }
      out << std::endl << indent(level) << "<process "
          << "id=\""   << reinterpret_cast<size_t>(p)  << "\" "
          << "type=\"" << type << "\" "
          << "name=\"" << p->name() << "\" "
          << "size=\"" << size << "\"/>";
    }
  }

  void xml_oarchive::register_variable(variable* v) {
    if (!registered_vars.contains(v)) {
      registered_vars.insert(v);

      if (v->process() != NULL) {
        register_process(v->process());
        // process id and the index are enough to restore the process variable
        out << std::endl  << indent(level) << "<variable "
            << "id=\""      << reinterpret_cast<size_t>(v) << "\" "
            << "process=\"" << reinterpret_cast<size_t>(v->process()) << "\" "
            << "index=\""   << v->process()->index_str(v->index()) << "\"/>";
      } else {
        // output the complete description
        char type;
        size_t size;
        if (dynamic_cast<finite_variable*>(v) != NULL) {
          type = 'F';
          size = static_cast<finite_variable*>(v)->size();
      } else if(dynamic_cast<vector_variable*>(v) != NULL) {
          type = 'V'; 
          size = static_cast<vector_variable*>(v)->size();
        } else {
          assert(false);
          size = 0;
        }
        out << std::endl << indent(level) << "<variable "
            << "id=\""   << reinterpret_cast<size_t>(v)  << "\" "
            << "process=\"\" "
            << "type=\"" << type << "\" "
            << "name=\"" << v->name() << "\" "
            << "size=\"" << size << "\"/>";
      }
    }
  }

  void xml_oarchive::register_variables(const forward_range<variable*>& vars) {
    finalize_header();
    foreach(variable* v, vars) register_variable(v);
  }

  void xml_oarchive::override_name(const char* name) {
    assert(name && *name); // !name.empty()
    if(custom_name.empty()) custom_name = name;
  }
    
  void xml_oarchive::start_element(const char* name) {
    finalize_header(); // finalize header, if any
    if (custom_name.empty()) {
      elements.push(name);
    } else {
      elements.push(custom_name);
      custom_name.clear();
    }
    out << std::endl << indent(level) << '<' << elements.top();
    inside_header = true;
    leaf = true;
    ++level;
  }

  void xml_oarchive::add_attribute(const char* name, const char* value) {
    out << ' ' << name << '=' << '"' << value << '"';
  }

  void xml_oarchive::add_attribute(const char* name, size_t value) {
    out << ' ' << name << '=' << '"' << value << '"';
  }

  void xml_oarchive::add_attribute(const char* name, int value) {
    out << ' ' << name << '=' << '"' << value << '"';
  }

  void xml_oarchive::end_element() {
    finalize_header();
    --level;
    if (!leaf) out << std::endl << indent(level);
    out << "</" << elements.top() << ">";
    elements.pop();
    leaf = false;
  }

  xml_oarchive& xml_oarchive::operator<<(const forward_range<variable*>& vars) {
    assert(!custom_name.empty());
    register_variables(vars);
    start_element(custom_name.c_str());
    finalize_header();
    foreach(variable* v, vars) out << reinterpret_cast<size_t>(v) << ' ';
    end_element();
    return *this;
  }

  xml_oarchive& xml_oarchive::operator<<(size_t x) {
    start_element("size_t");
    finalize_header();
    out << x;
    end_element();
    return *this;
  }

  void xml_oarchive::finalize_document() {
    if(!finalized) {
      assert(level == 1);
      finalized = true;
      out << std::endl << "</prl>" << std::endl;
    }
  }

} // namespace prl

