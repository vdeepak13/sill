#include "converters.hpp"

#include <sill/base/variable.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/finite_assignment.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/base/vector_assignment.hpp>
#include <sill/factor/table_factor.hpp>

#include <boost/python.hpp>
#include <boost/python/detail/prefix.hpp>
#include <boost/python/object/stl_iterator_core.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/python/stl_iterator.hpp>

using namespace boost::python;
using namespace sill;

#include <sill/macros_def.hpp>

// An STL input iterator over a python sequence 
// that allows us to check the type of the elements
template<typename T>
struct stl_type_checking_iterator
  : boost::iterator_facade<stl_type_checking_iterator<T>,
                           bool, std::input_iterator_tag, bool> {
  
  stl_type_checking_iterator()
    : impl_() { }

  // ob is the python sequence
  stl_type_checking_iterator(const boost::python::object& ob)
    : impl_(ob) { }
  
private:
  friend class boost::iterator_core_access;

  void increment() { impl_.increment(); }

  bool dereference() const  {
    extract<T> ex(this->impl_.current().get());
    return ex.check();
  }

  bool equal(const stl_type_checking_iterator& other) const {
    return impl_.equal(other.impl_);
  }

  boost::python::objects::stl_input_iterator_impl impl_;
};

template <typename V>
struct domain_to_list {
  static PyObject* convert(const std::set<V*>& dom) {
    list result;
    foreach(V* v, dom) {
      result.append(ptr(v));
    }
    return incref(result.ptr());
  }
  static const PyTypeObject* get_pytype() {
    return &PyList_Type;
  }
};

template <typename V>
struct vector_to_list {
  static PyObject* convert(const std::vector<V*>& vec) {
    list result;
    foreach(V* v, vec) {
      result.append(ptr(v));
    }
    return incref(result.ptr());
  }
  static const PyTypeObject* get_pytype() {
    return &PyList_Type;
  }
};

struct finite_assignment_to_dict {
  static PyObject* convert(const finite_assignment& a) {
    dict result;
    foreach(const finite_assignment::value_type& value, a) {
      result[ptr(value.first)] = value.second;
    }
    return incref(result.ptr());
  }
  static const PyTypeObject* get_pytype() {
    return &PyDict_Type;
  }
};

template <typename V>
struct extract_vector {
  static std::vector<V*> execute(PyObject* obj) {
    static std::vector<V*> val;
    return val;
    //return std::vector<V*>();
  }
};

template <typename Container>
struct from_python_sequence {
  typedef typename Container::value_type value_type;
  from_python_sequence() {
    boost::python::converter::registry::push_back(
      &convertible,
      &construct,
      boost::python::type_id<Container>()
    );
  }
  
  static void* convertible(PyObject* pyobj) {
    if (!(PyList_Check(pyobj) ||
          PyTuple_Check(pyobj) ||
          PyIter_Check(pyobj) ||
          PyRange_Check(pyobj))) {
      return 0;
    }
    object obj(boost::python::handle<>(boost::python::borrowed(pyobj)));
    stl_type_checking_iterator<value_type> it(obj), end;
    for (; it != end; ++it) {
      if (!*it) {
        return 0;
      }
    }
    return pyobj;
  }
  
  static void construct(
    PyObject* pyobj,
    boost::python::converter::rvalue_from_python_stage1_data* data) {

    void* storage =
      ((boost::python::converter::rvalue_from_python_storage<Container>*)data)
      ->storage.bytes;
    new (storage) Container();
    data->convertible = storage;
    Container& result = *((Container*)storage);
    object obj(boost::python::handle<>(boost::python::borrowed(pyobj)));
    stl_input_iterator<value_type> it(obj), end;
    std::copy(it, end, std::inserter(result, result.end()));
  }
};

template <typename Container>
struct from_python_dictionary {
  typedef typename Container::key_type key_type;
  typedef typename Container::mapped_type mapped_type;
  typedef typename Container::value_type value_type;

  from_python_dictionary() {
    boost::python::converter::registry::push_back(
      &convertible,
      &construct,
      boost::python::type_id<Container>()
    );
  }
  
  static void* convertible(PyObject* pyobj) {
    if (!PyDict_Check(pyobj)) {
      return 0;
    }
    dict obj(boost::python::handle<>(boost::python::borrowed(pyobj)));
    list items = obj.items();
    stl_input_iterator<tuple> it(items), end;
    for (; it != end; ++it) {
      tuple pair = *it;
      if (len(pair) != 2) {
        return 0;
      }
      extract<key_type>    xkey(pair[0]);
      extract<mapped_type> xmap(pair[1]);
      if (!xkey.check() || !xmap.check()) {
        return 0;
      }
    }
    return pyobj;
  }
  
  static void construct(
    PyObject* pyobj,
    boost::python::converter::rvalue_from_python_stage1_data* data) {

    void* storage =
      ((boost::python::converter::rvalue_from_python_storage<Container>*)data)
      ->storage.bytes;
    new (storage) Container();
    data->convertible = storage;
    Container& result = *((Container*)storage);

    dict obj(boost::python::handle<>(boost::python::borrowed(pyobj)));
    list items = obj.items();
    stl_input_iterator<tuple> it(items), end;
    for (; it != end; ++it) {
      tuple pair = *it;
      key_type key = extract<key_type>(pair[0]);
      mapped_type val = extract<mapped_type>(pair[1]);
      result.insert(std::make_pair(key, val));
    }
  }
};

void variable_converters() {
  to_python_converter<domain, domain_to_list<variable>, true>();
  to_python_converter<finite_domain, domain_to_list<finite_variable>, true>();
  to_python_converter<vector_domain, domain_to_list<vector_variable>, true>();

  to_python_converter<var_vector, vector_to_list<variable>, true>();
  to_python_converter<finite_var_vector, vector_to_list<finite_variable>, true>();
  to_python_converter<vector_var_vector, vector_to_list<vector_variable>, true>();

  to_python_converter<finite_assignment, finite_assignment_to_dict, true>();
  
  from_python_sequence<domain>();
  from_python_sequence<finite_domain>();
  from_python_sequence<vector_domain>();

  from_python_sequence<var_vector>();
  from_python_sequence<finite_var_vector>();
  from_python_sequence<vector_var_vector>();

  from_python_dictionary<finite_assignment>();
}

void container_converters() {
  from_python_sequence<std::vector<double> >();
  from_python_sequence<std::vector<size_t> >();
  from_python_sequence<std::vector<table_factor> >();
  // todo: to_python for these
}
