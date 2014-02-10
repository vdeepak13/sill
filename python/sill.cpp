#include <boost/python.hpp>

#include "converters.hpp"

void def_base();
void def_factor();
void def_graph();
void def_model();

BOOST_PYTHON_MODULE(sillpy) {
  def_base();
  def_factor();
  def_graph();
  def_model();
  variable_converters();
  container_converters();
  // what about hash functions?
}
