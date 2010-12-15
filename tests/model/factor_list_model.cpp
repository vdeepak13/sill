#include <iostream>

#include <boost/array.hpp>

#include <prl/variable.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/datastructure/dense_table.hpp>
#include <prl/model/factor_list_model.hpp>

/**
 * \file factor_list_model.cpp Factor List Model test
 */
int main() {

  using namespace prl;
  using boost::array;

  // Create a universe.
  universe u;

  // Create some variables and factors
  finite_variable_h x0 = u.new_finite_variable(2);
  finite_variable_h x1 = u.new_finite_variable(2);
  finite_variable_h x2 = u.new_finite_variable(2);
  finite_variable_h x3 = u.new_finite_variable(2);
  finite_variable_h x4 = u.new_finite_variable(2);
  array<finite_variable_h, 1> a0 = {{x0}};
  array<double, 2> v0 = {{.3, .7}};
  array<finite_variable_h, 1> a1 = {{x1}};
  array<double, 2> v1 = {{.5, .5}};
  array<finite_variable_h, 2> a12 = {{x1, x2}};
  array<double, 4> v12 = {{.8, .2, .2, .8}};
  array<finite_variable_h, 3> a123 = {{x1, x2, x3}};
  array<double, 8> v123 = {{.1, .1, .3, .5, .9, .9, .7, .5}};
  array<finite_variable_h, 3> a034 = {{x0, x3, x4}};
  array<double, 8> v034 = {{.6, .1, .2, .1, .4, .9, .8, .9}};

  table_factor f0 = make_dense_table_factor(a0, v0);
  table_factor f1 = make_dense_table_factor(a1, v1);
  table_factor f12 = make_dense_table_factor(a12, v12);
  table_factor f123 = make_dense_table_factor(a123, v123);
  table_factor f034 = make_dense_table_factor(a034, v034);

  factor_list_model<table_factor> flm(make_domain(x0,x1,x2,x3,x4));
  flm.add_factor(f0);
  flm.add_factor(f1);
  flm.add_factor(f12);
  flm.add_factor(f123);
  flm.add_factor(f034);

  std::cout << "Factor list model:\n";
  flm.print(std::cout);
  std::cout << std::endl << "Saving copy of this model." << std::endl;

  factor_list_model<table_factor> flm2(flm);

  finite_assignment a;
  a[x0] = 0;
  a[x2] = 1;
  flm.condition(a);
  std::cout << "Conditioned model on assignment: " << a << " to get:"
            << std::endl;
  flm.print(std::cout);
  std::cout << std::endl;

  a[x1] = 0;
  a[x3] = 0;
  a[x4] = 0;
  flm.condition(a);
  std::cout << "Conditioned model on assignment: " << a << " to get:"
            << std::endl;
  flm.print(std::cout);
  std::cout << std::endl;

  std::cout << "Copy of original model: ";
  flm2.print(std::cout);
  std::cout << std::endl;

}
