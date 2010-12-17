#include <iostream>
#include <prl/base/finite_variable.hpp>
#include <prl/base/assignment.hpp>
#include <prl/base/universe.hpp>
#include <prl/factor/table_factor.hpp>

int main(int argc, char** argv) {

  using namespace prl;
  using namespace std;

  cout << std::log(-0.01) << endl;
  cout << std::log(0.0) << endl;


  // Choose the dimensions of the grid.
  const int width = 30; // 100;
  const int height = 30; // 100;

  // Create a universe.
  universe u;

  // Make variables corresponding to the x and y locations at two time
  // steps (t and t + 1).
  finite_variable* x_t  = u.new_finite_variable(width);
  finite_variable* x_tp = u.new_finite_variable(width);
  finite_variable* y_t  = u.new_finite_variable(height);
  finite_variable* y_tp = u.new_finite_variable(height);

  // Make a prior location distribution that puts unit mass on a
  // central grid square.
  finite_domain pos_t;
  pos_t.insert(x_t);
  pos_t.insert(y_t);
  table_factor location(pos_t, 0.0);
  finite_assignment a;
  a[x_t] = width / 2;
  a[y_t] = height / 2;
  location(a) = 1;
  
  // Make a motion model which puts uniform mass on a cell's neighbors.
  cout << "Initializing motion model..." << flush;
  finite_domain pos_tp;
  pos_tp.insert(x_tp);
  pos_tp.insert(y_tp);
  finite_domain pos_ttp = set_union(pos_t, pos_tp);
  table_factor motion_model(pos_ttp, 0.0);
  int x, y;
  for (a[x_t]=x=0; x<width; ++a[x_t], ++x) {
    for (a[y_t]=y=0; y<height; ++a[y_t], ++y) {
      // Distribute the probability mass evenly among neighbors.
      size_t n = 0; // number of cells
      n = (min(x+2, width) - max(x-1, 0)) * (min(y+2, height) - max(y-1, 0));
      for (a[x_tp] = max(x-1, 0); 
	   a[x_tp] < size_t(min(x+2, width)); ++a[x_tp])
	for (a[y_tp] = max(y-1, 0);
	     a[y_tp] < size_t(min(y+2, height)); ++a[y_tp])
	  motion_model(a)= 1.0 / n;
    }
  }
  // cout << " (" << motion_model.table().num_explicit_elts()
  //	  << " explicit elements)" << endl;

  // Create a mapping to rename the variables.
  finite_var_map var_map;
  var_map[x_tp] = x_t;
  var_map[y_tp] = y_t;

  // Evolve the location distribution.
  for (int t = 1; t < 50; ++t) {
    cout << "time: " << t << endl;
    table_factor prediction = location * motion_model;
    table_factor roll_up = sum(prediction, pos_t);
    location = roll_up.subst_args(var_map);
  }

  cout << location << endl;

  return EXIT_SUCCESS;
}
