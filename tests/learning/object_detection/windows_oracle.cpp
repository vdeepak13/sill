#include <iostream>

#include <boost/timer.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/object_detection/image.hpp>
#include <sill/learning/object_detection/image_oracle.hpp>
#include <sill/learning/object_detection/random_windows_oracle.hpp>
#include <sill/learning/object_detection/sliding_windows_oracle.hpp>

#include <sill/macros_def.hpp>

/**
 * Test of sliding_windows_oracle and random_windows_oracle
 */
int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  size_t window_h = 2;
  size_t window_w = 2;

  std::string filename
    = argc > 1 ? argv[1] : "../../../../tests/data/image_varsize.txt";
  universe u;
  copy_ptr<image_oracle> o_ptr(new image_oracle(filename, u));
  std::vector<record> data;
  while(o_ptr->next())
    data.push_back(o_ptr->current());
  cout << "Original images:" << endl;
  foreach(const record& img, data) {
    image::write(cout, img);
    cout << endl;
  }

  cout << "-------------------------\n\n"
       << "sliding_windows_oracle using all images:\n" << endl;
  o_ptr->reset();
  vector_var_vector var_order(image::create_var_order(u, window_h, window_w, 1,
                                                      vector_var_vector()));
  sliding_windows_oracle sliding_o1(o_ptr, var_order, window_h, window_w);
  while(sliding_o1.next())
    image::write(cout, sliding_o1.current());

  cout << "-------------------------\n\n"
       << "sliding_windows_oracle using only the first image:\n" << endl;
  o_ptr->reset();
  o_ptr->next();
  sliding_windows_oracle sliding_o2(o_ptr->current(), var_order, window_h,
                                    window_w);
  while(sliding_o2.next())
    image::write(cout, sliding_o2.current());

  cout << "-------------------------\n\n"
       << "sliding_windows_oracle using only the first image and adding "
       << "class variable with value 100:\n" << endl;
  o_ptr->reset();
  o_ptr->next();
  sliding_windows_oracle::parameters swo_params;
  swo_params.class_variable = u.new_finite_variable(101);
  swo_params.label = 100;
  sliding_windows_oracle sliding_o3(o_ptr->current(), var_order, window_h,
                                    window_w, swo_params);
  while(sliding_o3.next())
    cout << sliding_o3.current().assignment() << endl;

  cout << "-------------------------\n\n"
       << "random_windows_oracle using all images:\n" << endl;
  random_windows_oracle::parameters rwo_params;
  rwo_params.random_seed = 5627654;
  random_windows_oracle random_o(data, var_order, window_h, window_w,
                                 rwo_params);
  for (size_t i = 0; i < 4; ++i) {
    random_o.next();
    image::write(cout, random_o.current());
  }

}
