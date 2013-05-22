#include <iostream>

#include <boost/timer.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/object_detection/image.hpp>
#include <sill/learning/object_detection/image_oracle.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  cout << "Variable-sized images:\n" << endl;
  std::string filename
    = (argc > 1) ? argv[1] : "../../../../tests/data/image_varsize.txt";
  universe u;
  boost::shared_ptr<std::vector<record<> > >
    data_ptr(load_images(filename, u));
  foreach(const record<>& img, *data_ptr) {
    image::write(std::cout, img);
    std::cout << std::endl;
  }

  cout << "\nFixed-sized images:\n" << endl;
  filename = "../../../../tests/data/image_fixedsize.txt";
  data_ptr = load_images(filename, u);
  foreach(const record<>& img, *data_ptr) {
    image::write(std::cout, img);
    std::cout << std::endl;
  }

}
