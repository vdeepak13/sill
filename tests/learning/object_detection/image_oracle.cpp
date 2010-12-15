#include <iostream>

#include <boost/timer.hpp>

#include <prl/base/universe.hpp>
#include <prl/learning/dataset/vector_dataset.hpp>
#include <prl/learning/dataset/data_conversions.hpp>
#include <prl/learning/object_detection/image.hpp>
#include <prl/learning/object_detection/image_oracle.hpp>

#include <prl/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace prl;
  using namespace std;

  cout << "Variable-sized images:\n" << endl;
  std::string filename
    = (argc > 1) ? argv[1] : "../../../../tests/data/image_varsize.txt";
  universe u;
  boost::shared_ptr<std::vector<record> >
    data_ptr(load_images(filename, u));
  foreach(const record& img, *data_ptr) {
    image::write(std::cout, img);
    std::cout << std::endl;
  }

  cout << "\nFixed-sized images:\n" << endl;
  filename = "../../../../tests/data/image_fixedsize.txt";
  data_ptr = load_images(filename, u);
  foreach(const record& img, *data_ptr) {
    image::write(std::cout, img);
    std::cout << std::endl;
  }

}
