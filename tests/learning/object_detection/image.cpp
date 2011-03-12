#include <iostream>

#include <boost/timer.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/dataset/data_loader.hpp>
#include <sill/learning/object_detection/image.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;
  boost::timer timer;

  std::string filename
    = argc > 1 ? argv[1] : "../../../../tests/data/image_test.sum";
  universe u;
  boost::shared_ptr<vector_dataset<> > data_ptr
    = data_loader::load_symbolic_dataset<vector_dataset<> >(filename, u);

  cout  << "For each image, print original image, rescaled image (1/2 size), "
        << "rescaled image (3/4 size), integral representation of image."
        << endl;
  for (vector_dataset<>::record_iterator it = data_ptr->begin();
       it != data_ptr->end(); ++it) {
    record<> r(*it);
    image::write(cout, r);
    cout << endl;
    image::set_view(r, 0, 0,
                    image::true_height(r) / 2, image::true_width(r) / 2,
                    .5, .5);
    image::write(cout, r);
    cout << endl;
    image::set_view(r, 0, 0,
                    3 * image::true_height(r) / 4, 3 * image::true_width(r) / 4,
                    .75, .75);
    image::write(cout, r);
    cout << endl;
    image::reset_view(r);
    image::raw2integral(r);
    image::write(cout, r);
    cout << "\n------------------" << endl;
  }
}
