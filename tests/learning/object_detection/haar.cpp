#include <iostream>

#include <boost/timer.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/data_loader.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/object_detection/haar.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;
  boost::timer timer;

  std::string filename
    = argc > 1 ? argv[1] : "../../../../tests/data/image_test2.sum";
  universe u;
  boost::shared_ptr<vector_dataset> data_ptr
    = data_loader::load_symbolic_dataset<vector_dataset>(filename, u);
  for (int i = 2; i < argc; ++i) {
    symbolic_oracle
      o(*(data_loader::load_symbolic_oracle(argv[i],
                                            data_ptr->datasource_info())));
    while(o.next())
      data_ptr->insert(o.current());
  }
  data_ptr->randomize(20938471);

  dataset_view ds_train(*data_ptr);
  ds_train.set_record_range(0, data_ptr->size() / 2);
  dataset_view ds_test(*data_ptr);
  ds_test.set_record_range(data_ptr->size() / 2, data_ptr->size());
  statistics stats(ds_train);

  timer.restart();
  haar<> h(stats);

  cout << "Trained Haar classifier in " << timer.elapsed()
       << " seconds to get training accuracy of "
       << h.train_accuracy() << ", now testing:" << endl
       << "true label\tpredicted label" << endl;

  size_t class_var_index
    = data_ptr->record_index(data_ptr->finite_class_variables().front());

  size_t pright = 0;
  size_t ptotal = 0;
  size_t nright = 0;
  size_t ntotal = 0;
  foreach(const record& example, ds_test.records()) {
    size_t predicted = h.predict(example);
    size_t truth = example.finite(class_var_index);
    if (truth == 0) {
      ++ntotal;
      if (predicted == truth)
        nright++;
    } else {
      ++ptotal;
      if (predicted == truth)
        pright++;
    }
  }
  cout << "Test accuracy on negative examples = "
       << ((double)(nright) / ntotal) << endl
       << "Test accuracy on positive examples = "
       << ((double)(pright) / ptotal) << endl;

  cout << "Saving haar...";
  h.save("haar_test.txt");
  cout << "loading haar...";
  h.load("haar_test.txt", *data_ptr);
  cout << "testing haar again...";
  pright = 0;
  ptotal = 0;
  nright = 0;
  ntotal = 0;
  foreach(const record& example, data_ptr->records()) {
    size_t predicted = h.predict(example);
    size_t truth = example.finite(class_var_index);
    if (truth == 0) {
      ++ntotal;
      if (predicted == truth)
        nright++;
    } else {
      ++ptotal;
      if (predicted == truth)
        pright++;
    }
  }
  cout << "Test accuracy on negative examples = "
       << ((double)(nright) / ntotal) << endl
       << "Test accuracy on positive examples = "
       << ((double)(pright) / ptotal) << endl;

}
