#include <sill/factor/moment_gaussian.hpp>
#include <sill/learning/dataset/sequence_memory_dataset.hpp>
#include <sill/learning/dataset/vector_sequence_record.hpp>
#include <sill/learning/parameter/markov_chain_mle.hpp>
#include <sill/parsers/string_functions.hpp>

#include <boost/filesystem.hpp>
#include <map>

#include <sill/macros_def.hpp>

using namespace sill;

// the dataset that will store sequences of vector variables
typedef sequence_memory_dataset<vector_dataset<> > dataset_type;

// map from sign to dataset
typedef std::map<std::string, dataset_type> dataset_map;

// map from sign to a markov chain model
typedef std::map<std::string, markov_chain<moment_gaussian> > model_map;

void load_subject(const std::string& dir,
                  const symbolic_format& format,
                  dataset_map& datasets) {
  using namespace boost::filesystem;
  path p(dir);
  std::vector<vector_discrete_process*> processes = format.vector_discrete_proc_vec();
  vector_sequence_record<> record(processes);
  for (directory_iterator it(p), end; it != end; ++it) {
    std::string filename = it->path().filename().string();
    std::string sign = filename.substr(0, filename.find('-'));
    load_tabular(it->path().string(), format, record);
    if (!datasets.count(sign)) {
      datasets[sign].initialize(processes);
    }
    datasets[sign].insert(record);
  }
}

std::string predict(const model_map& models,
                    const vector_sequence_record<>& r) {
  std::string best_sign;
  double best_ll = -std::numeric_limits<double>::infinity();
  foreach(const model_map::value_type& p, models) {
    double ll = log(p.second(r));
    if (ll > best_ll) {
      best_ll = ll;
      best_sign = p.first;
    }
  }
  return best_sign;
}

int main(int argc, char** argv) {

  if (argc < 3) {
    std::cerr << "Usage: auslan <directory> <smoothing>" << std::endl;
    return -1;
  }

  size_t num_subjects = 9;
  size_t num_train = 7;
  size_t num_test = 2;
  std::string rootdir = argv[1];
  double smoothing = sill::parse_string<double>(argv[2]);

  // load the format
  symbolic_format format;
  universe u;
  format.load_config(rootdir + "/auslan.cfg", u);
  std::vector<vector_discrete_process*> processes = format.vector_discrete_proc_vec();

  // load the data
  dataset_map datasets;
  for (size_t i = 1; i <= num_subjects; ++i) {
    std::cout << "Loading subject " << i << std::endl;
    std::string subject_dir = rootdir + "/auslan2-mld/tctodd" + to_string(i);
    load_subject(subject_dir, format, datasets);
  }
  
  // train the models
  model_map models;
  markov_chain_mle<moment_gaussian> learner(processes, 1);
  foreach(dataset_map::value_type& p, datasets) {
    std::cout << "Training model for sign " << p.first << std::endl;
    std::string sign = p.first;
    size_t count = p.second.size() * num_train / num_subjects;
    learner.learn(p.second.subset(0, count), smoothing, models[sign]);
  }

  // test the models
  size_t sum_correct = 0;
  size_t sum_test = 0;
  foreach(dataset_map::value_type& p, datasets) {
    std::string sign = p.first;
    size_t num_correct = 0;
    size_t num_test = 0;
    size_t count = p.second.size() * num_train / num_subjects;
    dataset_type::slice_view_type test = p.second.subset(count, p.second.size());
    foreach(const vector_sequence_record<>& r, test.records(processes)) {
      std::string prediction = predict(models, r);
      num_correct += (sign == prediction);
      ++num_test;
    }
    sum_correct += num_correct;
    sum_test += num_test;
    std::cout << sign << ": "
              << num_correct << " / " << num_test << " correct" << std::endl;
  }
  std::cout << "Overall: " << sum_correct << " / " << sum_test 
            << " (" << 100 * sum_correct / sum_test << "%)" << std::endl;
  
  return 0;
}
