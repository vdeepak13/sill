#include <sill/factor/table_factor.hpp>
#include <sill/model/naive_bayes.hpp>
#include <sill/learning/dataset/finite_memory_dataset.hpp>
#include <sill/learning/dataset/finite_dataset_io.hpp>
#include <sill/learning/factor_mle/table_factor.hpp>
#include <sill/learning/parameter/naive_bayes_em.hpp>
#include <sill/learning/parameter/naive_bayes_learner.hpp>

#include <sill/macros_def.hpp>

using namespace sill;
void normalize(const finite_dataset& in_ds,
               finite_variable* class_var,
               std::size_t samples_per_class,
               finite_memory_dataset& out_ds) {
  // extract the indices by each 
  std::vector<std::vector<size_t> > indices(class_var->size());
  size_t index = 0;
  foreach(const finite_record& r, in_ds.records(make_vector(class_var))) {
    indices[r.values[0]].push_back(index++);
  }

  // draw random samples (slow!!!)
  out_ds.initialize(in_ds.arg_vector(), samples_per_class * class_var->size());
  foreach(const std::vector<size_t>& ind, indices) {
    for (size_t i = 0; i < samples_per_class; ++i) {
      index = ind[rand() % ind.size()];
      out_ds.insert(in_ds.record(index));
    }
  }
}

/*
 * This example performs clustering on the Car dataset and attempts to recover
 * the true classes in an unsupervised manner.
 */
int main(int argc, char** argv) {
  using namespace sill;
  std::string data_dir = argc > 1 ? argv[1] : "";
  size_t num_clusters = argc > 2 ? atoi(argv[2]) : 4;
  size_t num_restarts = argc > 4 ? atoi(argv[4]) : 1;

  // load the dataset format
  universe u;
  symbolic_format format;
  format.load_config(data_dir + "car.cfg", u);

  // load the dataset
  finite_memory_dataset ds1;
  load(data_dir + "car.data", format, ds1);
  
  // create the naive Bayes EM learner with the latent variable label
  // and features all variables in the dataset other than "class"
  finite_variable* class_var = format.finite_var("class");
  finite_domain feature_vars = format.finite_vars();
  feature_vars.erase(class_var);
  finite_variable* label_var = u.new_finite_variable("label", num_clusters);
  naive_bayes_em<table_factor> learner(label_var, feature_vars);

  // normalize the dataset
  finite_memory_dataset ds;
  normalize(ds1, class_var, 300, ds);
  
  // train the model on the normalized data,
  // using the default regularization parameters (set to 0)
  naive_bayes<table_factor> best_model;
  double best_ll = -std::numeric_limits<double>::infinity();
  naive_bayes_em<table_factor>::param_type params;
  if (argc > 3) { params.max_iters = atoi(argv[3]); }
  for (unsigned seed = 0; seed < num_restarts; ++seed) {
    naive_bayes<table_factor> model;
    double ll = learner.learn(ds, params, model);
    std::cout << "Seed " << seed
              << ": ll=" << ll
              << " (" << learner.num_iters() << " iterations)"
              << std::endl;
    params.seed = seed;
    if (ll > best_ll) {
      best_ll = ll;
      best_model = model;
    }
  }

  std::cout << "best ll " << best_ll << std::endl;
  //std::cout << best_model << std::endl;

  // compute a human-readable representation of the clusters
  mat counts = arma::zeros(label_var->size(), class_var->size());
  finite_assignment a;
  foreach(const finite_record& r, ds.records(ds.arg_vector())) {
    r.extract(a);
    table_factor posterior = best_model.posterior(a);
    assert(posterior.size() == num_clusters);
    for(size_t i = 0; i < num_clusters; ++i) {
      counts(i, a[class_var]) += posterior(i);
    }
  }

  // compute the label map
  mat dist = counts / repmat(sum(counts, 1), 1, class_var->size());
  std::cout << dist << std::endl;
  std::vector<arma::uword> label_map(num_clusters);
  for (size_t i = 0; i < num_clusters; ++i) {
    mat(dist.row(i)).max(label_map[i]);
  }

  // train a classifier using the (fully observed) data
  naive_bayes_learner<table_factor> observed_learner(class_var, feature_vars);
  naive_bayes<table_factor> observed_model;
  observed_learner.learn(ds, observed_model);
  
  // evaluate the accuracy of the two classifiers
  size_t correct_observed = 0;
  size_t correct_clusters = 0;
  foreach(const finite_record& r, ds.records(ds.arg_vector())) {
    r.extract(a);
    correct_observed +=
      arg_max(observed_model.posterior(a))[class_var] == a[class_var];
    correct_clusters +=
      label_map[arg_max(best_model.posterior(a))[label_var]] == a[class_var];
  }

  std::cout << "correct observed: " << correct_observed << std::endl;
  std::cout << "correct clusters: " << correct_clusters << std::endl;

  return 0;
}
