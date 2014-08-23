#ifndef SILL_LEARNING_DISCRIMINATIVE_LOGISTIC_REGRESSION_HPP
#define SILL_LEARNING_DISCRIMINATIVE_LOGISTIC_REGRESSION_HPP

#include <algorithm>

#include <sill/functional.hpp>
#include <sill/learning/dataset_old/ds_oracle.hpp>
#include <sill/learning/dataset_old/vector_dataset.hpp>
#include <sill/learning/discriminative/binary_classifier.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

// Set to true to print debugging information.
#define DEBUG_LOGISTIC_REGRESSION 0

namespace sill {

  struct logistic_regression_parameters {

    //! Number of initial iterations to run;
    //! if using stochastic, then this is multiplied by the dataset size
    //!  (default = 1000)
    size_t init_iterations;

    //! 1: L_1
    //! 2: L_2 supported currently
    //!  (default = L_2)
    size_t regularization;

    //! regularization parameter
    //!  (default = 0)
    double lambda;

    /**
     * 0 = gradient descent,
     * 1 = Newton's method,
     * 2 = stochastic gradient descent
     *  (default = 2)
     */
    size_t method;

    //! Learning rate in (0,1]
    //!  (default = .1)
    double eta;

    /**
     * Rate at which to decrease the learning rate
     * (by multiplying ETA by MU each round).
     * Note: Setting INIT_ITERATIONS causes this to be reset
     * to the default, so this should be set after INIT_ITERATIONS.
     *  (default = exp(log(10^-4) / INIT_ITERATIONS),
     *         or .999 if INIT_ITERATIONS == 0)
     */
    double mu;

    //! Range [-PERTURB_INIT,PERTURB_INIT] within
    //! which to choose perturbed values for initial parameters
    //!  (default = 0 for L_1, .001 for L_2)
    double perturb_init;

    //! Amount of change in average log likelihood
    //! below which algorithm will consider itself converged
    //!  (default = .000001)
    double convergence;

    //! Used to make the algorithm deterministic
    //!  (default = time)
    double random_seed;

    logistic_regression_parameters()
      : init_iterations(1000), regularization(2), lambda(0), method(2),
        eta(.1), mu(choose_mu()), perturb_init(.001), convergence(.000001) {
      std::time_t time_tmp;
      time(&time_tmp);
      random_seed = time_tmp;
    }

    bool valid() const {
      if (regularization != 1 && regularization != 2)
        return false;
      if (lambda < 0)
        return false;
      if (method > 2)
        return false;
      if (eta <= 0 || eta > 1)
        return false;
      if (mu <= 0 || mu > 1)
        return false;
      if (perturb_init < 0)
        return false;
      if (convergence < 0)
        return false;
      return true;
    }

    double choose_mu() const {
      if (init_iterations == 0)
        return .999;
      else
        return exp(-4. * std::log(10.) / init_iterations);
    }

    void save(std::ofstream& out) const {
      out << init_iterations << " " << regularization << " "
          << lambda << " " << method << " " << eta << " " << mu << " "
          << perturb_init << " " << convergence << " " << random_seed
          << "\n";
    }

    void load(std::ifstream& in) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> init_iterations))
        assert(false);
      if (!(is >> regularization))
        assert(false);
      if (!(is >> lambda))
        assert(false);
      if (!(is >> method))
        assert(false);
      if (!(is >> eta))
        assert(false);
      if (!(is >> mu))
        assert(false);
      if (!(is >> perturb_init))
        assert(false);
      if (!(is >> convergence))
        assert(false);
      if (!(is >> random_seed))
        assert(false);
    }

  }; // struct logistic_regression_parameters

  /**
   * Class for learning a logistic regression model.
   *  - The parameters may be set to support different optimization methods
   *    and different types of regularization.
   *
   * TODO: THE CONFIDENCE() FUNCTIONS DO NOT HANDLE K-VALUED VECTOR VARIABLES
   *       PROPERLY!
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo serialization
   * @todo confidence-rated predictions, regularization, different optimization
   *       objectives
   */
  template <typename LA = dense_linear_algebra<> >
  class logistic_regression : public binary_classifier<LA> {

    // Public types
    //==========================================================================
  public:

    typedef LA la_type;

    typedef binary_classifier<la_type> base;

    typedef typename base::record_type record_type;

    // Protected data members
    //==========================================================================
  protected:

    // Data from base class:
    using base::label_;
    using base::label_index_;

    logistic_regression_parameters params;

    //! datasource.finite_list()
    finite_var_vector finite_seq;

    //! datasource.vector_list()
    vector_var_vector vector_seq;

    //! List of indices of non-class finite variables
    //! @todo See if it's faster just to compare with the class variable index.
    std::vector<size_t> finite_indices;

    //! finite_offset[j] = first index in w_fin for finite variable j,
    //!  as indexed in finite_indices
    //!  (so finite_offset[j] + k is the index for value k)
    std::vector<size_t> finite_offset;

    //! vector_offset[j] = first index in w_vec for vector variable j
    //!  (so vector_offset[j] + k is the index for value k of variable j)
    std::vector<size_t> vector_offset;

    //! Current eta
    double eta;

    //! Lambda
    double lambda;

    //! Weights w for finite part of x in predict(x) = sigma(w'x + b)
    std::vector<double> w_fin;

    //! Weights w for vector part of x in predict(x) = sigma(w'x + b)
    std::vector<double> w_vec;

    //! Offset b in predict(x) = sigma(w'x + b)
    double b;

    //! Gradient for w_fin
    std::vector<double> grad_fin;

    //! Gradient for w_vec
    std::vector<double> grad_vec;

    //! Gradient for b
    double grad_b;

    //! Total weight of training data (for batch learners),
    //! or total weight of training data used so far (for online learners).
    double total_train;

    //! Current training accuracy (for batch learners),
    //! or weighted sum of training accuracies so far (online).
    double train_acc;

    //! Current training log likelihood (batch),
    //! or weighted sum o training log likelihoods so far (online).
    double train_log_like;

    //! Current iteration number (from 0); i.e., number of iterations completed.
    size_t iteration_;

    //! Dataset (for batch mode when given an oracle)
    vector_dataset_old<la_type>* ds_ptr;

    //! Dataset
    const dataset<la_type>& ds;

    //! Dataset oracle (for online mode when given a dataset)
    ds_oracle<la_type>* ds_o_ptr;

    //! Data oracle
    oracle<la_type>& o;

    // Protected methods
    //==========================================================================

    //! Initialize stuff, and do the initial iterations of learning.
    void init();

    //! Gradient descent.
    bool step_gradient_descent();

    //! Stochastic gradient descent.
    bool step_stochastic_gradient_descent();

    // Public methods
    //==========================================================================
  public:

    // Virtual methods from base classes (*means pure virtual):
    //  From learner:
    //   name()*            x
    //   is_online()*       x
    //   training_time()     
    //   random_seed()      x
    //   save(), load()
    //  From classifier:
    //   is_confidence_rated()
    //   train_accuracy()
    //   set_confidences()
    //  From binary_classifier:
    //   create()*           x
    //   predict()*          x
    //   confidence()        x
    //   predict_raw()
    //   probability()       x

    // Constructors and destructors
    //==========================================================================

    /**
     * Constructor which builds nothing but may be used to create other
     * instances.
     * @param parameters    algorithm parameters
     */
    explicit logistic_regression(logistic_regression_parameters params
                                 = logistic_regression_parameters())
      : params(params),
        train_acc(-1), train_log_like(-std::numeric_limits<double>::max()),
        iteration_(0), ds_ptr(new vector_dataset_old<la_type>()), ds(*ds_ptr),
        ds_o_ptr(new ds_oracle<la_type>(*ds_ptr)), o(*ds_o_ptr) { }

    /**
     * Constructor.
     * @param stats         a statistics class for the training dataset
     * @param parameters    algorithm parameters
     */
    explicit logistic_regression(dataset_statistics<la_type>& stats,
                                 logistic_regression_parameters params
                                 = logistic_regression_parameters())
      : base(stats.get_dataset()), params(params),
        finite_seq(stats.get_dataset().finite_list()),
        vector_seq(stats.get_dataset().vector_list()), total_train(0),
        train_acc(-1), train_log_like(-std::numeric_limits<double>::max()),
        iteration_(0), ds_ptr(NULL), ds(stats.get_dataset()),
        ds_o_ptr(new ds_oracle<la_type>(stats.get_dataset())), o(*ds_o_ptr) {
      init();
    }

    /**
     * Constructor.
     * @param o    data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param parameters    algorithm parameters
     */
    logistic_regression(oracle<la_type>& o, size_t n,
                        logistic_regression_parameters params
                        = logistic_regression_parameters())
      : base(o), params(params),
        finite_seq(o.finite_list()), vector_seq(o.vector_list()),
        total_train(0), train_acc(-1),
        train_log_like(-std::numeric_limits<double>::max()), iteration_(0),
        ds_ptr(new vector_dataset_old<la_type>(o.datasource_info())), ds(*ds_ptr),
        ds_o_ptr(NULL), o(o) {
      switch(params.method) {
      case 0:
      case 1:
        for (size_t i(0); i < n; ++i)
          if (o.next())
            ds_ptr->insert(o.current(), o.weight());
          else
            break;
        break;
      case 2:
        break;
      default:
        assert(false);
      }
      init();
    }

    ~logistic_regression() {
      if (ds_o_ptr != NULL)
        delete(ds_o_ptr);
      if (ds_ptr != NULL)
        delete(ds_ptr);
    }

    //! Train a new binary classifier of this type with the given data.
    boost::shared_ptr<binary_classifier<la_type> > create(dataset_statistics<la_type>& stats) const {
      boost::shared_ptr<binary_classifier<la_type> >
        bptr(new logistic_regression(stats, this->params));
      return bptr;
    }

    //! Train a new binary classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<binary_classifier<la_type> > create(oracle<la_type>& o, size_t n) const {
      boost::shared_ptr<binary_classifier<la_type> >
        bptr(new logistic_regression(o, n, this->params));
      return bptr;
    }

    // Getters and helpers
    //==========================================================================

    //! Return a name for the algorithm without template parameters.
    std::string name() const {
      return "logistic_regression";
    }

    //! Return a name for the algorithm with comma-separated template parameters
    //! (e.g., objective).
    std::string fullname() const {
      return name();
    }

    //! Returns true iff learner is naturally an online learner.
    bool is_online() const;

    //! Returns training accuracy (or estimate of it).
    double train_accuracy() const;

    operator std::string() const {
      std::ostringstream out; out << *this; return out.str();
    }

    //! Print classifier
    template <typename Char, typename Traits>
    void write(std::basic_ostream<Char, Traits>& out) const {
      out << "Logistic Regression\n";
      out << "  w (finite) =";
      foreach(double v, w_fin)
        out << " " << v;
      out << "\n  w (vector) =";
      foreach(double v, w_vec)
        out << " " << v;
      out << "\n  b = " << b << "\n"
          << " training accuracy = " << train_accuracy() << "\n";
    }

    // Learning and mutating operations
    //==========================================================================

    //! Reset the random seed in this algorithm's parameters and in its
    //! random number generator.
    void random_seed(double value) {
      params.random_seed = value;
    }

    // Prediction methods
    //==========================================================================

    //! Predict the 0/1 label of a new example.
    std::size_t predict(const record_type& example) const {
      return (confidence(example) > 0 ? 1 : 0);
    }

    //! Predict the 0/1 label of a new example.
    std::size_t predict(const assignment& example) const {
      return (confidence(example) > 0 ? 1 : 0);
    }

    //! Value indicating the confidence of the prediction, with
    //!  predict() == (confidence() > 0) ? 1 : 0.
    //! If the classifier does not have actual confidence ratings,
    //!  then this should be any value with the correct sign.
    double confidence(const record_type& example) const;

    //! Value indicating the confidence of the prediction, with
    //!  predict() == (confidence() > 0) ? 1 : 0.
    //! If the classifier does not have actual confidence ratings,
    //!  then this should be any value with the correct sign.
    double confidence(const assignment& example) const;

    //! Predict the probability of the class variable having value 1.
    double probability(const record_type& example) const {
      return 1. / (1. + exp(-1. * confidence(example)));
    }

    double probability(const assignment& example) const {
      return 1. / (1. + exp(-1. * confidence(example)));
    }

    // Methods for iterative learners
    //==========================================================================

    //! Returns the current iteration number (from 0)
    //!  (i.e., the number of learning iterations completed).
    //! ITERATIVE ONLY: This must be implemented by iterative learners.
    size_t iteration() const {
      return iteration_;
    }

    /*
    //! Returns the total time elapsed after each iteration.
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    std::vector<double> elapsed_times() const {
    }
    */

    //! Does the next step of training (updating the current classifier).
    //! @return true iff the learner may be trained further
    //! ITERATIVE ONLY: This must be implemented by iterative learners.
    bool step();

    /*
    //! Resets the data source to be used in future rounds of training.
    //! @param  n   max number of examples which may be drawn from the oracle
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    void reset_datasource(oracle<la_type>& o, size_t n) { assert(false); }

    //! Resets the data source to be used in future rounds of training.
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    void reset_datasource(dataset_statistics<la_type>& stats) { assert(false); }
    */

    // Save and load methods
    //==========================================================================

    using base::save;
    using base::load;

    //! Output the classifier to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    void save(std::ofstream& out, size_t save_part = 0,
              bool save_name = true) const;

    /**
     * Input the classifier from a human-readable file.
     * @param in          input filestream for file holding the saved learner
     * @param ds          datasource used to get variables
     * @param load_part   0: load function (default), 1: engine, 2: shell
     * This assumes that the learner name has already been checked.
     * @return true if successful
     */
    bool load(std::ifstream& in, const datasource& ds, size_t load_part);

  }; // class logistic_regression

  // Free functions
  //==========================================================================

  template <typename Char, typename Traits, typename LA>
  std::basic_ostream<Char,Traits>&
  operator<<(std::basic_ostream<Char, Traits>& out,
             const logistic_regression<LA>& lr) {
    lr.write(out);
    return out;
  }

  //==========================================================================
  // Implementations of methods in logistic_regression
  //==========================================================================

  // Protected methods
  //==========================================================================

  template <typename LA>
  void logistic_regression<LA>::init() {
    assert(params.valid());
    if (ds.num_finite() > 1) {
      finite_offset.push_back(0);
      for (size_t j = 0; j < ds.num_finite(); ++j)
        if (j != label_index_) {
          finite_indices.push_back(j);
          finite_offset.push_back(finite_offset.back()+finite_seq[j]->size());
        }
      finite_offset.pop_back();
    }
    if (ds.num_vector() > 0) {
      vector_offset.push_back(0);
      for (size_t j = 0; j < ds.num_vector()-1; ++j)
        vector_offset.push_back(vector_offset.back()+vector_seq[j]->size());
    }
    lambda = params.lambda;
    if (params.regularization == 1 || params.regularization == 2) {
      if (lambda <= 0) {
        if (DEBUG_LOGISTIC_REGRESSION)
          std::cerr << "logistic_regression was told to"
                    << " use regularization but given lambda = " << lambda
                    << std::endl;
        assert(false);
        return;
      }
    }

    w_fin.resize(ds.finite_dim() - label_->size());
    w_vec.resize(ds.vector_dim());
    b = 0;
    grad_fin.resize(w_fin.size());
    grad_vec.resize(w_vec.size());

    if (params.perturb_init > 0) {
      boost::mt11213b rng(static_cast<unsigned>(params.random_seed));
      boost::uniform_real<double> uniform_dist;
      uniform_dist = boost::uniform_real<double>
        (-1 * params.perturb_init, params.perturb_init);
      foreach(double& v, w_fin)
        v = uniform_dist(rng);
      foreach(double& v, w_vec)
        v = uniform_dist(rng);
      b = uniform_dist(rng);
    }

    eta = params.eta;
    train_acc = 0;
    train_log_like = 0;

    switch(params.method) {
    case 0:
    case 1:
      for (size_t i = 0; i < ds.size(); ++i)
        total_train += ds.weight(i);
      break;
    case 2:
      break;
    default:
      assert(false);
    }

    // TODO: Change this once there are non-iterative methods.
    while(iteration_ < params.init_iterations)
      if (!(step()))
        break;
  }

  template <typename LA>
  bool logistic_regression<LA>::step_gradient_descent() {
    double prev_train_log_like = 0;
    for (size_t j(0); j < grad_fin.size(); ++j)
      grad_fin[j] = 0;
    for (size_t j(0); j < grad_vec.size(); ++j)
      grad_vec[j] = 0;
    grad_b = 0;
    train_acc = 0;
    train_log_like = 0;
    for (size_t i = 0; i < ds.size(); ++i) {
      const record_type& rec = ds[i];
      double v(confidence(rec));
      const std::vector<size_t>& findata = rec.finite();
      const vec& vecdata = rec.vector();
      double bin_label = (findata[label_index_] > 0 ? 1 : -1);
      train_acc += ((v > 0) ^ (bin_label == -1) ? ds.weight(i) : 0);
      train_log_like -= ds.weight(i) * std::log(1. + exp(-bin_label * v));
      if (train_log_like == -std::numeric_limits<double>::infinity())
        std::cerr << ""; // TODO: DEBUG THIS:
      /*
        e.g.,
        ./run_learner --learner batch_booster --train_data /Users/jbradley/data/uci/adult/adult-train.sum --test_data /Users/jbradley/data/uci/adult/adult-test.sum --weak_learner log_reg --learner_objective ada
      */
      v = ds.weight(i) * bin_label / (1. + exp(bin_label * v));
      // update gradient
      for (size_t j = 0; j < finite_indices.size(); ++j) {
        size_t val = findata[finite_indices[j]];
        grad_fin[finite_offset[j] + val] += v;
      }
      for (size_t j = 0; j < w_vec.size(); ++j)
        grad_vec[j] += vecdata[j] * v;
      grad_b += v;
    }
    train_acc /= total_train;
    train_log_like /= total_train;
    if (fabs(train_log_like - prev_train_log_like) < params.convergence) {
      if (DEBUG_LOGISTIC_REGRESSION)
        std::cerr << "logistic_regression converged: training log likelihood "
                  << "changed from " << prev_train_log_like << " to "
                  << train_log_like << "; exiting early (iteration "
                  << iteration_ << ")." << std::endl;
      return false;
    }
    prev_train_log_like = train_log_like;
    // update gradients for regularization
    switch(params.regularization) {
    case 1:
      std::cerr << "NOT IMPLEMENTED YET" << std::endl;
      assert(false);
      break;
    case 2:
      for (size_t j(0); j < grad_fin.size(); ++j)
        grad_fin[j] -= lambda * w_fin[j];
      for (size_t j(0); j < grad_vec.size(); ++j)
        grad_vec[j] -= lambda * w_vec[j];
      grad_b -= lambda * b;
      break;
    default:
      break;
    }
    // update weights
    for (size_t j(0); j < grad_fin.size(); ++j)
      w_fin[j] += eta * grad_fin[j];
    for (size_t j(0); j < grad_vec.size(); ++j)
      w_vec[j] += eta * grad_vec[j];
    b += eta * grad_b;
    eta *= params.mu;
    return true;
  } // end of function: bool step_gradient_descent()

  template <typename LA>
  bool logistic_regression<LA>::step_stochastic_gradient_descent() {
    if (!(o.next()))
      return false;
    const record_type& rec = o.current();
    double ex_weight(o.weight());
    total_train += ex_weight;
    double v(confidence(rec));
    const std::vector<size_t>& findata = rec.finite();
    const vec& vecdata = rec.vector();
    double bin_label = (findata[label_index_] > 0 ? 1 : -1);
    train_acc += ((v > 0) ^ (bin_label == -1) ? ex_weight : 0);
    train_log_like -= ex_weight * std::log(1. + exp(-bin_label * v));
    v = ex_weight * bin_label / (1. + exp(bin_label * v));
    // update weights
    for (size_t j(0); j < grad_fin.size(); ++j)
      grad_fin[j] = 0;
    for (size_t j = 0; j < finite_indices.size(); ++j) {
      size_t val(findata[finite_indices[j]]);
      grad_fin[finite_offset[j] + val] += v;
    }
    for (size_t j = 0; j < w_vec.size(); ++j)
      grad_vec[j] += vecdata[j] * v;
    grad_b += v;
    // update weights for regularization
    switch(params.regularization) {
    case 1:
      std::cerr << "NOT IMPLEMENTED YET" << std::endl;
      assert(false);
      break;
    case 2:
      for (size_t j(0); j < grad_fin.size(); ++j)
        grad_fin[j] -= lambda * w_fin[j];
      for (size_t j(0); j < grad_vec.size(); ++j)
        grad_vec[j] -= lambda * w_vec[j];
      grad_b -= lambda * b;
      break;
    default:
      break;
    }
    // update weights
    for (size_t j(0); j < grad_fin.size(); ++j)
      w_fin[j] += eta * grad_fin[j];
    for (size_t j(0); j < grad_vec.size(); ++j)
      w_vec[j] += eta * grad_vec[j];
    b += eta * grad_b;
    eta *= params.mu;
    ++iteration_;
    return true;
  } // end of function: bool step_stochastic_gradient_descent()

  // Getters and helpers
  //==========================================================================

  template <typename LA>
  bool logistic_regression<LA>::is_online() const {
    switch(params.method) {
    case 0:
    case 1:
      return false;
    case 2:
      return true;
    default:
      assert(false);
      return false;
    }
  }

  template <typename LA>
  double logistic_regression<LA>::train_accuracy() const {
    switch(params.method) {
    case 0:
    case 1:
      return train_acc;
    case 2:
      return (total_train == 0 ? -1 : train_acc / total_train);
    default:
      assert(false);
      return -1;
    }
  }

  // Prediction methods
  //==========================================================================

  template <typename LA>
  double logistic_regression<LA>::confidence(const record_type& example) const {
    double v(b);
    const std::vector<size_t>& findata = example.finite();
    for (size_t j(0); j < finite_indices.size(); ++j) {
      size_t val = findata[finite_indices[j]];
      v += w_fin[finite_offset[j] + val];
    }
    const vec& vecdata = example.vector();
    for (size_t j(0); j < w_vec.size(); ++j)
      v += w_vec[j] * vecdata[j];
    return v;
  }

  template <typename LA>
  double logistic_regression<LA>::confidence(const assignment& example) const {
    double v(b);
    const finite_assignment& fa = example.finite();
    for (size_t j(0); j < finite_indices.size(); ++j) {
      size_t val = safe_get(fa, finite_seq[finite_indices[j]]);
      v += w_fin[finite_offset[j] + val];
    }
    //      for (size_t j = 0; j < w_fin.size(); ++j)
    //        v += w_fin[j] * example[finite_seq[finite_indices[j]]];
    const vector_assignment& va = example.vector();
    for (size_t j(0); j < w_vec.size(); ++j) {
      const vec& vecdata = safe_get(va, vector_seq[j]);
      for (size_t j2(0); j2 < vector_seq[j]->size(); ++j2) {
        size_t ind(vector_offset[j] + j2);
        v += w_vec[ind] * vecdata[j2];
      }
    }
    return v;
  }

  // Methods for iterative learners
  //==========================================================================

  template <typename LA>
  bool logistic_regression<LA>::step() {
    switch(params.method) {
    case 0:
      return step_gradient_descent();
    case 1:
      std::cerr << "Newton's method not yet implemented." << std::endl;
      assert(false);
      return false;
    case 2:
      return step_stochastic_gradient_descent();
    default:
      assert(false);
      return false;
    }
  }

  // Save and load methods
  //==========================================================================

  template <typename LA>
  void logistic_regression<LA>::save(std::ofstream& out, size_t save_part,
                                 bool save_name) const {
    base::save(out, save_part, save_name);
    params.save(out);
    out << eta << " " << w_fin << " " << w_vec << " " << b
        << " " << train_acc << " " << train_log_like << " " << iteration_
        << " " << total_train << "\n";
  }

  template <typename LA>
  bool logistic_regression<LA>::load(std::ifstream& in, const datasource& ds, size_t load_part) {
    if (!(base::load(in, ds, load_part)))
      return false;
    finite_seq = ds.finite_list();
    vector_seq = ds.vector_list();
    finite_offset.clear();
    finite_indices.clear();
    if (ds.num_finite() > 1) {
      finite_offset.push_back(0);
      for (size_t j = 0; j < ds.num_finite(); ++j)
        if (j != label_index_) {
          finite_indices.push_back(j);
          finite_offset.push_back(finite_offset.back()+finite_seq[j]->size());
        }
      finite_offset.pop_back();
    }
    vector_offset.clear();
    if (ds.num_vector() > 0) {
      vector_offset.push_back(0);
      for (size_t j = 0; j < ds.num_vector() - 1; ++j)
        vector_offset.push_back(vector_offset.back()+vector_seq[j]->size());
    }
    params.load(in);
    std::string line;
    getline(in, line);
    std::istringstream is(line);
    if (!(is >> eta))
      assert(false);
    assert(eta > 0 && eta <= 1);
    read_vec(is, w_fin);
    read_vec(is, w_vec);
    if (!(is >> b))
      assert(false);
    if (!(is >> train_acc))
      assert(false);
    if (!(is >> train_log_like))
      assert(false);
    if (!(is >> iteration_))
      assert(false);
    if (!(is >> total_train))
      assert(false);
    return true;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_LOGISTIC_REGRESSION_HPP
