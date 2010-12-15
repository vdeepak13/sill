
#ifndef PRL_LEARNING_DISCRIMINATIVE_LOGISTIC_REGRESSION_HPP
#define PRL_LEARNING_DISCRIMINATIVE_LOGISTIC_REGRESSION_HPP

#include <algorithm>

#include <prl/functional.hpp>
#include <prl/learning/dataset/ds_oracle.hpp>
#include <prl/learning/dataset/vector_dataset.hpp>
#include <prl/learning/discriminative/binary_classifier.hpp>
#include <prl/learning/discriminative/free_functions.hpp>
#include <prl/stl_io.hpp>

#include <prl/macros_def.hpp>

// Set to true to print debugging information.
#define DEBUG_LOGISTIC_REGRESSION 0

namespace prl {

  // forward declarations
  class logistic_regression;
  template <typename Char, typename Traits>
  std::basic_ostream<Char,Traits>&
  operator<<(std::basic_ostream<Char, Traits>& out,
             const logistic_regression& lr);

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
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo serialization
   * @todo confidence-rated predictions, regularization, different optimization
   *       objectives
   */
  class logistic_regression : public binary_classifier {

    typedef binary_classifier base;

    // Protected data members
    //==========================================================================
  protected:

    // Data from base class:
    //  finite_variable* label_
    //  size_t label_index_

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
    vector_dataset* ds_ptr;

    //! Dataset
    const dataset& ds;

    //! Dataset oracle (for online mode when given a dataset)
    ds_oracle* ds_o_ptr;

    //! Data oracle
    oracle& o;

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
        iteration_(0), ds_ptr(new vector_dataset()), ds(*ds_ptr),
        ds_o_ptr(new ds_oracle(*ds_ptr)), o(*ds_o_ptr) { }

    /**
     * Constructor.
     * @param stats         a statistics class for the training dataset
     * @param parameters    algorithm parameters
     */
    explicit logistic_regression(statistics& stats,
                                 logistic_regression_parameters params
                                 = logistic_regression_parameters())
      : base(stats.get_dataset()), params(params),
        finite_seq(stats.get_dataset().finite_list()),
        vector_seq(stats.get_dataset().vector_list()), total_train(0),
        train_acc(-1), train_log_like(-std::numeric_limits<double>::max()),
        iteration_(0), ds_ptr(NULL), ds(stats.get_dataset()),
        ds_o_ptr(new ds_oracle(stats.get_dataset())), o(*ds_o_ptr) {
      init();
    }

    /**
     * Constructor.
     * @param o    data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param parameters    algorithm parameters
     */
    logistic_regression(oracle& o, size_t n,
                        logistic_regression_parameters params
                        = logistic_regression_parameters())
      : base(o), params(params),
        finite_seq(o.finite_list()), vector_seq(o.vector_list()),
        total_train(0), train_acc(-1),
        train_log_like(-std::numeric_limits<double>::max()), iteration_(0),
        ds_ptr(new vector_dataset(o.datasource_info())), ds(*ds_ptr),
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
    boost::shared_ptr<binary_classifier> create(statistics& stats) const {
      boost::shared_ptr<binary_classifier>
        bptr(new logistic_regression(stats, this->params));
      return bptr;
    }

    //! Train a new binary classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<binary_classifier> create(oracle& o, size_t n) const {
      boost::shared_ptr<binary_classifier>
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
    std::size_t predict(const record& example) const {
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
    double confidence(const record& example) const;

    //! Value indicating the confidence of the prediction, with
    //!  predict() == (confidence() > 0) ? 1 : 0.
    //! If the classifier does not have actual confidence ratings,
    //!  then this should be any value with the correct sign.
    double confidence(const assignment& example) const;

    //! Predict the probability of the class variable having value 1.
    double probability(const record& example) const {
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
    void reset_datasource(oracle& o, size_t n) { assert(false); }

    //! Resets the data source to be used in future rounds of training.
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    void reset_datasource(statistics& stats) { assert(false); }
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

  template <typename Char, typename Traits>
  std::basic_ostream<Char,Traits>&
  operator<<(std::basic_ostream<Char, Traits>& out,
             const logistic_regression& lr) {
    lr.write(out);
    return out;
  }

} // end of namespace: prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_LEARNING_DISCRIMINATIVE_LOGISTIC_REGRESSION_HPP
