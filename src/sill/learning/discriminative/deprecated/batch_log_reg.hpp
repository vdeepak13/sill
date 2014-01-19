
#ifndef SILL_LEARNING_DISCRIMINATIVE_BATCH_LOG_REG_HPP
#define SILL_LEARNING_DISCRIMINATIVE_BATCH_LOG_REG_HPP

#include <algorithm>

#include <sill/assignment.hpp>
#include <sill/datastructure/concepts.hpp>
#include <sill/stl_io.hpp>
#include <sill/functional.hpp>
#include <sill/learning/discriminative/binary_classifier.hpp>
#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/discriminative.hpp>

#include <sill/macros_def.hpp>

/**
 * \file batch_log_reg.hpp Batch Logistic Regression
 *                          using stochastic gradient descent
 */

namespace sill {

  // forward declarations
  class batch_log_reg;
  template <typename Char, typename Traits>
  std::basic_ostream<Char,Traits>&
  operator<<(std::basic_ostream<Char, Traits>& out, const batch_log_reg& lr);

  /**
   * Class for learning a logistic regression model using stochastic gradient
   * descent for parameter estimation.
   *  - Minimizes squared error.
   *  - Starts gradient descent with values near 0 (but perturbed a bit).
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo serialization
   * @todo confidence-rated predictions, regularization, different optimization
   *       objectives
   * @todo Finite variables should really be handled via indicators for each
   *       possible value; change this to do that.
   *    RIGHT HERE NOW: DEBUG THIS ON ADULT DATASET (PROBLEM WITH FINITE VARS?)
   *                    Change this to handle finite variables correctly,
   *                    then change this to use Newton's method.
   *
   * DEPRECATED
   */
  class batch_log_reg : public binary_classifier {

    //! Print debugging info
    static const bool debug = false;

    typedef binary_classifier base;

  public:
    //! Type of data source
    typedef dataset data_type;

    BOOST_STATIC_ASSERT(std::numeric_limits<double>::has_infinity);

  public:
    /**
     * PARAMETERS
     * - ETA (double): learning rate in (0,1]
     *    (default = .1)
     * - INIT_ITERATIONS (size_t): number of initial iterations to run;
     *    if using stochastic, then this is multiplied by the dataset size
     *    (default = 1000)
     * - MU (double): rate at which to decrease the learning rate
     *    (by multiplying ETA by MU each round).
     *    Note: Setting INIT_ITERATIONS causes this to be reset
     *    to the default, so this should be set after INIT_ITERATIONS.
     *    (default = exp(log(10^-4) / INIT_ITERATIONS),
     *            or .999 if INIT_ITERATIONS == 0)
     * - PERTURB_INIT (double): range [-PERTURB_INIT,PERTURB_INIT] within
     *    which to choose perturbed values for initial parameters
     *    (default = .001)
     * - LAMBDA (double): regularization parameter
     *    (default = 0)
     * - STOCHASTIC (bool): use stochastic gradient descent instead of
     *    a batch Newton's method
     *    (default = false)
     * - CONVERGENCE (double): amount of change in average log likelihood
     *    below which algorithm will consider itself converged.
     *    TODO: For stochastic gradient descent, this currently does nothing;
     *    make a new class for online stochastic gradient descent since it is
     *    not clear how to make the convergence measures equivalent.
     *    (default = .000001)
     * - RANDOM_SEED (double): used to make the algorithm deterministic
     *    (default = time)
     */
    class parameters {
    protected:
      double eta_;
      size_t init_iterations_;
      double mu_;
      double perturb_init_;
      double lambda_;
      bool stochastic_;
      double convergence_;
      double random_seed_;
      //! Reset mu_ to default
      void reset_mu() {
        if (init_iterations_ == 0)
          mu_ = .999;
        else
          mu_ = exp(-4. * log(10.) / init_iterations_);
      }
    public:
      parameters() {
        eta_ = .1;
        init_iterations_ = 1000;
        reset_mu();
        perturb_init_ = .001;
        lambda_ = 0;
        stochastic_ = false;
        convergence_ = .000001;
        std::time_t time_tmp;
        time(&time_tmp);
        random_seed_ = time_tmp;
      }
      parameters& eta(double value) {
        assert(value > 0 && value <= 1); eta_ = value; return *this;
      }
      parameters& init_iterations(size_t value) {
        init_iterations_ = value; reset_mu(); return *this;
      }
      parameters& mu(double value) {
        assert(value > 0 && value <= 1); mu_ = value; return *this;
      }
      parameters& perturb_init(double value) {
        assert(value >= 0); perturb_init_ = value; return *this;
      }
      parameters& lambda(double value) {
        assert(value >= 0); lambda_ = value; return *this;
      }
      parameters& stochastic(bool value) {
        stochastic_ = value; return *this;
      }
      parameters& convergence(double value) {
        assert(value >= 0); convergence_ = value; return *this;
      }
      parameters& random_seed(double value) {
        random_seed_ = value; return *this;
      }

      double eta() const { return eta_; }
      size_t init_iterations() const { return init_iterations_; }
      double mu() const { return mu_; }
      double perturb_init() const { return perturb_init_; }
      double lambda() const { return lambda_; }
      bool stochastic() const { return stochastic_; }
      double convergence() const { return convergence_; }
      double random_seed() const { return random_seed_; }

      void save(std::ofstream& out) const {
        out << eta_ << " " << init_iterations_ << " " << mu_ << " "
            << perturb_init_ << " " << lambda_ << " "
            << stochastic_ << " "
            << convergence_ << " " << random_seed_;
      }
      void load(std::istringstream& is) {
        if (!(is >> eta_))
          assert(false);
        if (!(is >> init_iterations_))
          assert(false);
        if (!(is >> mu_))
          assert(false);
        if (!(is >> perturb_init_))
          assert(false);
        if (!(is >> lambda_))
          assert(false);
        if (!(is >> stochastic_))
          assert(false);
        if (!(is >> convergence_))
          assert(false);
        if (!(is >> random_seed_))
          assert(false);
      }
    }; // class parameters

    /////////////////////// PRIVATE DATA AND METHODS ////////////////////

  private:
    parameters params;

    //! List of indices of non-class finite variables
    std::vector<size_t> finite_indices;

    //! finite_offset[j] = first index in w_fin for finite variable j,
    //!  as indexed in finite_indices
    //!  (so finite_offset[j] + k is the index for value k)
    std::vector<size_t> finite_offset;

    //! Current eta
    double eta;
    //! Weights w for finite part of x in predict(x) = sigma(w'x + b)
    std::vector<double> w_fin;
    //! Weights w for vector part of x in predict(x) = sigma(w'x + b)
    std::vector<double> w_vec;
    //! Offset b in predict(x) = sigma(w'x + b)
    double b;

    //! Training log likelihood
    double train_log_like;

    //! Initialize stuff
    void init(statistics& stats) {
      const dataset& ds = stats.dataset();
      base::init(ds, false);

      assert(ds.size() > 0);
      finite_offset.push_back(0);
      for (size_t j = 0; j < ds.num_finite(); ++j)
        if (j != class_variable_index) {
          finite_indices.push_back(j);
          finite_offset.push_back(finite_offset.back()+finite_seq[j]->size());
        }
      finite_offset.pop_back();

      w_fin.resize(ds.finite_dim() - class_variable->size());
      w_vec.resize(ds.vector_dim());
      b = 0;

      if (params.perturb_init() > 0) {
        boost::mt11213b rng(static_cast<unsigned>(params.random_seed()));
        boost::uniform_real<double> uniform_dist;
        uniform_dist = boost::uniform_real<double>
          (-1 * params.perturb_init(), params.perturb_init());
        foreach(double& v, w_fin)
          v = uniform_dist(rng);
        foreach(double& v, w_vec)
          v = uniform_dist(rng);
        b = uniform_dist(rng);
      }
    }

    //! build using stochastic gradient descent
    void build_stochastic(statistics& stats) {
      const dataset& ds = stats.dataset();
      eta = params.eta();
      double lambda = params.lambda() / ds.size();
      size_t i = 0; // index into dataset ds
      train_acc = 0;
      train_log_like = 0;
      for (size_t t = 0; t < params.init_iterations(); ++t) {
        double v = b;
        const record& rec = ds[i];
        const std::vector<size_t>& findata = rec.finite();
        for (size_t j = 0; j < finite_indices.size(); ++j) {
          size_t val = findata[finite_indices[j]];
          v += w_fin[finite_offset[j] + val];
        }
        const vec& vecdata = rec.vector();
        for (size_t j = 0; j < w_vec.size(); ++j)
          v += w_vec[j] * vecdata[j];
        double bin_label = (findata[class_variable_index] > 0 ? 1 : -1);
        v = ds.weight(i) * eta * bin_label / (1. + exp(bin_label * v));
        // update weights
        for (size_t j = 0; j < finite_indices.size(); ++j) {
          for (size_t k = 0; k < finite_seq[j]->size(); ++k)
            w_fin[finite_offset[j] + k] -=
              eta * lambda * w_fin[finite_offset[j] + k];
          size_t val = findata[finite_indices[j]];
          w_fin[finite_offset[j] + val] += val * v;
        }
        for (size_t j = 0; j < w_vec.size(); ++j)
          w_vec[j] += vecdata[j] * v
            - eta * lambda * w_vec[j];
        b += v;
        eta *= params.mu();
        ++i;
        if (i == ds.size())
          i = 0;
      }
    } // end of function: void build_stochastic()

    //! batch gradient descent
    //! @todo Newton's method instead
    void build_batch(statistics& stats) {
      const dataset& ds = stats.dataset();
      eta = params.eta();
      // TODO: have this build using a public step() function
      double total_train = 0;
      for (size_t i = 0; i < ds.size(); ++i)
        total_train += ds.weight(i);
      double prev_train_log_like = 0;
        //        = - std::numeric_limits<double>::infinity();
      for (size_t t = 0; t < params.init_iterations(); ++t) {
        std::vector<double> grad_fin(w_fin.size(),0);
        std::vector<double> grad_vec(w_vec.size(),0);
        double grad_b = 0;
        train_acc = 0;
        train_log_like = 0;
        for (size_t i = 0; i < ds.size(); ++i) {
          double v = b;
          const record& rec = ds[i];
          const std::vector<size_t>& findata = rec.finite();
          for (size_t j = 0; j < finite_indices.size(); ++j) {
            size_t val = findata[finite_indices[j]];
            v += w_fin[finite_offset[j] + val];
          }
          const vec& vecdata = rec.vector();
          for (size_t j = 0; j < w_vec.size(); ++j)
            v += w_vec[j] * vecdata[j];
          double bin_label = (findata[class_variable_index] > 0 ? 1 : -1);
          train_acc += ( (v > 0) ^ (bin_label == -1) ? ds.weight(i) : 0);
          train_log_like -= ds.weight(i) * log(1. + exp(-bin_label * v));
          if (train_log_like == -std::numeric_limits<double>::infinity())
            std::cerr << ""; // TODO: DEBUG THIS:
          /*
            e.g., ./run_learner --learner batch_booster --train_data
                         ../../../tests/data/uci/adult-train.sum
                         --test_data
                         ../../../tests/data/uci/adult-test.sum
                         --weak_learner log_reg --learner_objective
                         ada
           */
          v = ds.weight(i) * bin_label / (1. + exp(bin_label * v));
          // update gradient
          for (size_t j = 0; j < finite_indices.size(); ++j) {
            size_t val = findata[finite_indices[j]];
            grad_fin[finite_offset[j] + val] += val * v;
          }
          for (size_t j = 0; j < w_vec.size(); ++j)
            grad_vec[j] += vecdata[j] * v;
          grad_b += v;
        }
        train_acc /= total_train;
        train_log_like /= total_train;
        if (fabs(train_log_like - prev_train_log_like) < params.convergence()) {
          if (debug)
            std::cerr << "batch_log_reg converged: training log likelihood "
                      << "changed from " << prev_train_log_like << " to "
                      << train_log_like << "; exiting early (iteration "
                      << t << ")." << std::endl;
          return;
        }
        prev_train_log_like = train_log_like;
        // update weights
        for (size_t j = 0; j < finite_indices.size(); ++j) {
          for (size_t k = 0; k < finite_seq[j]->size(); ++k) {
            size_t ind = finite_offset[j] + k;
            w_fin[ind] +=
              eta * (grad_fin[ind] - params.lambda() * w_fin[ind]);
          }
        }
        for (size_t j = 0; j < w_vec.size(); ++j)
          w_vec[j] += eta * (grad_vec[j] - params.lambda() * w_vec[j]);
        b += eta * grad_b;
        eta *= params.mu();
      }
      if (debug)
        std::cerr << "batch_log_reg exited with last change in training log "
                  << "likelihood from " << prev_train_log_like << " to "
                  << train_log_like << std::endl;
    } // end of function: void build_batch()

    /////////// PUBLIC METHODS: BatchBinaryClassifier interface ////////////

  public:
    /**
     * Constructor which builds nothing but may be used to create other
     * instances.
     * @param parameters    algorithm parameters
     */
    batch_log_reg(parameters params = parameters())
      : params(params) {
    }
    /**
     * Constructor.
     * @param stats         a statistics class for the training dataset
     * @param parameters    algorithm parameters
     */
    batch_log_reg(statistics& stats, parameters params = parameters())
      : params(params) {
      init(stats);
      if (params.stochastic())
        build_stochastic(stats);
      else
        build_batch(stats);
    }
    BASE_CREATE_FUNCTIONS2(batch_log_reg);

    //! Reset the random seed in this algorithm's parameters and in its
    //! random number generator.
    void random_seed(double value) {
      params.random_seed(value);
    }

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
    double confidence(const record& example) const {
      double v = b;
      const std::vector<size_t>& findata = example.finite();
      for (size_t j = 0; j < finite_indices.size(); ++j) {
        size_t val = findata[finite_indices[j]];
        v += w_fin[finite_offset[j] + val];
      }
//      for (size_t j = 0; j < w_fin.size(); ++j)
//        v += w_fin[j] * findata[finite_indices[j]];
      const vec& vecdata = example.vector();
      for (size_t j = 0; j < w_vec.size(); ++j)
        v += w_vec[j] * vecdata[j];
      return v;
    }
    //! Value indicating the confidence of the prediction, with
    //!  predict() == (confidence() > 0) ? 1 : 0.
    //! If the classifier does not have actual confidence ratings,
    //!  then this should be any value with the correct sign.
    double confidence(const assignment& example) const {
      double v = b;
      const finite_assignment& fa = example.finite();
      for (size_t j = 0; j < finite_indices.size(); ++j) {
        size_t val = fa[finite_seq[finite_indices[j]]];
        v += w_fin[finite_offset[j] + val];
      }
//      for (size_t j = 0; j < w_fin.size(); ++j)
//        v += w_fin[j] * example[finite_seq[finite_indices[j]]];
      const vector_assignment& va = example.vector();
      for (size_t j = 0; j < w_vec.size(); ++j) {
        const vec& vecdata = va[vector_seq[j]];
        v += w_vec[j] * vecdata[0];
      }
      return v;
    }

    /////////////////// BinaryRegressor interface ///////////////////////
    //! Predict the probability of the class variable having value 1.
    double probability(const record& example) const {
      return 1. / (1. + exp(-1. * confidence(example)));
    }
    double probability(const assignment& example) const {
      return 1. / (1. + exp(-1. * confidence(example)));
    }

    operator std::string() const {
      std::ostringstream out; out << *this; return out.str();
    }

    //! Print classifier
    template <typename Char, typename Traits>
    void write(std::basic_ostream<Char, Traits>& out) const {
      out << "Logistic Regression";
      if (params.stochastic())
        out << " (stochastic):\n";
      else
        out << " (batch):\n";
      out << "  w (finite) =";
      foreach(double v, w_fin)
        out << " " << v;
      out << "\n  w (vector) =";
      foreach(double v, w_vec)
        out << " " << v;
      out << "\n  b = " << b << "\n"
          << " training accuracy = " << train_acc << "\n";
    }

    using base::save;
    //! Output the classifier to a human-readable file which can be reloaded.
    //! NOTE: This does not save the state of the random number generator.
    void save(std::ofstream& out) const {
      out << "batch_log_reg\n";
      params.save(out);
      out << " " << class_variable_index
          << " " << eta << " " << w_fin << " " << w_vec << " " << b
          << " " << train_log_like << " " << train_acc << "\n";
    }
    /**
     * Input the classifier from a human-readable file.
     * @param in   input filestream for file holding the saved classifier
     * @param ds   datasource used to get variables and variable orderings
     */
    void load(std::ifstream& in, const datasource& ds) {
      finite_seq = ds.finite_list();
      vector_seq = ds.vector_list();
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      params.load(is);
      if (!(is >> class_variable_index))
        assert(false);
      assert(class_variable_index < ds.num_finite());
      class_variable = finite_seq[class_variable_index];
      if (!(is >> eta))
        assert(false);
      assert(eta > 0 && eta <= 1);
      read_vec(is, w_fin);
      read_vec(is, w_vec);
      if (!(is >> b))
        assert(false);
      if (!(is >> train_log_like))
        assert(false);
      if (!(is >> train_acc))
        assert(false);
      confidence_rated_ = false;
    }

    /**
     * Input the classifier from a human-readable file.
     * @param filename  file holding the saved classifier
     * @param ds        datasource used to get variables and variable orderings
     */
    void load(const std::string& filename, const datasource& ds) {
      std::ifstream in(filename.c_str(), std::ios::in);
      std::string line;
      getline(in, line);
      assert(line.compare("batch_log_reg") == 0);
      load(in, ds);
      in.close();
    }

    //! Return a name for the algorithm.
    std::string name() const {
      return "batch_log_reg";
    }

  }; // class batch_log_reg

  template <typename Char, typename Traits>
  std::basic_ostream<Char,Traits>&
  operator<<(std::basic_ostream<Char, Traits>& out, const batch_log_reg& lr) {
    lr.write(out);
    return out;
  }

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_BATCH_LOG_REG_HPP
