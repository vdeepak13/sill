
#ifndef SILL_LEARNING_DISCRIMINATIVE_BINARY_BOOSTER_HPP
#define SILL_LEARNING_DISCRIMINATIVE_BINARY_BOOSTER_HPP

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>

//#include <sill/learning/dataset/concepts.hpp>
#include <sill/learning/dataset/dataset.hpp>
#include <sill/learning/dataset/ds_oracle.hpp>
#include <sill/learning/dataset/oracle.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/binary_classifier.hpp>
#include <sill/learning/discriminative/booster.hpp>
//#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/discriminative.hpp>
#include <sill/learning/discriminative/load_functions.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * BINARY BOOSTER PARAMETERS
   * From booster_parameters:
   *  - SMOOTHING
   *  - RANDOM_SEED
   *  - INIT_ITERATIONS
   *  - CONVERGENCE_ZERO
   */
  struct binary_booster_parameters : public booster_parameters {

    //! Specifies weak learner type
    //!  (required)
    boost::shared_ptr<binary_classifier<> > weak_learner;

    binary_booster_parameters() { }

    virtual bool valid() const {
      if (!booster_parameters::valid())
        return false;
      if (weak_learner.get() == NULL)
        return false;
      return true;
    }

    void save(std::ofstream& out) const {
      if (weak_learner.get() == NULL) {
        std::cerr << "Error: cannot save binary_booster which does not have a"
                  << " weak learner type." << std::endl;
        assert(false);
        return;
      }
      booster_parameters::save(out);
      weak_learner->save(out);
    }

    void load(std::ifstream& in, const datasource& ds) {
      booster_parameters::load(in);
      weak_learner = load_binary_classifier(in, ds);
    }

  };  // struct binary_booster_parameters

  /**
   * Binary batch boosting algorithm.
   *
   * @param Objective      optimization objective defining the booster
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo serialization
   */
  template <typename Objective>
  class binary_booster : public binary_classifier<>, public booster {

    // Public types
    //==========================================================================
  public:

    typedef binary_classifier<>::la_type la_type;
    typedef binary_classifier<>::record_type record_type;

    // Protected data members
    //==========================================================================
  protected:

    // Data from base class:
    //  finite_variable* label_
    //  size_t label_index_

    // These data members must be handled by child classes:
    //  - smoothing: This must be initialized by the children.
    //  - iteration_: This must be incremented by the children.
    //  - base_hypotheses, alphas: These must be added by the children.
    //  - timing: The initial value is set here, but further ones must be
    //            added by the children.

    //! Indicates if weak learner is confidence rated
    bool wl_confidence_rated;

    //! uniform distribution over [0, 1]
    boost::uniform_real<double> uniform_prob;

    //! Value for smoothing confidences or weights alpha
    double smoothing;

    //! Current iteration number (from 0); i.e., number of iterations completed.
    size_t iteration_;

    //! Base hypotheses
    std::vector<boost::shared_ptr<binary_classifier<> > > base_hypotheses;

    //! Weights for base hypotheses
    std::vector<double> alphas;

    //! Timing: timing[0] = time booster was initialized;
    //!         timing[i] = time i^th round (from 1) finished
    std::vector<double> timing;

    // Public methods
    //==========================================================================
  public:

    // Virtual methods from base classes (*means pure virtual):
    //  From learner:
    //   name()*
    //   is_online()*
    //   training_time()
    //   random_seed()
    //   save(), load()
    //  From classifier:
    //   is_confidence_rated()
    //   train_accuracy()
    //   set_confidences()
    //  From singlelabel_classifier:
    //   predict()*
    //  From binary_classifier:
    //   create()*
    //   confidence()
    //   predict_raw()
    //   probability()
    //  From iterative_learner:
    //   iteration()*
    //   elapsed_times()*
    //   step()*
    //   reset_datasource()*
    //   train_accuracies()
    //   test_accuracies()
    //  From booster:
    //   edges()*
    //   norm_constants()

    // Constructors
    //==========================================================================

    explicit binary_booster(const binary_booster_parameters& params)
      : binary_classifier<>(),
        uniform_prob(boost::uniform_real<double>(0,1)), smoothing(0),
        iteration_(0) {
      if (params.weak_learner.get() != NULL)
        wl_confidence_rated = params.weak_learner->is_confidence_rated();
      /*
      assert(params.weak_learner.get() != NULL);
      wl_confidence_rated = params.weak_learner->is_confidence_rated();
      */
      std::time_t time_tmp;
      time(&time_tmp);
      timing.push_back(time_tmp);
    }

    binary_booster(const datasource& ds,
                   const binary_booster_parameters& params)
      : binary_classifier<>(ds),
        uniform_prob(boost::uniform_real<double>(0,1)), smoothing(0),
        iteration_(0) {
      if (params.weak_learner.get() != NULL)
        wl_confidence_rated = params.weak_learner->is_confidence_rated();
      /*
      assert(params.weak_learner.get() != NULL);
      wl_confidence_rated = params.weak_learner->is_confidence_rated();
      */
      std::time_t time_tmp;
      time(&time_tmp);
      timing.push_back(time_tmp);
    }

    // Getters and helpers
    //==========================================================================

    //! Returns the current iteration number (from 0)
    //!  (i.e., the number of learning iterations completed).
    size_t iteration() const { return iteration_; }

    //! Returns the total training time (so far) for the learner.
    double training_time() const {
      if (timing.size() == 0)
        return 0;
      return (timing.back() - timing.front());
    }

    //! Returns the total time elapsed after each iteration.
    std::vector<double> elapsed_times() const {
      if (timing.size() == 0)
        return std::vector<double>();
      std::vector<double> times(timing.size()-1);
      for (size_t t = 0; t < times.size(); ++t) {
        times[t] = timing[t+1] - timing[0];
      }
      return times;
    }

    //! Returns the edge from each iteration.
    //! Note: This is virtual since not all booster objectives have
    //!       an inverse_alpha() function.
    virtual std::vector<double> edges() const {
      std::vector<double> edges_;
      for (size_t t = 0; t < alphas.size(); ++t)
        edges_.push_back(Objective::inverse_alpha(alphas[t], smoothing));
      return edges_;
    }

    //! Computes the accuracy after each iteration on a test set.
    std::vector<double> test_accuracies(const dataset<la_type>& testds) const {
      if (testds.size() == 0) {
        std::cerr << "binary_booster::test_accuracies() called with an empty"
                  << " dataset." << std::endl;
        assert(false);
        return std::vector<double>();
      }
      std::vector<double> test_acc(iteration_, 0);
      typename dataset<la_type>::record_iterator testds_end = testds.end();
      for (typename dataset<la_type>::record_iterator testds_it
             = testds.begin();
           testds_it != testds_end; ++testds_it) {
        double s = 0;
        const record_type& example = *testds_it;
        for (size_t t = 0; t < iteration_; t++) {
          if (wl_confidence_rated)
            s += base_hypotheses[t]->confidence(example);
          else
            s += alphas[t] * base_hypotheses[t]->confidence(example);
          if ((s > 0) ^ (label(example) == 0))
            test_acc[t]++;
        }
      }
      foreach(double& acc, test_acc)
        acc /= testds.size();
      return test_acc;
    }

    // Prediction methods
    //==========================================================================

    //! Predict the 0/1 label of a new example.
    std::size_t predict(const assignment& example) const {
      return (predict_raw(example) > 0 ? 1 : 0);
    }

    //! Predict the 0/1 label of a new example.
    std::size_t predict(const record_type& example) const {
      return (predict_raw(example) > 0 ? 1 : 0);
    }

    //! Returns predict_raw().
    double confidence(const assignment& example) const {
      return predict_raw(example);
    }

    //! Returns predict_raw().
    double confidence(const record_type& example) const {
      return predict_raw(example);
    }

    //! Returns the weighted sum of weak predictions.
    double predict_raw(const assignment& example) const {
      double pred = 0;
      if (wl_confidence_rated)
        for (size_t t = 0; t < iteration_; t++)
          pred += base_hypotheses[t]->confidence(example);
      else
        for (size_t t = 0; t < iteration_; t++)
          pred += alphas[t] * base_hypotheses[t]->confidence(example);
      return pred;
    }

    //! Returns the weighted sum of weak predictions.
    double predict_raw(const record_type& example) const {
      double pred = 0;
      if (wl_confidence_rated)
        for (size_t t = 0; t < iteration_; t++)
          pred += base_hypotheses[t]->confidence(example);
      else
        for (size_t t = 0; t < iteration_; t++)
          pred += alphas[t] * base_hypotheses[t]->confidence(example);
      return pred;
    }

    //! Predict the probability of the class variable having value +1.
    double probability(const record_type& example) const {
      return Objective::probability(label(example), predict_raw(example));
    }

    double probability(const assignment& example) const {
      return Objective::probability(label(example), predict_raw(example));
    }

    // Save and load methods
    //==========================================================================

    using binary_classifier<>::save;
    using binary_classifier<>::load;

    //! Output the learner to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    virtual void save(std::ofstream& out, size_t save_part = 0,
                      bool save_name = true) const {
      binary_classifier<>::save(out, save_part, save_name);
      out << (wl_confidence_rated ?"1":"0") << " " << smoothing << " "
          << iteration_ << " " << alphas << " " << timing << "\n";
      for (size_t t = 0; t < iteration_; ++t)
        base_hypotheses[t]->save(out);
    }

    /**
     * Input the learner from a human-readable file.
     * @param in          input filestream for file holding the saved learner
     * @param ds          datasource used to get variables
     * @param load_part   0: load function (default), 1: engine, 2: shell
     * This assumes that the learner name has already been checked.
     * @return true if successful
     */
    virtual bool
    load(std::ifstream& in, const datasource& ds, size_t load_part) {
      if (!(base::load(in, ds, load_part)))
        return false;
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> wl_confidence_rated))
        assert(false);
      if (!(is >> smoothing))
        assert(false);
      if (!(is >> iteration_))
        assert(false);
      read_vec(is, alphas);
      read_vec(is, timing);
      base_hypotheses.resize(iteration_);
      for (size_t t = 0; t < iteration_; ++t)
        base_hypotheses[t] = load_binary_classifier(in, ds);
      uniform_prob = boost::uniform_real<double>(0,1);
      return true;
    }

  }; // class binary_booster

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_BINARY_BOOSTER_HPP
