
#ifndef SILL_LEARNING_DISCRIMINATIVE_MULTICLASS_BOOSTER_HPP
#define SILL_LEARNING_DISCRIMINATIVE_MULTICLASS_BOOSTER_HPP

#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

//#include <sill/learning/dataset/concepts.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/discriminative/binary_classifier.hpp>
#include <sill/learning/discriminative/booster.hpp>
//#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/discriminative.hpp>
#include <sill/learning/discriminative/multiclass_classifier.hpp>
#include <sill/learning/discriminative/tree_sampler.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Multiclass boosting algorithm interface.
   *
   * @param Objective      optimization objective defining the booster
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo serialization
   */
  template <typename Objective>
  class multiclass_booster : public multiclass_classifier<>, public booster {

    // Public types
    //==========================================================================
  public:

    typedef multiclass_classifier<>::la_type la_type;
    typedef multiclass_classifier<>::record_type record_type;

    // Protected data members
    //==========================================================================
  protected:

    // These data members must be handled by child classes:
    //  - smoothing: This must be initialized by the children.
    //  - iteration_: This must be incremented by the children.
    //  - alphas: These must be added by the children.
    //  - timing: The initial value is set here, but further ones must be
    //            added by the children.

    // Data from base class:
    //  finite_variable* label_
    //  size_t label_index_

    //! Number of classes
    size_t nclasses_;

    //! Indicates if weak learner is confidence rated
    bool wl_confidence_rated;

    //! Random seed used to initialize the random number generator
    //    double random_seed_;
    //! random number generator
    //    boost::mt11213b rng;
    //! uniform distribution over [0, 1]
    boost::uniform_real<double> uniform_prob;

    //! uniform distribution over {true, false}
    boost::bernoulli_distribution<double> bernoulli_dist;

    //! Value for smoothing confidences or weights alpha
    double smoothing;

    //! Current iteration number (from 0); i.e., number of iterations completed.
    size_t iteration_;

    //! Weights for base hypotheses
    std::vector<double> alphas;

    //! Timing: timing[0] = time booster was initialized;
    //!         timing[i] = time i^th round (from 1) finished
    std::vector<double> timing;

    // Protected methods
    //==========================================================================

    //! Called at the end of a successful step().
    void end_step() {
      ++iteration_;
      std::time_t time_tmp;
      time(&time_tmp);
      timing.push_back(time_tmp);
    }

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
    //  From multiclass_classifier:
    //   create()*
    //   confidences()
    //   predict_raws()
    //   probabilities()
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

    // Constructors and destructors
    //==========================================================================

    explicit multiclass_booster(const classifier<>* wl_ptr)
      : multiclass_classifier<>(), nclasses_(nclasses()),
        uniform_prob(boost::uniform_real<double>(0,1)),
        bernoulli_dist(boost::bernoulli_distribution<double>(.5)),
        smoothing(0), iteration_(0) {
      assert(wl_ptr != NULL);
      wl_confidence_rated = wl_ptr->is_confidence_rated();
      assert(nclasses_ > 1);
      std::time_t time_tmp;
      time(&time_tmp);
      timing.push_back(time_tmp);
    }

    multiclass_booster(const datasource& ds, const classifier<>* wl_ptr)
      : multiclass_classifier<>(ds), nclasses_(nclasses()),
        uniform_prob(boost::uniform_real<double>(0,1)),
        bernoulli_dist(boost::bernoulli_distribution<double>(.5)),
        smoothing(0), iteration_(0) {
      assert(wl_ptr != NULL);
      wl_confidence_rated = wl_ptr->is_confidence_rated();
      assert(nclasses_ > 1);
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

    // Save and load methods
    //==========================================================================

    using multiclass_classifier<>::save;
    using multiclass_classifier<>::load;

    //! Output the learner to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    virtual void save(std::ofstream& out, size_t save_part = 0,
                      bool save_name = true) const {
      multiclass_classifier<>::save(out, save_part, save_name);
      out << (wl_confidence_rated ?"1":"0") << " " << smoothing << " "
          << iteration_ << " " << alphas << " " << timing << "\n";
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
      if (!(multiclass_classifier<>::load(in, ds, load_part)))
        return false;
      nclasses_ = nclasses();
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
      uniform_prob = boost::uniform_real<double>(0,1);
      bernoulli_dist = boost::bernoulli_distribution<double>(.5);
      return true;
    }

  }; // class multiclass_booster

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_MULTICLASS_BOOSTER_HPP
