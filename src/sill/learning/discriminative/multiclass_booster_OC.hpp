
#ifndef SILL_LEARNING_DISCRIMINATIVE_MULTICLASS_BOOSTER_OC_HPP
#define SILL_LEARNING_DISCRIMINATIVE_MULTICLASS_BOOSTER_OC_HPP

#include <sill/learning/discriminative/load_functions.hpp>
#include <sill/learning/discriminative/multiclass_booster.hpp>
#include <sill/math/permutations.hpp>
#include <sill/math/statistics.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * MULTICLASS BOOSTER OC PARAMETERS
   * From booster_parameters:
   *  - SMOOTHING
   *  - RANDOM_SEED
   *  - INIT_ITERATIONS
   *  - CONVERGENCE_ZERO
   */
  struct multiclass_booster_OC_parameters : public booster_parameters {

    //! New variable used to create binary views of the class variable
    //!  (required)
    finite_variable* binary_label;

    //! Specifies weak learner type
    //!  (required)
    boost::shared_ptr<binary_classifier<> > weak_learner;

    multiclass_booster_OC_parameters() : binary_label(NULL) { }

    virtual bool valid() const {
      if (!booster_parameters::valid())
        return false;
      if (binary_label == NULL)
        return false;
      if (weak_learner.get() == NULL)
        return false;
      return true;
    }

    void save(std::ofstream& out) const {
      if (weak_learner.get() == NULL) {
        std::cerr << "Error: cannot save multiclass_booster_OC which does not"
                  << " have a weak learner type." << std::endl;
        assert(false);
        return;
      }
      booster_parameters::save(out);
      weak_learner->save(out);
    }

    //! Note: This cannot load the binary_label parameter.  It must be set
    //!       afterwards by the user.
    void load(std::ifstream& in, const datasource& ds) {
      booster_parameters::load(in);
      weak_learner = load_binary_classifier(in, ds);
    }

  };  // struct multiclass_booster_OC_parameters

  /**
   * Multiclass boosting algorithm interface for Output Coding (.OC) boosters.
   *
   * @param Objective      optimization objective defining the booster
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo serialization
   * @todo In boosting on MNIST, Freund and Schapire report that
   *       it is often a good idea to use the same WL in multiple rounds
   *       (but with a different coloring, where the coloring is similar to
   *       that used by *.OC boosters).  I should look into this.
   *       They are talking about this w.r.t. the proto_knn learner,
   *       but perhaps it would apply to others as well.
   */
  template <typename Objective>
  class multiclass_booster_OC : public multiclass_booster<Objective> {

    // Public types
    //==========================================================================
  public:

    typedef multiclass_booster<Objective> base;

    typedef typename base::la_type la_type;
    typedef typename base::record_type record_type;

    // Protected data members
    //==========================================================================
  protected:

    // These data members must be handled by child classes:
    //  - rng: This must be seeded, saved, and loaded by the children.
    //  - smoothing: This must be initialized by the children.
    //  - iteration_: This must be incremented by the children (by calling
    //                end_step()).
    //  - alphas: These must be added by the children.
    //  - timing: The initial value is set here, but further ones must be
    //            added by the children.
    //  - base_hypotheses, colorings: These must be added by the children.

    // Data from base class:
    //  finite_variable* label_
    //  size_t label_index_
    //  size_t nclasses_
    //  bool wl_confidence_rated
    //  boost::uniform_real<double> uniform_prob
    //  boost::bernoulli_distribution<double> bernoulli_dist
    //  double smoothing
    //  size_t iteration_
    //  std::vector<double> alphas
    //  std::vector<double> timing

    using base::label_;
    using base::label_index_;
    using base::nclasses_;
    using base::wl_confidence_rated;
    using base::uniform_prob;
    using base::bernoulli_dist;
    using base::smoothing;
    using base::iteration_;
    using base::alphas;
    using base::timing;
    using base::label;

    /*
    // Import stuff from multiclass_classifier
    using base::label_;
    using base::label_index_;
    using base::finite_seq;
    using base::vector_seq;
    using base::train_acc;
    // Import stuff from multiclass_booster
    using base::random_seed_;
    using base::rng;
    using base::uniform_prob;
    using base::bernoulli_dist;
    using base::smoothing;
    using base::iter;
    using base::alphas;
    using base::timing;
    using base::label;
    */

    //! random number generator
    mutable boost::mt11213b rng;

    //! Base hypotheses
    std::vector<boost::shared_ptr<binary_classifier<> > > base_hypotheses;
    //! Colorings for labels: colorings[t][l] = 0/1 for label l on round t
    std::vector<std::vector<size_t> > colorings;

    //! Temp vector of size nclasses_ to avoid reallocation during predictions
    mutable vec tmp_vector;

    // Protected methods
    //==========================================================================

    //! Choose 0/1 coloring uniformly at random from colorings with floor(k/2)
    //! of the labels set to 0 (as Schapire (1997) did).
    std::vector<size_t> choose_coloring() {
      std::vector<size_t> perm(randperm(nclasses_,rng));
      std::vector<size_t> coloring(nclasses_,0);
      for (size_t j = 0; j < (size_t)ceil(nclasses_/2.); ++j)
        coloring[perm[j]] = 1;
      return coloring;
    }

    //! Set given vector (which must already be of size nclasses_) to
    //! the raw prediction.
    void my_predict_raws(const assignment& example,
                         vec& pred) const {
      for (size_t l = 0; l < nclasses_; ++l)
        pred[l] = 0;
      for (size_t t = 0; t < iteration_; ++t) {
        const std::vector<size_t>& coloring = colorings[t];
        size_t wl = base_hypotheses[t]->predict(example);
        for (size_t l = 0; l < nclasses_; ++l)
          if (wl == coloring[l])
            pred[l] += alphas[t];
      }
    }

    void my_predict_raws(const record_type& example,
                         vec& pred) const {
      for (size_t l = 0; l < nclasses_; ++l)
        pred[l] = 0;
      for (size_t t = 0; t < iteration_; ++t) {
        const std::vector<size_t>& coloring = colorings[t];
        size_t wl = base_hypotheses[t]->predict(example);
        for (size_t l = 0; l < nclasses_; ++l)
          if (wl == coloring[l])
            pred[l] += alphas[t];
      }
    }

    // Public methods
    //==========================================================================
  public:

    // Virtual methods from base classes (*means pure virtual):
    //  From learner:
    //   name()*
    //   is_online()*
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
    //   step()*
    //   reset_datasource()*
    //   train_accuracies()
    //   test_accuracies()
    //  From booster:
    //   edges()
    //   norm_constants()

    // Constructors and destructors
    //==========================================================================

    explicit
    multiclass_booster_OC(const multiclass_booster_OC_parameters& params)
      : base(params.weak_learner.get()) { }

    multiclass_booster_OC(const datasource& ds,
                          const multiclass_booster_OC_parameters& params)
      : base(ds, params.weak_learner.get()) {
      tmp_vector.resize(nclasses_);
    }

    // Prediction methods
    //==========================================================================

    //! Predict the label of a new example.
    std::size_t predict(const assignment& example) const {
      my_predict_raws(example, tmp_vector);
      return max_index(tmp_vector, rng);
    }

    //! Predict the label of a new example.
    std::size_t predict(const record_type& example) const {
      my_predict_raws(example, tmp_vector);
      return max_index(tmp_vector, rng);
    }

    //! Returns a prediction whose value indicates the label
    //! (like confidence()) but whose magnitude may differ.
    //! For boosters, this is the weighted sum of base predictions.
    //! If this is not implemented, then it returns confidence(example).
    vec predict_raws(const record_type& example) const {
      my_predict_raws(example, tmp_vector);
      return tmp_vector;
    }

    vec predict_raws(const assignment& example) const {
      my_predict_raws(example, tmp_vector);
      return tmp_vector;
    }

    //! Compute accuracy for each iteration on a test set.
    //! If ntest is not given, this assumes the oracle has a limit.
    std::vector<double>
    test_accuracies(oracle<la_type>& o,
                    size_t ntest = std::numeric_limits<size_t>::max()) const {
      std::vector<double> test_acc(iteration_, 0);
      size_t n = 0;
      while (o.next() && n < ntest) {
        for (size_t l = 0; l < nclasses_; ++l)
          tmp_vector[l] = 0;
        for (size_t t = 0; t < iteration_; ++t) {
          size_t wl = base_hypotheses[t]->predict(o.current());
          for (size_t l = 0; l < nclasses_; ++l)
            if (wl == colorings[t][l])
              tmp_vector[l] += alphas[t];
          if (max_index(tmp_vector,rng) == label(o.current()))
            test_acc[t] += 1;
        }
        ++n;
      }
      if (n > 0)
        for (size_t t = 0; t < iteration_; ++t)
          test_acc[t] /= n;

      return test_acc;
    }

    //! Compute accuracy for each iteration on a test set:
    std::vector<double>
    test_accuracies(const dataset<la_type>& testds) const {
      if (testds.size() == 0)
        return std::vector<double>();

      std::vector<double> test_acc(iteration_, 0);
      for (size_t i = 0; i < testds.size(); ++i) {
        for (size_t l = 0; l < nclasses_; ++l)
          tmp_vector[l] = 0;
        for (size_t t = 0; t < iteration_; ++t) {
          size_t wl = base_hypotheses[t]->predict(testds[i]);
          for (size_t l = 0; l < nclasses_; ++l)
            if (wl == colorings[t][l])
              tmp_vector[l] += alphas[t];
          if (max_index(tmp_vector,rng) == label(testds,i))
            test_acc[t] += 1;
        }
      }
      for (size_t t = 0; t < iteration_; ++t)
        test_acc[t] /= testds.size();

      return test_acc;
    }

    // Save and load methods
    //==========================================================================

    using base::save;
    using base::load;

    //! Output the learner to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    virtual void save(std::ofstream& out, size_t save_part = 0,
                      bool save_name = true) const {
      base::save(out, save_part, save_name);
      for (size_t t = 0; t < iteration_; ++t)
        base_hypotheses[t]->save(out);
      for (size_t t = 0; t < iteration_; ++t)
        out << colorings[t] << " ";
      out << "\n";
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
      base_hypotheses.resize(iteration_);
      for (size_t t = 0; t < iteration_; ++t)
        base_hypotheses[t] = load_binary_classifier(in, ds);
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      colorings.resize(iteration_);
      for (size_t t = 0; t < iteration_; ++t)
        read_vec(is, colorings[t]);
      tmp_vector.resize(nclasses_);
      return true;
    }

  }; // class multiclass_booster_OC

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_MULTICLASS_BOOSTER_OC_HPP
