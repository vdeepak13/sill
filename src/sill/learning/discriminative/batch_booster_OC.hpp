
#ifndef SILL_LEARNING_DISCRIMINATIVE_BATCH_BOOSTER_OC_HPP
#define SILL_LEARNING_DISCRIMINATIVE_BATCH_BOOSTER_OC_HPP

#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/multiclass_booster_OC.hpp>

#include <sill/macros_def.hpp>

// Set to 1 to print debugging info
#define DEBUG_BATCH_BOOSTER_OC 0

namespace sill {

  /**
   * BATCH_BOOSTER_OC_PARAMETERS
   * From multiclass_booster_OC_parameters:
   *  - BINARY_LABEL
   *  - WEAK_LEARNER
   *  - SMOOTHING
   *  - RANDOM_SEED
   *  - INIT_ITERATIONS
   *  - CONVERGENCE_ZERO
   */
  struct batch_booster_OC_parameters : public multiclass_booster_OC_parameters {

    //! If > 0, train WL with this many examples
    //!  (default = 0)
    size_t resampling;

    //! If true, scale value for RESAMPLING by log(t)
    //! where t is the number of rounds
    //!  (default = false)
    bool scale_resampling;

    batch_booster_OC_parameters()
      : resampling(0), scale_resampling(false) { }

    void save(std::ofstream& out) const {
      multiclass_booster_OC_parameters::save(out);
      out << resampling << " " << (scale_resampling?"1":"0") << "\n";
    }

    void load(std::ifstream& in, const datasource& ds) {
      multiclass_booster_OC_parameters::load(in, ds);
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> resampling))
        assert(false);
      if (!(is >> scale_resampling))
        assert(false);
    }

  };  // struct batch_booster_OC_parameters

  /**
   * Batch boosting algorithm for multiclass labels which uses binary weak
   * learners and output coding.
   *
   * To create, for example, AdaBoost.OC with traditional decision trees,
   * construct:
   * batch_booster_OC<boosting::adaboost,
   *                  decision_tree<discriminative::objective_information> >
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @param Objective  optimization criterion defining the booster
   * @todo serialization
   */
  template <typename Objective>
  class batch_booster_OC : public multiclass_booster_OC<Objective> {

    // Protected data members
    //==========================================================================
  protected:

    typedef multiclass_booster_OC<Objective> base;
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
    using base::rng;
    using base::base_hypotheses;
    using base::colorings;
    using base::tmp_vector;

    using base::label;
    using base::choose_coloring;
    using base::end_step;

    batch_booster_OC_parameters params;

    //! Number of training examples
    size_t ntrain;

    //! For loading saved classifier without associated data or
    //! for learning from an oracle.
    dataset* ds_ptr;

    //! For loading saved classifier without associated data or
    //! for learning from an oracle.
    statistics* stats_ptr;

    //! Stats for dataset
    statistics& stats;

    //! Dataset (from stats)
    const dataset& ds;

    //! Normalized distribution over training examples
    //! distribution[i][l] = weight of example i for label l
    std::vector<std::vector<double> > distribution;

    //! Temporary distribution over examples, once coloring has been applied
    //! distribution[i] = weight of example i
    tree_sampler resampler;

    //! Temporary vector of size ntrain to hold weak learner predictions
    std::vector<size_t> tmp_wl_preds;

    // Protected methods
    //==========================================================================

    void init() {
      assert(params.binary_label != NULL);
      assert(params.weak_learner.get() != NULL);
      params.set_smoothing(ntrain, label_->size());
      smoothing = params.smoothing;
      rng.seed(static_cast<unsigned>(params.random_seed));

      typename tree_sampler::parameters tree_sampler_params;
      tree_sampler_params.random_seed =
        boost::uniform_int<int>(0,std::numeric_limits<int>::max())(rng);
      tree_sampler_params.allow_sampling = (params.resampling > 0);
      resampler = tree_sampler(ntrain, tree_sampler_params);
      distribution.resize(ntrain);
      double distribution_norm = 0;
      for (size_t i = 0; i < ntrain; ++i)
        distribution_norm += ds.weight(i);
      distribution_norm *= (nclasses_ - 1);
      for (size_t i = 0; i < ntrain; ++i) {
        distribution[i].resize(nclasses_);
        for (size_t j = 0; j < nclasses_; ++j)
          if (label(ds,i) != j)
            distribution[i][j] = ds.weight(i) / distribution_norm;
      }

      tmp_wl_preds.resize(ntrain);
      if (DEBUG_BATCH_BOOSTER_OC) {
        std::vector<double> label_distrib(nclasses_, 0);
        for (size_t i = 0; i < ntrain; ++i)
          label_distrib[label(ds,i)] += ds.weight(i);
        std::cerr << "batch_booster_OC::init(): distribution over labels = "
                  << label_distrib << std::endl;
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
    //  From multiclass_classifier:
    //   create()*
    //   confidences()
    //   probabilities()
    //  From iterative_learner:
    //   step()*
    //   reset_datasource()*
    //   train_accuracies()
    //  From booster:
    //   edges()
    //   norm_constants()

    // Constructors
    //==========================================================================

    /**
     * Constructor for a multiclass batch booster without associated data;
     * useful for:
     *  - creating other instances
     *  - loading a saved booster
     * @param params        algorithm parameters
     */
    explicit batch_booster_OC(batch_booster_OC_parameters params
                              = batch_booster_OC_parameters())
      : base(params), params(params), ntrain(0),
        ds_ptr(new vector_dataset()), stats_ptr(new statistics(*ds_ptr)),
        stats(*stats_ptr), ds(*ds_ptr) { }

    /**
     * Constructor for a multiclass batch booster.
     * @param stats         a statistics class for the training dataset
     * @param params        algorithm parameters
     */
    explicit batch_booster_OC(statistics& stats,
                              batch_booster_OC_parameters params
                              = batch_booster_OC_parameters())
      : base(stats.get_dataset(), params), params(params),
        ntrain(stats.get_dataset().size()), ds_ptr(NULL), stats_ptr(NULL),
        stats(stats), ds(stats.get_dataset()) {
      init();
      for (size_t t = 0; t < params.init_iterations; t++) {
        if (!(step()))
          break;
      }
    }

    /**
     * Constructor for a multiclass batch booster.
     * @param o    data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param params        algorithm parameters
     */
    batch_booster_OC(oracle& o, size_t n,
                     batch_booster_OC_parameters params
                     = batch_booster_OC_parameters())
      : base(o, params), params(params),
        ds_ptr(new vector_dataset(o.datasource_info())),
        stats_ptr(new statistics(*ds_ptr)), stats(*stats_ptr), ds(*ds_ptr) {
      for (size_t i = 0; i < n; ++i) {
        if (o.next())
          ds_ptr->insert(o.current().finite(), o.current().vector());
        else {
          std::cerr << "batch_booster called with an oracle with fewer than "
                    << "the max n=" << n << " examples in it." << std::endl;
          break;
        }
      }
      ntrain = ds_ptr->size();
      init();
      for (size_t t = 0; t < params.init_iterations; t++) {
        if (!(step()))
          break;
      }
    }

    ~batch_booster_OC() {
      if (stats_ptr != NULL)
        delete(stats_ptr);
      if (ds_ptr != NULL)
        delete(ds_ptr);
    }

    //! Train a new multiclass classifier of this type with the given data.
    boost::shared_ptr<multiclass_classifier> create(statistics& stats) const {
      boost::shared_ptr<multiclass_classifier>
        bptr(new batch_booster_OC<Objective>(stats, this->params));
      return bptr;
    }

    //! Train a new multiclass classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<multiclass_classifier> create(oracle& o, size_t n) const {
      boost::shared_ptr<multiclass_classifier>
        bptr(new batch_booster_OC<Objective>(o, n, this->params));
      return bptr;
    }

    // Getters and helpers
    //==========================================================================

    //! Return a name for the algorithm without template parameters.
    std::string name() const { return "batch_booster_OC"; }

    //! Return a name for the algorithm with comma-separated template parameters
    //! (e.g., objective).
    std::string fullname() const {
      return name() + "<" + Objective::name() + ">";
    }

    //! Returns true iff learner is naturally an online learner.
    bool is_online() const { return false; }

    //! Get the parameters of the algorithm
    batch_booster_OC_parameters& get_parameters() { return params; }

    // Learning and mutating operations
    //==========================================================================

    //! Resets the random seed in the learner's random number generator
    //! and parameters.
    void random_seed(double value) {
      params.random_seed = value;
      rng.seed(static_cast<unsigned>(value));
    }

    //! Run next iteration of boosting.
    //! @return  false iff the booster may not be trained further
    bool step() {
      // Choose coloring and build distribution over examples
      std::vector<size_t> coloring(choose_coloring());
      colorings.push_back(coloring);
      double total_w = 0;
      for (size_t i = 0; i < ntrain; ++i) {
        double w = 0;
        for (size_t j = 0; j < nclasses_; ++j)
          if (coloring[label(ds,i)] != coloring[j])
            w += distribution[i][j];
        resampler.set(i,w);
        total_w += w;
//        ex_distribution[i] = w;
      }
      if (total_w == 0)
        return false;
      resampler.commit_update();

      // Train weak learner
      size_t m_t((size_t)(params.resampling * std::log(exp(1.) +iteration_)));
      params.weak_learner->random_seed(uniform_prob(rng));
      dataset_view ds_view(ds);
      ds_view.set_binary_coloring(label_, params.binary_label, coloring);
      if (m_t > 0 && m_t < ds.size()) {
        // Use resampling
        std::vector<size_t> indices(m_t);
        for (size_t i = 0; i < m_t; ++i)
          indices[i] = resampler.sample();
        dataset_view ds_view2(ds_view);
        ds_view2.set_record_indices(indices);
        statistics stats_view(ds_view2);
        base_hypotheses.push_back(params.weak_learner->create(stats_view));
      } else {
        // Do not use resampling
        ds_view.set_weights(resampler.distribution());
        statistics stats_view(ds_view);
        base_hypotheses.push_back(params.weak_learner->create(stats_view));
      }

      // Compute edge and alpha
      double edge = 0;
      for (size_t i = 0; i < ntrain; ++i) {
        size_t pred = base_hypotheses.back()->predict(ds_view[i]);
        tmp_wl_preds[i] = pred;
        for (size_t j = 0; j < nclasses_; ++j)
          edge += distribution[i][j]
            * ((coloring[label(ds,i)] == pred ? 0 : 1) +
               (coloring[j] == pred ? 1 : 0));
      }
      edge = .5 * (1 - edge);
      // Deal with edge = -.5, 0, .5
      if (abs(edge) <= params.convergence_zero) {
        if (DEBUG_BATCH_BOOSTER_OC)
          std::cerr << "batch_booster_OC exited early because a base hypothesis"
                    << " had an edge of 0." << std::endl;
        base_hypotheses.pop_back();
        return false;
      } else if (abs(edge + .5) <= params.convergence_zero) {
        if (DEBUG_BATCH_BOOSTER_OC)
          std::cerr << "batch_booster_OC exited early because a base hypothesis"
                    << " had an edge of -.5." << std::endl;
        // TODO: We could get rid of all of the other base hypotheses,
        //       but it doesn't really matter.
        alphas.push_back(- discriminative::BIG_DOUBLE);
        end_step();
        return false;
      } else if (abs(edge - .5) <= params.convergence_zero) {
        if (DEBUG_BATCH_BOOSTER_OC)
          std::cerr << "batch_booster_OC exited early because a base hypothesis"
                    << " had an edge of .5." << std::endl;
        // TODO: We could get rid of all of the other base hypotheses,
        //       but it doesn't really matter.
        alphas.push_back(discriminative::BIG_DOUBLE);
        end_step();
        return false;
      }

      // Choose weight alpha for weak hypothesis
      double alpha = Objective::alpha(edge, params.smoothing);
      alphas.push_back(alpha);

      // Update distribution
      double distribution_norm = 0;
      for (size_t i = 0; i < ntrain; ++i)
        for (size_t j = 0; j < nclasses_; ++j)
          if (label(ds,i) != j) {
            distribution[i][j] = Objective::weight_update
              (distribution[i][j], alpha, 0,
               (coloring[label(ds,i)] == tmp_wl_preds[i] ? 0 : 1) +
               (coloring[j] == tmp_wl_preds[i] ? 1 : 0));
            distribution_norm += distribution[i][j];
          }
      for (size_t i = 0; i < ntrain; ++i)
        for (size_t j = 0; j < nclasses_; ++j)
          distribution[i][j] /= distribution_norm;

      end_step();
      if (DEBUG_BATCH_BOOSTER_OC)
        std::cerr << iteration_ << "\t" << edge << "\t" << alphas.back()
                  << std::endl;
      return true;
    }

    //! Resets the data source to be used in future rounds of training.
    //! @param  n   max number of examples which may be drawn from the oracle
    void reset_datasource(oracle& o, size_t n) {
      assert(false);
      // TODO: IMPLEMENT THIS
    }

    //! Resets the data source to be used in future rounds of training.
    void reset_datasource(statistics& stats) {
      assert(false);
      // TODO: IMPLEMENT THIS
    }

    // Save and load methods
    //==========================================================================

    using base::save;
    using base::load;

    //! Output the classifier to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    void save(std::ofstream& out, size_t save_part = 0,
              bool save_name = true) const {
      base::save(out, save_part, save_name);
      params.save(out);
      if (save_part == 1) {
        out << distribution.size() << " ";
        for (size_t i = 0; i < distribution.size(); ++i)
          out << distribution[i] << " ";
        out << "\n";
        resampler.save(out);
      }
    }

    /**
     * Input the classifier from a human-readable file.
     * @param in          input filestream for file holding the saved learner
     * @param ds          datasource used to get variables
     * @param load_part   0: load function (default), 1: engine, 2: shell
     * This assumes that the learner name has already been checked.
     * @return true if successful
     */
    bool load(std::ifstream& in, const datasource& ds, size_t load_part) {
      if (!(base::load(in, ds, load_part)))
        return false;
      params.load(in, ds);
      if (load_part == 1) {
        std::string line;
        getline(in, line);
        std::istringstream is(line);
        distribution.resize(ntrain);
        for (size_t i = 0; i < ntrain; ++i)
          read_vec(is, distribution[i]);
        getline(in, line);
        is.clear();
        is.str(line);
        resampler.load(in);
      }
      rng.seed(static_cast<unsigned>(params.random_seed));
      tmp_wl_preds.resize(ntrain);
      return true;
    }

  }; // class batch_booster_OC

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_BATCH_BOOSTER_OC_HPP
