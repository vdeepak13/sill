
#ifndef SILL_LEARNING_DISCRIMINATIVE_BATCH_BOOSTER_HPP
#define SILL_LEARNING_DISCRIMINATIVE_BATCH_BOOSTER_HPP

#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/discriminative/binary_booster.hpp>
#include <sill/learning/discriminative/tree_sampler.hpp>

#include <sill/macros_def.hpp>

// Set to 1 to print out debugging information
#define BATCH_BOOSTER_DEBUG 0

namespace sill {

  /**
   * BATCH BOOSTER PARAMETERS
   * From binary_booster_parameters:
   *  - WEAK_LEARNER
   *  - SMOOTHING
   *  - RANDOM_SEED
   *  - INIT_ITERATIONS
   *  - CONVERGENCE_ZERO
   */
  struct batch_booster_parameters : public binary_booster_parameters {

    //! If > 0, train WL with this many examples
    //!  (default = 0)
    size_t resampling;

    //! If true, scale value for RESAMPLING by log(t)
    //! where t is the number of rounds
    //!  (default = false)
    bool scale_resampling;

    batch_booster_parameters()
      : resampling(0), scale_resampling(false) { }

    virtual bool valid() const {
      if (!binary_booster_parameters::valid())
        return false;
      if (smoothing < 0)
        return false;
      if (convergence_zero < 0)
        return false;
      return true;
    }

    void set_smoothing(size_t ntrain, size_t nlabels) {
      if (smoothing == -1)
        smoothing = 1. / (2. * ntrain * nlabels);
    }

    void save(std::ofstream& out) const {
      binary_booster_parameters::save(out);
      out << resampling << " " << (scale_resampling ? "1" : "0") << "\n";
    }

    void load(std::ifstream& in, const datasource& ds) {
      binary_booster_parameters::load(in, ds);
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> resampling))
        assert(false);
      if (!(is >> scale_resampling))
        assert(false);
    }

  };  // struct batch_booster_parameters

  /**
   * Batch boosting algorithm for binary labels.
   *
   * To create, for example, batch AdaBoost with traditional decision trees,
   * construct:
   * batch_booster<boosting::adaboost<>,
   *               decision_tree<discriminative::objective_information> >
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @param Objective      optimization objective defining the booster
   * @todo serialization
   * @todo Fix confidence-rated predictions.
   */
  template <typename Objective>
  class batch_booster : public binary_booster<Objective> {

    // Public types
    //==========================================================================
  public:

    typedef binary_booster<Objective> base;

    typedef typename base::la_type la_type;
    typedef typename base::record_type record_type;

    // Protected data members
    //==========================================================================
  protected:

    // Data from base class:
    //  finite_variable* label_
    //  size_t label_index_
    //  size_t iteration_
    //  bool wl_confidence_rated
    //  std::vector<double> timing
    //  boost::shared_ptr<binary_classifier> wl_ptr
    //  boost::uniform_real<double> uniform_prob
    //  double smoothing
    //  std::vector<boost::shared_ptr<binary_classifier> > base_hypotheses
    //  std::vector<double> alphas

    // Import stuff from multiclass_classifier
    using base::label_;
    using base::label_index_;
    using base::label;
    using base::predict_raw;

    // Import stuff from binary_classifier
    using base::confidence;

    // Import stuff from binary_booster
    using base::iteration_;
    using base::wl_confidence_rated;
    using base::timing;
    using base::uniform_prob;
    using base::smoothing;
    using base::base_hypotheses;
    using base::alphas;

    batch_booster_parameters params;

    //! random number generator
    boost::mt11213b rng;

    //! For loading saved classifier without associated data or
    //! for learning from an oracle.
    dataset<la_type>* ds_ptr;

    //! For loading saved classifier without associated data or
    //! for learning from an oracle.
    dataset_statistics<la_type>* stats_ptr;

    //! Stats for dataset
    dataset_statistics<la_type>& stats;

    //! Dataset (from stats)
    const dataset<la_type>& ds;

    //! Normalized distribution over training examples
    tree_sampler resampler;

    // Protected methods
    //==========================================================================

    void init() {
      params.set_smoothing(ds.size(), label_->size());
      assert(params.valid());
      smoothing = params.smoothing;
      rng.seed(static_cast<unsigned>(params.random_seed));

      tree_sampler::parameters tree_sampler_params;
      tree_sampler_params.random_seed =
        boost::uniform_int<int>(0,std::numeric_limits<int>::max())(rng);
      tree_sampler_params.allow_sampling = (params.resampling > 0);
      resampler = tree_sampler(ds.size(), tree_sampler_params);
      for (size_t i = 0; i < ds.size(); i ++)
        resampler.set(i,ds.weight(i));
      resampler.commit_update();
    }

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
    //   save(), load()
    //   random_seed()
    //  From classifier:
    //   is_confidence_rated()
    //   train_accuracy()
    //   set_confidences()
    //  From binary_classifier:
    //   create()*
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
     * Constructor for a binary batch booster without associated data; useful
     * for:
     *  - creating other instances
     *  - loading a saved booster
     * @param params     algorithm parameters
     */
    explicit batch_booster(batch_booster_parameters params
                           = batch_booster_parameters())
      : base(params), params(params), ds_ptr(new vector_dataset<la_type>()),
        stats_ptr(new dataset_statistics<la_type>(*ds_ptr)), stats(*stats_ptr), ds(*ds_ptr) { }

    /**
     * Constructor for a binary batch booster.
     * @param stats         a statistics class for the training dataset
     * @param params        algorithm parameters
     */
    explicit batch_booster(dataset_statistics<la_type>& stats,
                           batch_booster_parameters params
                           = batch_booster_parameters())
      : base(stats.get_dataset(), params),
        params(params), ds_ptr(NULL), stats_ptr(NULL),
        stats(stats), ds(stats.get_dataset()) {
      init();
      for (size_t t = 0; t < params.init_iterations; t++)
        if (!(step()))
          break;
    }

    /**
     * Constructor for a binary batch booster.
     * @param o    data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param params        algorithm parameters
     */
    batch_booster(oracle<la_type>& o, size_t n,
                  batch_booster_parameters params = batch_booster_parameters())
      : base(o, params),
        params(params), ds_ptr(new vector_dataset<la_type>(o.datasource_info())),
        stats_ptr(new dataset_statistics<la_type>(*ds_ptr)), stats(*stats_ptr), ds(*ds_ptr) {
      for (size_t i = 0; i < n; ++i) {
        if (o.next())
          ds_ptr->insert(o.current().finite(), o.current().vector());
        else {
          std::cerr << "batch_booster called with an oracle with fewer than "
                    << "the max n=" << n << " examples in it." << std::endl;
          break;
        }
      }
      init();
      for (size_t t = 0; t < params.init_iterations; t++)
        if (!(step()))
          break;
    }

    ~batch_booster() {
      if (stats_ptr != NULL)
        delete(stats_ptr);
      if (ds_ptr != NULL)
        delete(ds_ptr);
    }

    //! Train a new binary classifier of this type with the given data.
    boost::shared_ptr<binary_classifier<> > create(dataset_statistics<la_type>& stats) const {
      boost::shared_ptr<binary_classifier<> >
        bptr(new batch_booster<Objective>(stats, this->params));
      return bptr;
    }

    //! Train a new binary classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<binary_classifier<> > create(oracle<la_type>& o, size_t n) const {
      boost::shared_ptr<binary_classifier<> >
        bptr(new batch_booster<Objective>(o, n, this->params));
      return bptr;
    }

    // Getters and helpers
    //==========================================================================

    //! Return a name for the algorithm without template parameters.
    std::string name() const { return "batch_booster"; }

    //! Return a name for the algorithm with comma-separated template parameters
    //! (e.g., objective).
    std::string fullname() const {
      return name() + "<" + Objective::name() + ">";
    }

    //! Returns true iff learner is naturally an online learner.
    bool is_online() const { return false; }

    // Learning and mutating operations
    //==========================================================================

    //! Resets the random seed in the learner's random number generator
    //! and parameters.
    void random_seed(double value) {
      params.random_seed = value;
      rng.seed(static_cast<unsigned>(params.random_seed));
    }

    //! Run next iteration of boosting.
    //! @return  false iff the booster may not be trained further
    bool step() {
      // Train weak learner
      size_t m_t((size_t)(params.resampling * std::log(exp(1.) +iteration_)));
      double edge;
      params.weak_learner->random_seed
        (boost::uniform_int<int>(0, std::numeric_limits<int>::max())(rng));
      const vec& distribution = resampler.distribution();
      if (BATCH_BOOSTER_DEBUG) {
        double min_dist = 1;
        for (size_t i(0); i < distribution.size(); ++i) {
          if (min_dist > distribution[i])
            min_dist = distribution[i];
        }
        std::cerr << " min value in distribution = " << min_dist << std::endl;
        resampler.check_validity();
      }
      typename dataset<la_type>::record_iterator_type ds_end = ds.end();
      if (m_t > 0 && m_t < ds.size()) {
        // Use resampling
        std::vector<size_t> indices(m_t);
        for (size_t i = 0; i < m_t; ++i)
          indices[i] = resampler.sample();
        dataset_view<la_type> ds_view(ds);
        ds_view.set_record_indices(indices);
        if (BATCH_BOOSTER_DEBUG) {
          double zeros = 0;
          double ones = 0;
          for (size_t i = 0; i < ds_view.size(); ++i)
            if (ds_view.finite(i,label_index_) == 0)
              zeros += 1;
            else
              ones += 1;
          zeros /= ds_view.size();
          ones /= ds_view.size();
          std::cerr << "WL training set class distribution: (" << zeros << ", "
                    << ones << ")" << std::endl;
          zeros = 0;
          ones = 0;
          for (size_t i = 0; i < ds.size(); ++i)
            if (ds.finite(i,label_index_) == 0)
              zeros += distribution[i];
            else
              ones += distribution[i];
          std::cerr << "True training set class distribution: (" << zeros
                    << ", " << ones << ")" << std::endl;
        }
        dataset_statistics<la_type> stats_view(ds_view);
        base_hypotheses.push_back(params.weak_learner->create(stats_view));
        // TODO: Eventually, we should store WL predictions so we don't
        //       call predict() twice (once for edges and once to update
        //       distribution).
        edge = 0;
        size_t i = 0;
        for (typename dataset<la_type>::record_iterator_type ds_it = ds.begin();
             ds_it != ds_end; ++ds_it) {
          edge += (base_hypotheses.back()->predict(*ds_it) == label(ds,i) ?
                   distribution[i] : 0);
          ++i;
        }
        edge -= .5;
        if (BATCH_BOOSTER_DEBUG) {
          double tmpacc = 0;
          size_t i = 0;
          for (typename dataset<la_type>::record_iterator_type ds_it = ds.begin();
               ds_it != ds_end; ++ds_it) {
            tmpacc += (base_hypotheses.back()->predict(*ds_it) == label(ds,i) ?
                       1 : 0);
            ++i;
          }
          tmpacc /= ds.size();
          std::cerr << " Accuracy on unweighted training set = " << tmpacc
                    << std::endl;
        }
      } else {
        // Do not use resampling
        dataset_view<la_type> ds_view(ds);
        ds_view.set_weights(resampler.distribution());
        dataset_statistics<la_type> stats_view(ds_view);
        base_hypotheses.push_back(params.weak_learner->create(stats_view));
        edge = base_hypotheses.back()->train_accuracy() - .5;
      }
      // Deal with edge = -.5, 0, .5
      if (fabs(edge) <= params.convergence_zero) {
        if (BATCH_BOOSTER_DEBUG)
          std::cerr << "batch_booster exited early because a base hypothesis"
                    << " had an edge of 0." << std::endl;
        base_hypotheses.pop_back();
        return false;
      } else if (fabs(edge + .5) <= params.convergence_zero) {
        if (BATCH_BOOSTER_DEBUG)
          std::cerr << "batch_booster exited early because a base hypothesis"
                    << " had an edge of -.5." << std::endl;
        // TODO: We could get rid of all of the other base hypotheses,
        //       but it doesn't really matter.
        if (!wl_confidence_rated)
          alphas.push_back(- discriminative::BIG_DOUBLE);
        end_step();
        return false;
      } else if (fabs(edge - .5) <= params.convergence_zero) {
        if (BATCH_BOOSTER_DEBUG)
          std::cerr << "batch_booster exited early because a base hypothesis"
                    << " had an edge of .5." << std::endl;
        // TODO: We could get rid of all of the other base hypotheses,
        //       but it doesn't really matter.
        if (!wl_confidence_rated)
          alphas.push_back(discriminative::BIG_DOUBLE);
        end_step();
        return false;
      }

      // Choose weight alpha for weak hypothesis
      double alpha = 1;
      if (!wl_confidence_rated) {
        alpha = Objective::alpha(edge, params.smoothing);
        alphas.push_back(alpha);
      }

      // Update distribution
      if (Objective::can_update) {
        size_t i = 0;
        for (typename dataset<la_type>::record_iterator_type ds_it = ds.begin();
             ds_it != ds_end; ++ds_it) {
          double tmpw = Objective::weight_update
            (distribution[i], alpha, label(ds,i),
             base_hypotheses.back()->confidence(*ds_it));
          if (tmpw == std::numeric_limits<double>::infinity())
            resampler.set(i, discriminative::BIG_DOUBLE);
          else if (tmpw == - std::numeric_limits<double>::infinity())
            resampler.set(i, - discriminative::BIG_DOUBLE);
          else
            resampler.set(i,tmpw);
  /*
          resampler.set(i,
                        Objective::weight_update
                        (distribution[i], alpha, label(ds,i),
                         base_hypotheses.back()->confidence(*ds_it)));
  */
          ++i;
        }
      } else {
        size_t i = 0;
        for (typename dataset<la_type>::record_iterator_type ds_it = ds.begin();
             ds_it != ds_end; ++ds_it) {
          resampler.set(i, Objective::weight(label(ds,i), predict_raw(*ds_it)));
          ++i;
        }
      }
      resampler.commit_update();

      end_step();
      if (BATCH_BOOSTER_DEBUG)
        std::cerr << iteration_ << "\t" << edge << "\t" << alpha << std::endl;
      return true;
    }

    //! Resets the data source to be used in future rounds of training.
    //! @param  n   max number of examples which may be drawn from the oracle
    void reset_datasource(oracle<la_type>& o, size_t n) {
      assert(false);
      // TODO: IMPLEMENT THIS
    }

    //! Resets the data source to be used in future rounds of training.
    void reset_datasource(dataset_statistics<la_type>& stats) {
      assert(false);
      // TODO: IMPLEMENT THIS
    }

    // Save and load methods
    //==========================================================================

    using base::save;
    using base::load;

    //! Output the learner to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    void save(std::ofstream& out, size_t save_part = 0,
              bool save_name = true) const {
      base::save(out, save_part, save_name);
      params.save(out);
      if (save_part == 1)
        resampler.save(out);
    }

    /**
     * Input the learner from a human-readable file.
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
      if (load_part == 1)
        resampler.load(in);
      rng.seed(static_cast<unsigned>(params.random_seed));
      return true;
    }

  }; // class batch_booster

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_BATCH_BOOSTER_HPP
