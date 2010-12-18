
#ifndef SILL_LEARNING_DISCRIMINATIVE_FILTERING_BOOSTER_HPP
#define SILL_LEARNING_DISCRIMINATIVE_FILTERING_BOOSTER_HPP

#include <sill/learning/discriminative/binary_booster.hpp>

#include <sill/macros_def.hpp>

// Set to 1 to print out debugging information
#define FILTERING_BOOSTER_DEBUG 0

namespace sill {

  /**
   * FILTERING_BOOSTER_PARAMETERS
   * From binary_booster_parameters:
   *  - WEAK_LEARNER
   *  - SMOOTHING
   *  - RANDOM_SEED
   *  - INIT_ITERATIONS
   *  - CONVERGENCE_ZERO
   */
  struct filtering_booster_parameters : public binary_booster_parameters {

    //! Train WL with this many examples (> 0)
    //!  (default = 100)
    size_t m_t;

    //! If true, scale value for M_T by log(t) where t is the number of rounds
    //!  (default = false)
    bool scale_m_t;

    /**
     * 0 = filter M_T examples as usual;
     * 1 = use weighted examples, drawing until sum of weights >= M_T;
     * 2 = use weighted examples, drawing M_T examples
     *  (default = 0)
     */
    size_t weight_m_t;

    /**
     * Estimate weak hypothesis edges with this many examples;
     * if 0, then use an adaptive sampling algorithm
     *  (default = 100)
     */
    size_t n_t;

    //! If true, scale value for N_T by log(t) where t is the number of rounds
    //!  (default = false)
    bool scale_n_t;

    /**
     * 0 = filter N_T examples as usual;
     * 1 = use weighted examples, drawing until sum of weights >= N_T;
     * 2 = use weighted examples, drawing N_T examples
     *  (default = 1)
     */
    size_t weight_n_t;

    //! Confidence parameter (probability with which this
    //! is allowed to fail). This is used if M_T or N_T are not fixed.
    //!  (default = .05)
    double delta;

    //! Target accuracy. This is used if M_T or N_T are not fixed.
    //!  (default = .95)
    double target_acc;

    filtering_booster_parameters()
      : m_t(100), scale_m_t(false), weight_m_t(0), n_t(100), scale_n_t(false),
        weight_n_t(1), delta(.05), target_acc(.95) {
    }

    bool valid() const {
      if (!binary_booster_parameters::valid())
        return false;
      if (m_t <= 0)
        return false;
      if (weight_m_t >= 3)
        return false;
      if (weight_n_t >= 3)
        return false;
      if (delta <= 0)
        return false;
      return true;
    }

    void save(std::ofstream& out) const {
      binary_booster_parameters::save(out);
      out << m_t << " " << (scale_m_t ? "1" : "0") << " " << weight_m_t << " "
          << n_t << " " << (scale_n_t ? "1" : "0") << " " << weight_n_t
          << " " << delta << " " << target_acc << "\n";
    }

    void load(std::ifstream& in, const datasource& ds) {
      binary_booster_parameters::load(in, ds);
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> m_t))
        assert(false);
      if (!(is >> scale_m_t))
        assert(false);
      if (!(is >> weight_m_t))
        assert(false);
      if (!(is >> n_t))
        assert(false);
      if (!(is >> scale_n_t))
        assert(false);
      if (!(is >> weight_n_t))
        assert(false);
      if (!(is >> delta))
        assert(false);
      if (!(is >> target_acc))
        assert(false);
    }

  };  // struct filtering_booster_parameters

  /**
   * Filtering boosting algorithm for binary labels.
   *
   * To create, for example, FilterBoost with traditional decision trees,
   * construct:
   * filtering_booster<dataset, boosting::filterboost,
   *                   decision_tree<discriminative::objective_information> >
   *
   * @param Objective      optimization objective defining the booster
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo serialization
   * @todo pass in the type of dataset which filteringbooster uses to collect
   *       datapoints for the weak learner?
   * @todo This does the termination check if it is using filtering,
   *       but it could really do it for weighting as well (at least for
   *       weighting = 1).
   * @todo Fix confidence-rated predictions.
   */
  template <typename Objective>
  class filtering_booster : public binary_booster<Objective> {

    // Protected data members
    //==========================================================================
  protected:

    typedef binary_booster<Objective> base;
    using base::label_;
    using base::label_index_;
    using base::iteration_;
    using base::wl_confidence_rated;
    using base::timing;
    using base::uniform_prob;
    using base::smoothing;
    using base::base_hypotheses;
    using base::alphas;

    /*
    // Import stuff from booster
    using base::wl_ptr;
    using base::rng;
    using base::uniform_prob;
    using base::smoothing;
    using base::iter;
    using base::base_hypotheses;
    using base::alphas;
    using base::wl_confidence_rated;
    using base::timing;
    // Import stuff from binary_classifier
    using base::confidence_rated_;
    // Import stuff from multiclass_classifier
    using base::label_;
    using base::label_index_;
    using base::finite_seq;
    using base::vector_seq;
    using base::train_acc;
    using base::label;
    using base::predict_raw;
    */

    filtering_booster_parameters params;

    //! random number generator
    boost::mt11213b rng;

    //! Dataset passed to weak learner (avoids reallocation each round)
    vector_dataset ds;

    //! Dataset oracle (for batch mode)
    ds_oracle* ds_o_ptr;

    //! Data oracle
    oracle& o;

    //! Average p_t.
    //! (Note this is average over p_t, not p_t computed from
    //! an average over examples used since avg(1/x) != 1/avg(x).)
    std::vector<double> p_ts;

    // Protected data members and methods: filter
    //==========================================================================

/*
    // forward declaration
    double probability(const assignment& example);
    //! Functor to pass to filter_oracle
    struct filter_functor {
    private:
      const filtering_booster& fb;
    public:
      filter_functor(const filtering_booster& fb) : fb(fb) { }
      double operator()(const record& example) const {
        return fb.probability(example);
      }
    };
    //! Filter
    //! TODO: Try building this directly into filtering_booster to see how much
    //!       faster that is.
    filter_oracle<filter_functor, vec> filter;
*/

    // Filter parameters.
    //! Max number of examples filter may draw on each call.
    size_t filter_limit;

    //! Weight of last example returned by filter.
    double filter_weight;

    //! Sum of filter_weight for all examples generated during this iteration of
    //! boosting; reset in step() function.
    double sum_filter_weights;

    //! Number of examples used to generate last example returned by filter.
    //! This is also set in step() for weighting = 1.
    size_t filter_count;

    //! Generate a filtered example for weighting > 0.
    bool filter2() {
      if (!(o.next())) {
        filter_weight = 0;
        return false;
      }
      filter_weight = Objective::weight(base::label(o.current()),
                                        base::predict_raw(o.current()));
      return true;
    }

    //! Generate a filtered example for weighting = 0.
    bool filter0() {
      filter_count = 0;
      do {
        if (!(o.next()) || ++filter_count > filter_limit)
          return false;
        filter_weight = Objective::weight(base::label(o.current()),
                                          base::predict_raw(o.current()));
        sum_filter_weights += filter_weight;
      } while (uniform_prob(rng) > filter_weight);
      return true;
    }

    // Protected methods
    //==========================================================================

    void init() {
      params.set_smoothing(1000, label_->size());
      assert(params.valid());
      smoothing = params.smoothing;
      rng.seed(static_cast<unsigned>(params.random_seed));

      ds.set_finite_class_variable(label_);
      ds.set_vector_class_variable();
      if (params.weight_m_t > 0)
        ds.make_weighted();
      filter_limit = std::numeric_limits<size_t>::max();
    }

    //! Called at the end of a successful step().
    void end_step() {
      iteration_++;
      std::time_t time_tmp;
      time(&time_tmp);
      timing.push_back(time_tmp);
    }

    //! Called when the filter fails during boosting.
    void filter_failed(size_t weighting) {
      if (FILTERING_BOOSTER_DEBUG) {
        if ((weighting == 0 || weighting == 1) && filter_count > filter_limit)
          std::cerr << "filtering_booster exited early because it "
                    << "reached the target accuracy "
                    << "(with high probability)." << std::endl;
        else
          std::cerr << "filtering_booster exited early because the "
                    << "oracle was depleted." << std::endl;
      }
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

    // Constructors and destructors
    //==========================================================================

    /**
     * Constructor for a binary filtering booster without associated data;
     * useful for:
     *  - creating other instances
     *  - loading a saved booster
     * @param params        algorithm parameters
     */
    explicit filtering_booster(filtering_booster_parameters params
                               = filtering_booster_parameters())
      : base(params),
        params(params), ds(), ds_o_ptr(new ds_oracle(ds)), o(*ds_o_ptr) {
    }

    /**
     * Constructor for a binary filtering booster.
     * @param o    training data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param params    algorithm parameters
     */
    filtering_booster(oracle& o, size_t n,
                      filtering_booster_parameters params
                      = filtering_booster_parameters())
      : base(o, params), params(params),
        ds(o.datasource_info()), ds_o_ptr(NULL), o(o) {
      init();
      for (size_t t = 0; t < this->params.init_iterations; ++t)
        if (!(step()))
          break;
    }

    /**
     * Constructor for a binary filtering booster.
     * @param stats     a statistics class for the training dataset
     * @param params    algorithm parameters
     */
    explicit filtering_booster(statistics& stats,
                               filtering_booster_parameters params
                               = filtering_booster_parameters())
      : base(stats.get_dataset(), params),
        params(params), ds(stats.get_dataset().datasource_info()),
        ds_o_ptr(new ds_oracle(stats.get_dataset())), o(*ds_o_ptr) {
      assert(stats.get_dataset().is_weighted() == false);
      init();
      for (size_t t = 0; t < this->params.init_iterations; ++t)
        if (!(step()))
          break;
    }

    ~filtering_booster() {
      if (ds_o_ptr != NULL)
        delete(ds_o_ptr);
    }

    //! Train a new binary classifier of this type with the given data.
    boost::shared_ptr<binary_classifier> create(statistics& stats) const {
      boost::shared_ptr<binary_classifier>
        bptr(new filtering_booster<Objective>(stats, this->params));
      return bptr;
    }

    //! Train a new binary classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<binary_classifier> create(oracle& o, size_t n) const {
      boost::shared_ptr<binary_classifier>
        bptr(new filtering_booster<Objective>(o, n, this->params));
      return bptr;
    }

    // Getters and helpers
    //==========================================================================

    //! Return a name for the algorithm without template parameters.
    std::string name() const { return "filtering_booster"; }

    //! Return a name for the algorithm with comma-separated template parameters
    //! (e.g., objective).
    std::string fullname() const {
      return name() + "<" + Objective::name() + ">";
    }

    //! Returns true iff learner is naturally an online learner.
    bool is_online() const { return true; }

    //! Returns an estimate of p_t from each iteration.
    std::vector<double> norm_constants() const { return p_ts; }

    // Learning and mutating operations
    //==========================================================================

    //! Reset the random seed in this algorithm's parameters and in its
    //! random number generator.
    void random_seed(double value) {
      params.random_seed = value;
      rng.seed(static_cast<unsigned>(params.random_seed));
    }

    //! Run next iteration of boosting.
    bool step() {
      // Get a dataset with m examples for the weak learner.
      size_t m = (params.scale_m_t
                  ? (size_t)(params.m_t * std::log(exp(1.) + iteration_))
                  : params.m_t);
      ds.clear();
      sum_filter_weights = 0; // Estimate p_t using examples from filter.
      size_t n_examples_used = 0; // For estimating p_t
      double delta = params.delta / (3 * (iteration_ + 1) * (iteration_ + 2));
      filter_limit = (params.target_acc < 1
                      ? (size_t)((-2. / (1.-params.target_acc)) * std::log(delta))
                      : std::numeric_limits<size_t>::max());
      if (params.weight_m_t == 2) {
        // Draw m weighted examples to make a dataset
        for (size_t i = 0; i < m; i++) {
          if (!filter2()) {
            filter_failed(params.weight_m_t);
            return false;
          }
          ds.insert(o.current().finite(), o.current().vector(), filter_weight);
          sum_filter_weights += filter_weight;
        }
        n_examples_used += m;
      } else if (params.weight_m_t == 1) {
        // Draw examples until weight >= m to make a dataset
        double total_weight(0);
        filter_count = 0;
        do {
          if (!filter2()) {
            filter_failed(params.weight_m_t);
            return false;
          }
          total_weight += filter_weight;
          ++filter_count;
          if (filter_count / total_weight > filter_limit) {
            filter_failed(params.weight_m_t);
            return false;
          }
          ds.insert(o.current().finite(), o.current().vector(), filter_weight);
        } while (total_weight < m);
        sum_filter_weights += total_weight;
        n_examples_used += filter_count;
      } else { // params.weight_m_t == 0
        // Draw m filtered examples to make a dataset
        for (size_t i = 0; i < m; i++) {
          if (!filter0()) {
            filter_failed(params.weight_m_t);
            return false;
          }
          n_examples_used += filter_count;
          ds.insert(o.current().finite(), o.current().vector());
        }
      }

      // Train the weak learner
      params.weak_learner->random_seed
        (boost::uniform_int<int>(0, std::numeric_limits<int>::max())(rng));
      statistics stats(ds);
      base_hypotheses.push_back(params.weak_learner->create(stats));

      // Compute the weak learner's edge (by n_t or adaptive sampling)
      double edge = 0;
      binary_classifier& base_hypothesis = *(base_hypotheses.back());
      if (params.n_t > 0) {
        // Use n examples
        size_t n = (params.scale_n_t
                    ? (size_t)(params.n_t * std::log(exp(1.) + iteration_))
                    : params.n_t);
        if (params.weight_n_t == 2) {
          // Draw n weighted examples
          double total_weight = 0;
          for (size_t i = 0; i < n; i++) {
            if (!filter2()) {
              filter_failed(params.weight_n_t);
              base_hypotheses.pop_back();
              return false;
            }
            if (base_hypothesis.predict(o.current()) ==
                o.current().finite()[label_index_])
              edge += filter_weight;
            total_weight += filter_weight;
          }
          sum_filter_weights += total_weight;
          n_examples_used += n;
          edge = edge / total_weight - .5;
        } else if (params.weight_n_t == 1) {
          // Draw examples until weight >= n
          double total_weight = 0;
          filter_count = 0;
          do {
            if (!filter2()) {
              filter_failed(params.weight_n_t);
              base_hypotheses.pop_back();
              return false;
            }
            total_weight += filter_weight;
            ++filter_count;
            if (filter_count / total_weight > filter_limit) {
              filter_failed(params.weight_n_t);
              base_hypotheses.pop_back();
              return false;
            }
            if (base_hypothesis.predict(o.current()) ==
                o.current().finite()[label_index_])
              edge += filter_weight;
          } while (total_weight < n);
          sum_filter_weights += total_weight;
          n_examples_used += filter_count;
          edge = edge / total_weight - .5;
        } else { // params.weight_n_t == 0
          // Draw n filtered examples
          for (size_t i = 0; i < n; i++) {
            if (!filter0()) {
              filter_failed(params.weight_n_t);
              base_hypotheses.pop_back();
              return false;
            }
            n_examples_used += filter_count;
            if (base_hypothesis.predict(o.current()) ==
                o.current().finite()[label_index_])
              ++edge;
          }
          edge = edge / n - .5;
        }
      } else {
        // Use adaptive sampling
        edge = 0;
        assert(false);
      }
      // Deal with edge = 0, -.5, .5
      if (absval(edge) <= params.convergence_zero) {
        if (FILTERING_BOOSTER_DEBUG)
          std::cerr << "Warning: filtering_booster had a base hypothesis"
                    << " with an edge of 0; base hypothesis discarded."
                    << "  This could be caused by the weak learner not being"
                    << " powerful enough to improve accuracy further, or"
                    << " it could be caused by m_t, n_t being too small."
                    << std::endl;
        base_hypotheses.pop_back();
        return false;
      } else if (absval(edge + .5) <= params.convergence_zero) {
        if (FILTERING_BOOSTER_DEBUG)
          std::cerr << "filtering_booster exited early because a base "
                    << "hypothesis had an edge of -.5." << std::endl;
        // TODO: We could get rid of all of the other base hypotheses,
        //       but it doesn't really matter.
        if (!wl_confidence_rated)
          alphas.push_back(- discriminative::BIG_DOUBLE);
        end_step();
        return false;
      } else if (absval(edge - .5) <= params.convergence_zero) {
        if (FILTERING_BOOSTER_DEBUG)
          std::cerr << "filtering_booster exited early because a base "
                    << "hypothesis had an edge of .5." << std::endl;
        // TODO: We could get rid of all of the other base hypotheses,
        //       but it doesn't really matter.
        if (!wl_confidence_rated)
          alphas.push_back(discriminative::BIG_DOUBLE);
        end_step();
        return false;
      }

      // Compute weight alpha
      if (!wl_confidence_rated)
        alphas.push_back(Objective::alpha(edge, params.smoothing));
      p_ts.push_back(sum_filter_weights / n_examples_used);

      end_step();
      if (FILTERING_BOOSTER_DEBUG)
        std::cerr << "iteration " << iteration_ << "\t edge = " << edge
                  << ", alpha = " << alphas.back() << ", p_t = "
                  << p_ts.back() << std::endl;
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

    //! Output the learner to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    void save(std::ofstream& out, size_t save_part = 0,
              bool save_name = true) const {
      base::save(out, save_part, save_name);
      params.save(out);
      out << p_ts << "\n";
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
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      read_vec(is, p_ts);
      rng.seed(static_cast<unsigned>(params.random_seed));
      return true;
    }

  }; // class filtering_booster

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_FILTERING_BOOSTER_HPP
