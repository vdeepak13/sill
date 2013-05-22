
#ifndef SILL_LEARNING_DISCRIMINATIVE_FILTERING_BOOSTER_OC_HPP
#define SILL_LEARNING_DISCRIMINATIVE_FILTERING_BOOSTER_OC_HPP

#include <sill/learning/dataset/ds_oracle.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/multiclass_booster_OC.hpp>

#include <sill/macros_def.hpp>

// Set to 1 to print debugging info
#define DEBUG_FILTERING_BOOSTER_OC 0

namespace sill {

  /**
   * FILTERING_BOOSTER_OC_PARAMETERS
   * From multiclass_booster_OC_parameters:
   *  - BINARY_LABEL
   *  - WEAK_LEARNER
   *  - SMOOTHING
   *  - RANDOM_SEED
   *  - INIT_ITERATIONS
   *  - CONVERGENCE_ZERO
   * @tparam LA  Linear algebra type specifier
   *              (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  struct filtering_booster_OC_parameters
    : public multiclass_booster_OC_parameters<LA> {

    typedef LA la_type;

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

    //! Estimate weak hypothesis edges with this many examples;
    //! if 0, then use an adaptive sampling algorithm
    //!  (default = 100)
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

    filtering_booster_OC_parameters()
      : m_t(100), scale_m_t(false), weight_m_t(0), n_t(100),
        scale_n_t(false), weight_n_t(1), delta(.05), target_acc(.95) { }

    bool valid() const {
      if (!multiclass_booster_OC_parameters<la_type>::valid())
        return false;
      if (m_t == 0)
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
      out << m_t << " " << (scale_m_t ? "1" : "0") << " " << weight_m_t << " "
          << n_t << " " << (scale_n_t ? "1" : "0") << " " << weight_n_t
          << " " << delta << " " << target_acc << "\n";
    }

    void load(std::ifstream& in, const datasource& ds) {
      multiclass_booster_OC_parameters<la_type>::load(in, ds);
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

  };  // struct filtering_booster_OC_parameters

  /**
   * Filtering boosting algorithm for multiclass labels which uses binary weak
   * learners and output coding.
   *
   * To create, for example, FilterBoost.OC with traditional decision trees,
   * construct:
   * filtering_booster_OC<boosting::filterboost,
   *                      decision_tree<discriminative::objective_information> >
   * @tparam Objective  optimization criterion defining the booster
   * @tparam LA  Linear algebra type specifier
   *              (default = dense_linear_algebra<>)
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo serialization
   */
  template <typename Objective, typename LA = dense_linear_algebra<> >
  class filtering_booster_OC : public multiclass_booster_OC<Objective,LA> {

    // Public types
    //==========================================================================
  public:

    typedef LA la_type;

    typedef multiclass_booster_OC<Objective,la_type> base;

    typedef typename base::record_type record_type;

    typedef filtering_booster_OC_parameters<la_type> parameters;

    // Protected data members
    //==========================================================================
  protected:

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

    parameters params;

    //! Dataset passed to weak learner (avoids reallocation each round)
    vector_dataset<la_type> ds;

    //! Dataset oracle (for batch mode)
    ds_oracle<la_type>* ds_o_ptr;

    //! Data oracle
    oracle<la_type>& o;

    //! Average p_t.
    //! These are calculated from the examples used to calculate edges,
    //!  not from those used to train the weak learner.
    //! (Note this is average over p_t, not p_t computed from
    //! an average over examples used since avg(1/x) != 1/avg(x).)
    std::vector<double> p_ts;

    // Protected data members and methods: filter
    //==========================================================================

    // These filter parameters are shared between filterWL() and filter():
    //! Max number of examples filter may draw on each call.
    size_t filter_limit;

    //! Weight of last example returned by filter (sum over all y' != y).
    double filter_weight;

    //! Weight of last example returned by filterWL (sum over y s.t.
    //! coloring(y') != coloring(y)).
    double wl_filter_weight;

    //! Sum of filter_weight for all examples generated during this iteration of
    //! boosting; reset in step() function.
    //! (For examples from filterWL(), this includes sum over all y' != y,
    //!  not just those such that coloring(y') != coloring(y).)
    double sum_filter_weights;

    //! Number of examples used to generate last example returned by filter.
    size_t filter_count;

    //! q_t'((x,y),y') for last example (x,y) returned by filter().
    //! (This is used by getEdge().)
    vec filter_distribution;

    //! Set filter_weight (but not filter_distribution) using current example
    //! in oracle -- for WL examples (not estimating edges).
    //! @todo This could be sped up by only running predict_raw() on the
    //!       labels which do not match the coloring of the true label.
    void compute_filter_distributionWL() {
      this->my_predict_raws(o.current(), tmp_vector);
      size_t y(label(o.current()));
      filter_weight = 0;
      wl_filter_weight = 0;
      for (size_t j = 0; j < nclasses_; ++j)
        if (colorings[iteration_][j] != colorings[iteration_][y])
          wl_filter_weight +=
            Objective::weight(1, tmp_vector[y] - tmp_vector[j]);
        else
          filter_weight += Objective::weight(1, tmp_vector[y] - tmp_vector[j]);
      filter_weight += wl_filter_weight - .5;
      filter_weight /= (nclasses_ - 1);
      wl_filter_weight /= (nclasses_ - 1);
    }

    //! Set filter_weight, filter_distribution using current example in oracle.
    void compute_filter_distribution() {
      this->my_predict_raws(o.current(), tmp_vector);
      size_t y(label(o.current()));
      filter_weight = 0;
      for (size_t j = 0; j < nclasses_; ++j) {
        filter_distribution[j] =
          Objective::weight(1, tmp_vector[y] - tmp_vector[j]);
        filter_weight += filter_distribution[j];
      }
      filter_weight -= .5; // to account for j == y
      filter_weight /= (nclasses_ - 1);
    }

    //! Generate a filtered example for the weak learner for weighting > 0.
    bool filterWL2() {
      if (!(o.next())) {
        filter_failed(params.weight_m_t, false);
        return false;
      }
      compute_filter_distributionWL();
      return true;
    }

    //! Generate a filtered example for the weak learner for weighting = 0.
    bool filterWL0() {
      filter_count = 0;
      double total_filter_weight(0);
      do {
        if (!(o.next())) {
          filter_failed(params.weight_m_t, false);
          return false;
        }
        compute_filter_distributionWL();
        total_filter_weight += filter_weight;
        ++filter_count;
        if (filter_count / total_filter_weight > filter_limit) {
          filter_failed(params.weight_m_t, true);
          return false;
        }
      } while (uniform_prob(rng) > wl_filter_weight);
      sum_filter_weights += total_filter_weight;
      return true;
    }

    //! Generate a filtered example for estimating edges for weighting > 0.
    bool filter2() {
      if (!(o.next())) {
        filter_failed(params.weight_n_t, false);
        return false;
      }
      compute_filter_distribution();
      return true;
    }

    //! Generate a filtered example for estimating edges for weighting = 0.
    bool filter0() {
      filter_count = 0;
      do {
        if (!(o.next())) {
          filter_failed(params.weight_n_t, false);
          return false;
        }
        if (++filter_count > filter_limit) {
          filter_failed(params.weight_n_t, true);
          return false;
        }
        compute_filter_distribution();
        sum_filter_weights += filter_weight;
      } while (uniform_prob(rng) > filter_weight);
      return true;
    }

    //! Called when the filter fails during boosting.
    void filter_failed(size_t weighting, bool at_target_acc = false) {
      if (DEBUG_FILTERING_BOOSTER_OC) {
        if ((weighting == 0 || weighting == 1) && at_target_acc)
          std::cerr << "filtering_booster_OC exited early because it "
                    << "reached the target accuracy "
                    << "(with high probability)." << std::endl;
        else
          std::cerr << "filtering_booster_OC exited early because the "
                    << "oracle was depleted." << std::endl;
      }
    }

    // Protected methods
    //==========================================================================

    void init() {
      assert(params.valid());
      assert(params.binary_label != NULL);
      params.set_smoothing(1000, label_->size());
      smoothing = params.smoothing;
      rng.seed(static_cast<unsigned>(params.random_seed));

      ds.set_finite_class_variable(label_);
      ds.set_vector_class_variable();
      if (params.weight_m_t > 0)
        ds.make_weighted();
      filter_limit = std::numeric_limits<size_t>::max();
      filter_distribution.set_size(nclasses_);
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

    // Constructors and destructors
    //==========================================================================

    /**
     * Constructor for a multiclass filtering booster without associated data;
     * useful for:
     *  - creating other instances
     *  - loading a saved booster
     * @param params        algorithm parameters
     */
    explicit filtering_booster_OC(parameters params
                                  = parameters())
      : base(params), params(params), ds(),
        ds_o_ptr(new ds_oracle<la_type>(ds)), o(*ds_o_ptr) { }

    /**
     * Constructor for a multiclass filtering booster.
     * @param o    training data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param params        algorithm parameters
     */
    filtering_booster_OC(oracle<la_type>& o, size_t n,
                         parameters params
                         = parameters())
      : base(o, params), params(params), ds(o.datasource_info()),
        ds_o_ptr(NULL), o(o) {
      init();
      for (size_t t = 0; t < this->params.init_iterations; ++t) {
        if (!(step()))
          break;
      }
    }

    /**
     * Constructor for a multiclass filtering booster.
     * @param stats         a statistics class for the training dataset
     * @param parameters    algorithm parameters
     */
    explicit filtering_booster_OC(dataset_statistics<la_type>& stats,
                                  parameters params
                                  = parameters())
      : base(stats.get_dataset(), params), params(params),
        ds(stats.get_dataset().datasource_info()),
        ds_o_ptr(new ds_oracle<la_type>(stats.get_dataset())), o(*ds_o_ptr) {
      assert(stats.get_dataset().is_weighted() == false);
      init();
      for (size_t t = 0; t < this->params.init_iterations; ++t)
        if (!(step()))
          break;
    }

    ~filtering_booster_OC() {
      if (ds_o_ptr != NULL)
        delete(ds_o_ptr);
    }

    //! Train a new multiclass classifier of this type with the given data.
    boost::shared_ptr<multiclass_classifier<la_type> > create(dataset_statistics<la_type>& stats) const {
      boost::shared_ptr<multiclass_classifier<la_type> >
        bptr(new filtering_booster_OC<Objective,la_type>(stats, this->params));
      return bptr;
    }

    //! Train a new multiclass classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<multiclass_classifier<la_type> > create(oracle<la_type>& o, size_t n) const {
      boost::shared_ptr<multiclass_classifier<la_type> >
        bptr(new filtering_booster_OC<Objective,la_type>(o, n, this->params));
      return bptr;
    }

    // Getters and helpers
    //==========================================================================

    //! Return a name for the algorithm without template parameters.
    std::string name() const { return "filtering_booster_OC"; }

    //! Return a name for the algorithm with comma-separated template parameters
    //! (e.g., objective).
    std::string fullname() const {
      return name() + "<" + Objective::name() + ">";
    }

    //! Returns true iff learner is naturally an online learner.
    bool is_online() const { return true; }

    //! Get the parameters of the algorithm
    parameters& get_parameters() { return params; }

    //! Returns an estimate of the norm constant for the distribution on
    //! each iteration.
    //! E.g., p_t for filtering boosters and 1/Z_t for batch boosters.
    std::vector<double> norm_constants() const { return p_ts; }

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
      // Choose coloring.
      std::vector<size_t> coloring(choose_coloring());
      colorings.push_back(coloring);

      // Get a dataset with m examples for the weak learner.
      size_t m = (params.scale_m_t
                  ? (size_t)(params.m_t * std::log(exp(1.) + iteration_))
                  : params.m_t);
      ds.clear();
      sum_filter_weights = 0; // Estimate p_t using examples from filter.
      size_t n_examples_used = 0; // For estimating p_t
      double delta = params.delta / (3 * (iteration_ + 1) * (iteration_ + 2));
      filter_limit = (params.target_acc < 1
                      ? (size_t)((-2.*(nclasses_-1.) / (1.-params.target_acc))
                                 * std::log(delta))
                      : std::numeric_limits<size_t>::max());
      if (params.weight_m_t == 2) {
        // Draw m weighted examples to make a dataset
        for (size_t i = 0; i < m; i++) {
          if (!(filterWL2()))
            return false;
          ds.insert(o.current().finite(), o.current().vector(),
                    wl_filter_weight);
          sum_filter_weights += filter_weight;
        }
        n_examples_used += m;
      } else if (params.weight_m_t == 1) {
        // Draw examples until weight >= m to make a dataset
        double total_filter_weight(0);
        double total_wl_filter_weight(0);
        filter_count = 0;
        do {
          if (!filterWL2())
            return false;
          total_filter_weight += filter_weight;
          total_wl_filter_weight += wl_filter_weight;
          ++filter_count;
          if (filter_count / total_filter_weight > filter_limit) {
            filter_failed(params.weight_m_t, true);
            return false;
          }
          ds.insert(o.current().finite(), o.current().vector(),
                    wl_filter_weight);
        } while (total_wl_filter_weight < m);
        sum_filter_weights += total_filter_weight;
        n_examples_used += filter_count;
      } else { // params.weight_m_t == 0
        // Draw filtered examples to make a dataset
        for (size_t i = 0; i < m; i++) {
          if (!filterWL0())
            return false;
          n_examples_used += filter_count;
          ds.insert(o.current().finite(), o.current().vector());
        }
      }

      // Train the weak learner
      params.weak_learner->random_seed
        (boost::uniform_int<int>(0,std::numeric_limits<int>::max())(rng));
      dataset_view<la_type> ds_view(ds);
      ds_view.set_binary_coloring(label_, params.binary_label, coloring);
      dataset_statistics<la_type> stats(ds_view);
      base_hypotheses.push_back(params.weak_learner->create(stats));

      // Compute edge and alpha
      double edge = 0;
      binary_classifier<la_type>& base_hypothesis = *(base_hypotheses.back());
      if (params.n_t > 0) {
        // Use n examples
        size_t n = (params.scale_n_t
                    ? (size_t)(params.n_t * std::log(exp(1.) + iteration_))
                    : params.n_t);
        if (params.weight_n_t == 2) {
          // Draw n weighted examples to estimate the edge.
          double total_weight = 0;
          for (size_t i = 0; i < n; i++) {
            if (!(filter2())) {
              base_hypotheses.pop_back();
              return false;
            }
            total_weight += filter_weight;
            size_t y(label(o.current()));
            size_t pred(base_hypothesis.predict(o.current()));
            for (size_t j = 0; j < nclasses_; ++j)
              if (j != y)
                edge += filter_distribution[j] *
                  ((coloring[y] == pred ? 1 : 0) -
                   (coloring[j] == pred ? 1 : 0));
          }
          sum_filter_weights += total_weight;
          n_examples_used += n;
          edge = .5 * edge / ((nclasses_-1)*total_weight);
          // Note that we can move the above normalization (/ total_weight)
          // outside of the sum over i,j because of how the distributions over
          // (x,y) and y' != y relate.  (Same for weight_n_t = 1,0 too.)
        } else if (params.weight_n_t == 1) {
          // Draw examples until weight >= n to estimate the edge.
          double total_weight = 0;
          filter_count = 0;
          do {
            if (!filter2()) {
              base_hypotheses.pop_back();
              return false;
            }
            total_weight += filter_weight;
            ++filter_count;
            if (filter_count / total_weight > filter_limit) {
              filter_failed(params.weight_n_t, true);
              base_hypotheses.pop_back();
              return false;
            }
            size_t y(label(o.current()));
            size_t pred(base_hypothesis.predict(o.current()));
            for (size_t j = 0; j < nclasses_; ++j)
              if (j != y)
                edge += filter_distribution[j] *
                  ((coloring[y] == pred ? 1 : 0) -
                   (coloring[j] == pred ? 1 : 0));
          } while (total_weight < n);
          sum_filter_weights += total_weight;
          n_examples_used += filter_count;
          edge = .5 * edge / ((nclasses_-1)*total_weight);
        } else { // params.weight_n_t == 0
          // Draw n filtered examples to estimate the edge.
          for (size_t i = 0; i < n; i++) {
            if (!(filter0())) {
              base_hypotheses.pop_back();
              return false;
            }
            n_examples_used += filter_count;
            size_t y(label(o.current()));
            size_t pred(base_hypothesis.predict(o.current()));
            double tmp_edge(0);
            for (size_t j = 0; j < nclasses_; ++j) {
              if (j != y) {
                tmp_edge += filter_distribution[j] *
                  ((coloring[y] == pred ? 1 : 0) -
                   (coloring[j] == pred ? 1 : 0));
              }
            }
            tmp_edge /= ((nclasses_-1)*filter_weight);
            edge += tmp_edge;
          }
          edge = .5 * edge / n;
        }
      } else {
        // Use adaptive sampling
        edge = 0;
        assert(false);
      }
      // Deal with edge = 0, -.5, .5
      if (fabs(edge) <= params.convergence_zero) {
        if (DEBUG_FILTERING_BOOSTER_OC)
          std::cerr << "Warning: filtering_booster_OC had a base hypothesis"
                    << " with an edge of 0; base hypothesis discarded."
                    << "  This could be caused by the weak learner not being"
                    << " powerful enough to improve accuracy further, or"
                    << " it could be caused by m_t, n_t being too small."
                    << std::endl;
        base_hypotheses.pop_back();
        return false;
      } else if (fabs(edge + .5) <= params.convergence_zero) {
        if (DEBUG_FILTERING_BOOSTER_OC)
          std::cerr << "filtering_booster_OC exited early because a base "
                    << "hypothesis had an edge of -.5." << std::endl;
        // TODO: We could get rid of all of the other base hypotheses,
        //       but it doesn't really matter.
        alphas.push_back(- discriminative::BIG_DOUBLE);
        end_step();
        return false;
      } else if (fabs(edge - .5) <= params.convergence_zero) {
        if (DEBUG_FILTERING_BOOSTER_OC)
          std::cerr << "filtering_booster_OC exited early because a base "
                    << "hypothesis had an edge of .5." << std::endl;
        // TODO: We could get rid of all of the other base hypotheses,
        //       but it doesn't really matter.
        alphas.push_back(discriminative::BIG_DOUBLE);
        end_step();
        return false;
      }

      // Compute weight alpha
      double alpha = Objective::alpha(edge, params.smoothing);
      alphas.push_back(alpha);
      p_ts.push_back(sum_filter_weights / n_examples_used);

      ++iteration_;
      std::time_t time_tmp;
      time(&time_tmp);
      timing.push_back(time_tmp);
      if (DEBUG_FILTERING_BOOSTER_OC)
        std::cerr << "iteration " << iteration_ << "\t edge = " << edge
                  << ", alpha = " << alphas.back() << ", p_t = " << p_ts.back()
                  << std::endl;
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

    //! Output the classifier to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    void save(std::ofstream& out, size_t save_part = 0,
              bool save_name = true) const {
      base::save(out, save_part, save_name);
      params.save(out);
      out << p_ts << "\n";
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
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      read_vec(is, p_ts);
      rng.seed(static_cast<unsigned>(params.random_seed));
      filter_distribution.set_size(nclasses_);
      return true;
    }

  }; // class filtering_booster_OC

} // namespace sill

#undef DEBUG_FILTERING_BOOSTER_OC

#include <sill/macros_undef.hpp>

#include <sill/learning/discriminative/load_functions.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_FILTERING_BOOSTER_OC_HPP
