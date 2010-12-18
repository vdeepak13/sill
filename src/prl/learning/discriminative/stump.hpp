
#ifndef SILL_LEARNING_DISCRIMINATIVE_STUMP_HPP
#define SILL_LEARNING_DISCRIMINATIVE_STUMP_HPP

#include <algorithm>

#include <sill/functional.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/binary_classifier.hpp>
#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/discriminative.hpp>
#include <sill/stl_io.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/macros_def.hpp>

/**
 * \file stump.hpp Decision Stump
 *
 * Some design comments:
 *  - The scores and confidences are given by separate functions.  In many
 *    cases, these might be the same.  However, in boosting decision trees,
 *    one might wish to build the decision tree using, e.g., a mutual
 *    information heuristic but use an exponential loss for confidence-rated
 *    predictions with AdaBoost.
 */

// Set to true to print debugging information.
#define DEBUG_STUMP 0

namespace sill {

  struct stump_parameters {

    //! Value used for smoothing confidences
    //!  (default = 1 / (2 * # training examples * # labels))
    double smoothing;

    //! Used to make the algorithm deterministic
    //!  (default = time)
    double random_seed;

    stump_parameters() : smoothing(-1) {
      std::time_t time_tmp;
      time(&time_tmp);
      random_seed = time_tmp;
    }

    bool valid() const {
      if (smoothing < 0)
        return false;
      return true;
    }

    void save(std::ofstream& out) const {
      out << smoothing << " " << random_seed << "\n";
    }

    void load(std::ifstream& in) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> smoothing))
        assert(false);
      if (!(is >> random_seed))
        assert(false);
    }

    //! Sets smoothing to its default value for the given dataset info
    //! if it has not yet been set.
    void set_smoothing(size_t ntrain, size_t nlabels) {
      if (smoothing < 0)
        smoothing = 1. / (2. * ntrain * nlabels);
    }

  }; // struct stump_parameters

  /**
   * Class for learning decision stumps for binary data.
   *  - For discrete variables i, stumps are of form: I[record(i) == val]
   *  - For vector variables j, stumps are of form: I[record(j)[k] > val]
   *    - Note: This tries separate rules for each vector index k.
   * If some decision rules score equally well, this chooses each with equal
   * probability.
   *
   * @param Objective  class defining the optimization objective
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo serialization
   * @todo This should be easily extendible to multiclass data.
   * @todo The build() function could be made more efficient by using
   *       record_iterator instead of finite(,) and vector(,).
   */
  template <typename Objective = discriminative::objective_accuracy>
  class stump : public binary_classifier {

    concept_assert((sill::DomainPartitioningObjective<Objective>));

    // Protected data members
    //==========================================================================
  protected:

    typedef binary_classifier base;

    // Data from base class:
    //  finite_variable* label_
    //  size_t label_index_

    stump_parameters params;

    //! random number generator
    boost::mt11213b rng;

    //! uniform distribution over [0, 1]
    boost::uniform_real<double> uniform_prob;

    //! Type of variable used for the decision stump
    variable::variable_typenames split_var_type;

    //! Finite variable used for the stump (if finite)
    finite_variable* finite_split_var;

    //! Vector variable used for the stump (if vector)
    vector_variable* vector_split_var;

    //! Index of split_var in the dataset's records.
    size_t split_var_index;

    //! For finite split_var: value to test for
    //! If variable has this value, then predict predictA
    size_t split_finite;

    //! For vector split_var: index of component in vector to test
    //! (Each component of a vector variable is considered separately.)
    size_t split_vector_index;

    //! For vector split_var: value to test for
    //! If variable has value > split_vector, then predict predictA
    double split_vector;

    //! Confidences of predicting classes A, B.
    //! Note classes A may correspond to classes 0 or 1; ditto for B.
    double predictA, predictB;

    //! Training accuracy
    double train_acc;

    // Protected methods
    //==========================================================================

    void build(statistics& stats) {
      const dataset& ds = stats.get_dataset();

      params.set_smoothing(ds.size(), label_->size());
      assert(params.valid());
      stats.compute_order_stats();
      rng.seed(static_cast<unsigned>(params.random_seed));
      uniform_prob = boost::uniform_real<double>(0,1);

      // Find weight of classes 0,1
      std::vector<double> total_weights(2,0);
      for (size_t i = 0; i < ds.size(); i++)
        total_weights[label(ds,i)] += ds.weight(i);
      double total_weight = 0;
      foreach(double t, total_weights)
        total_weight += t;
      assert(total_weight > 0);

      if (DEBUG_STUMP)
        std::cerr << "stump::build(): weights of classes (0,1) = "
                  << total_weights << std::endl;

      // For all features, try all possible splits, and keep track of best
      double best_objective =
        - std::numeric_limits<double>::infinity();
      size_t num_best = 1; // used to randomly choose among best classifiers
      split_var_type = variable::FINITE_VARIABLE;
      finite_split_var = NULL;
      vector_split_var = NULL;
      foreach(finite_variable* v, ds.finite_variables()) {
        if (v == label_)
          continue;
        size_t v_index = ds.record_index(v);
        // weights[j][k] = weight of examples with value j, label k
        std::vector<std::vector<double> > weights(v->size()); // TODO: CHANGE THIS TO AVOID REALLOCATION
        foreach(std::vector<double>& w, weights)
          w.resize(label_->size());
        for (size_t i = 0; i < ds.size(); i++)
          weights[ds.finite(i,v_index)][label(ds,i)] += ds.weight(i);
        // choose decision rule
        for (size_t j = 0; j < v->size(); j++) {
          double cur_objective;
          if (weights[j][0] > weights[j][1])
            // then predict class 0 for value j
            cur_objective
              = Objective::unnormalized(weights[j][0], weights[j][1]);
          else
            // then predict class 1 for value j
            cur_objective
              = Objective::unnormalized(weights[j][1], weights[j][0]);
          if (total_weights[0] - weights[j][0]
              > total_weights[1] - weights[j][1])
            // then predict class 0 for all other values
            cur_objective
              += Objective::unnormalized(total_weights[0] - weights[j][0],
                                         total_weights[1] - weights[j][1]);
          else
            // then predict class 1 for all other values
            cur_objective
              += Objective::unnormalized(total_weights[1] - weights[j][1],
                                         total_weights[0] - weights[j][0]);
          if (cur_objective < best_objective)
            continue;
          if (cur_objective == best_objective) {
            ++num_best;
            if (uniform_prob(rng) > 1. / num_best)
              continue;
          } else
            num_best = 1;
          best_objective = cur_objective;
          split_var_type = variable::FINITE_VARIABLE;
          finite_split_var = v;
          split_var_index = v_index;
          split_finite = j;
          predictA = Objective::confidence(weights[j][0], weights[j][1]);
          predictB = Objective::confidence(total_weights[0] - weights[j][0],
                                           total_weights[1] - weights[j][1]);
          train_acc = std::max(weights[j][0], weights[j][1])
            + std::max(total_weights[0] - weights[j][0],
                       total_weights[1] - weights[j][1]);
          train_acc /= (total_weights[0] + total_weights[1]);
        }
      }
      foreach(vector_variable* v, ds.vector_variables()) {
        // consider each component j of the vector variable separately
        for (size_t j = 0; j < v->size(); j++) {
          size_t v_index = ds.record_index(v) + j;
          // weights[k] = weight of examples larger than cutoff with label k
          std::vector<double> weights(label_->size()); // CHANGE TO AVOID REALLOCATION
          for (size_t k = 0; k < label_->size(); k++)
            weights[k] = total_weights[k];
          for (size_t i = 0; i < ds.size() - 1; i++) {
            size_t r = stats.order_stats(v_index, i);
            // r index records in increasing order of variable value
            double cut = ds.vector(r,v_index);
            weights[ds.finite(r,label_index_)] -= ds.weight(r);
            if (ds.vector(stats.order_stats(v_index,i+1), v_index) == cut)
              continue; // todo: this reads each value twice (inefficient)
            // decide if we should use the rule value > cut
            double cur_objective;
            if (weights[0] > weights[1])
              // then predict class 0 for value > cut
              cur_objective = Objective::unnormalized(weights[0], weights[1]);
            else
              // then predict class 1 for value > cut
              cur_objective = Objective::unnormalized(weights[1], weights[0]);
            if (total_weights[0] - weights[0]
                > total_weights[1] - weights[1])
              // then predict class 0 for value <= cut
              cur_objective
                += Objective::unnormalized(total_weights[0] - weights[0],
                                           total_weights[1] - weights[1]);
            else
              // then predict class 1 for value <= cut
              cur_objective
                += Objective::unnormalized(total_weights[1] - weights[1],
                                           total_weights[0] - weights[0]);
            if (cur_objective < best_objective)
              continue;
            if (cur_objective == best_objective) {
              ++num_best;
              if (uniform_prob(rng) > 1. / num_best)
                continue;
            } else
              num_best = 1;
            best_objective = cur_objective;
            split_var_type = variable::VECTOR_VARIABLE;
            vector_split_var = v;
            split_var_index = v_index;
            split_vector_index = j;
            split_vector = cut;
            predictA = Objective::confidence(weights[0], weights[1]);
            predictB = Objective::confidence(total_weights[0] - weights[0],
                                             total_weights[1] - weights[1]);
            train_acc
              = std::max(weights[0], weights[1])
              + std::max(total_weights[0] - weights[0],
                         total_weights[1] - weights[1]);
            train_acc /= (total_weights[0] + total_weights[1]);
          }
        }
      }

      // Check to see if a rule was chosen; if not, then choose an arbitrary
      // one which predicts the majority class.
      // This deals with the case where all records are identical.
      if (finite_split_var == NULL && vector_split_var == NULL) {
        if (total_weights[1] > total_weights[0])
          best_objective =
            Objective::unnormalized(total_weights[1], total_weights[0]);
        else
          best_objective =
            Objective::unnormalized(total_weights[0], total_weights[1]);
        predictA = Objective::confidence(total_weights[0],total_weights[1]);
        predictB = Objective::confidence(total_weights[0],total_weights[1]);
        if (ds.vector_variables().size() > 0) {
          split_var_type = variable::VECTOR_VARIABLE;
          vector_split_var = *(ds.vector_variables().begin());
          split_var_index = ds.record_index(vector_split_var);
          split_vector_index = 0;
          split_vector = 0;
        } else if (ds.finite_variables().size() > 1) {
          split_var_type = variable::FINITE_VARIABLE;
          finite_domain temp = ds.finite_variables();
          temp.erase(label_);
          finite_split_var = *(temp.begin());
          split_var_index = ds.record_index(finite_split_var);
          split_finite = 0;
        } else
          assert(false);
      }
    } // end of function: void build()

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

    // Constructors and destructors
    //==========================================================================

    //! Constructor for an empty stump with the given parameters.
    explicit stump(stump_parameters params = stump_parameters())
      : base(), params(params) { }

    /**
     * Constructor for a decision stump.
     * @param stats     a statistics class for the training dataset
     * @param params    algorithm parameters
     */
    explicit stump(statistics& stats,
                   stump_parameters params = stump_parameters())
      : base(stats.get_dataset()), params(params) {
      build(stats);
    }

    /**
     * Constructor for a decision stump.
     * @param o    data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param params    algorithm parameters
     */
    stump(oracle& o, size_t n, stump_parameters params = stump_parameters())
      : base(o), params(params) {
      boost::shared_ptr<vector_dataset>
        ds_ptr(oracle2dataset<vector_dataset>(o,n));
      statistics stats(*ds_ptr);
      build(stats);
    }

    //! Train a new binary classifier of this type with the given data.
    boost::shared_ptr<binary_classifier> create(statistics& stats) const {
      boost::shared_ptr<binary_classifier>
        bptr(new stump<Objective>(stats, this->params));
      return bptr;
    }

    //! Train a new binary classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<binary_classifier> create(oracle& o, size_t n) const {
      boost::shared_ptr<binary_classifier>
        bptr(new stump<Objective>(o, n, this->params));
      return bptr;
    }

    // Getters and helpers
    //==========================================================================

    //! Return a name for the algorithm without template parameters.
    std::string name() const { return "stump"; }

    //! Return a name for the algorithm with comma-separated template parameters
    //! (e.g., objective).
    std::string fullname() const {
      return name() + "<" + Objective::name() + ">";
    }

    //! Returns true iff learner is naturally an online learner.
    bool is_online() const { return false; }

    //! Indicates if the predictions are confidence-rated.
    //! Note that confidence rating may be optimized for different objectives.
    bool is_confidence_rated() const { return Objective::confidence_rated(); }

    //! Returns training accuracy (or estimate of it).
    double train_accuracy() const { return train_acc; }

    // Learning and mutating operations
    //==========================================================================

    //! Reset the random seed in this algorithm's parameters and in its
    //! random number generator.
    void random_seed(double value) {
      params.random_seed = value;
      rng.seed(static_cast<unsigned>(params.random_seed));
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
    double confidence(const record& example) const {
      if (split_var_type == variable::FINITE_VARIABLE)
        if (example.finite()[split_var_index] == split_finite)
          return predictA;
        else
          return predictB;
      else if (split_var_type == variable::VECTOR_VARIABLE)
        if (example.vector()[split_var_index] > split_vector)
          return predictA;
        else
          return predictB;
      else
        assert(false);
      return - std::numeric_limits<double>::infinity();
    }

    //! Value indicating the confidence of the prediction, with
    //!  predict() == (confidence() > 0) ? 1 : 0.
    //! If the classifier does not have actual confidence ratings,
    //!  then this should be any value with the correct sign.
    double confidence(const assignment& example) const {
      if (split_var_type == variable::FINITE_VARIABLE)
        if (safe_get(example.finite(), finite_split_var) == split_finite)
          return predictA;
        else
          return predictB;
      else if (split_var_type == variable::VECTOR_VARIABLE) {
        const vec& vec_val = safe_get(example.vector(), vector_split_var);
        if (vec_val[split_vector_index] > split_vector)
          return predictA;
        else
          return predictB;
      } else
        assert(false);
      return - std::numeric_limits<double>::infinity();
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
      out << split_var_type << " " << split_var_index
          << " " << split_finite << " " << split_vector_index
          << " " << split_vector << " " << predictA << " " << predictB
          << " " << train_acc << "\n";
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
      params.load(in);
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      size_t tmpsize;
      if (!(is >> tmpsize))
        assert(false);
      split_var_type = static_cast<variable::variable_typenames>(tmpsize);
      if (!(is >> split_var_index))
        assert(false);
      if (!(is >> split_finite))
        assert(false);
      if (!(is >> split_vector_index))
        assert(false);
      switch(split_var_type) {
      case variable::FINITE_VARIABLE:
        if (split_var_index < ds.num_finite())
          finite_split_var = ds.finite_list()[split_var_index];
        else
          finite_split_var = NULL;
        vector_split_var = NULL;
        assert(split_finite < finite_split_var->size());
        break;
      case variable::VECTOR_VARIABLE:
        if (split_var_index < ds.num_vector())
          vector_split_var = ds.vector_list()[split_var_index];
        else
          vector_split_var = NULL;
        finite_split_var = NULL;
        assert(split_vector_index < ds.vector_dim());
        break;
      default:
        break;
      }
      if (!(is >> split_vector))
        assert(false);
      if (!(is >> predictA))
        assert(false);
      if (!(is >> predictB))
        assert(false);
      if (!(is >> train_acc))
        assert(false);
      rng.seed(static_cast<unsigned>(params.random_seed));
      uniform_prob = boost::uniform_real<double>(0,1);
      return true;
    }

  }; // class stump

} // end of namespace: prl

#undef DEBUG_STUMP

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_STUMP_HPP
