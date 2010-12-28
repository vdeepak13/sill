
#ifndef SILL_LEARNING_DISCRIMINATIVE_PROTO_KNN_HPP
#define SILL_LEARNING_DISCRIMINATIVE_PROTO_KNN_HPP

#include <algorithm>

#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/learning/discriminative/binary_classifier.hpp>
#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/discriminative.hpp>
#include <sill/learning/discriminative/tree_sampler.hpp>
#include <sill/math/matrix.hpp>
#include <sill/math/statistics.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

// Set to true to print debugging information.
#define DEBUG_PROTO_KNN 0

namespace sill {

  struct proto_knn_parameters {

    //! Number (> 0) of prototypes to pick
    //!  (default = 50)
    size_t n_proto;

    //! Number of candidate prototypes to consider when choosing each prototype
    //!  (default = 10)
    size_t n_cand;

    //! Used to make the algorithm deterministic
    //!  (default = time)
    double random_seed;

    proto_knn_parameters() : n_proto(50), n_cand(10) {
      std::time_t time_tmp;
      time(&time_tmp);
      random_seed = time_tmp;
    }

    bool valid() const {
      if (n_proto == 0)
        return false;
      if (n_cand == 0)
        return false;
      return true;
    }

    void save(std::ofstream& out) const {
      out << n_proto << " " << n_cand << " " << random_seed;
    }

    void load(std::ifstream& in) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> n_proto))
        assert(false);
      if (!(is >> n_cand))
        assert(false);
      if (!(is >> random_seed))
        assert(false);
    }

  }; // struct proto_knn_parameters

  /**
   * Class for learning a KNN classifier which uses a set of prototypes
   * instead of the full set of examples. This is based on the KNN weak
   * learner used by Freund and Schapire (1996).  It works as follows:
   *  - Let P be an initially empty set of prototype examples.
   *  - For k = 1,...,K
   *     - Select N_CAND examples at random.
   *     - Add the example which reduces the loss most to P.
   *  - Classify examples using KNN with the set P.
   * Note: This uses Euclidean distance between datapoints.  It is generally
   *       a good idea to normalize the data beforehand.
   *
   * @param Objective  class defining the optimization objective
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo serialization
   * @todo This should be easily extendible to multiclass data.
   * @todo Minor fix: If this is run on noisy data and chooses 2 prototypes
   *       with the same features but different labels, an arbitrary
   *       prototype will take precedence.  It should really choose randomly
   *       between the two.
   */
  template <typename Objective = discriminative::objective_accuracy>
  class proto_knn : public binary_classifier {

    concept_assert((sill::DomainPartitioningObjective<Objective>));

    // Protected data members
    //==========================================================================
  protected:

    // Data from base class:
    //  finite_variable* label_
    //  size_t label_index_

    typedef binary_classifier base;

    proto_knn_parameters params;

    //! random number generator
    boost::mt11213b rng;

    //! uniform distribution over [0, 1]
    boost::uniform_real<double> uniform_prob;

    //! datasource.finite_list()
    finite_var_vector finite_seq;

    //! datasource.vector_list()
    vector_var_vector vector_seq;

    //! Set of prototypes
    std::vector<record> prototypes;

    //! Prototype labels
    std::vector<size_t> prototype_labels;

    //! Confidences of predicting classes A, B.
    //! Note classes A may correspond to classes 0 or 1; ditto for B.
    double predictA, predictB;

    // Protected methods
    //==========================================================================

    //! Returns the Euclidean distance between two records.
    //! Finite values which differ have distance 1.
    //! This skips the class variable.
    double my_euclidean_distance(const record& r1, const record& r2) const {
      double d(0);
      const std::vector<size_t>& finite1 = r1.finite();
      const std::vector<size_t>& finite2 = r2.finite();
      const vec& vector1 = r1.vector();
      const vec& vector2 = r2.vector();
      if (finite1.size() != finite2.size() ||
          vector1.size() != vector2.size()) {
        std::cerr << "my_euclidean_distance() was called with non-matching"
                  << " (incomparable) records." << std::endl;
        assert(false);
        return - std::numeric_limits<double>::max();
      }
      for (size_t i = 0; i < finite1.size(); ++i)
        if (i != label_index_)
          d += (finite1[i] == finite2[i] ? 0 : 1);
      for (size_t i = 0; i < vector1.size(); ++i)
        d += (vector1[i] - vector2[i])*(vector1[i] - vector2[i]);
      if (d < 0) { return 0; } // for numerical precision (necessary?)
      return sqrt(d);
    }

    double my_euclidean_distance(const record& r1,const assignment& a) const {
      double d(0);
      const std::vector<size_t>& finite1 = r1.finite();
      const finite_assignment& finite2 = a.finite();
      const vec& vector1 = r1.vector();
      const vector_assignment& vector2 = a.vector();
      if (finite1.size() != finite2.size()) {
        std::cerr << "my_euclidean_distance() was called with non-matching"
                  << " (incomparable) records." << std::endl;
        assert(false);
        return - std::numeric_limits<double>::max();
      }
      for (size_t i = 0; i < finite1.size(); ++i)
        if (i != label_index_)
          d += (finite1[i] == safe_get(finite2, finite_seq[i]) ? 0 : 1);
      size_t i(0);
      for (size_t i2 = 0; i2 < vector2.size(); ++i2) {
        const vec& vec2 = safe_get(vector2, vector_seq[i2]);
        if (i + vec2.size() > vector1.size()) {
          std::cerr << "my_euclidean_distance() was called with non-matching"
                    << " (incomparable) records." << std::endl;
          assert(false);
          return - std::numeric_limits<double>::max();
        }
        for (size_t j = 0; j < vec2.size(); ++j) {
          d += (vector1[i] - vec2[j])*(vector1[i] - vec2[j]);
          ++i;
        }
      }
      if (d < 0) { return 0; } // for numerical precision (necessary?)
      return sqrt(d);
    }

    void init(const dataset& ds) {
      assert(params.valid());
      if (ds.size() < params.n_cand || ds.size() < params.n_proto) {
        std::cerr << "Dataset size for proto_knn is less than N_CAND or"
                  << " N_PROTO.  That is bad." << std::endl;
        assert(false);
        return;
      }
      rng.seed(static_cast<unsigned>(params.random_seed));
      uniform_prob = boost::uniform_real<double>(0,1);
    }

    //! Note: The choices of candidates and prototypes are made with
    //!       replacement, which is reasonable when the dataset size is
    //!       large enough.
    void build(const dataset& ds, const tree_sampler& sampler) {
      // CHOOSE AN INITIAL PROTOTYPE
      // We can choose the first prototype based on the max weight class,
      // so find weight of classes 0,1.
      std::vector<double> total_weights(2,0);
      for (size_t i = 0; i < ds.size(); i++)
        total_weights[label(ds,i)] += ds.weight(i);
      double total_weight = 0;
      foreach(double t, total_weights)
        total_weight += t;
      if (total_weight <= 0) {
        std::cerr << "proto_knn was given a dataset with all-zero weights."
                  << std::endl;
        assert(false);
        return;
      }
      std::vector<size_t> cs(params.n_cand); // candidate indices
      std::vector<double> c_objectives(params.n_cand); // objective values
      for (size_t i2 = 0; i2 < params.n_cand; ++i2) {
        cs[i2] = sampler.sample();
        const record& c = ds[cs[i2]];
        // Candidate c classifies all examples as its label.
        //  (So total_weights[label(c)] are correct.)
        c_objectives[i2] =
          Objective::objective(total_weights[label(c)],
                               total_weights[1 - label(c)],
                               0, 0);
      }
      size_t best_c(max_index(c_objectives, rng));
      prototypes.push_back(ds[cs[best_c]]);

      // MAINTAIN CURRENT PREDICTION on every record in ds
      // AND CURRENT DISTANCE from each record to the nearest prototype.
      size_t tmp_label(label(prototypes.back()));
      std::vector<size_t> cur_pred(ds.size(), tmp_label);
      std::vector<double> cur_dist(ds.size());
      for (size_t i = 0; i < ds.size(); ++i)
        cur_dist[i] = my_euclidean_distance(prototypes.back(), ds[i]);
      // contingency_table(i,j): i = truth, j = given label
      //  i.e., contingency_table(0,1) = false positives
      mat contingency_table(2,2);
      contingency_table(tmp_label, tmp_label) = total_weights[tmp_label];
      contingency_table(1-tmp_label, tmp_label) = total_weights[1-tmp_label];
      contingency_table(tmp_label, 1-tmp_label) = 0;
      contingency_table(1-tmp_label, 1-tmp_label) = 0;
      //      // Temp storage for candidates
      //      std::vector<size_t> new_pred(cur_pred);
      //      std::vector<double> new_dist(cur_dist);

      // CHOOSE THE REMAINING N_PROTO - 1 PROTOTYPES
      for (size_t k = 1; k < params.n_proto; ++k) {
        mat new_contingency_table;
        // Consider N_CAND possible prototypes
        for (size_t i2 = 0; i2 < params.n_cand; ++i2) {
          cs[i2] = sampler.sample();
          const record& c = ds[cs[i2]];
          tmp_label = label(c);
          new_contingency_table = contingency_table;
          // Compute objective after adding c to prototypes.
          // (Go through each datapoint, see if c is closer than the
          //  currently closest prototype, and adjust the counts if so.)
          for (size_t i = 0; i < ds.size(); ++i) {
            double d(my_euclidean_distance(c, ds[i]));
            if (d < cur_dist[i]) {
              //              new_dist[i] = d;
              if (tmp_label != cur_pred[i]) {
                size_t label_i(label(ds[i]));
                //                new_pred[i] = label_i;
                new_contingency_table(label_i, cur_pred[i]) -= ds.weight(i);
                new_contingency_table(label_i, tmp_label) += ds.weight(i);
              }
            }
          }
          c_objectives[i2] =
            Objective::objective(new_contingency_table(0,0),
                                 new_contingency_table(1,0),
                                 new_contingency_table(1,1),
                                 new_contingency_table(0,1));
        }
        // Take the best candidate
        best_c = max_index(c_objectives, rng);
        prototypes.push_back(ds[cs[best_c]]);
        // Update cur_pred, cur_dist, contingency_table
        // Note: These could be saved, but that would make this scale
        //       worse with N_CAND.
        const record& new_proto = prototypes.back();
        tmp_label = label(new_proto);
        for (size_t i = 0; i < ds.size(); ++i) {
          double d(my_euclidean_distance(new_proto, ds[i]));
          if (d < cur_dist[i]) {
            cur_dist[i] = d;
            if (tmp_label != cur_pred[i]) {
              size_t label_i(label(ds[i]));
              contingency_table(label_i, cur_pred[i]) -= ds.weight(i);
              contingency_table(label_i, tmp_label) += ds.weight(i);
              cur_pred[i] = tmp_label;
            }
          }
        }
      }

      // SET CONFIDENCE-RATED OUTPUTS (or normal ones if not confidence-rated)
      predictA = Objective::confidence(contingency_table(0,0),
                                       contingency_table(1,0));
      predictB = Objective::confidence(contingency_table(0,1),
                                       contingency_table(1,1));
      prototype_labels.resize(prototypes.size());
      for (size_t i = 0; i < prototypes.size(); ++i)
        prototype_labels[i] = label(prototypes[i]);
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
    //  From binary_classifier:
    //   create()*
    //   predict()*
    //   confidence()
    //   predict_raw()
    //   probability()

    // Constructors and destructors
    //==========================================================================

    //! Constructor for an empty proto_knn with the given parameters.
    explicit proto_knn(proto_knn_parameters params = proto_knn_parameters())
      : base(), params(params) { }

    /**
     * Constructor for a proto_knn classifier.
     * @param stats         a statistics class for the training dataset
     * @param parameters    algorithm parameters
     */
    explicit proto_knn(statistics& stats,
              proto_knn_parameters params = proto_knn_parameters())
      : base(stats.get_dataset()), params(params),
        finite_seq(stats.get_dataset().finite_list()),
        vector_seq(stats.get_dataset().vector_list()) {
      const dataset& ds = stats.get_dataset();
      init(ds);
      tree_sampler::parameters sampler_params;
      sampler_params.random_seed =
        boost::uniform_int<int>(0,std::numeric_limits<int>::max())(rng);
      tree_sampler sampler;
      if (ds.is_weighted())
        sampler = tree_sampler(ds.weights(), ds.size(), sampler_params);
      else
        sampler = tree_sampler(vec(ds.size(),1), ds.size(), sampler_params);

      build(ds, sampler);
    }

    /**
     * Constructor for a proto_knn classifier.
     * Note that this does not check to make sure the dataset distribution
     * matches that of the sampler; it only checks that they have the same
     * size.
     * @param stats         a statistics class for the training dataset
     * @param sampler       a tree sampler for the dataset distribution
     * @param parameters    algorithm parameters
     */
    proto_knn(statistics& stats, const tree_sampler& sampler,
              proto_knn_parameters params = proto_knn_parameters())
      : base(stats.get_dataset(), Objective::confidence_rated()), params(params),
        finite_seq(stats.get_dataset().finite_list()),
        vector_seq(stats.get_dataset().vector_list()) {
      const dataset& ds = stats.get_dataset();
      init(ds);
      if (ds.size() != sampler.distribution().size()) {
        std::cerr << "proto_knn constructor was passed a tree_sampler"
                  << " whose distribution did not match the given dataset."
                  << std::endl;
        assert(false);
        return;
      }
      build(ds, sampler);
    }

    /**
     * Constructor for a proto_knn classifier.
     * @param o    data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param parameters    algorithm parameters
     */
    proto_knn(oracle& o, size_t n,
              proto_knn_parameters params = proto_knn_parameters())
      : base(o), params(params),
        finite_seq(o.finite_list()), vector_seq(o.vector_list()) {
      boost::shared_ptr<vector_dataset>
        ds_ptr(oracle2dataset<vector_dataset>(o,n));
      init(*ds_ptr);
      tree_sampler::parameters sampler_params;
      sampler_params.random_seed
        = boost::uniform_int<int>(0,std::numeric_limits<int>::max())(rng);
      tree_sampler sampler(vec(ds_ptr->size(),1), ds_ptr->size(),
                           sampler_params);

      build(*ds_ptr, sampler);
    }

    //! Train a new binary classifier of this type with the given data.
    boost::shared_ptr<binary_classifier> create(statistics& stats) const {
      boost::shared_ptr<binary_classifier>
        bptr(new proto_knn<Objective>(stats, this->params));
      return bptr;
    }

    //! Train a new binary classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<binary_classifier> create(oracle& o, size_t n) const {
      boost::shared_ptr<binary_classifier>
        bptr(new proto_knn<Objective>(o, n, this->params));
      return bptr;
    }

    // Getters and helpers
    //==========================================================================

    //! Return a name for the algorithm without template parameters.
    std::string name() const { return "proto_knn"; }

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
      double min_dist(std::numeric_limits<double>::max());
      size_t min_i(0);
      for (size_t i = 0; i < prototypes.size(); ++i) {
        double temp_dist(my_euclidean_distance(prototypes[i], example));
        if (temp_dist < min_dist) {
          min_dist = temp_dist;
          min_i = i;
        }
      }
      if (prototype_labels[min_i] == 0)
        return predictA;
      else
        return predictB;
    }

    //! Value indicating the confidence of the prediction, with
    //!  predict() == (confidence() > 0) ? 1 : 0.
    //! If the classifier does not have actual confidence ratings,
    //!  then this should be any value with the correct sign.
    double confidence(const assignment& example) const {
      double min_dist(std::numeric_limits<double>::max());
      size_t min_i(0);
      for (size_t i = 0; i < prototypes.size(); ++i) {
        double temp_dist(my_euclidean_distance(prototypes[i], example));
        if (temp_dist < min_dist) {
          min_dist = temp_dist;
          min_i = i;
        }
      }
      if (prototype_labels[min_i] == 0)
        return predictA;
      else
        return predictB;
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
      std::cerr << "proto_knn::save() has not been implemented yet"
                << std::endl;
      assert(false);
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
      finite_seq = ds.finite_list();
      vector_seq = ds.vector_list();
      params.load(in);
      std::cerr << "proto_knn::load() has not been implemented yet"
                << std::endl;
      assert(false);
    }

  }; // class proto_knn

} // end of namespace: prl

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_PROTO_KNN_HPP
