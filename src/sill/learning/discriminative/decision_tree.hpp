
#ifndef SILL_LEARNING_DISCRIMINATIVE_DECISION_TREE_HPP
#define SILL_LEARNING_DISCRIMINATIVE_DECISION_TREE_HPP

#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/discriminative/stump.hpp>

#include <sill/macros_def.hpp>

// Set to true to print debugging information.
#define DEBUG_DECISION_TREE 0

namespace sill {

  struct decision_tree_parameters {

    //! Value used for smoothing confidences in stumps
    //!  (default = 1 / (2 * # training examples * # labels))
    double smoothing;

    //! Used to make the algorithm deterministic
    //!  (default = time)
    double random_seed;

    //! Percent of data to hold out for pruning the
    //! tree after building it using the rest of the data
    //!  (default = 1/3)
    double percent_prune;

    decision_tree_parameters() : smoothing(-1), percent_prune(1./3.) {
      std::time_t time_tmp;
      time(&time_tmp);
      random_seed = time_tmp;
    }

    bool valid() const {
      if (smoothing < 0)
        return false;
      if (percent_prune < 0 || percent_prune >= 1)
        return false;
      return true;
    }

    //! Sets smoothing to its default value for the given dataset info
    //! if it has not yet been set.
    void set_smoothing(size_t ntrain, size_t nlabels) {
      if (smoothing < 0)
        smoothing = 1. / (2. * ntrain * nlabels);
    }

    void save(std::ofstream& out) const {
      out << smoothing << " " << random_seed << " " << percent_prune << "\n";
    }

    void load(std::ifstream& in) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> smoothing))
        assert(false);
      if (!(is >> random_seed))
        assert(false);
      if (!(is >> percent_prune))
        assert(false);
    }

  }; // struct decision_tree_parameters

  /**
   * Decision tree for binary data.
   * This builds a tree using part of the data (greedily maximizing the given
   * criterion) and then prunes the tree using the remaining part of the data
   * (to maximize the criterion on the held-out data).
   *
   * @param Objective  class defining the optimization objective
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @see stump
   * @todo serialization
   * @todo This should be easily extendible to multiclass data.
   * @todo Would we ever want to templatize this and have stump replaceable by
   *       any binary classifier?
   */
  template <typename Objective = discriminative::objective_information>
  class decision_tree : public binary_classifier {

    concept_assert((sill::DomainPartitioningObjective<Objective>));

    // Protected data members
    //==========================================================================
  protected:

    // Data from base class:
    //  finite_variable* label_
    //  size_t label_index_

    typedef binary_classifier base;

    //! Type of decision stump used in decision tree nodes.
    typedef stump<Objective> stump_type;

    decision_tree_parameters params;

    //! random number generator
    boost::mt11213b rng;

    //! uniform distribution over [big negative int, big int]
    boost::uniform_int<> uniform_int;

    //! Training accuracy
    double train_acc;

    //! Class which represents a node of the tree.
    //! When a treenode is destructed, it destructs all of its descendants.
    //! When a treenode is copied, it does a deep copy.
    class treenode {
    public:

      //! Stump at this node
      stump<Objective> s;

      //! If childA == NULL, then this is a leaf.
      //! If stump returns 1, go to this child
      treenode* childA;

      //! If stump returns -1, go to this child
      treenode* childB;

      treenode() : childA(NULL), childB(NULL) { }

      explicit treenode(stump<Objective> s) : s(s), childA(NULL), childB(NULL) { }

      //! Does a deep copy
      treenode(const treenode& t)
        : s(s.t), childA(new treenode(*(s.childA))),
          childB(new treenode(*(s.childB))) {
      }

      ~treenode() {
        if (childA == NULL)
          delete(childA);
        if (childB == NULL)
          delete(childB);
      }

      void set_children(treenode* childA, treenode* childB) {
        assert(childA != NULL && childB != NULL);
        this->childA = childA; this->childB = childB;
      }

      double confidence(const record& example) const {
        if (childA == NULL)
          return s.confidence(example);
        else
          if (s.confidence(example) > 0)
            return childA->confidence(example);
          else
            return childB->confidence(example);
      }

      double confidence(const assignment& example) const {
        if (childA == NULL)
          return s.confidence(example);
        else
          if (s.confidence(example) > 0)
            return childA->confidence(example);
          else
            return childB->confidence(example);
      }

      void save(std::ofstream& out) const {
        s.save(out, false);
        if (childA != NULL && childB != NULL) {
          out << "P\n";
          childA->save(out);
          childB->save(out);
        } else
          out << "C\n";
      }

      void load(std::ifstream& in, const datasource& ds) {
        s.load(in, ds, false);
        std::string line;
        getline(in, line);
        if (line.compare("P") == 0) {
          childA = new treenode();
          childA->load(in, ds);
          childB = new treenode();
          childB->load(in, ds);
        } else {
          childA = NULL;
          childB = NULL;
        }
      }

    }; // class treenode

    treenode* root;

    // Protected methods
    //==========================================================================

    /**
     * Recursively build the decision tree.
     * @param test_results  <1 classified as 1, 1 as 0, 0 as 0, 0 as 1>
     * @return  pointer to root of this subtree
     */
    treenode*
    build_recursive(const dataset_view& train_view,
                    const dataset_view& test_view,
                    boost::tuple<size_t,size_t,size_t,size_t>& test_results) {
      // Train a stump at 'node' (which has not yet been allocated)
      stump_parameters stump_params;
      stump_params.smoothing = params.smoothing;
      stump_params.random_seed = uniform_int(rng);
      dataset_statistics stats(train_view);
      treenode* node = new treenode(stump_type(stats, stump_params));

      // Split the data into those labeled 0/1 by the stump.
      // Count the number indices0 which really have label 0 as cnt0.
      std::vector<size_t> indices0;
      std::vector<size_t> indices1;
      size_t cnt0 = 0, cnt1 = 1;
      for (size_t i = 0; i < train_view.size(); ++i)
        if (node->s.confidence(train_view[i]) > 0) {
          indices0.push_back(i);
          if (train_view[i].finite(label_index_) == 0)
            ++cnt0;
        } else {
          indices1.push_back(i);
          if (train_view[i].finite(label_index_) == 1)
            ++cnt1;
        }
      // Ditto for test points.
      std::vector<size_t> test_indices0;
      std::vector<size_t> test_indices1;
      size_t test_cnt0 = 0, test_cnt1 = 1;
      for (size_t i = 0; i < test_view.size(); ++i)
        if (node->s.confidence(test_view[i]) > 0) {
          test_indices0.push_back(i);
          if (test_view[i].finite(label_index_) == 0)
            ++test_cnt0;
        } else {
          test_indices1.push_back(i);
          if (test_view[i].finite(label_index_) == 1)
            ++test_cnt1;
        }
      // Compute objective at this level
      double objective_value(Objective::objective
                             (test_cnt1, test_indices1.size() - test_cnt1,
                              test_cnt0, test_indices0.size() - test_cnt0));
      test_results.get<0>() = test_cnt1;
      test_results.get<1>() = test_indices1.size() - test_cnt1;
      test_results.get<2>() = test_cnt0;
      test_results.get<3>() = test_indices0.size() - test_cnt0;
      // Recurse on node's children if each child will have training and test
      // datasets with both 0- and 1-labeled examples.
      if (indices0.size() > 0 && indices1.size() > 0 &&
          test_indices0.size() > 0 && test_indices1.size() > 0 &&
          cnt0 > 0 && cnt1 > 0 && test_cnt0 > 0 && test_cnt1 > 0 &&
          indices0.size() > cnt0 && indices1.size() > cnt1 &&
          test_indices0.size() > test_cnt0 &&
          test_indices1.size() > test_cnt1) {
        dataset_view train_view0(train_view);
        dataset_view test_view0(test_view);
        train_view0.set_record_indices(indices0);
        test_view0.set_record_indices(test_indices0);
        boost::tuple<size_t,size_t,size_t,size_t> test_results0;
        node->childB = build_recursive(train_view0, test_view0, test_results0);
        dataset_view train_view1(train_view);
        dataset_view test_view1(test_view);
        train_view1.set_record_indices(indices1);
        test_view1.set_record_indices(test_indices1);
        boost::tuple<size_t,size_t,size_t,size_t> test_results1;
        node->childA = build_recursive(train_view1, test_view1, test_results1);
        // Compute objective when using children
        double children_objective_value
          (Objective::objective
           (test_results0.get<0>() + test_results1.get<0>(),
            test_results0.get<1>() + test_results1.get<1>(),
            test_results0.get<2>() + test_results1.get<2>(),
            test_results0.get<3>() + test_results1.get<3>()));
        if (children_objective_value > objective_value) {
          test_results.get<0>() =
            test_results0.get<0>() + test_results1.get<0>();
          test_results.get<1>() =
            test_results0.get<1>() + test_results1.get<1>();
          test_results.get<2>() =
            test_results0.get<2>() + test_results1.get<2>();
          test_results.get<3>() =
            test_results0.get<3>() + test_results1.get<3>();
        } else {
          delete(node->childA);
          delete(node->childB);
          node->childA = NULL;
          node->childB = NULL;
        }
      }
      return node;
    }

    /**
     *
     */
    void build(dataset_statistics& stats) {
      params.set_smoothing(stats.get_dataset().size(), label_->size());
      assert(params.valid());
      rng.seed(static_cast<unsigned>(params.random_seed));
      uniform_int = boost::uniform_int<int>
        (-(std::numeric_limits<int>::max()-1),
         (std::numeric_limits<int>::max()-1));

      dataset_view train_view(stats.get_dataset());
      train_view.set_record_range
        (0, (size_t)floor((1-params.percent_prune) * stats.get_dataset().size()));
      dataset_view test_view(stats.get_dataset());
      test_view.set_record_range
        ((size_t)floor((1-params.percent_prune) * stats.get_dataset().size()),
         stats.get_dataset().size());
      boost::tuple<size_t,size_t,size_t,size_t> test_results;
      root = build_recursive(train_view, test_view, test_results);
      train_acc = (test_results.get<0>() + test_results.get<2>())
        / (test_results.get<0>() + test_results.get<1>() +
           test_results.get<2>() + test_results.get<3>());
    }

    // Constructors and destructors
    //==========================================================================
  public:

    //! Constructor for an empty decision tree with the given parameters.
    explicit decision_tree(decision_tree_parameters params
                           = decision_tree_parameters())
      : base(), params(params), root(NULL) { }

    /**
     * Constructor for a decision tree.
     * @param stats         a statistics class for the training dataset
     * @param parameters    algorithm parameters
     */
    explicit decision_tree(dataset_statistics& stats,
                           decision_tree_parameters params
                           = decision_tree_parameters())
      : base(stats.get_dataset()), params(params), root(NULL) {
      build(stats);
    }

    /**
     * Constructor for a decision tree.
     * @param o    data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param parameters    algorithm parameters
     */
    decision_tree(oracle& o, size_t n,
                  decision_tree_parameters params = decision_tree_parameters())
      : base(o), params(params), root(NULL) {
      boost::shared_ptr<vector_dataset>
        ds_ptr(oracle2dataset<vector_dataset>(o,n));
      dataset_statistics stats(*ds_ptr);
      build(stats);
    }

    ~decision_tree() {
      if (root != NULL)
        delete(root);
    }

    //! Train a new binary classifier of this type with the given data.
    boost::shared_ptr<binary_classifier> create(dataset_statistics& stats) const {
      boost::shared_ptr<binary_classifier>
        bptr(new decision_tree<Objective>(stats, this->params));
      return bptr;
    }

    //! Train a new binary classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<binary_classifier> create(oracle& o, size_t n) const {
      boost::shared_ptr<binary_classifier>
        bptr(new decision_tree<Objective>(o, n, this->params));
      return bptr;
    }

    // Getters and helpers
    //==========================================================================

    //! Return a name for the algorithm without template parameters.
    std::string name() const { return "decision_tree"; }

    //! Return a name for the algorithm with comma-separated template parameters
    //! (e.g., objective).
    std::string fullname() const {
      return name() + "<" + Objective::name() + ">";
    }

    //! Returns true iff learner is naturally an online learner.
    bool is_online() const { return false; }

    //! Indicates if the predictions are confidence-rated.
    //! Note that confidence rating may be optimized for different objectives.
    bool is_confidence_rated() const { return false; }

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
      if (root == NULL) return 0;
      return root->confidence(example);
    }

    //! Value indicating the confidence of the prediction, with
    //!  predict() == (confidence() > 0) ? 1 : 0.
    //! If the classifier does not have actual confidence ratings,
    //!  then this should be any value with the correct sign.
    double confidence(const assignment& example) const {
      if (root == NULL) return 0;
      return root->confidence(example);
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
      out << train_acc << "\n";
      if (root == NULL)
        out << "N\n";
      else {
        out << "R\n";
        root->save(out);
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
      params.load(in);
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> train_acc))
        assert(false);
      getline(in, line);
      if (line.compare("N") == 0)
        root = NULL;
      else if (line.compare("R") == 0) {
        root = new treenode();
        root->load(in, ds);
      } else {
        assert(false);
      }
      return true;
    }

  }; // class decision_tree

} // namespace sill

#undef DEBUG_DECISION_TREE

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_DECISION_TREE_HPP
