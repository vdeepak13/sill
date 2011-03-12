
#ifndef SILL_OBJECT_DETECTION_HAAR_HPP
#define SILL_OBJECT_DETECTION_HAAR_HPP

#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
//#include <sill/learning/discriminative/concepts.hpp>
#include <sill/learning/discriminative/binary_classifier.hpp>
#include <sill/learning/discriminative/discriminative.hpp>
#include <sill/learning/object_detection/image.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

// Set to true to print debugging information.
#define DEBUG_HAAR 0

namespace sill {

  struct haar_parameters {

    //! Use 2-rectangle features
    //!  (default = true)
    bool two_rectangle;

    //! Use 3-rectangle features
    //!  (default = true)
    bool three_rectangle;

    //! Use 4-rectangle features
    //!  (default = true) 
    bool four_rectangle;

    //! Value used for smoothing confidences
    //!  (default = 1 / (2 * # training examples * # labels))
    double smoothing;

    //!  (default = time)
    double random_seed;

    haar_parameters()
      : two_rectangle(true), three_rectangle(true), four_rectangle(true),
        smoothing(-1) {
      std::time_t time_tmp;
      time(&time_tmp);
      random_seed = time_tmp;
    }

    bool valid() const {
      if (smoothing < 0)
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
      out << two_rectangle << " " << three_rectangle << " "
          << four_rectangle << " " << smoothing << " " << random_seed
          << "\n";
    }

    void load(std::ifstream& in) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> two_rectangle))
        assert(false);
      if (!(is >> three_rectangle))
        assert(false);
      if (!(is >> four_rectangle))
        assert(false);
      if (!(is >> smoothing))
        assert(false);
      if (!(is >> random_seed))
        assert(false);
    }

  }; // struct haar_parameters

  /**
   * Base classifier for image data using haar-like features, as in
   * Viola & Jones' work on face detection.
   * Images must be in image record format and in integral form.
   *
   * This class also has functions useful for re-featurizing datasets.
   *
   * @param Objective  class defining the optimization objective
   * \author Joseph Bradley
   * @see image.hpp
   * @todo Change this to choose randomly to break ties between classifiers.
   */
  template <typename Objective = discriminative::objective_accuracy>
  class haar : public binary_classifier<> {

    //    concept_assert((sill::DomainPartitioningObjective<Objective>));

    // Public types
    //==========================================================================
  public:

    typedef binary_classifier<> base;

    typedef base::la_type la_type;
    typedef base::record_type record_type;

    /////////////////////// PROTECTED DATA AND METHODS ////////////////////

  protected:

    // Data from base class:
    //  finite_variable* label_
    //  size_t label_index_

    haar_parameters params;

    //! Height of image (view)
    size_t window_h;
    //! Width of image (view)
    size_t window_w;
    //! Info for feature chosen: list of coordinates of pixels
    //! (i,j) actually means (i-1,j-1); (0,*) and (*,0) are ignored
    std::vector<std::pair<size_t,size_t> > coordinates;
    //! Variables corresponding to the coordinates
    vector_var_vector coordinate_vars;
    //! Info for feature chosen: list of multipliers for pixel values
    std::vector<double> multipliers;
    //! Confidences of predicting classes A, B.
    //! Note classes A may correspond to classes 0 or 1; ditto for B.
    double predictA, predictB;

    //! Optimization objective value for training data
    double train_objective;
    //! Training accuracy
    double train_acc;

    //! Compute objective and predictions for given points (i*,j*), multipliers
    //! @param best              best <train_acc, objective, predictA, predictB>
    //!                          found so far
    //!                          (set by this function if a better set is found)
    //! @param best_coordinates  best coordinates found so far
    //!                          (set by this function if a better set is found)
    template <typename Distribution>
    void
    check(const dataset<la_type>& ds, const Distribution& distribution,
          const std::vector<std::pair<size_t,size_t> >& cur_coordinates,
          const std::vector<double>& cur_multipliers,
          boost::tuple<double, double, double, double>&
            best,
          std::vector<std::pair<size_t,size_t> >& best_coordinates) const {
      double cur_predictA = 0, cur_predictB = 0;
      double cur_train_acc = 0, cur_objective = 0;
      // posA = positive examples in set A (val > 0), etc.
      double posA = 0, posB = 0, negA = 0, negB = 0;
      size_t i = 0; // indexes ds/distribution
      typename dataset<la_type>::record_iterator end = ds.end();
      for (typename dataset<la_type>::record_iterator it = ds.begin();
           it != end; ++it) {
        double val = 0;
        for (size_t j = 0; j < cur_coordinates.size(); ++j)
          if (cur_coordinates[j].first > 0 && cur_coordinates[j].second > 0)
            val += cur_multipliers[j] *
              image::get_pixel(*it, cur_coordinates[j].first - 1,
                                    cur_coordinates[j].second - 1);
        if (val > 0)
          if (ds.finite(i,label_index_) == 0)
            negA += distribution[i];
          else
            posA += distribution[i];
        else
          if (ds.finite(i,label_index_) == 0)
            negB += distribution[i];
          else
            posB += distribution[i];
        ++i;
      }
      cur_predictA = Objective::confidence(negA,posA);
      cur_predictB = Objective::confidence(negB,posB);
      double total = posA + posB + negA + negB;
      if (cur_predictA > 0) { // if A means positive
        if (cur_predictB > 0) { // if B means positive
          cur_train_acc = (posA + posB) / total;
          cur_objective = Objective::objective(posA, negA, posB, negB);
        } else {
          cur_train_acc = (posA + negB) / total;
          cur_objective = Objective::objective(posA, negA, negB, posB);
        }
      } else {
        if (cur_predictB > 0) { // if B means positive
          cur_train_acc = (negA + posB) / total;
          cur_objective = Objective::objective(negA, posA, posB, negB);
        } else {
          cur_train_acc = (negA + negB) / total;
          cur_objective = Objective::objective(negA, posA, negB, posB);
        }
      }
      if (cur_objective > best.get<0>()) {
        best.get<0>() = cur_train_acc;
        best.get<1>() = cur_objective;
        best.get<2>() = cur_predictA;
        best.get<3>() = cur_predictB;
        best_coordinates = cur_coordinates;
      }
    }

    /**
     * Choose best 2-rectangle.
     * Vertices are ordered as:
     * A B C  or  A D
     * D E F      B E
     *            C F
     * @return <coordinates, multipliers, train_acc, objective,
     *          predictA, predictB>
     */
    template <typename Distribution>
    boost::tuple<std::vector<std::pair<size_t,size_t> >,
                 std::vector<double>,
                 double, double, double, double>
    choose2(const dataset<la_type>& ds, const Distribution& distribution) {
      std::vector<std::pair<size_t,size_t> > best_coordinates(6);
      std::vector<double> multiplier(6);
      multiplier[0] = -1;
      multiplier[1] = 2;
      multiplier[2] = -1;
      multiplier[3] = 1;
      multiplier[4] = -2;
      multiplier[5] = 1;
      boost::tuple<double, double, double, double> best;
      best.get<0>() = 0;
      best.get<1>() = - std::numeric_limits<double>::infinity();
      best.get<2>() = 0;
      best.get<3>() = 0;
      std::vector<std::pair<size_t,size_t> > cur_coordinates(6);
      // foreach (i1,:)
      for (size_t i1 = 0; i1 <= window_h; ++i1) {
        // foreach (i2,:) s.t. exists (i3,:)
        for (size_t i2 = i1+1; i2 < ceil((window_h + i1 + 1) / 2.); ++i2) {
          // i3 = i2 + (i2-i1)
          // foreach j1,j2
          for (size_t j1 = 0; j1 < window_w; ++j1) {
            cur_coordinates[0].first = i1;
            cur_coordinates[1].first = i2;
            cur_coordinates[2].first = 2 * i2 - i1;
            cur_coordinates[3].first = i1;
            cur_coordinates[4].first = i2;
            cur_coordinates[5].first = 2 * i2 - i1;
            cur_coordinates[0].second = j1;
            cur_coordinates[1].second = j1;
            cur_coordinates[2].second = j1;
            for (size_t j2 = j1+1; j2 <= window_w; ++j2) {
              cur_coordinates[3].second = j2;
              cur_coordinates[4].second = j2;
              cur_coordinates[5].second = j2;
              // 2-rectangle (i1 i2 i3, j1 j2)
              if (DEBUG_HAAR)
                std::cerr << cur_coordinates << std::endl;
              check(ds, distribution, cur_coordinates, multiplier, best,
                    best_coordinates);
            }
          }
          // foreach j1,j2,j3
          for (size_t j1 = 0; j1 < window_w-1; ++j1) {
            cur_coordinates[0].first = i1;
            cur_coordinates[1].first = i1;
            cur_coordinates[2].first = i1;
            cur_coordinates[3].first = i2;
            cur_coordinates[4].first = i2;
            cur_coordinates[5].first = i2;
            cur_coordinates[0].second = j1;
            cur_coordinates[3].second = j1;
            for (size_t j2 = j1+1; j2 < ceil((window_w + j1 + 1) / 2.); ++j2) {
              cur_coordinates[4].second = j2;
              cur_coordinates[5].second = 2 * j2 - j1;
              cur_coordinates[1].second = j2;
              cur_coordinates[2].second = 2 * j2 - j1;
              // j3 = j2 + (j2-j1)
              // 2-rectangle (i1 i2, j1 j2 j3)  // print coordinates to debug
              if (DEBUG_HAAR)
                std::cerr << cur_coordinates << std::endl;
              check(ds, distribution, cur_coordinates, multiplier, best,
                    best_coordinates);
            }
          }
        }
        // foreach (i2,:) s.t. does not exist (i3,:)
        for (size_t i2 = (size_t)(ceil((window_h + i1 + 1) / 2.));
             i2 <= window_h; ++i2)
          // foreach j1,j2,j3
          for (size_t j1 = 0; j1 < window_w-1; ++j1) {
            cur_coordinates[0].first = i1;
            cur_coordinates[1].first = i1;
            cur_coordinates[2].first = i1;
            cur_coordinates[3].first = i2;
            cur_coordinates[4].first = i2;
            cur_coordinates[5].first = i2;
            cur_coordinates[0].second = j1;
            cur_coordinates[3].second = j1;
            for (size_t j2 = j1+1; j2 < ceil((window_w + j1 + 1) / 2.); ++j2) {
              cur_coordinates[1].second = j2;
              cur_coordinates[2].second = 2 * j2 - j1;
              cur_coordinates[4].second = j2;
              cur_coordinates[5].second = 2 * j2 - j1;
              // j3 = j2 + (j2-j1)
              // 2-rectangle (i1 i2, j1 j2 j3)  // print coordinates to debug
              if (DEBUG_HAAR)
                std::cerr << cur_coordinates << std::endl;
              check(ds, distribution, cur_coordinates, multiplier, best,
                    best_coordinates);
            }
          }
      }
      return boost::make_tuple(best_coordinates, multiplier,
                               best.get<0>(), best.get<1>(),
                               best.get<2>(), best.get<3>());
    }

    /**
     * Choose best 3-rectangle.
     * Vertices are ordered as:
     * A B C D
     * E F G H
     * @return <coordinates, multipliers, train_acc, objective,
     *          predictA, predictB>
     */
    template <typename Distribution>
    boost::tuple<std::vector<std::pair<size_t,size_t> >,
                 std::vector<double>,
                 double, double, double, double>
    choose3(const dataset<la_type>& ds, const Distribution& distribution) {
      std::vector<std::pair<size_t,size_t> > best_coordinates(8);
      std::vector<double> multiplier(8);
      multiplier[0] = 1;
      multiplier[1] = -3;
      multiplier[2] = 3;
      multiplier[3] = -1;
      multiplier[4] = -1;
      multiplier[5] = 3;
      multiplier[6] = -3;
      multiplier[7] = 1;
      boost::tuple<double, double, double, double> best;
      best.get<0>() = 0;
      best.get<1>() = - std::numeric_limits<double>::infinity();
      best.get<2>() = 0;
      best.get<3>() = 0;
      std::vector<std::pair<size_t,size_t> > cur_coordinates(8);
      // foreach (j1,:)
      for (size_t j1 = 0; j1 < window_h; ++j1) {
        // foreach (j2,:) s.t. exists (j3,:), (j4,:)
        for (size_t j2 = j1+1; j2 < ceil((window_w + 2 * j1 + 1) / 3.); ++j2) {
          // j3 = j2 + (j2-j1)
          // j4 = j2 + 2 * (j2-j1)
          cur_coordinates[0].second = j1;
          cur_coordinates[1].second = j2;
          cur_coordinates[2].second = 2 * j2 - j1;
          cur_coordinates[3].second = 3 * j2 - 2 * j1;
          cur_coordinates[4].second = j1;
          cur_coordinates[5].second = j2;
          cur_coordinates[6].second = 2 * j2 - j1;
          cur_coordinates[7].second = 3 * j2 - 2 * j1;
          // foreach i1,i2
          for (size_t i1 = 0; i1 < window_h; ++i1) {
            cur_coordinates[0].first = i1;
            cur_coordinates[1].first = i1;
            cur_coordinates[2].first = i1;
            cur_coordinates[3].first = i1;
            for (size_t i2 = i1+1; i2 <= window_h; ++i2) {
              cur_coordinates[4].first = i2;
              cur_coordinates[5].first = i2;
              cur_coordinates[6].first = i2;
              cur_coordinates[7].first = i2;
              // 3-rectangle (i1 i2, j1 j2 j3 j4)
              if (DEBUG_HAAR)
                std::cerr << cur_coordinates << std::endl;
              check(ds, distribution, cur_coordinates, multiplier, best,
                    best_coordinates);
            }
          }
        }
      }
      return boost::make_tuple(best_coordinates, multiplier,
                               best.get<0>(), best.get<1>(),
                               best.get<2>(), best.get<3>());
    }

    /**
     * Choose best 4-rectangle.
     * Vertices are ordered as:
     * A B C
     * D E F
     * G H I
     * @return <coordinates, multipliers, train_acc, objective,
     *          predictA, predictB>
     */
    template <typename Distribution>
    boost::tuple<std::vector<std::pair<size_t,size_t> >,
                 std::vector<double>,
                 double, double, double, double>
    choose4(const dataset<la_type>& ds, const Distribution& distribution) {
      std::vector<std::pair<size_t,size_t> > best_coordinates(9);
      std::vector<double> multiplier(9);
      multiplier[0] = -1;
      multiplier[1] = 2;
      multiplier[2] = -1;
      multiplier[3] = 2;
      multiplier[4] = -4;
      multiplier[5] = 2;
      multiplier[6] = -1;
      multiplier[7] = 2;
      multiplier[8] = -1;
      boost::tuple<double, double, double, double> best;
      best.get<0>() = 0;
      best.get<1>() = - std::numeric_limits<double>::infinity();
      best.get<2>() = 0;
      best.get<3>() = 0;
      std::vector<std::pair<size_t,size_t> > cur_coordinates(9);
      // foreach (i1,:), (i2,:) s.t. exists (i3,:)
      for (size_t i1 = 0; i1 < window_h; ++i1) {
        for (size_t i2 = i1+1; i2 < ceil((window_h + i1 + 1) / 2.); ++i2) {
          // i3 = i2 + (i2-i1)
          cur_coordinates[0].first = i1;
          cur_coordinates[1].first = i2;
          cur_coordinates[2].first = 2 * i2 - i1;
          cur_coordinates[3].first = i1;
          cur_coordinates[4].first = i2;
          cur_coordinates[5].first = 2 * i2 - i1;
          cur_coordinates[6].first = i1;
          cur_coordinates[7].first = i2;
          cur_coordinates[8].first = 2 * i2 - i1;
          // foreach (j1,:), (j2,:) s.t. exists (j3,:)
          for (size_t j1 = 0; j1 < window_w; ++j1) {
            for (size_t j2 = j1+1; j2 < ceil((window_w + j1 + 1) / 2.); ++j2) {
              // j3 = j2 + (j2-j1)
              cur_coordinates[0].second = j1;
              cur_coordinates[1].second = j1;
              cur_coordinates[2].second = j1;
              cur_coordinates[3].second = j2;
              cur_coordinates[4].second = j2;
              cur_coordinates[5].second = j2;
              cur_coordinates[6].second = 2 * j2 - j1;
              cur_coordinates[7].second = 2 * j2 - j1;
              cur_coordinates[8].second = 2 * j2 - j1;
              // 4-rectangle (i1 i2 i3, j1 j2 j3)
              if (DEBUG_HAAR)
                std::cerr << cur_coordinates << std::endl;
              check(ds, distribution, cur_coordinates, multiplier, best,
                    best_coordinates);
            }
          }
        }
      }
      return boost::make_tuple(best_coordinates, multiplier,
                               best.get<0>(), best.get<1>(),
                               best.get<2>(), best.get<3>());
    }

    /**
     * Choose best Haar-like feature.
     */
    template <typename Distribution>
    void build(const dataset<la_type>& ds,
               const Distribution& distribution) {
      // TODO: move some of these things to a binary_classifier init()
      const vector_var_vector& vector_seq = ds.vector_list();
      assert(ds.size() > 0);
      window_h = image::height(ds[0]);
      window_w = image::width(ds[0]);
      params.set_smoothing(ds.size(), label_->size());
      assert(params.valid());

      boost::tuple<std::vector<std::pair<size_t,size_t> >,
                   std::vector<double>,
                   double, double, double, double> best;
      boost::tuple<std::vector<std::pair<size_t,size_t> >,
                   std::vector<double>,
                   double, double, double, double>
        best_tmp;
      best.get<3>() = - std::numeric_limits<double>::infinity();
      // TODO: optimize the choose*() functions
      if (params.two_rectangle) {
        best_tmp = choose2(ds, distribution);
        if (best_tmp.get<3>() > best.get<3>())
          best = best_tmp;
      }
      if (params.three_rectangle) {
        best_tmp = choose3(ds, distribution);
        if (best_tmp.get<3>() > best.get<3>())
          best = best_tmp;
      }
      if (params.four_rectangle) {
        best_tmp = choose4(ds, distribution);
        if (best_tmp.get<3>() > best.get<3>())
          best = best_tmp;
      }
      coordinates = best.get<0>();
      coordinate_vars.clear();
      if (vector_seq.size() == window_h * window_w + 1) {
        // Only set coordinate_vars if images are not scaled (so variables
        //  are meaningful).
        coordinate_vars.resize(coordinates.size());
        for (size_t j = 0; j < coordinates.size(); ++j)
          if (coordinates[j].first > 0 && coordinates[j].second > 0)
            coordinate_vars[j] =
              image::get_var(vector_seq, window_w,
                             coordinates[j].first - 1,
                             coordinates[j].second - 1);
          else
            coordinate_vars[j] = NULL;
      }
      multipliers = best.get<1>();
      predictA = best.get<4>();
      predictB = best.get<5>();
      train_objective = best.get<3>();
      train_acc = best.get<2>();
    }

    /////////// PUBLIC METHODS: BatchBinaryClassifier interface ////////////
    // Virtual methods from base classes (*means pure virtual):
    //  From learner:
    //   name()*
    //   is_online()*
    //   training_time()     x
    //   random_seed()       x
    //   save(), load()
    //  From classifier:
    //   is_confidence_rated()
    //   train_accuracy()
    //   set_confidences()
    //  From singlelabel_classifier:
    //   predict()*          x
    //  From binary_classifier:
    //   create()*           x
    //   confidence()        x
    //   predict_raw()
    //   probability()

  public:
    /**
     * Constructor for empty Haar-like feature; useful for:
     *  - creating other instances
     *  - loading a saved classifier
     * @param parameters    algorithm parameters
     */
    explicit haar(haar_parameters params = haar_parameters())
      : base(), params(params) {
    }
    /**
     * Constructor for Haar-like feature.
     * @param stats         a statistics class for the training dataset
     * @param parameters    algorithm parameters
     */
    explicit haar(dataset_statistics<la_type>& stats, haar_parameters params = haar_parameters())
      : base(stats.get_dataset()), params(params) {
      build(stats.get_dataset(), make_constant<double>(1));
    }
    /**
     * Constructor for Haar-like feature.
     * @param o    data oracle
     * @param n    max number of examples which should be drawn from the oracle
     * @param parameters    algorithm parameters
     */
    haar(oracle<la_type>& o, size_t n, haar_parameters params = haar_parameters())
      : base(o), params(params) {
      vector_dataset<la_type> ds;
      oracle2dataset(o, n, ds);
      build(ds, make_constant<double>(1));
    }

    //! Train a new binary classifier of this type with the given data.
    boost::shared_ptr<binary_classifier<> > create(dataset_statistics<la_type>& stats) const {
      boost::shared_ptr<binary_classifier<> >
        bptr(new haar<Objective>(stats, this->params));
      return bptr;
    }
    //! Train a new binary classifier of this type with the given data.
    //! @param n  max number of examples which should be drawn from the oracle
    boost::shared_ptr<binary_classifier<> > create(oracle<la_type>& o, size_t n) const {
      boost::shared_ptr<binary_classifier<> >
        bptr(new haar<Objective>(o, n, this->params));
      return bptr;
    }

    ////----------------- HELPER & INFO METHODS ----------------////

    //! Return a name for the algorithm without template parameters.
    std::string name() const { return "haar"; }
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

    ////----------------- LEARNING (MUTATING) METHODS ----------------////

    //! Reset the random seed in this algorithm's parameters and in its
    //! random number generator.
    void random_seed(double value) {
      params.random_seed = value;
    }

    ////----------------- PREDICTION METHODS ----------------////

    //! Predict the 0/1 label of a new example.
    std::size_t predict(const record_type& example) const {
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
    double confidence(const record_type& example) const {
      double s = 0;
      for (size_t i = 0; i < coordinates.size(); ++i)
        if (coordinates[i].first > 0 && coordinates[i].second > 0)
          s += multipliers[i] *
            image::get_pixel(example, coordinates[i].first - 1,
                                  coordinates[i].second - 1);
      if (s > 0)
        return predictA;
      else
        return predictB;
    }
    //! Value indicating the confidence of the prediction, with
    //!  predict() == (confidence() > 0) ? 1 : 0.
    //! If the classifier does not have actual confidence ratings,
    //!  then this should be any value with the correct sign.
    //! NOTE: Be careful using this since images care less about variables
    //!  than about variable orderings.
    double confidence(const assignment& example) const {
      if (coordinate_vars.size() == 0) {
        std::cerr << "haar<>::confidence(const assignment&) may not be "
                  << "called if haar was constructed using image views or "
                  << "scaled images since those render variables meaningless"
                  << std::endl;
      }
      double s = 0;
      const vector_assignment& va = example.vector();
      vector_domain vd(keys(va));
      for (size_t i = 0; i < coordinates.size(); ++i)
        if (coordinate_vars[i] != NULL) {
          if (!(vd.count(coordinate_vars[i]))) {
            assert(false);
          }
          s += multipliers[i] * (safe_get(va, coordinate_vars[i]))[0];
        }
      if (s > 0)
        return predictA;
      else
        return predictB;
    }

    ////----------------- SAVE & LOAD METHODS ----------------////

    using base::save;
    using base::load;

    //! Output the classifier to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    void save(std::ofstream& out, size_t save_part = 0,
              bool save_name = true) const {
      base::save(out, save_part, save_name);
      params.save(out);
      out << window_h << " " << window_w
          << " " << multipliers << " " << predictA << " " << predictB
          << " " << train_objective << " " << train_acc;
      for (size_t j = 0; j < coordinates.size(); ++j)
        out << " " << coordinates[j];
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
    bool load(std::ifstream& in, const datasource& ds, size_t load_part) {
      if (!(base::load(in, ds, load_part)))
        return false;
      params.load(in);
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> window_h))
        assert(false);
      if (!(is >> window_w))
        assert(false);
      read_vec(is, multipliers);
      if (!(is >> predictA))
        assert(false);
      if (!(is >> predictB))
        assert(false);
      if (!(is >> train_objective))
        assert(false);
      if (!(is >> train_acc))
        assert(false);
      coordinates.resize(multipliers.size());
      for (size_t j = 0; j < multipliers.size(); ++j)
        read_pair(is, coordinates[j]);
      coordinate_vars.clear();
      if (ds.num_vector() == window_h * window_w + 1) {
        coordinate_vars.resize(coordinates.size());
        for (size_t j = 0; j < coordinates.size(); ++j)
          if (coordinates[j].first > 0 && coordinates[j].second > 0)
            coordinate_vars[j] =
              image::get_var(ds.vector_list(), window_h,
                             coordinates[j].first - 1,
                             coordinates[j].second - 1);
          else
            coordinate_vars[j] = NULL;
      }
      return true;
    }

  };  // class haar

} // end of namespace: prl

#undef DEBUG_HAAR

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_OBJECT_DETECTION_HAAR_HPP
