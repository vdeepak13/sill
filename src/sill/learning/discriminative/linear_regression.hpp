
#ifndef SILL_LEARNING_DISCRIMINATIVE_LINEAR_REGRESSION_HPP
#define SILL_LEARNING_DISCRIMINATIVE_LINEAR_REGRESSION_HPP

#include <sill/base/variable_type_group.hpp>
#include <sill/datastructure/mutable_queue.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/record_conversions.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>
#include <sill/math/linear_algebra/armadillo.hpp>
#include <sill/math/statistics.hpp>
#include <sill/optimization/conjugate_gradient.hpp>
#include <sill/optimization/gradient_descent.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  struct linear_regression_parameters {

    //! Number of initial iterations to run (for iterative learning methods).
    //! This is ignored if the learning method is non-iterative.
    //!  (default = 100)
    size_t init_iterations;

    /**
     * Objective:
     *  - 2: least-squares (objective = (truth - prediction)^2)
     *     (default)
     */
    size_t objective;

    //! Regularization.
    //! 0: none
    //! 1: L_1
    //! 2: L_2
    //!  (default = L_2)
    size_t regularization;

    //! Regularization parameter, proportional to the number of pseudoexamples.
    //!  (default = .001)
    double lambda;

    //! If true, then regularize the mean as well (but this is not
    //! recommended).
    //!  (default = false);
    bool regularize_mean;

    /**
     * Cross validation score type (when doing cross-validation):
     *  - 0: Mean of CV score.
     *     (default)
     *  - 1: Median of CV score. (This is more robust than the mean.)
     *       (This uses the Median Absolute Deviation instead of the std error
     *       to measure variance.)
     */
    size_t cv_score_type;

    /**
     * When creating a parameter grid for choosing regularization via CV,
     * make the grid be on a log scale.
     *  (default = true)
     */
    bool cv_log_scale;

    /**
     * Optimization method:
     *  - 0: matrix inversion (only applicable for least squares with no
     *        regularization or with L-2 regularization)
     *  - 1: batch gradient descent with line search
     *  - 2: batch conjugate gradient with line search
     *     (default)
     */
    size_t opt_method;

    //! Range [-PERTURB_INIT,PERTURB_INIT] within
    //! which to choose perturbed values for initial parameters.
    //! Note: This should generally not be used with L1 regularization.
    //!  (default = 0)
    double perturb_init;

    //! Amount of change in average log likelihood
    //! below which algorithm will consider itself converged.
    //!  (default = .000001)
    double convergence_zero;

    //! (default = time)
    double random_seed;

    /**
     * Print debugging info.
     *  - 0: none (default)
     *  - 1: some
     *  - higher: reverts to highest level of debugging
     * Sets optimization debugging to level to debug-1.
     */
    size_t debug;

    linear_regression_parameters()
      : init_iterations(100), objective(2), regularization(2), lambda(.001),
        regularize_mean(false), cv_score_type(0), cv_log_scale(true),
        opt_method(2), perturb_init(0),
        convergence_zero(.000001), random_seed(time(NULL)), debug(0) { }

    bool valid() const {
      if (objective != 2)
        return false;
      if (regularization > 2)
        return false;
      if (lambda < 0)
        return false;
      if (cv_score_type > 1)
        return false;
      if (opt_method > 2)
        return false;
      if (opt_method == 0)
        if ((objective != 2) || (regularization != 0 && regularization != 2))
          return false;
      if (perturb_init < 0)
        return false;
      if (convergence_zero < 0)
        return false;
      return true;
    }

    void save(std::ofstream& out) const {
      out << init_iterations << " " << objective << " " << regularization << " "
          << lambda << " " << regularize_mean << " " << cv_score_type << " "
          << cv_log_scale << " "
          << opt_method << " " << perturb_init << " " << convergence_zero
          << " " << random_seed << " " << debug
          << "\n";
    }

    void load(std::ifstream& in) {
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> init_iterations))
        assert(false);
      if (!(is >> objective))
        assert(false);
      if (!(is >> regularization))
        assert(false);
      if (!(is >> lambda))
        assert(false);
      if (!(is >> regularize_mean))
        assert(false);
      if (!(is >> cv_score_type))
        assert(false);
      if (!(is >> cv_log_scale))
        assert(false);
      if (!(is >> opt_method))
        assert(false);
      if (!(is >> perturb_init))
        assert(false);
      if (!(is >> convergence_zero))
        assert(false);
      if (!(is >> random_seed))
        assert(false);
      if (!(is >> debug))
        assert(false);
    }

  }; // struct linear_regression_parameters

  std::ostream&
  operator<<(std::ostream& out, const linear_regression_parameters& params);

  /**
   * Class for learning a multiclass linear regression model y ~ Ax + b.
   * The parameters may be set to support different optimization methods
   * and different types of regularization; this defines different algorithms
   * such as:
   *  - Ridge regression: set objective = 2, regularization = 2
   *  - LASSO: set objective = 2, regularization = 1
   * This minimizes squared error + regularization.
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   *
   * @todo Make an interface for linear regression methods.
   * @todo Support finite-valued inputs x.
   * @todo Override copy constructor, assignment operator.
   * @todo Add L-BFGS support, and combine some of the pointers.
   */
  class linear_regression {

    // Public types
    //==========================================================================
  public:

    typedef dense_linear_algebra<> la_type;

    typedef record<la_type> record_type;

    //! Optimization variables (which fit the OptimizationVector concept).
    //! These are the regression weights: y ~ Ax + b
    struct opt_vector {

      // Types and data
      //------------------------------------------------------------------------

      struct size_type {
        //! Size of concatenated y values.
        size_t ysize;

        //! Size of concatenated x values.
        size_t xsize;

        size_type(size_t ysize, size_t xsize)
          : ysize(ysize), xsize(xsize) { }

        bool operator==(const size_type& other) const {
          return ((ysize == other.ysize) && (xsize == other.xsize));
        }

        bool operator!=(const size_type& other) const {
          return (!operator==(other));
        }
      };

      mat A; // |Y| x |X|

      vec b;

      // Constructors
      //------------------------------------------------------------------------

      opt_vector() { }

      opt_vector(size_type s, double default_val)
        : A(s.ysize, s.xsize), b(s.ysize) {
        A.fill(default_val);
        b.fill(default_val);
      }

      opt_vector(const mat& A, const vec& b) : A(A), b(b) { }

      // Getters and non-math setters
      //------------------------------------------------------------------------

      //! Returns true iff this instance equals the other.
      bool operator==(const opt_vector& other) const {
        return (equal(A, other.A) && equal(b, other.b));
      }

      //! Returns false iff this instance equals the other.
      bool operator!=(const opt_vector& other) const {
        return !operator==(other);
      }

      size_type size() const {
        return size_type(A.n_rows, A.n_cols);
      }

      //! Resize the data.
      void resize(const size_type& newsize) {
        A.set_size(newsize.ysize, newsize.xsize);
        b.set_size(newsize.ysize);
      }

      // Math operations
      //------------------------------------------------------------------------

      //! Sets all elements to this value.
      opt_vector& operator=(double d) {
        A = d;
        b = d;
        return *this;        
      }

      //! Addition.
      opt_vector operator+(const opt_vector& other) const {
        return opt_vector(A + other.A, b + other.b);
      }

      //! Addition.
      opt_vector& operator+=(const opt_vector& other) {
        A += other.A;
        b += other.b;
        return *this;
      }

      //! Subtraction.
      opt_vector operator-(const opt_vector& other) const {
        return opt_vector(A - other.A, b - other.b);
      }

      //! Subtraction.
      opt_vector& operator-=(const opt_vector& other) {
        A -= other.A;
        b -= other.b;
        return *this;
      }

      //! Multiplication by a scalar value.
      opt_vector operator*(double d) const {
        return opt_vector(A * d, b * d);
      }

      //! Multiplication by a scalar value.
      opt_vector& operator*=(double d) {
        A *= d;
        b *= d;
        return *this;
      }

      //! Division by a scalar value.
      opt_vector operator/(double d) const {
        if (d == 0)
          throw std::invalid_argument
            ("linear_regression::opt_vector divide by 0.");
        return opt_vector(A / d, b / d);
      }

      //! Division by a scalar value.
      opt_vector& operator/=(double d) {
        if (d == 0)
          throw std::invalid_argument
            ("linear_regression::opt_vector divide by 0.");
        A /= d;
        b /= d;
        return *this;
      }

      //! Inner product with a value of the same size.
      double dot(const opt_vector& other) const {
        return (sill::dot(A, other.A) + sill::dot(b, other.b));
      }

      //! Element-wise multiplication with another value of the same size.
      opt_vector& elem_mult(const opt_vector& other) {
        A %= other.A;
        b %= other.b;
        return *this;
      }

      //! Element-wise reciprocal (i.e., change v to 1/v).
      opt_vector& reciprocal() {
        for (size_t i(0); i < A.n_rows; ++i) {
          for (size_t j(0); j < A.n_cols; ++j) {
            double& val = A(i,j);
            assert(val != 0);
            val = 1. / val;
          }
        }
        for (size_t i(0); i < b.size(); ++i) {
          double& val = b[i];
          assert(val != 0);
          val = 1. / val;
        }
        return *this;
      }

      //! Returns the L1 norm.
      double L1norm() const {
        return norm(A, 1) + norm(b, 1);
      }

      //! Returns the L2 norm.
      double L2norm() const {
        return sqrt(dot(*this));
      }

      //! Returns a struct of the same size but with values replaced by their
      //! signs (-1 for negative, 0 for 0, 1 for positive).
      opt_vector sign() const {
        opt_vector ov(*this);
        for (size_t i(0); i < A.size(); ++i)
          ov.A(i) = (A(i) > 0 ? 1 : (A(i) == 0 ? 0 : -1) );
        foreach(double& val, ov.b)
          val = (val > 0 ? 1 : (val == 0 ? 0 : -1) );
        return ov;
      }

      //! Sets all values to 0.
      void zeros() {
        this->operator=(0.);
      }

      void print(std::ostream& out) const {
        out << " A:\n" << A << "\n"
            << " b:\n" << b << "\n";
      }

      //! Print info about this vector (for debugging).
      void print_info(std::ostream& out) const {
        out << "A.size: [" << A.n_rows << ", " << A.n_cols << "], "
            << "b.size: " << b.size() << "\n";
      }

    }; // struct opt_vector

    // Protected types
    //==========================================================================
  protected:

    //! Objective functor usable with optimization routines.
    class objective_functor {

      const linear_regression& lr;

      mutable vec yvec;

      mutable mat tmpmat;

    public:
      objective_functor(const linear_regression& lr)
        : lr(lr), yvec(zeros<vec>(lr.Yvec_size)),
          tmpmat(zeros<mat>(lr.Ydata().n_rows, lr.Ydata().n_cols)) { }

      //! Computes the value of the objective at x.
      double objective(const opt_vector& x) const {
        double obj(0);

        switch(lr.params.objective) {
        case 2: // least-squares
          tmpmat = lr.Ydata();
          tmpmat -= lr.Xdata() * trans(x.A);
          for (size_t i(0); i < tmpmat.n_rows; ++i)
            tmpmat.row(i) -= x.b;
//            tmpmat.subtract_row(i, x.b);
          tmpmat %= tmpmat;
          if (lr.data_weights.size() == 0)
            obj += accu(tmpmat);
          else
            obj += dot(lr.data_weights,
                       tmpmat * ones<vec>(lr.data_weights.size()));
          break;
        default:
          throw std::invalid_argument
            ("linear_regression was given a bad objective parameter");
        }

        switch(lr.params.regularization) {
        case 0:
          break;
        case 1:
          obj += lr.params.lambda * norm(x.A,1);
          if (lr.params.regularize_mean)
            obj += lr.params.lambda * norm(x.b,1);
          break;
        case 2:
          obj += lr.params.lambda * .5 * sqr(norm(x.A,2));
          if (lr.params.regularize_mean)
            obj += lr.params.lambda * .5 * sqr(norm(x.b,2));
          break;
        default:
          assert(false);
        }
        obj /= lr.total_train_weight;

        return obj;
      } // objective()

    }; // class objective_functor

    //! Gradient functor usage with optimization routines.
    struct lr_gradient_functor {

      lr_gradient_functor(const linear_regression& lr)
        : lr(lr), yvec(zeros<vec>(lr.Yvec_size)),
          tmpmat(zeros<mat>(lr.Ydata().n_rows, lr.Ydata().n_cols)) { }

      //! Computes the gradient of the function at x, multiplies it by w,
      //! and adds it to grad.
      //! @param grad  The gradient is added to this OptVector.
      void add_gradient(opt_vector& grad, const opt_vector& x, double w) const {

        w /= lr.total_train_weight;

        switch(lr.params.objective) {
        case 2: // least-squares
          tmpmat = lr.Xdata() * trans(x.A);
          tmpmat -= lr.Ydata();
          if (lr.data_weights.size() == 0) {
            for (size_t i(0); i < tmpmat.n_rows; ++i) {
              tmpmat.row(i) += x.b;
//              tmpmat.add_row(i, x.b);
              grad.A += 2 * w * tmpmat.row(i) * trans(lr.Xdata().row(i));
//              grad.A += 2 * outer_product(tmpmat.row(i), lr.Xdata().row(i));
            }
            grad.b -= 2 * w * sum(tmpmat, 1);
          } else {
            for (size_t i(0); i < tmpmat.n_rows; ++i) {
              tmpmat.row(i) += x.b;
//              tmpmat.add_row(i, x.b);
              grad.A += (2 * w * lr.data_weights[i])
                * tmpmat.row(i) * trans(lr.Xdata().row(i));
//                * outer_product(tmpmat.row(i), lr.Xdata().row(i));
            }
            grad.b -= 2 * w * trans(tmpmat) * lr.data_weights;
          }
          break;
        default:
          throw std::invalid_argument
            ("linear_regression was given a bad objective parameter");
        }

        w *= lr.params.lambda;
        switch(lr.params.regularization) {
        case 0:
          break;
        case 1:
          for (size_t i(0); i < x.b.size(); ++i) {
            for (size_t j(0); j < x.A.n_cols; ++j) {
              if (x.A(i,j) > 0)
                grad.A(i,j) += w;
              else if (x.A(i,j) > 0)
                grad.A(i,j) -= w;
            }
          }
          if (lr.params.regularize_mean) {
            for (size_t i(0); i < x.b.size(); ++i) {
              if (x.b[i] > 0)
                grad.b[i] += w;
              else if (x.b[i] < 0)
                grad.b[i] -= w;
            }
          }
          break;
        case 2:
          grad.A += w * x.A;
          if (lr.params.regularize_mean)
            grad.b += w * x.b;
          break;
        default:
          assert(false);
        }
      } // gradient()

      //! Computes the gradient of the function at x.
      //! @param grad  Location in which to store the gradient.
      void gradient(opt_vector& grad, const opt_vector& x) const {
        grad.zeros();
        add_gradient(grad, x, 1);
      }

    private:
      const linear_regression& lr;
      mutable vec yvec;
      mutable mat tmpmat;

    }; // class lr_gradient_functor

    // Protected data members
    //==========================================================================

    linear_regression_parameters params;

    mutable boost::mt11213b rng;

    //! Output variables Y.
    vector_var_vector Yvec;

    //! Total size of Yvec.
    size_t Yvec_size;

    //! Input variables X.
    vector_var_vector Xvec;

    //! Total size of Xvec.
    size_t Xvec_size;

    //! vector_offset[j] = first index in weights for vector variable Y_j
    //!  (so vector_offset[j] + k is the index for element k of variable Y_j)
//    std::vector<size_t> Yvec_offset;

    //! Y training data: nrecords x |Y|; this is owned by this class.
    mat Ydata_own;

    //! X training data: nrecords x |X|; this is owned by this class.
    mat Xdata_own;

    //! Y training data: nrecords x |Y|; this is external data.
    const mat* Ydata_ptr;

    //! X training data: nrecords x |X|; this is external data.
    const mat* Xdata_ptr;

    //! Training data weights
    vec data_weights;

    //! Weights A,b defining this regressor: y ~ Ax + b
    opt_vector weights_;

    //! For all generic optimization methods
    objective_functor* obj_functor_ptr;

    //! For all generic optimization methods
    lr_gradient_functor* grad_functor_ptr;

    //! For batch gradient descent (stores weights)
    gradient_descent<opt_vector, objective_functor, lr_gradient_functor>*
    gradient_descent_ptr;

    //! For batch conjugate gradient (stores weights)
    conjugate_gradient<opt_vector, objective_functor, lr_gradient_functor>*
    conjugate_gradient_ptr;

    //! Current iteration; i.e., # iterations completed
    size_t iteration_;

    //! Total weight of training examples
    double total_train_weight;

    //! Current objective, averaged over training data.
    double train_obj;

    //! Vector of size x.
    mutable vec tmpx;

    // Protected methods
    //==========================================================================

    //! Return a const reference to the Y data matrix.
    const mat& Ydata() const;

    //! Return a const reference to the X data matrix.
    const mat& Xdata() const;

    //! Do learning via matrix inversion, without regularization for the mean.
    void train_matrix_inversion();

    //! Do learning via matrix inversion, with regularization for the mean.
    void train_matrix_inversion_with_mean();

    //! Learn stuff.
    //! @param  own_data  Set to false if using outside Ydata,Xdata.
    void init(const dataset<la_type>& ds, bool own_data);

    //! Take one step of gradient descent.
    //! @return  true iff learner may be trained further (and has not converged)
    bool step_gradient_descent();

    //! Take one step of conjugate gradient.
    //! @return  true iff learner may be trained further (and has not converged)
    bool step_conjugate_gradient();

    //! Constructor used by choose_lambda_ridge().
    linear_regression
    (const vector_var_vector& Yvec, const vector_var_vector& Xvec)
      : Yvec(Yvec), Yvec_size(vector_size(Yvec)),
        Xvec(Xvec), Xvec_size(vector_size(Xvec)),
        Ydata_ptr(NULL), Xdata_ptr(NULL),
        obj_functor_ptr(NULL), grad_functor_ptr(NULL),
        gradient_descent_ptr(NULL), conjugate_gradient_ptr(NULL),
        tmpx(Xvec_size) {
    }

    // Constructors and destructors
    //==========================================================================
  public:

    /**
     * Constructor for learning Y ~ X.  Y and X are specified by the dataset,
     * with Y being the vector class variables and X being all other vector
     * variables.
     * @param parameters    algorithm parameters
     */
    explicit linear_regression
    (const dataset<la_type>& ds,
     linear_regression_parameters params = linear_regression_parameters())
      : params(params),
        Yvec(ds.vector_class_variables()), Yvec_size(vector_size(Yvec)),
        Ydata_ptr(NULL), Xdata_ptr(NULL),
        obj_functor_ptr(NULL), grad_functor_ptr(NULL),
        gradient_descent_ptr(NULL), conjugate_gradient_ptr(NULL),
        iteration_(0), total_train_weight(0),
        train_obj(std::numeric_limits<double>::max()) {
      vector_domain tmpYdomain(Yvec.begin(), Yvec.end());
      foreach(vector_variable* v, ds.vector_list()) {
        if (tmpYdomain.count(v) == 0)
          Xvec.push_back(v);
      }
      Xvec_size = vector_size(Xvec);
      init(ds, true);
    }

    /**
     * Constructor which helps avoid copying data.
     * The data matrices may be constructed via dataset::get_value_matrix().
     * NOTE: If the learning method is matrix inversion with a regularized
     *       mean, then it is more efficient if Xdata has a vector of 1's
     *       pre-appended to it.
     * @param Ydata   matrix of Y (output) values corresponding to dataset ds
     * @param Xdata   matrix of X (input) values corresponding to dataset ds
     * @param parameters    algorithm parameters
     */
    linear_regression
    (const dataset<la_type>& ds, const mat& Ydata, const mat& Xdata,
     linear_regression_parameters params = linear_regression_parameters())
      : params(params),
        Yvec(ds.vector_class_variables()), Yvec_size(vector_size(Yvec)),
        Ydata_ptr(&Ydata), Xdata_ptr(&Xdata),
        obj_functor_ptr(NULL), grad_functor_ptr(NULL),
        gradient_descent_ptr(NULL), conjugate_gradient_ptr(NULL),
        iteration_(0), total_train_weight(0),
        train_obj(std::numeric_limits<double>::max()) {
      vector_domain tmpYdomain(Yvec.begin(), Yvec.end());
      foreach(vector_variable* v, ds.vector_list()) {
        if (tmpYdomain.count(v) == 0)
          Xvec.push_back(v);
      }
      Xvec_size = vector_size(Xvec);
      init(ds, false);
    }

    /**
     * Constructor which takes explicit Y, X to learn Y ~ X.
     * @param parameters    algorithm parameters
     */
    linear_regression
    (const dataset<la_type>& ds, const vector_var_vector& Yvec,
     const vector_var_vector& Xvec,
     linear_regression_parameters params = linear_regression_parameters())
      : params(params), Yvec(Yvec), Yvec_size(vector_size(Yvec)),
        Xvec(Xvec), Xvec_size(vector_size(Xvec)),
        Ydata_ptr(NULL), Xdata_ptr(NULL),
        obj_functor_ptr(NULL), grad_functor_ptr(NULL),
        gradient_descent_ptr(NULL), conjugate_gradient_ptr(NULL),
        iteration_(0), total_train_weight(0),
        train_obj(std::numeric_limits<double>::max()) {
      vector_domain tmpdom(Yvec.begin(), Yvec.end());
      tmpdom.insert(Xvec.begin(), Xvec.end());
      assert(ds.has_variables(tmpdom));
      assert(tmpdom.size() == Yvec.size() + Xvec.size()); // make sure disjoint
      init(ds, true);
    }

    /**
     * Constructor which both
     *  - takes explicit Y, X to learn Y ~ X and
     *  - helps avoid copying data.
     * The data matrices may be constructed via dataset::get_value_matrix().
     * NOTE: If the learning method is matrix inversion with a regularized
     *       mean, then it is more efficient if Xdata has a vector of 1's
     *       pre-appended to it.
     * @param Ydata   matrix of Y (output) values corresponding to dataset ds
     * @param Xdata   matrix of X (input) values corresponding to dataset ds
     * @param parameters    algorithm parameters
     */
    explicit linear_regression
    (const dataset<la_type>& ds, const vector_var_vector& Yvec,
     const vector_var_vector& Xvec, const mat& Ydata, const mat& Xdata,
     linear_regression_parameters params = linear_regression_parameters())
      : params(params), Yvec(Yvec), Yvec_size(vector_size(Yvec)),
        Xvec(Xvec), Xvec_size(vector_size(Xvec)),
        Ydata_ptr(&Ydata), Xdata_ptr(&Xdata),
        obj_functor_ptr(NULL), grad_functor_ptr(NULL),
        gradient_descent_ptr(NULL), conjugate_gradient_ptr(NULL),
        iteration_(0), total_train_weight(0),
        train_obj(std::numeric_limits<double>::max()) {
      vector_domain tmpdom(Yvec.begin(), Yvec.end());
      tmpdom.insert(Xvec.begin(), Xvec.end());
      assert(ds.has_variables(tmpdom));
      assert(tmpdom.size() == Yvec.size() + Xvec.size());
      init(ds, false);
    }

    //! Copy constructor.
    linear_regression(const linear_regression& lr)
      : params(lr.params), rng(lr.rng), Yvec(lr.Yvec), Yvec_size(lr.Yvec_size),
        Xvec(lr.Xvec), Xvec_size(lr.Xvec_size),
        Ydata_own(lr.Ydata_own), Xdata_own(lr.Xdata_own),
        Ydata_ptr(lr.Ydata_ptr), Xdata_ptr(lr.Xdata_ptr),
        data_weights(lr.data_weights),
        weights_(lr.weights_), obj_functor_ptr(NULL), grad_functor_ptr(NULL),
        gradient_descent_ptr(NULL), conjugate_gradient_ptr(NULL),
        iteration_(lr.iteration_), total_train_weight(lr.total_train_weight),
        train_obj(lr.train_obj), tmpx(lr.tmpx) {
      if (lr.obj_functor_ptr)
        obj_functor_ptr = new objective_functor(*(lr.obj_functor_ptr));
      if (lr.grad_functor_ptr)
        grad_functor_ptr = new lr_gradient_functor(*(lr.grad_functor_ptr));
      if (lr.gradient_descent_ptr)
        gradient_descent_ptr =
          new gradient_descent<opt_vector, objective_functor, lr_gradient_functor>
          (*(lr.gradient_descent_ptr));
      if (lr.conjugate_gradient_ptr)
        conjugate_gradient_ptr =
          new conjugate_gradient<opt_vector, objective_functor,lr_gradient_functor>
          (*(lr.conjugate_gradient_ptr));
    }

    ~linear_regression() {
      if (obj_functor_ptr)
        delete(obj_functor_ptr);
      if (grad_functor_ptr)
        delete(grad_functor_ptr);
      if (gradient_descent_ptr)
        delete(gradient_descent_ptr);
      if (conjugate_gradient_ptr)
        delete(conjugate_gradient_ptr);
    }

    //! Assignment operator.
    linear_regression& operator=(const linear_regression& lr) {
      params = lr.params;
      rng = lr.rng;
      Yvec = lr.Yvec;
      Yvec_size = lr.Yvec_size;
      Xvec = lr.Xvec;
      Xvec_size = lr.Xvec_size;
      Ydata_own = lr.Ydata_own;
      Xdata_own = lr.Xdata_own;
      Ydata_ptr = lr.Ydata_ptr;
      Xdata_ptr = lr.Xdata_ptr;
      data_weights = lr.data_weights;
      weights_ = lr.weights_;
      if (obj_functor_ptr) {
        delete(obj_functor_ptr);
        obj_functor_ptr = NULL;
      }
      if (lr.obj_functor_ptr)
        obj_functor_ptr = new objective_functor(*(lr.obj_functor_ptr));
      if (grad_functor_ptr) {
        delete(grad_functor_ptr);
        grad_functor_ptr = NULL;
      }
      if (lr.grad_functor_ptr)
        grad_functor_ptr = new lr_gradient_functor(*(lr.grad_functor_ptr));
      if (gradient_descent_ptr) {
        delete(gradient_descent_ptr);
        gradient_descent_ptr = NULL;
      }
      if (lr.gradient_descent_ptr)
        gradient_descent_ptr =
          new gradient_descent<opt_vector, objective_functor, lr_gradient_functor>
          (*(lr.gradient_descent_ptr));
      if (conjugate_gradient_ptr) {
        delete(conjugate_gradient_ptr);
        conjugate_gradient_ptr = NULL;
      }
      if (lr.conjugate_gradient_ptr)
        conjugate_gradient_ptr =
          new conjugate_gradient<opt_vector, objective_functor,lr_gradient_functor>
          (*(lr.conjugate_gradient_ptr));
      iteration_ = lr.iteration_;
      total_train_weight = lr.total_train_weight;
      train_obj = lr.train_obj;
      tmpx = lr.tmpx;
      return *this;
    } // end of operator=()

    // Getters and helpers
    //==========================================================================

    //! Output variables Y.
    const vector_var_vector& Yvector() const {
      return Yvec;
    }

    //! Input variables X.
    const vector_var_vector& Xvector() const {
      return Xvec;
    }

    //! Returns A,b, which define the regressor.
    const opt_vector& weights() const {
      return weights_;
    }

    //! Return a name for the algorithm without template parameters.
    std::string name() const {
      return "linear_regression";
    }

    //! Return a name for the algorithm with comma-separated template parameters
    //! (e.g., objective).
    std::string fullname() const {
      return name();
    }

    //! Returns true iff learner is naturally an online learner.
    bool is_online() const {
      switch(params.opt_method) {
      case 0:
        return false;
      case 1:
      case 2:
        return true;
      default:
        assert(false);
        return false;
      }
    }

    /*
    operator std::string() const {
      std::ostringstream out; out << *this; return out.str();
    }
    */

    //! Print classifier
    void print(std::ostream& out) const {
      out << "Linear Regression\n"
          << "  Y: " << Yvec << "\n"
          << "  X: " << Xvec << "\n"
          << "  weights:\n";
      weights_.print(out);
    }

    //! Returns the current training objective.
    double training_objective() const {
      return train_obj;
    }

    /**
     * Returns the set of input (non-class) variables which have non-zero
     * weights in the learned regressor.
     *
     * @tparam VariableType   This only returns inputs of this variable type.
     *
     * @todo Update this when this class supports finite variables.
     */
    template <typename VariableType>
    typename variable_type_group<VariableType>::domain_type
    get_dependencies() const;

    /**
     * This returns, in order of their weights in the learned regressor,
     * the top K vector input (non-class) variables, plus their weights
     * (the weight for input x being the sum of the weights in the A matrix
     * which are multiplied by x (1 for each output y) ).
     * If K = 0, then this returns all vector inputs.
     *
     * @tparam VariableType   This only returns inputs of this variable type.
     */
    template <typename VariableType>
    std::vector<std::pair<VariableType*, double> >
    get_dependencies(size_t K) const;

    /**
     * This returns the set of vector input (non-class) variables which have
     * non-zero weights in the learned regressor.
     */
    vector_domain get_vector_dependencies() const {
      vector_domain x;
      for (size_t i(0); i < Xvec.size(); ++i) {
        if (norm(weights_.A.col(i),1) > params.convergence_zero)
          x.insert(Xvec[i]);
      }
      return x;
    }

    /**
     * This returns, in order of their weights in the learned regressor,
     * the top K vector input (non-class) variables, plus their weights
     * (the weight for input x being the sum of the weights in the A matrix
     * which are multiplied by x (1 for each output y) ).
     * If K = 0, then this returns all vector inputs.
     */
    std::vector<std::pair<vector_variable*, double> >
    get_vector_dependencies(size_t K) const {
      if (K == 0)
        K = std::numeric_limits<size_t>::max();
      mutable_queue<vector_variable*, double> vqueue;
      for (size_t i(0); i < Xvec.size(); ++i) {
        vqueue.push(Xvec[i], norm(weights_.A.col(i),1));
      }
      std::vector<std::pair<vector_variable*, double> > x;
      while ((x.size() < K) && (vqueue.size() > 0)) {
        x.push_back(vqueue.top());
        vqueue.pop();
      }
      return x;
    }

    // Learning and mutating operations
    //==========================================================================

    //! Reset the random seed in this algorithm's parameters and in its
    //! random number generator.
    void random_seed(double value) {
      params.random_seed = value;
      rng.seed(static_cast<unsigned>(value));
    }

    // Prediction methods
    //==========================================================================

    //! Predict the output values y for a new example x.
    //! The values are in the order returned by Yvector().
    vec predict(const record_type& example) const {
//      vector_record2vector(example, Xvec, tmpx);
      example.vector_values(tmpx, Xvec);
      return (weights_.A * tmpx + weights_.b);
    }

    //! Predict the output values y for a new example x.
    //! The values are in the order returned by Yvector().
    vec predict(const assignment& example) const {
      vector_assignment2vector(example, Xvec, tmpx);
      return (weights_.A * tmpx + weights_.b);
    }

    //! Compute the mean squared error on the given data.
    //! @return <mean, std error>
    std::pair<double, double> mean_squared_error(const dataset<la_type>& testds) const {
      double s(0);
      double s2(0);
      double totalw(0);
      size_t i(0);
      vec tmpy(Yvec_size);
      tmpy.fill(0.);
      foreach(const record_type& r, testds.records()) {
//        vector_record2vector(r, Yvec, tmpy);
        r.vector_values(tmpy, Yvec);
        tmpy -= predict(r);
        double tmpval(testds.weight(i) * dot(tmpy, tmpy));
        totalw += testds.weight(i);
        s += tmpval;
        s2 += tmpval * tmpval;
        ++i;
      }
      if (totalw == 0)
        return std::make_pair(0.,0.);
      s /= totalw;
      s2 /= totalw;
      s2 = sqrt(s2 - s * s) / sqrt(totalw);
      return std::make_pair(s, s2);
    }

    /**
     * Compute the median squared error on the given data,
     * as well as the Median Absolute Deviation (MAD) of the squared errors.
     * The median and MAD are more robust estimates than the mean and standard
     * error.
     * @return <median squared error, MAD>
     * @todo Compute medians more efficiently!
     * @todo Support weighted datasets!
     */
    std::pair<double, double>
    median_squared_error(const dataset<la_type>& testds) const {
      assert(!testds.is_weighted());
      if (testds.size() == 0)
        return std::make_pair(0., 0.);
      std::vector<double> errors(testds.size(), 0.);
      size_t i(0);
      vec tmpy(Yvec_size);
      tmpy.fill(0.);
      foreach(const record_type& r, testds.records()) {
//        vector_record2vector(r, Yvec, tmpy);
        r.vector_values(tmpy, Yvec);
        tmpy -= predict(r);
        double tmpval(dot(tmpy, tmpy));
        errors[i] = tmpval;
        ++i;
      }
      return median_MAD(errors);
    }

    // Methods for iterative learners
    //==========================================================================

    //! Returns the current iteration number (from 0)
    //!  (i.e., the number of learning iterations completed).
    //! ITERATIVE ONLY: This must be implemented by iterative learners.
    size_t iteration() const {
      return iteration_;
    }

    /*
    //! Returns the total time elapsed after each iteration.
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    std::vector<double> elapsed_times() const {
    }
    */

    //! Does the next step of training (updating the current classifier).
    //! @return true iff the learner may be trained further
    //! ITERATIVE ONLY: This must be implemented by iterative learners.
    bool step();

    // Save and load methods
    //==========================================================================

    //    using base::save;
    //    using base::load;

    //! Output the classifier to a human-readable file which can be reloaded.
    //! @param save_part  0: save function (default), 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    //    void save(std::ofstream& out, size_t save_part = 0,
    //              bool save_name = true) const;

    /*
     * Input the classifier from a human-readable file.
     * @param in          input filestream for file holding the saved learner
     * @param ds          datasource used to get variables
     * @param load_part   0: load function (default), 1: engine, 2: shell
     * This assumes that the learner name has already been checked.
     * @return true if successful
     */
    //    bool load(std::ifstream& in, const datasource& ds, size_t load_part);

    // Methods for choosing regularization
    //==========================================================================

    /**
     * Choose the regularization parameter lambda via
     * cross-validation (when learning Y ~ X).
     * This calls either choose_lambda_cv() with 10 folds
     * or choose_lambda_ridge() with ds.size() folds, whichever is better.
     * It tries 10 lambdas between .001 and ds.size(), on a log scale.
     * If this can use choose_lambda_ridge(), then it does so, regardless
     * of what optimization method is specified in lr_params.
     *
     * @param Yvec        Y variables in Y ~ X.
     * @param Xvec        X variables in Y ~ X.
     * @param lambdas     Lambdas to try.
     * @param lr_params   Parameters for linear_regression.
     * @param ds          Training data.
     * @param random_seed This uses this random seed, not the one in the
     *                    algorithm parameters.
     *
     * @return  Best lambda value found.
     */
    static double
    choose_lambda_easy
    (const vector_var_vector& Yvec, const vector_var_vector& Xvec,
     const linear_regression_parameters& lr_params,
     const dataset<la_type>& ds, unsigned random_seed = time(NULL));

    /**
     * This is identical to the other choose_lambda_easy(), except that Y, X are
     * specified implicitly by the dataset.  Y are the class vector variables,
     * X are all other vector variables.
     *
     * @param lambdas     Lambdas to try.
     * @param lr_params   Parameters for linear_regression.
     * @param ds          Training data.
     * @param random_seed This uses this random seed, not the one in the
     *                    algorithm parameters.
     *
     * @return  Best lambda value found.
     */
    static double
    choose_lambda_easy
    (const linear_regression_parameters& lr_params,
     const dataset<la_type>& ds, unsigned random_seed = time(NULL));

    /**
     * Choose the regularization parameter lambda via brute-force
     * cross-validation (when learning Y ~ X).
     *
     * If you are using least-squares with L2 regularization,
     * it is a good idea to use choose_lambda_easy() or choose_lambda_ridge()
     * instead.
     *
     * @param all_lambdas (Return value.) All lambdas tried.
     * @param scores      (Return value.) Mean squared errors for the lambdas,
     *                    or whatever score you chose.
     * @param stderrs     (Return value.) Std errors of the MSE estimates, or
     *                    the measure of variance corresponding to your score.
     * @param n_folds     Number of CV folds (> 0) (<= dataset size)
     * @param lambdas     Lambdas to try.
     * @param lr_params   Parameters for linear_regression.
     * @param zoom        If > 0, try out an extra set of lambdas around the
     *                    best lambda from the initial set.  This will try about
     *                    the same number of extra lambdas as original lambdas.
     *                    Repeat zoom times.
     * @param random_seed This uses this random seed, not the one in the
     *                    algorithm parameters.
     * @return  Chosen lambda value.
     */
    static double
    choose_lambda_cv
    (vec& all_lambdas, vec& scores, vec& stderrs,
     const vector_var_vector& Yvec, const vector_var_vector& Xvec,
     size_t n_folds, const vec& lambdas,
     const linear_regression_parameters& lr_params,
     size_t zoom, const dataset<la_type>& ds, unsigned random_seed = time(NULL));

    /**
     * Choose the regularization parameter lambda for via LOOCV,
     * and do so efficiently using matrix ops.
     * This requires lr_params to have regularization = 2 (L2).
     * (It ignores the optimization method in lr_params.)
     *
     * @param all_lambdas (Return value.) All lambdas tried.
     * @param scores      (Return value.) Mean squared errors for the lambdas,
     *                    or whatever score you chose.
     * @param stderrs     (Return value.) Std errors of the MSE estimates.
     * @param Yvec        Y variables in Y ~ X.
     * @param Xvec        X variables in Y ~ X.
     * @param lambdas     Lambdas to try.
     * @param lr_params   Parameters for linear_regression.
     * @param zoom        If > 0, try out an extra set of lambdas around the
     *                    best lambda from the initial set.  This will try about
     *                    the same number of extra lambdas as original lambdas.
     *                    Repeat zoom times.
     * @param return_regressor  If true, this will return a pointer to a
     *                          trained regressor.
     * @param random_seed       This uses this random seed, not the one in the
     *                          algorithm parameters.
     * @return  <chosen lambda value, pointer to trained regressor>
     *          If return_regressor is set to false, then the pointer is NULL.
     */
    static std::pair<double, linear_regression*>
    choose_lambda_ridge
    (vec& all_lambdas, vec& scores, vec& stderrs,
     const vector_var_vector& Yvec, const vector_var_vector& Xvec,
     const vec& lambdas, const linear_regression_parameters& lr_params,
     size_t zoom, const dataset<la_type>& ds, bool return_regressor,
     unsigned random_seed = time(NULL));

    // Debugging methods
    //==========================================================================

    //! Set all weights to 0 (to make sure we are learning something).
    void set_weights_to_zero() {
      weights_ = 0;
    }

  }; // class linear_regression

  std::ostream&
  operator<<(std::ostream& out, const linear_regression& lr);

  //! Template specialization.
  template <>
  vector_domain
  linear_regression::get_dependencies<vector_variable>() const;

  //! Template specialization.
  template <>
  std::vector<std::pair<vector_variable*, double> >
  linear_regression::get_dependencies<vector_variable>(size_t K) const;


} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_LINEAR_REGRESSION_HPP
