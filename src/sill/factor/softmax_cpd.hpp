#ifndef SILL_SOFTMAX_CPD_HPP
#define SILL_SOFTMAX_CPD_HPP

#include <sill/global.hpp>
#include <sill/base/assignment.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/datastructure/finite_index.hpp>
#include <sill/datastructure/hybrid_index.hpp>
#include <sill/factor/factor.hpp>
#include <sill/factor/probability_table.hpp>
#include <sill/factor/traits.hpp>
#include <sill/factor/util/factor_mle.hpp>
#include <sill/learning/dataset/hybrid_dataset.hpp>
#include <sill/learning/dataset/hybrid_record.hpp>
#include <sill/math/constants.hpp>
#include <sill/math/function/softmax.hpp>

#include <armadillo>
#include <iostream>

namespace sill {

  /**
   * A factor that represents a conditional distribution over a categorical variable
   * given a set of vector variables. The conditional distribution is given by a
   * normalized exponential, p(y = j | x) \propto exp(b_j + x^T w_j).
   *
   * \tparam T a real type for representing each parameter
   *
   * \ingroup factor_types
   * \see Factor
   */
  template <typename T = double>
  class softmax_cpd : public factor {    
  public:
    // Public types
    //==========================================================================
    typedef T          real_type;
    typedef T          result_type;
    typedef variable   variable_type;
    typedef domain     domain_type;
    typedef var_vector var_vector_type;
    typedef assignment assignment_type;
    typedef softmax<T> param_type;

    /// IndexableFactor member types
    typedef hybrid_index<T> index_type;

    // LearnableFactor member types
    typedef hybrid_dataset<T> dataset_type;
    typedef hybrid_record<T> record_type;
    
    // Types to represent the parameters
    typedef arma::Mat<T> mat_type;
    typedef arma::Col<T> vec_type;
 
    // Constructors and conversion operators
    //==========================================================================
    
    /**
     * Default constructor. Creates an empty factor.
     */
    softmax_cpd()
      : head_(NULL) { }

    /**
     * Constructs a factor with the given label variable and vector tail
     * arguments, but no finite tail arguments.
     * Allocates the parameters but does not initialize their values.
     */
    softmax_cpd(finite_variable* head, const vector_var_vector& tail)
      : head_(NULL) {
      reset(head, finite_var_vector(), tail);
    }

    /**
     * Constructs a factor with the given label variable and tail arguments.
     * Allocates the parameters but does not initilize their values.
     */
    softmax_cpd(finite_variable* head,
                const finite_var_vector& finite_tail,
                const vector_var_vector& vector_tail)
      : head_(NULL) {
      reset(head, finite_tail, vector_tail);
    }

    /**
     * Constructs a factor with the given label variable and vector tail
     * arguments, but no finite tail arguments.
     * Sets the parameters to to the given parameter vector.
     */
    softmax_cpd(finite_variable* head,
                const vector_var_vector& tail,
                const param_type& param)
      : head_(head),
        vector_tail_(tail),
        param_(param) {
      assert(head);
      args_.insert(head);
      args_.insert(tail.begin(), tail.end());
      check_param();
    }


    /**
     * Constructs a factor with the given label variable and tail arguments.
     * Sets the parameters to the given parameter vector.
     */
    softmax_cpd(finite_variable* head,
                const finite_var_vector& finite_tail,
                const vector_var_vector& vector_tail,
                const param_type& param)
      : head_(head),
        finite_tail_(finite_tail),
        vector_tail_(vector_tail),
        param_(param) {
      assert(head);
      args_.insert(head);
      args_.insert(finite_tail.begin(), finite_tail.end());
      args_.insert(vector_tail.begin(), vector_tail.end());
      check_param();
    }

    /**
     * Exchanges the arguments and the parameters of two factors.
     */
    friend void swap(const softmax_cpd& f, const sofmax_cpd& g) {
      if (&f != &g) {
        using std::swap;
        swap(f.args_, g.args_);
        swap(f.head_, g.head_);
        swap(f.finite_tail_, g.finite_tail_);
        swap(f.vector_tail_, g.vector_tail_);
        swap(f.param_, g.param_);
      }
      return *this;
    }

    /**
     * Resets the content of this factor to the given finite and vector
     * arguments. The parameter values may become invalidated.
     */
    void reset(finite_variable* head,
               const finite_var_vector& finite_tail,
               const finite_var_vector& vector_tail) {
      assert(head);
      if (head_ != head ||
          finite_tail_ != finite_tail ||
          vector_tail_ != vector_tail) {
        args_.clear();
        args_.insert(head);
        args_.insert(finite_tail.begin(), finite_tail.end());
        args_.insert(vector_tail.begin(), vector_tail.end());
        head_ = head;
        finite_tail_ = finite_tail;
        vector_tail_ = vector_tail;
        size_t nf = vector_size(finite_tail) + vector_size(vector_tail);
        param_.reset(head->size(), nf);
      }
    }

    // Accessors and comparison operators
    //==========================================================================

    //! Returns the arguments set of this factor
    const domain& arguments() const {
      return args_;
    }

    //! Returns the head argument of this factor or NUL if the factor is empty.
    finite_variable* head() const {
      return label_;
    }

    //! Returns the finite tail arguments of this factor
    const finite_var_vector& finite_tail() const {
      return finite_tail_;
    }

    //! Returns the vector tail arguments of this factor
    const vector_var_vector& vector_tail() const {
      return vector_tail_;
    }

    //! Returns true if the factor is empty.
    bool empty() const {
      return !label_;
    }

    //! Returns the number of arguments of this factor or 0 if the factor is empty.
    size_t arity() const {
      return args_.size();
    }

    //! Returns the number of assignments to the head variable.
    size_t num_labels() const {
      return param_.num_labels();
    }

    //! Returns the dimensionality of the underlying feature vector.
    size_t num_features() const {
      return param_.num_features();
    }

    //! Returns true if the factor contains finite features.
    bool has_finite_features() const {
      return !finite_tail_.empty();
    }

    //! Returns the parameters of this factor.
    const param_type& param() const {
      return param_;
    }

    //! Provides mutable access to the parameters of this factor.
    param_type& param() const {
      return param_;
    }

    //! Returns the weight matrix.
    const mat_type& weight() const {
      return param_.weight();
    }

    //! Returns the weight matrix.
    mat_type& weight() {
      return param_.weight();
    }

    //! Returns the bias vector.
    const vec_type& bias() const {
      return param_.bias();
    }

    //! Returns the bias vector.
    vec_type& bias() {
      return param_.bias();
    }

    /**
     * Returns the value of the factor for the given index.
     * The first finite value is assumed to be the label.
     */
    T operator()(const hybrid_index<T>& index) const {
      if (has_finite_features()) {
        sparse_index<T> feature;
        extract_features(index, 1, features);
        return param_(features)[index.finite[0]];
      } else {
        return param_(index.vector)[index.finite[0]];
      }
    }

    /**
     * Returns the value of the factor (conditional probability) for the
     * given assignment.
     * \param strict if true, requires all the arguments to be present;
     *        otherwise, only the label variable must be present and the
     *        missing features are assumed to be 0.
     */
    T operator()(const assignment& a, bool strict = true) const {
      sparse_index<T> features;
      extract_features(a, strict, features);
      size_t finite = safe_get(a, head_);
      return param_(features)[finite];
    }

    /**
     * Returns the log-value of the factor for the given index.
     * The first finite value is assumed ot be the label.
     */
    T log(const hybrid_index<T>& index) const {
      return std::log(operator()(index));
    }

    /**
     * Returns the log of the value of the factor (conditional probability)
     * for the given assignment.
     * \param strict if true, requires all the arguments to be present;
     *        otherwise, only the label variable must be present and the
     *        missing features are assumed to be 0.
     */
    T log(const assignment& a, bool strict = true) const {
      return std::log(operator()(a, strict));
    }

    /**
     * Returns true if the two factors have the same argument vectors and
     * parameters.
     */
    friend bool operator==(const softmax_cpd& f, const softmax_cpd& g) {
      return f.finite_ == g.finite_
        && f.vector_ == g.vector_
        && f.param_ == g.param_;
    }

    /**
     * Returns true if the two factors do not have the same argument vectors
     * or parameters.
     */
    friend bool operator==(const softmax_cpd& f, const softmax_cpd& g) {
      return !(f == g);
    }

    // Indexing
    //==========================================================================

    /**
     * Computes a sparse vector of features for the given index.
     */
    void extract_features(const hybrid_index<T>& index,
                          size_t start,
                          sparse_index<T>& result) const {
      assert(index.finite.size() - start == finite_tail_.size());
      result.clear();
      result.resize(finite_tail_.size() + index.vector.size());
      size_t id = 0;
      for (size_t i = 0; i < finite_tail_.size(); ++i) {
        result.emplace_back(id + index.finite[i + start], 1);
        id += finite_tail_[i]->size();
      }
      for (size_t i = 0; i < index.num_vector(); ++i) {
        result.emplace_back(id++, index.vector[i]);
      }
    }

    /**
     * Computes a sparse vector of features for the given index.
     * \param strict If true, requires all the features to be present in the
     *        assignment. Otherwise, fills in 0 for the missing features.
     */
    void extract_features(const assignment& a,
                          bool strict,
                          sparse_index<T>& result) const {
      result.clear();
      result.reserve(finite_tail_.size() + vector_size(vector_tail_));

      // extract the finite features
      for (finite_variable* v : finite_tail_) {
        finite_assignment::const_iterator it = a.finite().find(v);
        if (it != a.finite().end()) {
          result.emplace_back(id + it->second, 1);
        } else if (strict) {
          throw std::invalid_argument(
            "The assignment does not contain the variable " + v->str()
          );
        }
        id += v->size();
      }

      // extract the vector features
      for (vector_variable* v : vector_tail_) {
        vector_assignment::const_iterator it = a.vector().find(v);
        if (it != a.vector().end()) {
          for (size_t i = 0; i < v->size(); ++i) {
            result.emplace_back(id + i, it->second[i]);
          }
        } else if (strict) {
          throw std::invalid_argument(
            "The assignment does not contain the variable " + v->str()
          );
        }
        id += v->size();
      }
    }

    /**
     * Checks if the dimensions of the parameters match this factor's arguments.
     * \throw std::runtime_error if some of the dimensions do not match.
     */
    void check_param() const {
      if (empty()) {
        if (param_.size_head() || param_.size_tail()) {
          throw std::runtime_error("The factor is empty but the parameters are not!");
        }
      } else {
        if (param_.size_head() != head_->size()) {
          throw std::runtime_error("Invalid number of labels");
        }
        size_t nf = vector_size(finite_tail_) + vector_size(vector_tail_);
        if (param_.size_tail() != nf) {
          throw std::runtime_error("Invalid number of features");
        }
      }
    }

    // Factor operations
    //==========================================================================

    /**
     * Returns true if the factor represents a valid distribution.
     * This is true if none of the parameters are infinite / nan.
     */
    bool is_normalizable() const {
      return param_.is_finite();
    }

    /**
     * Conditions the factor on the assignment to its tail variables.
     * \param strict if true, requires that all the tail arguments  are present
     *        in the assignment.
     */
    probability_matrix condition(const assignment& a) const {
      
      return probability_matrix(head, param_(index(a)));
    }

    //! Conditions the factor on the tail vector in the factor's internal ordering.
    probability_matrix condition(const vec_type& index) const {
      return probability_matrix(head, param_(index));
    }

    // Private members
    //==========================================================================
  private:
    //! The argument set of this factor
    domain args_;

    //! The finite arguments of this factor, starting with label variable.
    finite_var_vector finite_;

    //! The vector arguments of this factor, all of which belong to the tail.
    vector_var_vector vector_;
    
  }; // class softmax_cpd

  /**
   * Prints a human-readable representation of the CPD to a stream.
   * \relates softmax_cpd
   */
  template <typename T>
  std::ostream& operator<<(std::ostream& out, const softmax_cpd<T>& f) {
    out << "softmax("
        << f.head() << "|"
        << f.finite_tail() << ","
        << f.vector_tail() << ")" << std::endl
        << f.param();
    return out;
  }


  // Utility classes
  //==========================================================================
  /**
   * A utility class that represents a maximum-likelihood estimator of
   * a softmax conditional probability distribution. The maximum likelihood
   * estimate is computed iteratively using the specified optimization
   * class.
   */
  template <typename T>
  class factor_mle<softmax_cpd<T> > {
  public:
    typedef domain domain_type;
    typedef var_vector var_vector_type;
    typedef hybrid_dataset<T> dataset_type;
    typedef hybrid_record<T> record_type;
    struct param_type {
      T regul;
      // optimizer
      param_type(T regul = 0.1) : regul(regul) { }
    };
    
    /**
     * Creates a maximum-likelihood estimator with the given dataset,
     * parameters, and optimizer.
     */
    factor_mle(const dataset_type* ds,
               const param_type& params = param_type())
      : ds_(ds), params_(params) { }

    /**
     * Returns the conditional distribution p(head | tail).
     */
    F operator()(finite_variable* head,
                 const vector_var_vector& tail) const {
      dense_objective objective;
   
    }
               
              
    
    

  }

} // namespace sill  


#endif
