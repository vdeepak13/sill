#ifndef SILL_CRF_MODEL_HPP
#define SILL_CRF_MODEL_HPP

#include <iterator>
#include <map>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <sill/base/stl_util.hpp>
#include <sill/base/universe.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/model/crf_graph.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/model_functors.hpp>
#include <sill/range/forward_range.hpp>
#include <sill/range/transformed.hpp>

#include <sill/serialization/serialize.hpp>
#include <sill/serialization/list.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * This is a factor graph representation of a conditional random field (CRF)
   * for representing distributions P(Y | X).
   *
   * This permits CRFs of any form; the parametrization of the factors
   * is hidden within the factor type.  For example you could have:
   *  - a CRF with arbitrary factors:
   *     P(Y|X) \propto \prod_i f_i() where f_i() is arbitrary;
   *     f_i() are the parameters which need to be learned.
   *     - For this, use a factor which fits the CRFfactor concept.
   *  - a CRF with weighted features:
   *     P(Y|X) \propto \exp[\sum_i w_i * f_i()] where f_i() is arbitrary;
   *     f_i() are fixed features, and w_i are the parameters to be learned.
   *     - For this, use a factor which fits the CRFweighted_factor concept.
   *
   * @tparam FactorType  type of factor which fits the CRFfactor concept
   *
   * @see crf_graph
   * \ingroup model
   */
  template <typename FactorType>
  class crf_model
    : public crf_graph<typename FactorType::output_variable_type,
                       typename FactorType::input_variable_type,
                       typename FactorType::variable_type,
                       FactorType*> {

    concept_assert((sill::CRFfactor<FactorType>));

    /**
     * Print debugging info.
     *  - 0: none (default)
     *  - 1: some
     *  - higher values: revert to highest debugging mode
     */
    static const size_t debug = 0;

    // Public type declarations
    // =========================================================================
  public:

    //! CRF factor type
    typedef FactorType crf_factor;

    //! Type of output variable Y.
    typedef typename FactorType::output_variable_type output_variable_type;

    //! Type of input variable X.
    typedef typename FactorType::input_variable_type input_variable_type;

    //! Type of variable for both Y,X.
    typedef typename FactorType::variable_type variable_type;

    //! Type of domain for variables in Y.
    typedef typename FactorType::output_domain_type output_domain_type;

    //! Type of domain for variables in X.
    typedef typename FactorType::input_domain_type input_domain_type;

    //! Type of domain for variables in both Y,X.
    typedef typename FactorType::domain_type domain_type;

    //! The underlying CRF graph type (which is also the base).
    typedef crf_graph<typename FactorType::output_variable_type,
                      typename FactorType::input_variable_type,
                      typename FactorType::variable_type,
                      FactorType*> crf_graph_type;

    //! Type of assignment for variables in Y.
    typedef typename FactorType::output_assignment_type output_assignment_type;

    //! Type of assignment for variables in X.
    typedef typename FactorType::input_assignment_type input_assignment_type;

    //! Type of assignment for variables in both Y,X.
    typedef typename FactorType::assignment_type assignment_type;

    //! Type of record for variables in Y.
    typedef typename FactorType::output_record_type output_record_type;

    //! Type of record for variables in X.
    typedef typename FactorType::input_record_type input_record_type;

    //! Type of record for variables in both Y,X.
    typedef typename FactorType::record_type record_type;

    /**
     * The type which this factor f(Y,X) outputs to represent f(Y, X=x).
     * For finite Y, this will probably be table_factor;
     * for vector Y, this will probably be gaussian_base.
     */
    typedef typename FactorType::output_factor_type output_factor_type;

    // Graph types
    typedef typename crf_graph_type::vertex vertex;
    typedef typename crf_graph_type::edge edge;
    typedef typename crf_graph_type::vertex_property vertex_property;

    // Graph iterators
    typedef typename crf_graph_type::vertex_iterator     vertex_iterator;
    typedef typename crf_graph_type::neighbor_iterator   neighbor_iterator;
    typedef typename crf_graph_type::edge_iterator       edge_iterator;
    typedef typename crf_graph_type::in_edge_iterator    in_edge_iterator;
    typedef typename crf_graph_type::out_edge_iterator   out_edge_iterator;
    typedef typename crf_graph_type::neighbor2_iterator  neighbor2_iterator;
    typedef typename crf_graph_type::variable_vertex_iterator variable_vertex_iterator;

    //! Iterator over factors (permitting only const access).
    class factor_iterator
      : public std::iterator<std::forward_iterator_tag, const crf_factor> {

      const std::vector<crf_factor*>* factor_list_ptr;

      typename std::vector<crf_factor*>::const_iterator factor_it;

    public:
      //! Constructor.
      //! @param  is_end  If true, is end iterator.
      factor_iterator(const std::vector<crf_factor*>* factor_list_ptr,
                      bool is_end)
        : factor_list_ptr(factor_list_ptr) {
        assert(factor_list_ptr);
        if (is_end)
          factor_it = factor_list_ptr->end();
        else
          factor_it = factor_list_ptr->begin();
      }

      //! Prefix increment.
      factor_iterator& operator++() {
        if (factor_it == factor_list_ptr->end())
          return *this;
        ++factor_it;
        return *this;
      }

      //! Postfix increment.
      factor_iterator operator++(int) {
        factor_iterator tmp(*this);
        ++(*this);
        return tmp;
      }

      //! Returns a const reference to the current factor.
      const crf_factor& operator*() const {
        if (factor_it == factor_list_ptr->end())
          assert(false);
        return **factor_it;
      }

      //! Returns a const pointer to the current factor.
      const crf_factor* const operator->() const {
        if (factor_it == factor_list_ptr->end())
          assert(false);
        return *factor_it;
      }

      //! Returns truth if the two iterators are the same.
      bool operator==(const factor_iterator& it) const {
        if (factor_it != it.factor_it || factor_list_ptr != it.factor_list_ptr)
          return false;
        return true;
      }

      //! Returns truth if the two iterators are different.
      bool operator!=(const factor_iterator& it) const {
        return !operator==(it);
      }

    }; // class factor_iterator

    /**
     * Optimization variables (which fit the OptimizationVector concept).
     * These are the CRF factor weights.
     */
    struct opt_variables {

    protected:

      /**
       * These can be pointers into the weights stored by the factors,
       * or they can be owned by this opt_variables struct.
       */
      std::vector<typename crf_factor::optimization_vector*> factor_weights_;

      //! Indicates if this opt_variables struct owns the factor weights
      //! pointed to by factor_weights_ptr.
      bool own_data;

      friend class crf_model;

    public:

      // Types and special data access
      //------------------------------------------------------------------------

      typedef std::vector<typename crf_factor::optimization_vector::size_type>
        size_type;

      //! Returns a const reference to the weights for factor i.
      const typename crf_factor::optimization_vector&
      factor_weight(size_t i) const {
        assert(i < factor_weights_.size());
        return *(factor_weights_[i]);
      }

      //! Returns a mutable reference to the weights for factor i.
      typename crf_factor::optimization_vector&
      factor_weight(size_t i) {
        assert(i < factor_weights_.size());
        return *(factor_weights_[i]);
      }

      // Constructors and destructor
      //------------------------------------------------------------------------

      opt_variables() : own_data(true) { }

      opt_variables(size_type s, double default_val)
        : factor_weights_(s.size(), NULL), own_data(true) {
        for (size_t i(0); i < s.size(); ++i)
          factor_weights_[i] =
            new typename crf_factor::optimization_vector(s[i], default_val);
      }

      //! Copy constructor.
      //! The new struct owns its data.
      opt_variables(const opt_variables& ov)
        : factor_weights_(ov.factor_weights_.size(), NULL), own_data(true) {
        for (size_t i(0); i < ov.factor_weights_.size(); ++i)
          factor_weights_[i] =
            new typename crf_factor::optimization_vector
            (*(ov.factor_weights_[i]));
      }

      ~opt_variables() {
        if (own_data) {
          foreach(typename crf_factor::optimization_vector* ov, factor_weights_)
            delete(ov);
        }
      }

      //! Serialize members
      void save(oarchive & ar) const {
        ar << own_data;
        if (own_data) {
          ar << factor_weights_.size();
          foreach(typename crf_factor::optimization_vector* ov_ptr,
                  factor_weights_)
            ar << *ov_ptr;
        }
      }

      //! Deserialize members
      void load(iarchive & ar) {
        if (own_data) {
          foreach(typename crf_factor::optimization_vector* ov, factor_weights_)
            delete(ov);
          factor_weights_.clear();
        }
        ar >> own_data;
        if (own_data) {
          size_t fw_size;
          ar >> fw_size;
          factor_weights_.resize(fw_size, NULL);
          foreach(typename crf_factor::optimization_vector* ov_ptr,
                  factor_weights_) {
            ov_ptr = new typename crf_factor::optimization_vector();
            ar >> *ov_ptr;
          }
        }
      }

      // Getters and non-math setters
      //------------------------------------------------------------------------

      /**
       * Assignment operator.
       * If this struct owns its data, then this works like any assignment,
       * it will still own its data afterwards.
       * If this struct does not own its data, then the other struct must
       * have the same dimensions (including for each factor),
       * and each factor weight is copied over.
       */
      opt_variables& operator=(const opt_variables& other) {
        if (other.factor_weights_.size() != factor_weights_.size()) {
          if (own_data)
            resize(other.size());
          else
            assert(false);
        }
        for (size_t i(0); i < other.factor_weights_.size(); ++i) {
//          if (factor_weights_[i]->size() != other.factor_weights_[i]->size())
//            assert(false);
          factor_weights_[i]->operator=(*(other.factor_weights_[i]));
        }
        return *this;
      }

      //! Returns true iff this instance equals the other.
      bool operator==(const opt_variables& other) const {
        if (factor_weights_.size() != other.factor_weights_.size())
          return false;
        for (size_t i(0); i < factor_weights_.size(); ++i) {
          if (*(factor_weights_[i]) != *(other.factor_weights_[i]))
            return false;
        }
        return true;
      }

      //! Returns false iff this instance equals the other.
      bool operator!=(const opt_variables& other) const {
        return !operator==(other);
      }

      size_type size() const {
        size_type s;
        foreach(const typename crf_factor::optimization_vector* fwptr,
                factor_weights_)
          s.push_back(fwptr->size());
        return s;
      }

      //! Resize the data.
      //! This asserts false if this instance does not own its data.
      void resize(const size_type& newsize) {
        assert(own_data);
        size_t kept_size(std::min(factor_weights_.size(), newsize.size()));
        if (newsize.size() < factor_weights_.size()) {
          for (size_t i(newsize.size()); i < factor_weights_.size(); ++i)
            delete(factor_weights_[i]);
          factor_weights_.resize(newsize.size());
        } else if (newsize.size() > factor_weights_.size()) {
          factor_weights_.resize(newsize.size());
          for (size_t i(kept_size); i < newsize.size(); ++i)
            factor_weights_[i] =
              new typename crf_factor::optimization_vector(newsize[i], 0.);
        }
        for (size_t i(0); i < kept_size; ++i) {
          if (!(factor_weights_[i]->size() == newsize[i]))
            factor_weights_[i]->resize(newsize[i]);
        }
      }

      // Math operations
      //------------------------------------------------------------------------

      //! Sets all elements to this value.
      opt_variables& operator=(double d) {
        size_t nfactors(factor_weights_.size());
        for (size_t i(0); i < nfactors; ++i)
          factor_weights_[i]->operator=(d);
        return *this;
      }

      //! Addition.
      opt_variables operator+(const opt_variables& other) const {
        opt_variables ov(*this);
        ov += other;
        return ov;
      }

      //! Addition.
      opt_variables& operator+=(const opt_variables& other) {
        assert(factor_weights_.size() == other.factor_weights_.size());
        for (size_t i(0); i < other.factor_weights_.size(); ++i)
          factor_weights_[i]->operator+=(*(other.factor_weights_[i]));
        return *this;
      }

      //! Subtraction.
      opt_variables operator-(const opt_variables& other) const {
        opt_variables ov(*this);
        ov -= other;
        return ov;
      }

      //! Subtraction.
      opt_variables& operator-=(const opt_variables& other) {
        assert(factor_weights_.size() == other.factor_weights_.size());
        for (size_t i(0); i < other.factor_weights_.size(); ++i)
          factor_weights_[i]->operator-=(*(other.factor_weights_[i]));
        return *this;
      }

      //! Multiplication by a scalar value.
      opt_variables operator*(double d) const {
        opt_variables ov(*this);
        ov *= d;
        return ov;
      }

      //! Multiplication by a scalar value.
      opt_variables& operator*=(double d) {
        double tmpsize(factor_weights_.size());
        for (size_t i(0); i < tmpsize; ++i)
          factor_weights_[i]->operator*=(d);
        return *this;
      }

      //! Division by a scalar value.
      opt_variables operator/(double d) const {
        opt_variables ov(*this);
        ov /= d;
        return ov;
      }

      //! Division by a scalar value.
      opt_variables& operator/=(double d) {
        double tmpsize(factor_weights_.size());
        for (size_t i(0); i < tmpsize; ++i)
          factor_weights_[i]->operator/=(d);
        return *this;
      }

      //! Inner product with a value of the same size.
      double dot(const opt_variables& other) const {
        assert(factor_weights_.size() == other.factor_weights_.size());
        double val(0);
        for (size_t i(0); i < other.factor_weights_.size(); ++i)
          val += factor_weights_[i]->dot(*(other.factor_weights_[i]));
        return val;
      }

      //! Element-wise multiplication with another value of the same size.
      opt_variables& elem_mult(const opt_variables& other) {
        double tmpsize(factor_weights_.size());
        assert(tmpsize == other.factor_weights_.size());
        for (size_t i(0); i < tmpsize; ++i)
          factor_weights_[i]->elem_mult(*(other.factor_weights_[i]));
        return *this;
      }

      //! Element-wise reciprocal (i.e., change v to 1/v).
      opt_variables& reciprocal() {
        double tmpsize(factor_weights_.size());
        for (size_t i(0); i < tmpsize; ++i)
          factor_weights_[i]->reciprocal();
        return *this;
      }

      //! Returns the L1 norm.
      double L1norm() const {
        double val(0);
        for (size_t i(0); i < factor_weights_.size(); ++i)
          val += factor_weights_[i]->L1norm();
        return val;
      }

      //! Returns the L2 norm.
      double L2norm() const {
        return sqrt(dot(*this));
      }

      //! Returns a struct of the same size but with values replaced by their
      //! signs (-1 for negative, 0 for 0, 1 for positive).
      opt_variables sign() const {
        opt_variables ov(*this);
        for (size_t i(0); i < factor_weights_.size(); ++i)
          ov.factor_weights_[i]->operator=(factor_weights_[i]->sign());
        return ov;
      }

      //! Clears all data from this struct.
      void clear() {
        if (own_data) {
          foreach(typename crf_factor::optimization_vector* ov, factor_weights_)
            delete(ov);
        }
        factor_weights_.resize(0);
      }

      /**
       * "Zeros" this vector by calling this function on each factor's
       * optimization variables.
       */
      void zeros() {
        for (size_t i(0); i < factor_weights_.size(); ++i)
          factor_weights_[i]->zeros();
      }

      //! Print info about this vector (for debugging).
      void print_info(std::ostream& out) const {
        out << "crf_model::print_info: " << factor_weights_.size()
            << " factors with info:\n";
//      foreach(typename crf_factor::optimization_vector* v_ptr, factor_weights_)
//        v_ptr->print_info(out);
      }

    }; // struct opt_variables

    // Constructors, destructors, and assignment operators
    // =========================================================================
  public:

    /**
     * Creates a CRF model with no factors and no variables.
     * Use the add_factor method to add factors and variables.
     */
    crf_model()
      : crf_graph_type(), conditioned_model_valid(false) {
      if (debug)
        std::cerr << "WARNING: crf_model.debug is set to TRUE"
                  << " (which reduces efficiency significantly)." << std::endl;
      weights_.own_data = false;
    }

    /**
     * Creates a CRF model with the given structure, with default-initialized
     * factors.
     */
    explicit crf_model(const crf_graph_type& structure) {
      std::vector<crf_factor> fctrs;
      foreach(const vertex& v, structure.factor_vertices()) {
        fctrs.push_back(crf_factor(structure.output_arguments(v),
                                   structure.input_arguments(v)));
      }
      this->add_factors(fctrs);
      weights_.own_data = false;
    }

    //! Copy constructor.
    crf_model(const crf_model& other)
      : crf_graph_type(other), factors_(other.factors_),
        factor_v2i(other.factor_v2i) {
      init();
      conditioned_model_valid = other.conditioned_model_valid;
      if (other.conditioned_model_valid) {
        conditioned_model = other.conditioned_model;
        conditioned_model_vertex_map_ = other.conditioned_model_vertex_map_;
      }
    }

    //! Assignment operator.
    crf_model& operator=(const crf_model& other) {
      crf_graph_type::operator=(other);
      factors_ = other.factors_;
      factor_v2i = other.factor_v2i;
      init();
      conditioned_model_valid = other.conditioned_model_valid;
      if (other.conditioned_model_valid) {
        conditioned_model = other.conditioned_model;
        conditioned_model_vertex_map_ = other.conditioned_model_vertex_map_;
      }
      return *this;
    }

    // Serialization
    // =========================================================================

    //! Serialize members
    void save(oarchive & ar) const {
      crf_graph_type::save(ar);
      ar << factors_ << factor_v2i;
    } // save

    //! Deserialize members
    void load(iarchive & ar) {
      crf_graph_type::load(ar);
      ar >> factors_ >> factor_v2i;
      init();
    } // load

    // Getters and helpers
    // =========================================================================

    using crf_graph_type::size;
    using crf_graph_type::num_arguments;
    using crf_graph_type::arguments;
    using crf_graph_type::output_arguments;
    using crf_graph_type::input_arguments;

    //! Returns the factors in this model.
    const std::list<crf_factor>& factors() const { return factors_; }

    //! Returns the factor at the given vertex.
    //! This asserts false if the vertex does not contain a factor.
    const crf_factor& factor(const vertex& u) const {
      assert(this->is_factor_vertex(u));
      return *(this->operator[](u));
    }

    /**
     * Returns a mutable reference to the CRF factor weights.
     * (For optimization routines.)
     */
    opt_variables& weights() { return weights_; }

    /**
     * Returns a constant reference to the CRF factor weights.
     * (For optimization routines.)
     */
    const opt_variables& weights() const { return weights_; }

    //! Given a factor vertex, returns the index in weights for that factor.
    size_t factor_vertex2index(const vertex& u) const {
      return safe_get(factor_v2i, u);
    }

    //! Sets all weights to be in log space (or not, if not possible).
    //! @param log_space If true, set to log space; if false, set to real space.
    //! @return  true if successful, or false if not
    bool set_log_space(bool log_space) {
      if (log_space) {
        foreach(crf_factor& f, factors_) {
          if (!(f.convert_to_log_space()))
            return false;
        }
      } else {
        foreach(crf_factor& f, factors_) {
          if (!(f.convert_to_real_space()))
            return false;
        }
      }
      return true;
    }

    /**
     * This turns on the fixed_records option for all CRF factors.
     * NOTE: This can greatly speed up calls using records.  However, methods
     *       you call later on the CRF model/factors MUST use records
     *       of the same type (the same variable orderings), until you
     *       turn off this option.
     */
    void fix_records(const record_type& r) {
      foreach(crf_factor& f, factors_)
        f.fix_records(r);
    }

    /**
     * This turns off the fixed_records option for all CRF factors.
     */
    void unfix_records() {
      foreach(crf_factor& f, factors_)
        f.unfix_records();
    }

    // Probabilistic queries
    // =========================================================================

    /**
     * Return a decomposable model for P(Y | X = x).
     * WARNING: This assumes that the model P(Y | X = x) is tractable!
     * Note: This reference may no longer be valid after other calls to
     *       methods of this model.
     * @param x    an assignment to X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    const decomposable<output_factor_type>&
    condition(const input_assignment_type& x) const {
      condition_model(x);
      return conditioned_model;
    }

    /**
     * Return a decomposable model for P(Y | X = x).
     * WARNING: This assumes that the model P(Y | X = x) is tractable!
     * Note: This reference may no longer be valid after other calls to
     *       methods of this model.
     * @param x    record with an assignment to X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    const decomposable<output_factor_type>&
    condition(const input_record_type& x) const {
      condition_model(x);
      return conditioned_model;
    }

    /**
     * This is a helper function usable alongside the condition() methods.
     * The idea is:
     *  - Suppose your crf_model structure does not change, but you may
     *    alter the parameters.
     *  - Suppose you spend a lot of time conditioning your crf_model and
     *    then computing marginals using the resulting decomposable model.
     *  - Suppose these marginals are only over the output arguments of the
     *    the factors in your crf_model.
     *  - (This is what happens in parameter learning.)
     *  - Then, you can use this mapping to find a vertex in your decomposable
     *    model corresponding to each factor in your crf_model.
     * This may only be called AFTER condition() or another probabilistic query
     * which requires conditioning has been called.
     *
     * @return Vector whose corresponds to the CRF factor ordering returned by
     *         factors() and whose components are vertices within the
     *         decomposable model returned by condition().
     *         This mapping is guaranteed to remain valid until the structure
     *         of this crf_model changes.
     *         You can get the marginal for vertex v
     *         conditioned_model.marginal(v), etc.
     */
    const std::vector<typename decomposable<output_factor_type>::vertex>&
    conditioned_model_vertex_mapping() const {
      return conditioned_model_vertex_map_;
    }

    /**
     * Sample from P(Y | X = x).
     * WARNING: This currently assumes that the model is tractable!
     * @param x    an assignment to X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    template <typename RandomNumberGenerator>
    output_assignment_type
    sample(const input_assignment_type& x, RandomNumberGenerator& rng) const {
      condition_model(x);
      return conditioned_model.sample(rng);
    }

    // Losses
    //==========================================================================

    /**
     * Computes log P(Y = y | X = x).
     * WARNING: This assumes that the model P(Y | X = x) is tractable!
     * @param a    an assignment to Y,X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    double log_likelihood(const assignment_type& a, double base) const {
      condition_model(a);
      return conditioned_model.log_likelihood(a, base);
    }

    /**
     * Computes log P(Y = y | X = x).
     * WARNING: This assumes that the model P(Y | X = x) is tractable!
     * @param r    record with an assignment to Y,X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    double log_likelihood(const record_type& r, double base) const {
      condition_model(r);
      return conditioned_model.log_likelihood(r, base);
    }

    /**
     * Computes log P(Y = y | X = x) using log base e.
     * WARNING: This assumes that the model P(Y | X = x) is tractable!
     * @param a    an assignment to Y,X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    double log_likelihood(const assignment_type& a) const {
      return log_likelihood(a, exp(1.));
    }

    /**
     * Computes log P(Y = y | X = x) using log base e.
     * WARNING: This assumes that the model P(Y | X = x) is tractable!
     * @param r    record with an assignment to Y,X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    double log_likelihood(const record_type& r) const {
      return log_likelihood(r, exp(1.));
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected log likelihood E[log P(Y|X)].
     */
    model_log_likelihood_functor<crf_model>
    log_likelihood(double base = exp(1.)) const {
      return model_log_likelihood_functor<crf_model>(*this, base);
    }

    /**
     * Computes the per-label accuracy (average over Y variables).
     * WARNING: This assumes that the model P(Y | X = x) is tractable!
     * @param a    an assignment to Y,X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    double per_label_accuracy(const assignment_type& a) const {
      condition_model(a);
      return conditioned_model.per_label_accuracy(a);
    }

    /**
     * Computes the per-label accuracy (average over Y variables).
     * WARNING: This assumes that the model P(Y | X = x) is tractable!
     * @param r    record with an assignment to Y,X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    double per_label_accuracy(const record_type& r) const {
      condition_model(r);
      return conditioned_model.per_label_accuracy(r);
    }

    /**
     * Computes the per-label accuracy of predicting Y given Z and X,
     * where this model is of P(Y,Z|X).
     * WARNING: This assumes that the model P(Y | X = x) is tractable!
     * @param r    record with an assignment to Y,X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    double per_label_accuracy(const record_type& r,
                              const output_domain_type& Z) const {
      condition_model(r);
      return conditioned_model.per_label_accuracy(r, Z);
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected per-label accuracy of predicting Y given X.
     */
    model_per_label_accuracy_functor<crf_model>
    per_label_accuracy() const {
      return model_per_label_accuracy_functor<crf_model>(*this);
    }

    /**
     * Returns 1 if this predicts all Y variable values correctly and 0 o.w.
     * WARNING: This assumes that the model P(Y | X = x) is tractable!
     * @param a    an assignment to Y,X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    size_t accuracy(const assignment_type& a) const {
      condition_model(a);
      return conditioned_model.accuracy(a);
    }

    /**
     * Returns 1 if this predicts all Y variable values correctly and 0 o.w.
     * WARNING: This assumes that the model P(Y | X = x) is tractable!
     * @param r    record with an assignment to Y,X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    size_t accuracy(const record_type& r) const {
      condition_model(r);
      return conditioned_model.accuracy(r);
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected all-or-nothing accuracy of predicting Y given X.
     */
    model_accuracy_functor<crf_model>
    accuracy() const {
      return model_accuracy_functor<crf_model>(*this);
    }

    /**
     * Computes the mean squared error (mean over Y variables);
     * for finite data, this is the same as per-label accuracy.
     * WARNING: This assumes that the model P(Y | X = x) is tractable!
     * @param a    an assignment to Y,X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    double mean_squared_error(const assignment_type& a) const {
      condition_model(a);
      return conditioned_model.mean_squared_error(a);
    }

    /**
     * Computes the mean squared error of predicting Y given Z and X,
     * where this model is P(Y,Z|X).
     * WARNING: This assumes that the model P(Y | X = x) is tractable!
     * @param a    an assignment to Y,Z,X
     * @todo Make this safer (in terms of tractability) and more efficient.
     */
    double mean_squared_error(const assignment_type& a,
                              const output_domain_type& Z) const {
      condition_model(a);
      return conditioned_model.mean_squared_error(a,Z);
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected mean squared error of predicting Y given X.
     */
    model_mean_squared_error_functor<crf_model>
    mean_squared_error() const {
      return model_mean_squared_error_functor<crf_model>(*this);
    }

    //! Computes the log pseudolikelihood of this model for the given record.
    double
    log_pseudolikelihood(const record_type& r, double base = exp(1.)) const {
      double pl = 0;
      foreach(output_variable_type y, output_arguments())
        pl += log_pseudolikelihood_component(y, r, base);
      return pl;
    }

    //! Computes the log pseudolikelihood of this model for the given record.
    double
    log_pseudolikelihood(const assignment_type& a, double base = exp(1.)) const{
      double pl = 0;
      foreach(output_variable_type y, output_arguments())
        pl += log_pseudolikelihood_component(y, a, base);
      return pl;
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected log pseudolikelihood.
     */
    model_log_pseudolikelihood_functor<crf_model>
    log_pseudolikelihood(double base = exp(1.)) const {
      return model_log_pseudolikelihood_functor<crf_model>(*this, base);
    }

    //! Computes the log pseudolikelihood component for Yi:
    //!  log P(Yi | neighbors of Yi in Y,X)
    double
    log_pseudolikelihood_component(output_variable_type Yi,
                                   const record_type& r,
                                   double base = exp(1.)) const {
      typename crf_factor::output_factor_type f(make_domain(Yi), 1);
      typename crf_factor::output_factor_type tmpf;
      foreach(const typename crf_graph_type::vertex& neighbor_v,
              neighbors(Yi)){
        const typename crf_factor::output_factor_type& neighbor_f =
          this->operator[](neighbor_v)->condition(r);
        neighbor_f.restrict
          (r, set_difference(neighbor_f.arguments(), make_domain(Yi)), tmpf);
        f *= tmpf;
      }
      f.normalize();
      return f.logv(r) / std::log(base);
    }

    //! Computes the log pseudolikelihood component for Yi:
    //!  log P(Yi | neighbors of Yi in Y,X)
    double
    log_pseudolikelihood_component(output_variable_type Yi,
                                   const assignment_type& a,
                                   double base = exp(1.)) const {
      typename crf_factor::output_factor_type f(make_domain(Yi), 1);
      typename crf_factor::output_factor_type tmpf;
      foreach(const typename crf_graph_type::vertex& neighbor_v,
              this->neighbors(Yi)){
        const typename crf_factor::output_factor_type& neighbor_f =
          this->operator[](neighbor_v)->condition(a);
        neighbor_f.restrict
          (a, set_difference(neighbor_f.arguments(), make_domain(Yi)), tmpf);
        f *= tmpf;
      }
      f.normalize();
      return f.logv(a) / std::log(base);
    }

    // Mutating methods
    // =========================================================================

    using crf_graph_type::add_factor;

    /**
     * Add a factor to this factor graph.
     * All the variables in the factor are added to this graphical model,
     * potentially changing the domain.
     * Note: If you are adding many factors, it is more efficient to use
     *       add_factors().
     * @return vertex for this factor
     */
    vertex add_factor(const crf_factor& factor) {
      conditioned_model_valid = false;
      factors_.push_back(factor);
      if (!factor.fixed_value())
        weights_.factor_weights_.push_back(&(factors_.back().weights()));
      vertex v =
        this->crf_graph_type::add_factor(factor.output_arguments(),
                                         factor.input_arguments_ptr(),
                                         &(factors_.back()));
      factor_v2i[v] = factors_.size() - 1;
      return v;
    }

    /**
     * Add a range of factors to this factor graph.
     * All the variables in the factors are added to this graphical model,
     * potentially changing the domain.
     */
    void add_factors(const forward_range<crf_factor>& fctrs) {
      conditioned_model_valid = false;
      foreach(const crf_factor& f, fctrs) {
        factors_.push_back(f);
        if (!f.fixed_value())
          weights_.factor_weights_.push_back(&(factors_.back().weights()));
        vertex v =
          crf_graph_type::add_factor_no_check(f.output_arguments(),
                                              f.input_arguments_ptr(),
                                              &(factors_.back()));
        factor_v2i[v] = factors_.size() - 1;
      }
      // Check to make sure Y,X stay separate.
      if (!set_disjoint(Y_, X_)) {
        throw std::invalid_argument
          (std::string("crf_model::add_factor() given overlapping Y,X domains:")
           + "\nY: " + to_string(Y_) + "\nX: " + to_string(X_) + "\n");
      }
    }

    //! Clears all factors and variables from this model.
    void clear() {
      factors_.clear();
      factor_v2i.clear();
      weights_.clear();
      this->crf_graph_type::clear();
      conditioned_model_valid = false;
    }

    /**
     * Removes factors whose arguments are included in other factors.
     */
    void simplify() {
      std::vector<vertex> removed_vertices(crf_graph_type::simplify());
      if (removed_vertices.size() != 0) {
        std::vector<crf_factor> new_factors;
        foreach(const vertex& v, removed_vertices)
          new_factors.push_back(*(this->operator[](v)));
        clear();
        add_factors(new_factors);
      }
    }

    /**
     * Simplifies the model by removing unary factors (over a single Y argument)
     * if another factor contains that Y argument (as well as the X arguments).
     * This only removes unary factors whose argument is in vars.
     * @todo Make this more efficient.
     */
    void simplify_unary(const output_domain_type& vars) {
      std::set<vertex> removed_vertices;
      foreach(output_variable_type y, vars) {
        foreach(const vertex& u, neighbors(y)) {
          if (output_arguments(u).size() == 1 &&
              output_arguments(u).count(y)) {
            // Check for a covering factor.
            foreach(const vertex& v, neighbors2(u)) {
              if (removed_vertices.count(v))
                continue;
              if (output_arguments(v).count(y) &&
                  includes(input_arguments(v), input_arguments(u))) {
                removed_vertices.insert(u);
                break;
              }
            }
          }
        }
      }
      if (removed_vertices.size() != 0) {
        std::vector<crf_factor> new_factors;
        foreach(const vertex& v, removed_vertices)
          new_factors.push_back(*(this->operator[](v)));
        clear();
        add_factors(new_factors);
      }
    } // simplify_unary

    /**
     * Relabels outputs Y, inputs X so that:
     *  - inputs may become outputs (if variable_type = output_variable_type)
     *  - outputs may become inputs (if variable_type = input_variable_type).
     * The entire argument set remains the same.
     *
     * Note: new_Y and new_X must be disjoint,
     *       and their union must be a superset of the union of the old Y,X.
     */
    void relabel_outputs_inputs(const output_domain_type& new_Y,
                                const input_domain_type& new_X) {
      if (!crf_factor::valid_output_input_relabeling
          (output_arguments(), input_arguments(), new_Y, new_X)) {
        throw std::invalid_argument("crf_model::relabel_outputs_inputs given new_Y,new_X whose union did not equal the union of the old Y,X.");
      }
      std::list<crf_factor> new_factors(factors_);
      foreach(crf_factor& f, new_factors) {
        f.relabel_outputs_inputs(new_Y, new_X);
      }
      this->clear();
      this->add_factors(new_factors);
      conditioned_model_valid = false;
    } // relabel_outputs_inputs

    //! Prints the arguments and factors of the model.
    template <typename OutputStream>
    void print(OutputStream& out) const {
      crf_graph_type::print(out);
      out << "Factors:\n";
      foreach(const crf_factor& f, factors())
        out << f;
    }

    // NOTE: See crf_graph for more methods I could potentially add.

    // UNSAFE mutating methods
    // WARNING: Do not use these unless you know what you are doing.
    // =========================================================================

    /**
     * Mutable access to the factors in this model.
     * WARNING: Changing the factor arguments can cause fatal errors!
     *          This should only be used for, e.g., calling
     *          (factor).fixed_value().
     */
    std::list<crf_factor>& factors() { return factors_; }

    // Protected data
    // =========================================================================
  protected:

    using crf_graph_type::Y_;
    using crf_graph_type::X_;

    // Private types
    // =========================================================================
  private:

    //! Functor used to make a transformed range for conditioning factors.
    //! @tparam SampleType  input record or assignment type
    template <typename SampleType>
    struct factor_conditioner
      : public std::unary_function<const crf_factor&,const output_factor_type&>{
      const SampleType* r_ptr;
      mutable const output_factor_type* tmpf_ptr;
      factor_conditioner()
        : r_ptr(NULL), tmpf_ptr(NULL) { }
      explicit factor_conditioner(const SampleType& r)
        : r_ptr(&r), tmpf_ptr(NULL) { }
      const output_factor_type& operator()(const crf_factor& f) const {
        assert(r_ptr);
        tmpf_ptr = &(f.condition(*r_ptr));
        return (*tmpf_ptr);
      }
    }; // struct factor_conditioner

    // Private data
    // =========================================================================

    //! All the factors.
    std::list<crf_factor> factors_;

    //! OptimizationVector which contains pointers to the factors' weights.
    opt_variables weights_;

    //! Map: factor vertex in crf_graph --> index in weights_ for that factor
    std::map<vertex, size_t> factor_v2i;

    /**
     * This makes multiple calls which compute P(Y | X = x) more efficient.
     * It allows this CRF to make use of the same underlying decomposable
     * graph for inference without redoing unnecessary computation.
     *
     * Note: Methods which do conditioning must check to make sure this is
     *       valid and recompute it if not.
     *       Methods which change the CRF structure must invalidate this.
     *       See the conditioned_model_valid bit.
     */
    mutable decomposable<output_factor_type> conditioned_model;

    //! Mapping returned by conditioned_model_vertex_mapping().
    //! This let decomposable::replace_factors() avoid searching for
    //! clique covers when it is called to condition this CRF.
    mutable std::vector<typename decomposable<output_factor_type>::vertex>
      conditioned_model_vertex_map_;

    //! Indicates if the above model's structure is valid.
    mutable bool conditioned_model_valid;

    // Private methods
    // =========================================================================

    // Import method from base class.
//    using crf_graph_type::simplify;

    //! Helper method for copy constructor, assignment operator,
    //! and deserialization.
    void init() {
      if (debug)
        std::cerr << "WARNING: crf_model.debug is set to TRUE"
                  << " (which reduces efficiency significantly)." << std::endl;
      weights_.clear();
      weights_.own_data = false;

      // factor_i2v is for setting factor pointers in crf_graph.
      std::vector<vertex> factor_i2v(factor_v2i.size(),
                                     crf_graph_type::null_vertex());
      typedef typename std::map<vertex, size_t>::value_type v_size_pair;
      foreach(const v_size_pair& v_i, factor_v2i) {
        assert(v_i.second < factor_i2v.size());
        if (factor_i2v[v_i.second] != crf_graph_type::null_vertex()) { // DEBUGGING
          std::cerr << "v_i: " << v_i << "\n"
                    << "factor_i2v[v_i.second]: " << factor_i2v[v_i.second]
                    << "\n"
                    << std::endl;
        }
        assert(v_i.first != crf_graph_type::null_vertex()); // DEBUGGING
        factor_i2v[v_i.second] = v_i.first;
      }

      size_t j = 0;
      foreach(crf_factor& f, factors_) {
        if (!f.fixed_value())
          weights_.factor_weights_.push_back(&(f.weights()));
        assert(factor_i2v[j] != crf_graph_type::null_vertex());
        crf_graph_type::operator[](factor_i2v[j]) = &f;
        ++j;
      }

      conditioned_model_valid = false;
    } // init

    //! Sets the conditioned_model to be valid for the given datapoint.
    //! @tparam SampleType  input record or assignment type
    template <typename SampleType>
    void condition_model(const SampleType& r) const {
      if (debug) {
        // Check to make sure each factor is normalizable.
        foreach(const crf_factor& f, factors()) {
          const output_factor_type& tmpf = f.condition(r);
          if (!tmpf.is_normalizable()) {
            std::cerr << "crf_model::condition_model() tried to condition"
                      << " a CRF factor and got an unnormalizable result.\n"
                      << "CRF factor:\n"
                      << f << "\n"
                      << "resulting factor:\n"
                      << tmpf << std::endl;
            throw normalization_error
              (std::string("crf_model::condition_model ran into a factor") +
               " which could not be normalized after conditioning.");
          }
        }
      }
      if (conditioned_model_valid) {
        conditioned_model.replace_factors
          (make_transformed(factors(), factor_conditioner<SampleType>(r)),
           conditioned_model_vertex_map_);
      } else {
        conditioned_model.clear();
        conditioned_model *=
          make_transformed(factors(), factor_conditioner<SampleType>(r));
        conditioned_model_valid = true;
        set_conditioned_model_vertex_mapping(conditioned_model_vertex_map_);
      }
    } // condition_model

    //! See conditioned_model_vertex_mapping().
    void set_conditioned_model_vertex_mapping
    (std::vector<typename decomposable<output_factor_type>::vertex>& vm) const {
      typedef typename decomposable<output_factor_type>::vertex cm_vertex_type;
      assert(conditioned_model_valid);
      vm.clear();
      foreach(const crf_factor& f, factors()) {
        cm_vertex_type
          v(conditioned_model.find_clique_cover(f.output_arguments()));
        if (v == conditioned_model.null_vertex()) {
          std::cerr << "crf_model::set_conditioned_model_vertex_mapping"
                    << " could not find a clique cover in conditioned_model"
                    << " for this factor:\n"
                    << f
                    << "conditioned_model.arguments: "
                    << conditioned_model.arguments() << std::endl;
          throw std::runtime_error
            (std::string("crf_model::set_conditioned_model_vertex_mapping") +
             " failed due to internal error!");
        }
        vm.push_back(v);
      }
    }

  }; // crf_model


  /**
   * Write a CRF to an output stream using its built-in print() function.
   */
  template <typename F>
  std::ostream& operator<<(std::ostream& out, const crf_model<F> crf) {
    crf.print(out);
    return out;
  }

  // Specializations of model functors for crf_model types
  //============================================================================

  template <typename F>
  struct model_conditional_log_likelihood_functor<crf_model<F> > {

    explicit
    model_conditional_log_likelihood_functor
    (const crf_model<F>& model,
     const typename crf_model<F>::output_domain_type& X,
     double base = exp(1.))
      : modelptr(&model), X(X), base(base) { }

    double operator()(const typename crf_model<F>::record_type r) const {
      assert(modelptr);
      return modelptr->conditional_log_likelihood(r, X, base);
    }

  private:
    const crf_model<F>* modelptr;
    typename crf_model<F>::output_domain_type X;
    double base;
  };

  template <typename F>
  struct model_per_label_accuracy_functor<crf_model<F> > {

    //! Constructor for accuracy w.r.t. all arguments of the model.
    explicit model_per_label_accuracy_functor(const crf_model<F>& model)
      : modelptr(&model) { }

    //! Constructor for accuracy of predicting Y given X, where the model
    //! is of P(Y,X).
    explicit
    model_per_label_accuracy_functor
    (const crf_model<F>& model,
     const typename crf_model<F>::output_domain_type& X)
      : modelptr(&model), X(X) { }

    double operator()(const typename crf_model<F>::record_type r) const {
      assert(modelptr);
      if (X.size() == 0)
        return modelptr->per_label_accuracy(r);
      else
        return modelptr->per_label_accuracy(r, X);
    }

  private:
    const crf_model<F>* modelptr;
    const typename crf_model<F>::output_domain_type X;
  };

  template <typename F>
  struct model_mean_squared_error_functor<crf_model<F> > {

    //! Constructor for mean squared error w.r.t. all arguments of the model.
    explicit model_mean_squared_error_functor(const crf_model<F>& model)
      : modelptr(&model) { }

    //! Constructor for mean squared error of predicting Y given X,
    //! where the model is P(Y,X).
    explicit
    model_mean_squared_error_functor
    (const crf_model<F>& model,
     const typename crf_model<F>::output_domain_type& X)
      : modelptr(&model), X(X) { }

    double operator()(const typename crf_model<F>::record_type r) const {
      assert(modelptr);
      if (X.size() == 0)
        return modelptr->mean_squared_error(r);
      else
        return modelptr->mean_squared_error(r, X);
    }

  private:
    const crf_model<F>* modelptr;
    typename crf_model<F>::output_domain_type X;
  };

}  // namespace sill

#include <sill/macros_undef.hpp>

#endif // SILL_CRF_MODEL_HPP
