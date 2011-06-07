#ifndef SILL_GIBBS_SAMPLER_HPP
#define SILL_GIBBS_SAMPLER_HPP

#include <map>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

#include <sill/base/random_assignment.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/math/permutations.hpp>
#include <sill/model/interfaces.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  // Forward declaration
  class table_factor;

  /**
   * Class for doing Gibbs sampling for a factorized model.
   *
   * @tparam F  Type of factor.
   *
   * \author Joseph Bradley
   */
  template <typename F>
  class gibbs_sampler {

    // Public types
    //==========================================================================
  public:

    typedef F factor_type;
    typedef typename F::variable_type    variable_type;
    typedef typename F::domain_type      domain_type;
    typedef typename F::var_vector_type  var_vector_type;
    typedef typename F::assignment_type  assignment_type;
    typedef typename F::record_type      record_type;

    //! Parameters
    struct parameters {

      parameters()
        : init_r(), random_seed(time(NULL)) { }

      //! Initial state of the sampler.
      //! If empty, reverts to default (random).
      //! Note: If set, then this specifies the variable ordering in the
      //!       samples.  Otherwise, the ordering is arbitrary.
      //!  (default = random)
      record_type init_r;

      //! (default = time)
      unsigned random_seed;

    }; // struct parameters

    // Public methods
    //==========================================================================
  public:

    //! Default constructor.
    gibbs_sampler()
      : uniform_prob(0,1), last_v(NULL) { }

    /**
     * Create a Gibbs sampler for the given model.
     * @param var_seq  If specified, use this variable sequence internally.
     *                 (This determines the indexing used by next_variable().)
     */
    gibbs_sampler(const factorized_model<factor_type>& model,
                  const var_vector_type& var_seq = var_vector_type(),
                  const parameters& params = parameters())
      : params(params), rng(params.random_seed), uniform_prob(0,1),
        var_sequence(var_seq), r(params.init_r), last_v((size_t)(-1)) {
      init(model);
    } // constructor

    //! Acts like a constructor.
    void reset(const factorized_model<factor_type>& model,
               const var_vector_type& var_seq = var_vector_type(),
               const parameters& params = parameters()) {
      this->params = params;
      rng.seed(params.random_seed);
      var_sequence = var_seq;
      r = params.init_r;
      last_v = (size_t)(-1);
      init(model);
    }

    virtual ~gibbs_sampler() { }

    /**
     * Get the next sample from the model (updating a single variable).
     */
    const record_type& next_sample() {
      last_v = next_variable();

      // Restrict the factors containing last_v, multiply them together,
      //  and normalize the resulting factor.
      factor_type& f = singleton_factors[last_v];
      factor_type& f_tmp = singleton_factors_tmp[last_v];
      f = 1;
      foreach(size_t fptr_i, var2factors[last_v]) {
        const factor_type* fptr = factor_ptrs[fptr_i];
        fptr->restrict
          (r,
           set_difference(fptr->arguments(), make_domain(var_sequence[last_v])),
           f_tmp);
        f *= f_tmp;
      }
      f.normalize();
      // Sample a new value for last_v.
      r.copy_from_assignment(f.sample(rng));
      return r;
    } // next_sample

    //! Return the current sample from the model.
    const record_type& current_sample() const { return r; }

    //! Return the last variable which was updated.
    //! This returns NULL if no samples have been taken.
    variable_type* last_variable() const {
      if (last_v < var_sequence.size())
        return var_sequence[last_v];
      else
        return NULL;
    }

    //! Reset the random seed.
    void random_seed(unsigned seed) {
      rng.seed(seed);
    }

    //! Resets the current sample to a random assignment.
    void randomize_current_sample() {
      domain_type args(r.variables());
      r = random_assignment(args, rng);
    }

    // Protected data and methods
    //==========================================================================
  protected:

    parameters params;

    //! random number generator
    boost::mt11213b rng;

    //! uniform distribution over [0, 1]
    boost::uniform_real<double> uniform_prob;

    //! List of pointers to factors in the model.
    std::vector<const factor_type*> factor_ptrs;

    //! Variable ordering.
    //! The next_variable method returns indices into this ordering.
    var_vector_type var_sequence;

    //! var2factors[index in var_sequence]
    //!  = indices in factor_ptrs for factors with that variable
    std::vector<std::vector<size_t> > var2factors;

    //! var2factors[v] = pointers to factors which include variable v
//    std::map<variable_type*, std::vector<const factor_type*> > var2factors;

    //! Current assignment to variables
    record_type r;

    //! Last variable which was updated
    size_t last_v;
    //    variable_type* last_v;

    //! singleton_factors[index into var_sequence]
    //!   = factor whose domain is only that variable
    //! (used to avoid reallocation)
    std::vector<factor_type> singleton_factors;

    //! Same as singleton_factors, but used for a separate purpose.
    std::vector<factor_type> singleton_factors_tmp;

    //! r_numberings[index in factor_ptrs]
    //!   = indices in r for factor's arguments (in the factor's natural order)
    std::vector<ivec> r_numberings;

    //! Returns the index of the next variable to update.
    virtual size_t next_variable() = 0;

    void init(const factorized_model<factor_type>& model) {
      // Set factor_ptrs, var_sequence.
      factor_ptrs.clear();
      domain_type vars;
      foreach(const factor_type& f, model.factors()) {
        vars.insert(f.arguments().begin(), f.arguments().end());
        factor_ptrs.push_back(&f);
      }
      if (var_sequence.size() == 0) {
        std::vector<size_t> perm(randperm(vars.size(), rng));
        var_sequence.resize(vars.size());
        size_t j = 0;
        foreach(variable_type* v, vars) {
          var_sequence[perm[j]] = v;
          ++j;
        }
      } else {
        assert(vars == make_domain(var_sequence));
      }

      // If no initial assignment was given, choose one uniformly at random.
      if (r.num_variables() == 0) {
        r = record_type(var_sequence);
        r = random_assignment(model.arguments(), rng);
      }

      // Set var2factors, r_numberings.
      std::map<variable_type*, size_t> var2index;
      for (size_t i = 0; i < var_sequence.size(); ++i) {
        var2index[var_sequence[i]] = i;
      }
      var2factors.clear();
      var2factors.resize(var_sequence.size());
      r_numberings.clear();
      r_numberings.resize(factor_ptrs.size());
      for (size_t i = 0; i < factor_ptrs.size(); ++i) {
        const factor_type& f = *(factor_ptrs[i]);
        foreach(variable_type* v, f.arguments()) {
          var2factors[var2index[v]].push_back(i);
        }
        f.set_record_indices(r, r_numberings[i]);
      }

      // Assert that each variable has at least one associated factor.
      foreach(const std::vector<size_t>& fs, var2factors) {
        assert(fs.size() > 0);
      }

      singleton_factors.clear();
      singleton_factors_tmp.clear();
      foreach(variable_type* v, var_sequence)
        singleton_factors.push_back(factor_type(make_domain(v)));
      singleton_factors_tmp = singleton_factors;
    } // init(model)

  }; // class gibbs_sampler

  /**
   * Class for doing Gibbs sampling for a factorized model.
   * The variables are sampled in a fixed order.
   *
   * @tparam F  Type of factor.
   */
  template <typename F>
  class sequential_gibbs_sampler : public gibbs_sampler<F> {

    using gibbs_sampler<F>::rng;

    // Public types
    //==========================================================================
  public:

    typedef gibbs_sampler<F> base;

    typedef F factor_type;
    typedef typename F::variable_type    variable_type;
    typedef typename F::domain_type      domain_type;
    typedef typename F::var_vector_type  var_vector_type;
    typedef typename F::assignment_type  assignment_type;

    typedef typename base::parameters    parameters;

    // Public methods
    //==========================================================================
  public:

    //! Default constructor.
    sequential_gibbs_sampler() { }

    /**
     * Create a Gibbs sampler for the given model.
     * @param var_order  order in which to sample the variables
     *                   (default = random)
     * @param params     parameters for the base class (gibbs_sampler)
     */
    sequential_gibbs_sampler(const factorized_model<factor_type>& model,
                             const var_vector_type& var_order
                             = var_vector_type(),
                             const parameters& params = parameters())
      : base(model, var_order, params), next_i(0) {
    }

    //! Acts like a constructor.
    void reset(const factorized_model<factor_type>& model,
               const var_vector_type& var_order = var_vector_type(),
               const parameters& params = parameters()) {
      base::reset(model, var_order, params);
      next_i = 0;
    }

    // Protected data and methods
    //==========================================================================
  protected:

    using base::var_sequence;

    //! Next index (into var_sequence) to return
    size_t next_i;

    //! Returns the index of the next variable to update.
    size_t next_variable() {
      if (var_sequence.size() == 0)
        return (size_t)(-1);
      size_t this_i(next_i);
      ++next_i;
      if (next_i >= var_sequence.size())
        next_i = 0;
      return this_i;
    }

  }; // class sequential_gibbs_sampler


  // Specialization of gibbs_sampler methods
  //============================================================================

  template <>
  const gibbs_sampler<table_factor>::record_type&
  gibbs_sampler<table_factor>::next_sample();

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
