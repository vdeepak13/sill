#ifndef SILL_GIBBS_SAMPLER_HPP
#define SILL_GIBBS_SAMPLER_HPP

#include <map>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>

#include <sill/base/random_assignment.hpp>
#include <sill/factor/concepts.hpp>
#include <sill/math/permutations.hpp>
#include <sill/model/interfaces.hpp>

//#include <sill/range/transformed.hpp>

#include <sill/macros_def.hpp>

namespace sill {

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

    //! Parameters
    struct parameters {

      parameters()
        : init_a(), random_seed(time(NULL)) { }

      //! Initial state of the sampler.
      //! If empty, reverts to default (random).
      //!  (default = random)
      assignment_type init_a;

      //! (default = time)
      unsigned random_seed;

    }; // struct parameters

    // Public methods
    //==========================================================================
  public:

    //! Default constructor.
    gibbs_sampler()
      : uniform_prob(0,1), last_v(NULL) { }

    //! Create a Gibbs sampler for the given model.
    gibbs_sampler(const factorized_model<factor_type>& model,
                  const parameters& params = parameters())
      : params(params), rng(params.random_seed), uniform_prob(0,1),
        a(params.init_a), last_v(NULL) {
      init(model);
    } // constructor

    //! Acts like a constructor.
    void reset(const factorized_model<factor_type>& model,
               const parameters& params = parameters()) {
      this->params = params;
      rng.seed(params.random_seed);
      a = params.init_a;
      last_v = NULL;
      init(model);
    }

    virtual ~gibbs_sampler() { }

    /**
     * Get the next sample from the model (updating a single variable).
     */
    const assignment_type& next_sample() {
      last_v = next_variable();
      // Restrict the factors containing last_v, multiply them together,
      //  and normalize the resulting factor.
      assignment_type tmp_a(a);
      tmp_a.erase(last_v);
      factor_type vfactor(1);
      foreach(const factor_type* fptr, var2factors[last_v])
        vfactor *= fptr->restrict(tmp_a);
      vfactor.normalize();
      // Sample a new value for last_v.
      a[last_v] = vfactor.sample(rng)[last_v];
      return a;
    } // next_sample

    //! Return the current sample from the model.
    const assignment_type& current_sample() const { return a; }

    //! Return the last variable which was updated.
    //! This returns NULL if no samples have been taken.
    variable_type* last_variable() const { return last_v; }

    //! Reset the random seed.
    void random_seed(unsigned seed) {
      rng.seed(seed);
    }

    //! Resets the current sample to a random assignment.
    void randomize_current_sample() const {
      domain_type args(keys(a));
      a = random_assignment(args, rng);
    }

    // Protected data and methods
    //==========================================================================
  protected:

    parameters params;

    //! random number generator
    boost::mt11213b rng;

    //! uniform distribution over [0, 1]
    boost::uniform_real<double> uniform_prob;

    //! var2factors[v] = pointers to factors which include variable v
    std::map<variable_type*, std::vector<const factor_type*> > var2factors;

    //! Current assignment to variables
    assignment_type a;

    //! Last variable which was updated
    variable_type* last_v;

    //! Function returning the next variable to update
    virtual variable_type* next_variable() = 0;

    void init(const factorized_model<factor_type>& model) {
      var2factors.clear();
      foreach(const factor_type& f, model.factors()) {
        foreach(variable_type* v, f.arguments()) {
          if (var2factors.count(v))
            var2factors[v].push_back(&f);
          else
            var2factors[v] = std::vector<const factor_type*>(1,&f);
        }
      }
      // Assert that each variable has at least one associated factor.
      assert(var2factors.size() == model.arguments().size());

      // If no initial assignment was given, choose one uniformly at random.
      if (a.size() == 0)
        a = random_assignment(model.arguments(), rng);
    }

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
      : base(model, params), var_order(var_order), next_i(0) {
      init(model);
    }

    //! Acts like a constructor.
    void reset(const factorized_model<factor_type>& model,
               const var_vector_type& var_order = var_vector_type(),
               const parameters& params = parameters()) {
      base::reset(model, params);
      this->var_order = var_order;
      next_i = 0;
      init(model);
    }

    // Protected data and methods
    //==========================================================================
  protected:

    //! Variables in order used for sampling
    var_vector_type var_order;

    //! Next index (into var_order) to return
    size_t next_i;

    void init(const factorized_model<factor_type>& model) {
      var_vector_type args;
      foreach(variable_type* v, model.arguments())
        args.push_back(v);
      if (var_order.size() == 0) {
        std::vector<size_t> r(randperm(args.size(), rng));
        this->var_order.resize(args.size());
        for (size_t j = 0; j < r.size(); ++j)
          this->var_order[j] = args[r[j]];
      } else
        assert(var_order.size() == args.size());
    }

    //! Function returning the next variable to update
    variable_type* next_variable() {
      if (var_order.size() == 0)
        return NULL;
      size_t this_i(next_i);
      ++next_i;
      if (next_i >= var_order.size())
        next_i = 0;
      return var_order[this_i];
    }

  }; // class sequential_gibbs_sampler

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
