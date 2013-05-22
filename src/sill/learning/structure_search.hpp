
#ifndef SILL_STRUCTURE_SEARCH_HPP
#define SILL_STRUCTURE_SEARCH_HPP

#include <sill/learning/decomposable_iterator.hpp>

/**
 * \file structure_search.hpp  Incomplete structure search class; I need to
 *                             reorganize structure search.
 */

/*

  STUFF TO DO:
  -There is a merge() function in decomposable!
   See where it is used and if it does what its documentation says,
   and decide if we should use it. Note: it removes the source.
  -We should have something in junction trees which maps each variable
   to a clique which contains it so that operations involving a single
   variable runs in time proportional to the size of the variable's subtree.
  -restrict maps should be changed so that only the max limit value means no
   restriction.
  -Try using a multidimensional array for data.hpp
  -I think the problem's I've been having have been caused by the copy
   constructor in decomposable_t:
   -Copying the decomposable model seems to screw up the iterator over
    edges, even those edges which are stored separately in a vector.
     (See email to Stano.)
   -These are things which should be copied by the decomposable_t copy
    constructor:
    -set_index
    -jt_ptr (junction_tree_t)
     -jt_vertex_info_t
      -aux_vertex_info_t (decomposable_t::vertex_info_t)
       -marginal_ptr
     -jt_undir_edge_info_t
      -aux_undir_edge_info_t (decomposable_t::edge_info_t)
       -marginal_ptr
     -jt_dir_edge_info_t
      -aux_dir_edge_info_t (not used here, but maybe later)
      -undir_info_ptr
  -change factors so that 'args' and 'arguments()' are in factor_t,
   not in each individual factor
  -Maybe it's a bad idea how restrict_map is done since it could be messed
   up if the table's dimensions are increased.

  STEP BACK & THINK ABOUT HOW WE WANT TO DO STRUCTURE SEARCH:
  -models
   -over X,E or just X
   -calibrated, uncalibrated
  -objectives
   -involving X,E or X|E or X|e
   -decomposable, non-decomposable
   -Some should be built in, but the user should also be able to specify
    new ones.  They should be classes with:
     -fields specifying: if decomposable, if model must be calibrated
     -function to compute objective initially
       -should take: entire model
     -function to recompute objective
       -should take: removed potentials, added potentials, entire model
   -think about some objectives:
     -qs_cond_entropy:
       -restrict to e, recalibrate, compute entropy
       -objective is essentially decomposable
       -recalibrating affects all cliques, but it just divides everything
        by a constant factor:  a log(a) + b log(b)    Let z = a + b
           (1/z)[a log(a/z) + b log(b/z)]
         = (1/z)[a log(a) - a log(z) + b log(b) - b log(z)]
         = -log(z) + (1/z)[a log(a) + b log(b)]
  -different initial models, then take steps to find better models
   -record stuff about the best models for each initial model?
    (for measuring the importance of trying every initial model)
   -restrict set of steps using L1 stuff, known locality, or other heuristics

  structure_search
   \--> decomposable_iterator

  structure_search:
   -should have built-in objectives
  decomposable_iterator:
   -needs to save references to potentials which were just removed and
    just added and make them publicly available
  decomposable:
   -needs to store local info about objective, e.g. entropy of potential

  NEW THOUGHTS ABOUT DECOMPOSABLE_ITERATOR AND STRUCTURE_SEARCH:
  -have local_step concept:
   -operator() to get value of step
   -apply() function to apply step to model
   -valid() function to check if step is still valid for model?
   -This concept should probably be the same for all possible steps and
    will store generic info about changes to cliques (& marginals?).
  -strategy concept:
   -collection of local_steps
   -one strategy for each type of local step, plus a concatenate strategy
   -Takes current model, objective, data.
  -init: free functions for iterating over initial models
   -empty, star
  -structure search:
   -take init, strategy, objective, data
   -foreach init,
    -maintain queue of possible steps & iterate:
     -create strategy for current model, and add all steps to queue
     -choose best step, take it, and delete invalid steps in queue
   -Maintaining queue of possible steps:
    -how should we check for invalid steps?
     -look at best step & check if valid
     -keep pointers to steps in each clique, and delete steps as soon as clique
      changes
     -Would #2 require less memory b/c it wouldn't store invalid
      steps, or would it require more since there would be tons of pointers?
      If we did #1, we'd have to store extra info about the relevant cliques
      in each local step object, which would take a lot more space that a
      single pointer in each clique.  So let's do #2.

  ORGANIZING DECOMPOSABLE MODELS & OTHER TYPES OF MODELS:
  -cluster graphs, JTs, CRFs, CRFs with tree structure, Markov nets, Bayes nets
  -hmmm.....

*/

namespace sill {

  //! Structure learning objectives.
  enum structure_objective_enum {
    /**
     * Entropy of P_{JT}(X | E = e)
     * For models over X,E
     */
    QS_COND_ENTROPY
  };

  // TODO: This expects a decomposable model, but it should eventually handle
  //   more general model types.
  /**
   * Class for structure learning via heuristic search.
   * Template parameter is factor type.
   *
   * \ingroup learning_structure
   */
  template <typename F>
  class structure_search {

  protected:

    // TODO: We need to decide how to store local objective info (like the
    //   entropy of potentials) in the decomposable model, after which we
    //   should modify these objectives to use that info.
    //   Or is it worth storing these?
    //! Abstract class for objectives for structure search.
    class objective_t {

    protected:

      //! Current value of objective.
      double value;

    public:

      //! True if objective is decomposable.
      const bool is_decomposable;
      //! True if objective requires calibrated model.
      const bool requires_calibrated;

      /**
       * Constructor.
       *
       * @param is_decomposable     True if objective is decomposable.
       * @param requires_calibrated True if objective is for calibrated model.
       */
      explicit objective_t(bool is_decomposable = false,
                           bool requires_calibrated = true)
	: is_decomposable(is_decomposable),
	  requires_calibrated(requires_calibrated)
      { }

      //! Assignment.
      objective_t& operator=(const objective_t& o) {
	value = o.value;
	return *this;
      }

      virtual ~objective_t() { }

      //! Recompute objective after non-local modification.
      virtual double recompute(const decomposable<F>& model,
			       const data_t& data = data_t()) {
	std::cerr << "Objective does not implement proper recompute function!"
		  << std::endl;
	assert(false);
      }

      //! Recompute objective after local modification.
      virtual double recompute
      (const std::list<F>& removed_potentials,
       const std::list<F>& added_potentials) {
	std::cerr << "Objective does not implement proper recompute function!"
		  << std::endl;
	assert(false);
      }
      // RIGHT HERE NOW: How can I use Range?  You can't have templated
      //  virtual functions.  I could put Range in the class template, but
      //  that doesn't really make sense.

      //! Return current objective value.
      double get_value() {
	return value;
      }

    };

    // TODO: We'll eventually want to store the normalization constant of
    //   the model, though this is currently unnecessary since the decomposable
    //   type is always calibrated.
    /**
     * Class for Query-Specific conditional entropy objective for
     * structure search.
     */
    class objective_qs_cond_entropy_t : public objective_t {

    protected:

      //! Evidence in query.
      assignment evidence;

      //! Computes objective for entire model.
      void initialize(const decomposable<F>& model) {
	decomposable<F> tmp_model(model);
	tmp_model.condition(evidence);
	this->value = tmp_model.entropy();
      }

    public:

      //! Constructor.  Computes initial objective.
      objective_qs_cond_entropy_t(const decomposable<F>& model,
				  const assignment& evidence)
	: objective_t(true, false), evidence(evidence) {
	initialize(model);
      }

      //! Recompute objective after non-local modification.
      double recompute(const decomposable<F>& model,
		       const data_t& data = data_t()) {
	initialize(model);
	return this->value;
      }

      /**
       * Recompute objective after local modification.
       * For decomposable models, this assumes that separator potentials are
       * divided into clique potentials or that removed separators are in
       * added_potentials and vice versa.
       */
      double recompute
      (const std::list<F&>& removed_potentials,
       const std::list<F&>& added_potentials) {
	
	for (typename std::list<F&>::const_iterator it
	       = removed_potentials.begin();
	     ++it; it != removed_potentials.end())
	  this->value -= it->entropy();
	for (typename std::list<F&>::const_iterator it
	       = added_potentials.begin();
	     ++it; it != added_potentials.end())
	  this->value += it->entropy();
	return this->value;
      }

    }; // class objective_qs_cond_entropy_t

    /**
     * Current model.
     * Use copy-on-write pointers to avoid unnecessary copies when saving best
     * model.
     */
    copy_ptr<decomposable<F> > model_ptr;

    //! Best model found so far.
    copy_ptr<decomposable<F> > best_model_ptr;

    //! Score of best model found so far.
    double best_score;

    //! Query variables (if given).
    domain query_vars;

    //! Evidence in query (if given).
    assignment evidence;

    //! Dataset.
    const data_t& data;

    //! Types of initial model structures to try.
    std::vector<structure_initial_enum> initial_structures;

    //! Types of steps allowed.
    std::vector<structure_step_enum> step_types;

    //! Structure learning objective.
    structure_objective_enum which_objective;

    //! Parameter estimation method.
    param_method_enum param_method;

    //! Max clique size allowed.
    domain::size_type max_clique_size;

    //! Amount by which to smooth marginals.
    double smooth;

    //! Iterator over initial structures.
    decomposable_iterator<F>
    initial_structure_it;

    //! Class for computing objective.
    copy_ptr<objective_t> objective_ptr;

    //! Number of search steps taken so far.
    std::size_t n_steps;

    //! Record of scores for current initial structure, from each round.
    copy_ptr<std::vector<double> > scores;

    //! Record of scores for each initial structure, from each round.
    std::list<copy_ptr<std::vector<double> > > all_scores;

  public:

    // TODO: Add param to constructor to choose what kind of structures
    //   (e.g. decomposable) to iterator over.
    /**
     * Constructor for query-specific structure learning via search:
     *  P(X | E = e).
     * Minimizes score specified by which_objective.
     * For each initial structure, performs heuristic search via the allowed
     *  local steps.
     *
     * @param query_vars         query variables (X)
     * @param evidence           instantiated evidence variables (e)
     * @param data               dataset
     * @param initial_structures types of initial structures to try
     * @param step_types         types of steps allowed
     * @param which_objective    structure learning objective
     * @param param_method       parameter estimation method
     * @param max_clique_size    max clique size allowed
     * @param num_steps          number of steps to take initially
     * @param smooth             amount by which to smooth marginals, default 0
     */
    structure_search(domain query_vars, assignment evidence,
		     const data_t& data,
		     std::vector<structure_initial_enum> initial_structures,
		     std::vector<structure_step_enum> step_types,
		     structure_objective_enum which_objective,
		     param_method_enum param_method,
		     domain::size_type max_clique_size = 2,
		     std::size_t num_steps = 0,
		     double smooth = 0)
      : query_vars(query_vars), evidence(evidence), data(data),
	initial_structures(initial_structures), step_types(step_types),
	which_objective(which_objective), param_method(param_method),
	max_clique_size(max_clique_size), smooth(smooth) {

      domain model_vars;
      switch(which_objective) {
      case QS_COND_ENTROPY:
	model_vars = query_vars.plus(evidence.keys());
	break;
      default:
	std::cerr << "Unknown value for param which_objective." << std::endl;
	assert(false);
      }
      // Make iterator over initial structures.
      initial_structure_it
	= decomposable_iterator<F>
	(data, model_vars, initial_structures, param_method, max_clique_size,
	 smooth);
      model_ptr
	= copy_ptr<decomposable<F> >(new decomposable<F>(*initial_structure_it));
      // Compute initial score of objective.
      switch(which_objective) {
      case QS_COND_ENTROPY:
	objective_ptr = copy_ptr<objective_t>
	  (new objective_qs_cond_entropy_t(*model_ptr, evidence));
	break;
      default:
	std::cerr << "Unknown value for param which_objective." << std::endl;
	assert(false);
      }
      n_steps = 0;
      best_score = std::numeric_limits<double>::max();
      best_model_ptr = copy_ptr<decomposable<F> >(new decomposable<F>());
      scores = copy_ptr<std::vector<double> >(new std::vector<double>());
      scores->push_back(objective_ptr->get_value());
      while (n_steps < num_steps) {
	if (!step())
	  break;
      }
    }

    /**
     * Take one search step.
     *
     * @return boolean indicating if the step was successful, or failed
     *         because e.g. no better model could be found
     */
    bool step() {
      // For each possible step (change in current model)
      //    Compute new parameters
      //    Compute score
      // Choose best new model
      // If no model is an improvement, then try the next initial structure

      decomposable_iterator<F>
	it(*model_ptr, data, step_types, param_method, max_clique_size, smooth);
      decomposable_iterator<F> end;
      double cur_best_score = scores->back();
      while (it != end) {
//	double score = objective.recompute(it.get_removed_potentials(),
//					   it.get_added_potentials());
	double score = objective_ptr->recompute(*it);

	std::cout << "  score = " << score << std::endl;

	if (score < cur_best_score) {
	  cur_best_score = score;
	  model_ptr = copy_ptr<decomposable<F> >(new decomposable<F>(*it));
	}
	++it;
      }

      if (cur_best_score == scores->back()) {
	if (scores->back() < best_score) {
	  best_score = scores->back();
	  best_model_ptr = model_ptr;
	}
	all_scores.push_back(scores);
	scores = copy_ptr<std::vector<double> >(new std::vector<double>());
	++initial_structure_it;
	if (initial_structure_it == end)
	  return false;
	model_ptr
	  = copy_ptr<decomposable<F> >(new decomposable<F>(*initial_structure_it));
	objective_ptr->recompute(*model_ptr);
	cur_best_score = objective_ptr->get_value();
      }
      scores->push_back(cur_best_score);
      ++n_steps;

      return true;
    }

    //! Initial structure.
    const std::vector<structure_initial_enum>& get_initial_structures() const {
      return initial_structures;
    }

    //! Types of steps allowed.
    const std::vector<structure_step_enum>& get_step_types() const {
      return step_types;
    }

    //! Structure learning objective.
    const structure_objective_enum get_which_objective() const {
      return which_objective;
    }

    //! Parameter estimation method.
    const param_method_enum get_param_method() const {
      return param_method;
    }

    //! Current model.
    const decomposable<F>& current_model() const {
      return *model_ptr;
    }

    //! Best model found so far.
    const decomposable<F>& get_best_model() const {
      return *model_ptr;
    }

    //! Query variables (if given).
    const domain get_query() const {
      return query_vars;
    }

    //! Evidence in query (if given).
    const assignment get_evidence() const {
      return evidence;
    }

    //! Number of search steps taken so far.
    const size_t get_n_steps() const {
      return n_steps;
    }

    //! Record of scores from each round for current initial structure.
    const std::vector<double>& get_scores() const {
      return *scores;
    }

    //! Record of scores from each round for previous initial structures.
    const std::list<copy_ptr<std::vector<double> > >& get_all_scores() const {
      return all_scores;
    }

    //! Current score for this round.
    const double current_score() const {
      return scores->back();
    }

  }; // class structure_search

} // namespace sill

#endif // #ifndef SILL_STRUCTURE_SEARCH_HPP
