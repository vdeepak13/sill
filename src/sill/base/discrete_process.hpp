namespace sill {
  template <typename V>
  class discrete_process;
}

#ifndef SILL_TIMED_PROCESS_HPP
#define SILL_TIMED_PROCESS_HPP

#include <set>
#include <map>
#include <climits> // for INT_MAX

#include <boost/lexical_cast.hpp>

#include <sill/base/concepts.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/base/process.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup base_types
  //! @{

  //! A constant that represents the current time step
  //! \relates discrete_process
  static const int current_step = INT_MAX - 1;

  //! A constant that represents the next time step
  //! \relates discrete_process
  static const int next_step = INT_MAX;

  /**
   * A process over a discrete timed steps.
   * \param V 
   *        The type of variables, used in this process.  The variable
   *        must have a single argument, size.  For other kinds of
   *        variables, the user needs to specialize this template.
   */
  template <typename V> 
  class discrete_process : public process {
    concept_assert((Variable<V>));

    // Public type and constant declarations
    //==========================================================================
  public:
    //! The variable type associated with this process
    typedef V variable_type;

    //! The type that represents an index
    typedef int index_type;

    // Private data members and constructors
    //==========================================================================
  private:
    //! The number of values that the process takes on at each step
    size_t size_;

    //! The instances of the process at different time steps
    mutable std::map<int, variable_type*> vars; 

    //! A special instance that represents the current time step
    mutable variable_type* var_current;

    //! A special instance that represents the next time step
    mutable variable_type* var_next;

  public:
    // documentation inherited from the base class
    void save(oarchive& ar) const{
      process::save(ar);
      ar << size_;
    }

    // documentation inherited from the base class
    void load(iarchive& ar) {
      // we only support loading of a new process
      assert(var_current == NULL);
      assert(var_next== NULL);
      assert(vars.empty());
      process::load(ar);
      ar >> size_;
    }

    // Public functions
    //==========================================================================
  public:
    //! Default constructor (only used by serialization)
    discrete_process():var_current(NULL), var_next(NULL) { }

    //! Constructs a generic process
    discrete_process(const std::string& name, size_t size) 
      : process(name), size_(size), var_current(NULL), var_next(NULL) { }
    
    //! Deletes the allocated processes
    ~discrete_process() {
      typedef std::pair<int, variable_type*> int_variable_pair;
      if(var_current) delete var_current;
      if(var_next) delete var_next;
      foreach(int_variable_pair p, vars) delete p.second;
    }

    //! Conversion to human-readable representation
    operator std::string() const {
      return "#P(" + name() + "|DT|" 
        + boost::lexical_cast<std::string>(size_) + ")";
    }

    //! Returns the number of assignments
    size_t size() const {
      return size_;
    }

    // documentation copied from the base class
    variable* at_any(const boost::any& index) const {
      return at(boost::any_cast<int>(index));
    }

    /**
     * Returns an instance of the process at the given time step.
     * This function is guaranteed to return the same variable
     * object in multiple invocations. 
     */
    variable_type* at(int step) const {
      if (step == current_step) return current();
      if (step == next_step) return next();
      variable_type*& var = vars[step];
      if (var == NULL) {
        std::string step_str = boost::lexical_cast<std::string>(step);
        process* p = const_cast<discrete_process*>(this);
        var = new variable_type(name() + ":" + step_str, size_, p, step);
      }
      return var;
    }

    /**
     * Returns a special instance that represents the generic
     * variable at the 'current' time step.
     */
    variable_type* current() const {
      if (var_current == NULL) {
        process* p = const_cast<discrete_process*>(this);
        var_current =
          new variable_type(name() + ":t", size_, p, int(current_step));
      }
      return var_current;
    }
    
    /**
     * Returns a special instance that represents the generic
     * variable at the 'next' time step.
     */
    variable_type* next() const {
      if (var_next == NULL) {
        process* p = const_cast<discrete_process*>(this);
        var_next =
          new variable_type(name() + ":t'", size_, p, int(next_step));
      }
      return var_next;
    }

    // documentation inherited from the base class
    void save_variable(oarchive& ar, variable* v) const {
      ar << boost::any_cast<int>(v->index());
    }

    // documentation inherited from the base class
    variable* load_variable(iarchive& ar) const {
      int index;
      ar >> index;
      return at(index);
    }

  }; // class discrete_process  

  //! A timed process over finite variables
  typedef discrete_process<finite_variable> finite_discrete_process;

  //! A timed process over vector variables
  typedef discrete_process<vector_variable> vector_discrete_process;
  
  /**
   * Template specialization for discrete process over arbitrary variable.
   * Only supports finite and vector variables at the moment.
   */
  template <>
  class discrete_process<variable> : public process {
    // Public type and constant declarations
    //==========================================================================
  public:
    //! The variable type associated with this process
    typedef variable variable_type;

    //! The type that represents an index
    typedef int index_type;

    // Private data members and constructors
    //==========================================================================
  private:
    finite_discrete_process* finite;
    vector_discrete_process* vector;

  public:
    // documentation inherited from the base class
    void save(oarchive& ar) const{
      process::save(ar);
      assert(false); // unsupported for now
//       if (finite) { ar << *finite; }
//       if (vector) { ar << *vector; }
    }

    // documentation inherited from the base class
    void load(iarchive& ar) {
      process::load(ar);
      assert(false); // unsupported for now
    }

    // Public functions
    //==========================================================================
  public:
    //! Default constructor (only used by serialization)
    discrete_process()
      : finite(NULL), vector(NULL) { }

    //! Constructs a finite process
    discrete_process(finite_discrete_process* finite)
      : process(finite->name()), finite(finite), vector(NULL) { }

    //! Constructs a vector process
    discrete_process(vector_discrete_process* vector)
      : process(vector->name()), finite(NULL), vector(vector) { }
    
    //! Deletes the allocated processes
    ~discrete_process() {
      if (finite) { delete finite; }
      if (vector) { delete vector; }
    }

    //! Returns true if the process is finite
    bool is_finite() const {
      return finite;
    }

    //! Returns true if the process is vector
    bool is_vector() const {
      return vector;
    }

    //! Casts this process to a finite one
    finite_discrete_process* as_finite() const {
      return finite;
    }
    
    //! Casts this process to a vector one
    vector_discrete_process* as_vector() const {
      return vector;
    }

    //! Conversion to human-readable representation
    operator std::string() const {
      return "#P(" + name() + "|DT|" 
        + boost::lexical_cast<std::string>(size()) + ")";
    }

    //! Returns the dimensionality / cardinality
    size_t size() const {
      if (finite) { return finite->size(); }
      if (vector) { return vector->size(); }
      return 0;
    }

    // documentation copied from the base class
    variable* at_any(const boost::any& index) const {
      if (finite) { return finite->at_any(index); }
      if (vector) { return vector->at_any(index); }
      return NULL;
    }

    /**
     * Returns an instance of the process at the given time step.
     * This function is guaranteed to return the same variable
     * object in multiple invocations. 
     */
    variable* at(int step) const {
      if (finite) { return finite->at(step); }
      if (vector) { return vector->at(step); }
      return NULL;
    }

    /**
     * Returns a special instance that represents the generic
     * variable at the 'current' time step.
     */
    variable* current() const {
      if (finite) { return finite->current(); }
      if (vector) { return vector->current(); }
      return NULL;
    }

    /**
     * Returns a special instance that represents the generic
     * variable at the 'next' time step.
     */
    variable* next() const {
      if (finite) { return finite->next(); }
      if (vector) { return vector->next(); }
      return NULL;
    }

    // documentation inherited from the base class
    void save_variable(oarchive& ar, variable* v) const {
      ar << boost::any_cast<int>(v->index());
    }

    // documentation inherited from the base class
    variable* load_variable(iarchive& ar) const {
      int index;
      ar >> index;
      return at(index);
    }

  }; // class discrete_process<variable>

  //! Returns a subset of variables at the specified time step.  All
  //! variables must be indexed by int, or else boost::bad_any_cast is thrown.
  //! \relates discrete_process
  template <typename Variable>
  std::set<Variable*> intersect(const std::set<Variable*>& vars, int step) {
    std::set<Variable*> result;
    foreach(Variable* v, vars)
      if (boost::any_cast<int>(v->index()) == step) result.insert(v);
    return result;
  }

  /**
   * Returns the processes associated with a set of variables.
   * \relates dicrete_process
   */
  template <typename V>
  std::set<discrete_process<V>*> discrete_processes(const std::set<V*>& vars) {
    return processes<discrete_process<V> >(vars);
  }

  /**
   * Splits the vector of processes into finite and vectors,
   * maintaining the ordering.
   * \relates discrete_process
   */
  inline void split(const std::vector<discrete_process<variable>*>& procs,
                    std::vector<finite_discrete_process*>& finite_procs,
                    std::vector<vector_discrete_process*>& vector_procs) {
    foreach(discrete_process<variable>* p, procs) {
      if (p->is_finite()) {
        finite_procs.push_back(p->as_finite());
      } else if (p->is_vector()) { 
        vector_procs.push_back(p->as_vector());
      }
    }
  }

  //! @} group base_types


  //! Serializer 
  template <typename V>
  oarchive& serialize(oarchive& ar, discrete_process<V>* const &p ) {
    ar << (dynamic_cast<process* const>(p));
    return ar;
  }
  
  template <typename V>
  iarchive& deserialize(iarchive& ar, discrete_process<V>* &p) {
    process* tmp = NULL;
    ar >> tmp;
    p = dynamic_cast<discrete_process<V>* >(tmp);
    return ar;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
