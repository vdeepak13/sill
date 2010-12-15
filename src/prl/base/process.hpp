#ifndef PRL_PROCESS_HPP
#define PRL_PROCESS_HPP

#include <iosfwd>
#include <string>
#include <set>
#include <map>
#include <vector>

#include <boost/any.hpp>

#include <prl/base/concepts.hpp> 
#include <prl/serialization/serialize.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  // forward declaration
  class variable;

  //! \addtogroup base_types
  //! @{

  /**
   * The base class of all processes. 
   * Processes are immutable.
   */
  class process {

    // Private data members and constructors
    //==========================================================================
  private:
    //! A concise, informative string to describe the process
    std::string name_;

    //! The id of this process as assigned by the universe
    size_t id_;
    
    //! Disable the assignment operator.
    process& operator=(const process& other);
    // if compilation fails here, you are probably trying manipulate
    // process objects rather than process pointers

    friend class universe;
    void set_id(size_t id) {
      id_ = id;
    }

  protected:
    //! Default constructor (only used by serialization)
    process() : id_(0) { }
    
    //! Creates a process with the given name
    process(const std::string& name)
      : name_(name), id_(0) { }

    // Public functions
    //==========================================================================
  public:
  
    enum process_typenames {TIMED_PROCESS_FINITE, TIMED_PROCESS_VECTOR};

    //! Returns the process type
    process_typenames get_process_type() const;
    
    //! Destructor.
    virtual ~process() { }

    //! Returns the name of this process
    const std::string& name() const {
      return name_;
    }
    
    //! Conversion to human-readable representation
    virtual operator std::string() const = 0;

    /**
     * Returns the process variable at the specified index.  The value
     * stored in the boost::any object must be of the exact type
     * index_type, used by the process (no conversion is performed).
     * This function name has an additional "any" specifier to avoid
     * potential argument conversion ambiguities in the process's
     * typesafe at() functions.
     */
    virtual variable* at_any(const boost::any& index) const = 0;

    /**
     * Saves the index of the variable to an archive.
     */
    virtual void save_variable(oarchive& ar, variable* v) const = 0;

    /**
     * Loads the variable from an archive.
     */
    virtual variable* load_variable(iarchive& ar) const = 0;

    /**
     * Serializes this process and all attached information. 
     * This performs a deep serialization of this process as opposed to 
     * just storing an ID. 
     */
    virtual void save(oarchive& ar) const{
      ar << name_;
    }
    
    /**
     * Deserializes this process and all attached information. 
     * This performs a deep serialization of this process as opposed to 
     * just storing an ID. 
     */
    virtual void load(iarchive& ar){
      ar >> name_;
    }
    
    size_t id() const {
      return id_;
    }



  }; // class process

  //! \relates process
  std::ostream& operator<<(std::ostream& out, process* p);

  // Free functions
  //============================================================================
  /**
   * Returns the processes associated with a set of variables.
   * \relates process
   */
  template <typename P, typename V>
  std::set<P*> processes(const std::set<V*>& variables) {
    concept_assert((Variable<V>));
    concept_assert((Process<P>));
    std::set<P*> result;
    foreach(V* v, variables) {
      P* p = dynamic_cast<P*>(v->process());
      assert(p);
      result.insert(p);
    }
    return result;
  }

  /**
   * Returns the variables, associated with the given processes.
   * Specifically, given a set of processes { p }, returns a set of variables
   * { p_index }.
   * \relates process
   */
  template <typename P>
  std::set<typename P::variable_type*> 
  variables(const std::set<P*>& processes, 
            const typename P::index_type& index) {
    concept_assert((Process<P>));
    std::set<typename P::variable_type*> result;
    foreach(P* p, processes) result.insert(p->at(index));
    return result;
  }

  /**
   * Returns the variables, associated with the given processes.
   * For each process p, stores the variable p_index into the output vector.
   * \relates process
   */
  template <typename P>
  std::vector<typename P::variable_type*>
  variables(const std::vector<P*>& procs, const typename P::index_type& index) {
    concept_assert((Process<P>));
    std::vector<typename P::variable_type*> result(procs.size());
    for(size_t i = 0; i < procs.size(); i++)
      result[i] = procs[i]->at(index);
    return result;
  }

  /**
   * Given a set of process variables, returns a new set of variables
   * that represent the state of the corresponding processes at the 
   * specified index.
   * \relates process
   */
  template <typename V>
  std::set<V*> subst_index(const std::set<V*>& variables, 
                           const boost::any& new_index) {
    concept_assert((Variable<V>));
    std::set<V*> result;
    foreach(V* v, variables) {
      V* new_v = dynamic_cast<V*>(v->process()->at_any(new_index));
      assert(new_v);
      result.insert(new_v);
    }
    assert(result.size() == variables.size());
    return result;
  }

  /**
   * Given a set of process variables, verifies that each variable in the
   * set has the specified index.
   * \relates process
   */
  template <typename V, typename Index>
  void check_index(const std::set<V*>& variables, const Index& index) {
    concept_assert((Variable<V>));
    foreach(V* v, variables) {
      Index i = boost::any_cast<Index>(v->index());
      assert(i == index);
    }
  }

  /**
   * Returns an object that maps variables from one index to another.
   * Specifically, given a set of processes {p }, returns an object that maps
   * the variable p[from] to the variable p[to], for each process in the set.
   * @param from the source index
   * @param to the target index
   * \relates process
   */
  template <typename P>
  std::map<typename P::variable_type*, typename P::variable_type*> 
  make_process_var_map(const std::set<P*>& processes,
                       const typename P::index_type& from,
                       const typename P::index_type& to) {
    std::map<typename P::variable_type*, typename P::variable_type*> result;
    foreach(P* p, processes)
      result[p->at(from)] = p->at(to);
    return result;
  }

  //! @} group base_types

//! Serializes the process* pointer. This only serializes an id.
//! The deserializer will look for the id in the universe
oarchive& operator<<(oarchive & ar, process* v);

//! Deserializes a process* pointer by reading an id from the archive.
//! The archive must have an attached universe
iarchive& operator>>(iarchive & ar, process* &p);

//! This serializes the pointer process* while also storing the dynamic
//! type information to correctly reconstruct the process datatype on 
//! deserialization.
void dynamic_deep_serialize(oarchive &a, process* i);

//! This deserializes the pointer process*, reconstructing the original
//! datatype of the process.
void dynamic_deep_deserialize(iarchive &a, process*& i);

} // namespace prl

#include <prl/macros_undef.hpp>

#endif
