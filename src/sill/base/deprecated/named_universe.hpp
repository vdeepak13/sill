#ifndef SILL_NAMED_UNIVERSE_HPP
#define SILL_NAMED_UNIVERSE_HPP

#include <string>
#include <stdexcept>

#include <boost/unordered_set.hpp>

#include <sill/global.hpp>
#include <sill/map.hpp>
#include <sill/base/universe.hpp>
#include <sill/base/process.hpp>
#include <sill/base/concepts.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class that stores variables by name. When the user requests
   * a new variable of a certain type, the universe checks 
   * if the variable with this name already exists. 
   * If the variable exists, the universe checks that the
   * the existing and the requested variables are type-compatible.
   * Otherwise, the universe simply returns the new variable object
   * and registers the variable for future use.
   */ 
  class named_universe : public universe {

  private:
    //! A map that stores the variables
    map<std::string, variable*> vars;

    //! A map that stores the processes
    map<std::string, process*> procs;

    //! A set that stores the duplicate variables
    boost::unordered_set<variable*> duplicate_vars;

    //! A set that stores the duplicate processes
    boost::unordered_set<process*> duplicate_procs;

  public:
    //! Default construtor
    named_universe() {}

    // Deletes the stored variables and processes
    ~named_universe();

    /**
     * Returns the variable with the given name.
     * \throw std::invalid_argument if the variable could not be found.
     */
    variable* operator[](const std::string& name);
    
    /**
     * Returns a finite variable with the given name and domain size.
     * \throw std::invalid_argument if the existing and the new variables
     *        are not type-compatible.
     */
    finite_variable* new_finite_variable(const std::string& name, size_t size);

    /**
     * Returns a vector variable with the given name and number of dimensions.
     * \throw std::invalid_argument if the existing and the new variables
     *        are not type-compatible.
     */
    vector_variable* new_vector_variable(const std::string& name, size_t size);

    // TODO: Fix the other routines.

    //! Returns true if there is a variable with the specified name.
    bool contains_variable(const std::string& name) const;

    //! Returns true if there is a process with the specified name.
    bool contains_process(const std::string& name) const;

    /**
     * Registers a variable with this universe. The variable becomes owned 
     * by this universe. If the variable belongs to a process, records the
     * process instead.
     * \param allow_duplicates
     *        if false, enforces that there is only one variable with the
     *        specified name. if true, allows multiple variables and returns
     *        the first one registered
     * \return v or a previously recorded variable, if there is one
     * \throw std::invalid_argument 
     *        if the specified and the existing variable are not type compatible
     */
    variable* add(variable* v, bool allow_duplicates = false);

    /**
     * Registers a process with this universe. The process becomes owned 
     * by this universe. 
     * \param allow_duplicates
     *        if false, enforces that there is only one process with the
     *        specified name. if true, allows multiple processs and returns
     *        the first one registered
     * \return p or a previously recorded process, if there is one
     * \throw std::invalid_argument 
     *        if the specified and the existing process are not type compatible
     */
    process* add(process* p, bool allow_duplicates = false);

    /**
     * Returns a variable whose name matches the name of the given variable.
     * Dynamically casts the result to the specified variable type.
     */
    template <typename V>
    V* add_variable(V* v, bool allow_duplicates = false) {
      concept_assert((Variable<V>));
      V* v_mine = dynamic_cast<V*>(add(v, allow_duplicates));
      assert(v_mine);
      return v_mine;
    }

    /**
     * Returns a process whose nameme matches the name of the given process.
     * Dynamically casts the result to the specified process type.
     */
    template <typename P>
    P* add_process(P* p, bool allow_duplicates = false) {
      concept_assert((Process<P>));
      P* p_mine = dynamic_cast<P*>(add(p, allow_duplicates)); 
      assert(p_mine);
      return p_mine;
    }

    /**
     * Records a set of variables and returns a mapping from the old to 
     * the new ones.
     */
    template <typename V>
    map<V*, V*> add_variables(const set<V*> variables, 
                              bool allow_duplicates = false) {
      concept_assert((Variable<V>));
      map<V*, V*> mapping;
      foreach(V* v, variables) 
        mapping[v] = add_variable(v, allow_duplicates);
      return mapping;
    }

  }; // class named_universe

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
