namespace sill {
  class universe;
}
#ifndef SILL_NAMED_UNIVERSE_HPP
#define SILL_NAMED_UNIVERSE_HPP

#include <string>
#include <stdexcept>
#include <map>

#include <boost/unordered_set.hpp>
#include <boost/lexical_cast.hpp>
#include <sill/global.hpp>
#include <sill/base/process.hpp>
#include <sill/base/concepts.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  class universe{

  private:
    //! A map that stores the variables
    std::map<std::string, variable*> vars;

    //! A map that stores the processes
    std::map<std::string, process*> procs;
    
    std::vector<variable*> vars_vector;
    std::vector<process*> procs_vector;
    
    size_t next_id() {
      return vars.size();
    }

    void register_variable_id(variable* v);

    void register_process_id(process* v);

  public:
    //! Default construtor
    universe() {}

    // Deletes the stored variables and processes
    ~universe();
    
    void save(oarchive& a) const;

    void load(iarchive& a);

    variable* var_from_id(size_t id) const{
      assert(id <  vars_vector.size());
      return vars_vector[id];
    }
    process* proc_from_id(size_t id) const{
      assert(id < procs_vector.size());
      return procs_vector[id];
    }

    variable* var_from_name(std::string &name) const{
      return safe_get(vars, name, (variable*)(NULL));
    }

    process* process_from_name(std::string &name) const{
      return safe_get(procs, name, (process*)(NULL));
    }


    /**
     * Returns a finite variable with the given name and domain size.
     * \throw std::invalid_argument if the existing and the new variables
     *        are not type-compatible.
     */
    finite_variable* new_finite_variable(const std::string& name, size_t size);


    finite_variable*
    new_finite_variable(size_t size) {
      return new_finite_variable(boost::lexical_cast<std::string>(next_id()), size);
    }

    finite_var_vector
    new_finite_variables(size_t n, size_t size) {
      finite_var_vector vars(n);
      foreach(finite_variable*& v, vars) 
        v = new_finite_variable(size);
      return vars;
    }

    finite_var_vector new_finite_variables(const std::vector<size_t>& sizes) {
      finite_var_vector vars(sizes.size());
      for(size_t i = 0; i < sizes.size(); i++) 
        vars[i] = new_finite_variable(sizes[i]);
      return vars;
    }

    vector_variable* new_vector_variable(size_t dim) {
      return new_vector_variable(boost::lexical_cast<std::string>(next_id()), dim);
    }
    
    vector_var_vector new_vector_variables(size_t n, size_t dim) {
      vector_var_vector vars(n);
      foreach(vector_variable*& v, vars) 
        v = new_vector_variable(dim);
      return vars;
    }
    
    vector_var_vector new_vector_variables(const std::vector<size_t>& dims) {
      vector_var_vector vars(dims.size());
      for(size_t i = 0; i < dims.size(); i++)
        vars[i] = new_vector_variable(dims[i]);
      return vars;
    }

    /**
     * Returns a vector variable with the given name and number of dimensions.
     * \throw std::invalid_argument if the existing and the new variables
     *        are not type-compatible.
     */
    vector_variable* new_vector_variable(const std::string& name, size_t size);

    /**
     * Return a variable of the specified type and the given size.
     * @tparam VarType  Variable type.
     * @param  size     Variable size.
     * @todo Standardize the new_*_variable() method interface so that this
     *       method can support variable names.
     */
    template <typename VarType>
    VarType* new_variable(size_t size);

    /**
     * Registers a variable with this universe. The variable becomes owned 
     * by this universe. 
     */
    void add_impl(variable* v);

    /**
     * Registers a process with this universe. The process becomes owned 
     * by this universe. 
     */
    void add_impl(process* p);

    /**
     * Registers a variable or a process with this universe. The object
     * becomes owned by this universe.
     * \returns the input object
     * \tparam T either process or a variable type
     */
    template <typename T>
    T* add(T* t) {
      add_impl(t);
      return t;
    }

  }; // class universe

  //! Specialization for finite variables.
  template <>
  finite_variable* universe::new_variable<finite_variable>(size_t size);

  //! Specialization for vector variables.
  template <>
  vector_variable* universe::new_variable<vector_variable>(size_t size);

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
