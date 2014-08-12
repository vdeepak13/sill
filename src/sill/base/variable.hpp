#ifndef SILL_VARIABLE_HPP
#define SILL_VARIABLE_HPP

#include <string>
#include <iosfwd>
#include <vector>
#include <set>
#include <map>

#include <boost/any.hpp>

#include <sill/base/concepts.hpp>
#include <sill/base/stl_util.hpp>
#include <sill/range/algorithm.hpp>
#include <sill/range/converted.hpp>
#include <sill/serialization/serialize.hpp>
#include <sill/range/forward_range.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  //! \addtogroup base_types
  //! @{

  // Forward declaration
  class process;

  /**
   * The base type of all variable information structures.
   * By design, variables are immutable.
   * \ingroup base_types
   */
  class variable {

  public:   
    //! Implements Variable::value_type
    typedef boost::any value_type;
    
    //! Destructor.
    virtual ~variable() { }

    //! Conversion to human-readable representation
    virtual operator std::string() const = 0;

    //! Returns true if the two variables are type-compatible
    virtual bool type_compatible(variable* v) const = 0;

    //! Returns the name of this variable
    const std::string& name() const {
      return name_;
    }
    
    //! Returns the dimensionality of the variable
    virtual size_t size() const = 0;
    
    //! Returns the process, with which this variable is associated
    //! \return the associated process or NULL if this is a free variable
    sill::process* process() const {
      return process_;
    }

    //! Returns true if this variable is associated with a process
    bool indexed() const {
      return process_ != NULL;
    }

    //! Returns the index of this variable in the associated process.
    //! Use boost::any_cast to cast the index to the original type.
    const boost::any& index() const {
      return index_;
    }

    //! Gets the id of this variable
    size_t id() const {
      return id_;
    }

    //! Serializes this variable and all attached information. 
    //! This performs a deep serialization of this variable as opposed to 
    //! just storing an ID. 
    virtual void save(oarchive& ar) const;

    //! deserializes this variable and all attached information. 
    //! This performs a deep deserialization. If this variable is part of a 
    //! process, the archive must be provided with the universe. 
    //! Otherwise the deserialization can be performed an attached universe.
    virtual void load(iarchive& ar);

    //! An enumeration over all variable types
    enum variable_typenames {FINITE_VARIABLE, VECTOR_VARIABLE};

    //! Returns the dynamic type of this variable as an enumeration
    variable_typenames get_variable_type() const;


    // Private data members and constructors
    //==========================================================================
  private:
    //! A concise, informative string to describe the variable
    std::string name_;
    
    //! The id of this variable as assigned by the universe
    size_t id_;

    //! The process, if any, with which this variable is associated
    sill::process* process_;

    //! The index of the variable in the process
    boost::any index_;

    //! give friend access to the pointer serializers
    friend oarchive& operator<<(oarchive& ar, variable* const &v );
    friend iarchive& operator>>(iarchive& ar, variable* &v);


    //! Disable the assignment operator.
    variable& operator=(const variable& other);
    // if compilation fails here, you are probably trying manipulate
    // variable objects rather than variable pointers

    friend class universe;
    void set_id(size_t id) {
      assert(id_ == 0);
      id_ = id;
    }

  protected:
    variable() : id_(0), process_(NULL) { }

    //! Creates a variable with the given name
    variable(const std::string& name)
      : name_(name), id_(0), process_(NULL) { }

    //! Creates a variable with the given name, process, and index
    variable(const std::string& name,
             sill::process* process,
             const boost::any& index)
      : name_(name), id_(0), process_(process), index_(index) { }

  }; // class variable

  /**
   * A set of arbitrary variables.
   */
  typedef std::set<variable*> domain;

  /**
   * A vector of arbitrary variables.
   * This type is used primarily to refer to variables in some specific order.
   */
  typedef std::vector<variable*> var_vector;

  /**
   * A map from variables to variables of the same type.
   * This kind of map is used to perform variable substitutions.
   */
  typedef std::map<variable*, variable*> var_map;


  // Free functions
  //============================================================================

  /**
   * Returns the intersection of a generic and a specific domain type
   * \relates variable
   */
  template <typename V>
  std::set<V*> intersect(const std::set<V*>& s, const domain& t) {
    concept_assert((Variable<V>));
    std::set<V*> u;
    sill::set_intersection(s, t, std::inserter(u, u.begin()));
    return u;
  }

  /**
   *  Returns the intersection of a generic and a specific domain type
   * \relates variable
   */
  template <typename V>
  std::set<V*> intersect(const domain& s, const std::set<V*>& t) {
    concept_assert((Variable<V>));
    return intersect(t, s);
  }

  /**
   * Returns the union of two (different) specific domain types
   * \relates variable
   */
  template <typename U, typename V>
  domain operator+(const std::set<U*>& s, const std::set<V*>& t) {
    concept_assert((Variable<U>));
    concept_assert((Variable<V>));
    domain u;
    sill::set_union(make_converted<variable*>(s),
                    make_converted<variable*>(t),
                    std::inserter(u, u.begin()));
    return u;
  }

  // TODO: this function should be renamed to make_set and put into 
  // set_operations.hpp
  // same goes for make_vector

  /**
   * A convenience function that returns a domain constructed from a
   * set of variables / processes.
   *
   * @return the set of all valid variable handles supplied to this method
   *
   * \relates variable
   */
  template<class V>
  std::set<V*> make_domain(V* v1 = NULL,
                           V* v2 = NULL,
                           V* v3 = NULL,
                           V* v4 = NULL,
                           V* v5 = NULL) {
    std::set<V*> set;
    if (v1 != NULL) set.insert(v1);
    if (v2 != NULL) set.insert(v2);
    if (v3 != NULL) set.insert(v3);
    if (v4 != NULL) set.insert(v4);
    if (v5 != NULL) set.insert(v5);
    return set;
  }

  /**
   * A function that converts a vector of variables to a set.
   * \relates variable
   */
  template <class V>
  std::set<V*> make_domain(const std::vector<V*>& vars) {
    return std::set<V*>(vars.begin(), vars.end());
  }

  /**
   * A function that converts a vector of variables to a set.
   * \relates variable
   */
  template <class V>
  std::set<V*> make_domain(const forward_range<V*>& vars) {
    return std::set<V*>(vars.begin(), vars.end());
  }

  /**
   * A function that returns an std::vector of variables / processes.
   * \relates variable
   */
  template <class V>
  std::vector<V*> make_vector(V* v1 = NULL,
                              V* v2 = NULL,
                              V* v3 = NULL,
                              V* v4 = NULL,
                              V* v5 = NULL) {
    std::vector<V*> vars;
    if (v1 != NULL) vars.push_back(v1);
    if (v2 != NULL) vars.push_back(v2);
    if (v3 != NULL) vars.push_back(v3);
    if (v4 != NULL) vars.push_back(v4);
    if (v5 != NULL) vars.push_back(v5);
    return vars;
  }

  /**
   * Converts a set of variables / processes to a vector.
   * \relates variable
   */
  template <class V>
  std::vector<V*> make_vector(const std::set<V*>& domain) {
    return std::vector<V*>(domain.begin(), domain.end());
  }

  /**
   * Substitutes variables in a domain.
   *
   * @param  vars  a set of variables
   * @param  map
   *         a map from (some of the) variables in vars to a new set
   *         of variables; this mapping must be 1:1, and each variable
   *         in vars must map to a type-compatible variable; any
   *         missing variable is assumed to map to itself
   * @return the image of vars under subst
   */
  template<class V>
  std::set<V*>
  subst_vars(const std::set<V*>& vars, const std::map<V*, V*>& map) {
    concept_assert((Variable<V>));
    if (vars.empty()) return vars;
    std::set<V*> new_vars;
    foreach(V* v, vars) {
      if(map.count(v)) {
        V* w = safe_get(map, v);
        if (!v->type_compatible(w)) {
          std::cerr << "Variables "
                    << v << "," << w
                    << " are not type-compatible." << std::endl;
          assert(false);
        }
        assert(new_vars.count(w) == 0);
        new_vars.insert(w);
      } else {
        assert(new_vars.count(v) == 0);
        new_vars.insert(v);
      }
    }
    return new_vars;
  }

//! @}

  /**
   * Outputs a human-readable representation of a variable to stream
   * \relates variable
   */
  std::ostream& operator<<(std::ostream& out, variable* h);

  //! Serializes the variable* pointer. This only serializes an id.
  //! The deserializer will look for the id in the universe
  oarchive& operator<<(oarchive& ar, variable* const &v );

  //! Deserializes a variable* pointer by reading an id from the archive.
  //! The archive must have an attached universe
  iarchive& operator>>(iarchive& ar, variable* &v);

  //! Serializes a variable type.
  oarchive& operator<<(oarchive& ar, const variable::variable_typenames& t);

  //! Deserializes a variable type.
  iarchive& operator>>(iarchive& ar, variable::variable_typenames& t);

  //! This serializes the pointer variable* while also storing the dynamic
  //! type information to correctly reconstruct the variable datatype on 
  //! deserialization.
  void dynamic_deep_serialize(oarchive &a, variable* const &i);

  //! This deserializes the pointer variable*, reconstructing the original
  //! datatype of the variable.
  void dynamic_deep_deserialize(iarchive &a, variable* &i);

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
