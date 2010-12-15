#ifndef PRL_VAR_MAP_HPP
#define PRL_VAR_MAP_HPP

#include <map>

#include <prl/base/stl_util.hpp>
#include <prl/base/finite_variable.hpp>
#include <prl/base/vector_variable.hpp>

namespace prl {

  // Forward declarations
  class variable;

  std::ostream&
  operator<<(std::ostream& out, const std::map<variable*, variable*>& map);

} // end namespace prl


namespace std {
  /**
   * A mapping between arbitrary types of variables
   * \ingroup base_types
   */
  template<>
  class map<prl::variable*, prl::variable*>
    : public prl::finite_var_map,
      public prl::vector_var_map{

    // Constructors and assignment/conversion operators
    //==========================================================================
  public:
    //! Creates an empty map
    map() {}

    //! Singleton constructor
    map(prl::finite_variable* v, prl::finite_variable* w) { 
      prl::finite_var_map::insert(std::make_pair(v,w));
    }

    //! Singleton constructor
    map(prl::vector_variable* v, prl::vector_variable* w) {
      prl::vector_var_map::insert(std::make_pair(v, w));
    }

    //! Constructs an map with the given finite component
    map(const prl::finite_var_map& a)
      : prl::finite_var_map(a) { }

    //! Constructs an map with the given vector component
    map(const prl::vector_var_map& a)
      : prl::vector_var_map(a) { }

    //! Assignment operator
    map& operator=(prl::finite_var_map a) {
      // clear the other components. since a is passed by value, &a != this
      clear(); 
      prl::finite_var_map::operator=(a);
      return *this;
    }

    //! Assignment operator
    map& operator=(prl::vector_var_map a) {
      // clear the other components. since a is passed by value, &a != this
      clear();
      prl::vector_var_map::operator=(a);
      return *this;
    }

    //! Conversion to human-readable representation
    operator std::string() const {
      using namespace prl;
      std::ostringstream out; out << *this; return out.str(); 
    }
      
    // Accessors
    //==========================================================================
    //! Returns the finite portion of this map
    const prl::finite_var_map& finite() const {
      return *this;
    }

    //! Returns the finite portion of this map
    prl::finite_var_map& finite() {
      return *this;
    }

    //! Returns the vector portion of this map
    const prl::vector_var_map& vector() const {
      return *this;
    }

    //! Returns the vector portion of this map
    prl::vector_var_map& vector() {
      return *this;
    }

    //! Returns the total number of elements in this map
    size_t size() const {
      return finite().size() + vector().size();
    }
    
    //! Returns true if the map is empty
    size_t empty() const {
      return finite().empty() && vector().empty();
    }

    // Queries
    //==========================================================================
    //! Returns true if the map contains a variable
    bool contains(prl::variable* v) const {
      prl::finite_variable* fv = dynamic_cast<prl::finite_variable*>(v);
      if (fv) return finite().count(fv) > 0;
      prl::vector_variable* vv = dynamic_cast<prl::vector_variable*>(v);
      if (vv) return vector().count(vv) > 0;
      assert(false); return false;
    }

    //! Returns the value associated with a variable
    prl::variable* operator[](prl::variable* v) const {
      prl::finite_variable* fv = dynamic_cast<prl::finite_variable*>(v);
      if (fv) return safe_get(finite(), fv);
      prl::vector_variable* vv = dynamic_cast<prl::vector_variable*>(v);
      if (vv) return safe_get(vector(), vv);
      assert(false); return NULL;
    }

    //! Equality test.
    bool operator==(const map& a) const {
      return finite() == a.finite() && vector() == a.vector();
    }

    //! Inequality test
    bool operator!=(const map& a) const {
      return !operator==(a);
    }

    // Mutators
    //==========================================================================
    //! Removes all elements from the map
    void clear() {
      prl::finite_var_map::clear();
      prl::vector_var_map::clear();
    }

  }; // class map<variable*, variable*>


} // namespace std

namespace prl {
  
  /**
   * Computes the union of two maps
   */
  template< >
  std::map<variable*, variable*> 
  map_union(const std::map< variable*, variable* > & a, 
            const std::map< variable*, variable*> & b) {
    // Initialize the output map
    std::map<variable*, variable*> output;
    output.finite() = map_union(a.finite(), b.finite());
    output.vector() = map_union(a.vector(), b.vector());
    return output;
  } // end of map_union

  /**
   * Computes the intersection of two maps
   */
  template< >
  std::map<variable*, variable*> 
  map_intersect(const std::map< variable*, variable* >& a, 
                const std::map< variable*, variable* >& b) {
    // Initialize the output map
    std::map<variable*, variable*> output;
    output.finite() = map_intersect(a.finite(), b.finite());
    output.vector() = map_intersect(a.vector(), b.vector());
    return output;
  } // end of map_union

  /**
   * Computes the difference between two maps
   */
  template< >
  std::map<variable*, variable*> 
  map_difference(const std::map< variable*, variable* >& a, 
                 const std::map< variable*, variable* >& b) {
    // Initialize the output map
    std::map<variable*, variable*> output;
    output.finite() = map_difference(a.finite(), b.finite());
    output.vector() = map_difference(a.vector(), b.vector());
    return output;
  } // end of map_union


  //! \relates map<variable*, variable*>
  inline std::ostream&
  operator<<(std::ostream& out, const std::map<variable*, variable*>& map) {
    out << map.finite();
    out << map.vector();
    return out;
  }

} // namespace prl

#endif
