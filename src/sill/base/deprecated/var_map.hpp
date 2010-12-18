#ifndef SILL_VAR_MAP_HPP
#define SILL_VAR_MAP_HPP

#include <map>

#include <sill/base/stl_util.hpp>
#include <sill/base/finite_variable.hpp>
#include <sill/base/vector_variable.hpp>

namespace sill {

  // Forward declarations
  class variable;

  std::ostream&
  operator<<(std::ostream& out, const std::map<variable*, variable*>& map);

} // end namespace sill


namespace std {
  /**
   * A mapping between arbitrary types of variables
   * \ingroup base_types
   */
  template<>
  class map<sill::variable*, sill::variable*>
    : public sill::finite_var_map,
      public sill::vector_var_map{

    // Constructors and assignment/conversion operators
    //==========================================================================
  public:
    //! Creates an empty map
    map() {}

    //! Singleton constructor
    map(sill::finite_variable* v, sill::finite_variable* w) { 
      sill::finite_var_map::insert(std::make_pair(v,w));
    }

    //! Singleton constructor
    map(sill::vector_variable* v, sill::vector_variable* w) {
      sill::vector_var_map::insert(std::make_pair(v, w));
    }

    //! Constructs an map with the given finite component
    map(const sill::finite_var_map& a)
      : sill::finite_var_map(a) { }

    //! Constructs an map with the given vector component
    map(const sill::vector_var_map& a)
      : sill::vector_var_map(a) { }

    //! Assignment operator
    map& operator=(sill::finite_var_map a) {
      // clear the other components. since a is passed by value, &a != this
      clear(); 
      sill::finite_var_map::operator=(a);
      return *this;
    }

    //! Assignment operator
    map& operator=(sill::vector_var_map a) {
      // clear the other components. since a is passed by value, &a != this
      clear();
      sill::vector_var_map::operator=(a);
      return *this;
    }

    //! Conversion to human-readable representation
    operator std::string() const {
      using namespace sill;
      std::ostringstream out; out << *this; return out.str(); 
    }
      
    // Accessors
    //==========================================================================
    //! Returns the finite portion of this map
    const sill::finite_var_map& finite() const {
      return *this;
    }

    //! Returns the finite portion of this map
    sill::finite_var_map& finite() {
      return *this;
    }

    //! Returns the vector portion of this map
    const sill::vector_var_map& vector() const {
      return *this;
    }

    //! Returns the vector portion of this map
    sill::vector_var_map& vector() {
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
    bool contains(sill::variable* v) const {
      sill::finite_variable* fv = dynamic_cast<sill::finite_variable*>(v);
      if (fv) return finite().count(fv) > 0;
      sill::vector_variable* vv = dynamic_cast<sill::vector_variable*>(v);
      if (vv) return vector().count(vv) > 0;
      assert(false); return false;
    }

    //! Returns the value associated with a variable
    sill::variable* operator[](sill::variable* v) const {
      sill::finite_variable* fv = dynamic_cast<sill::finite_variable*>(v);
      if (fv) return safe_get(finite(), fv);
      sill::vector_variable* vv = dynamic_cast<sill::vector_variable*>(v);
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
      sill::finite_var_map::clear();
      sill::vector_var_map::clear();
    }

  }; // class map<variable*, variable*>


} // namespace std

namespace sill {
  
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

} // namespace sill

#endif
