#ifndef SILL_MATH_COMPOSE_HPP
#define SILL_MATH_COMPOSE_HPP

#include <boost/shared_ptr.hpp>

#include <sill/math/interfaces.hpp>

namespace sill {

  //! Represents a composition of two vector function f(g(x))
  //! \ingroup math_functions
  class vector_function_composition : public vector_function {
  private:
    boost::shared_ptr<vector_function> f_ptr;
    boost::shared_ptr<vector_function> g_ptr;

  public:
    vector_function_composition(const vector_function& f,
                                const vector_function& g) 
      : f_ptr(f.clone()), g_ptr(g.clone()) { 
      assert(f.size_in() == g.size_out());
    }

    vector_function_composition* clone() const {
      return new vector_function_composition(*this);
    }

    operator std::string() const {
      return "vector_composition";
    }

    void value(const vec& x, vec& y) const {
      vec tmp;
      g_ptr->value(x, tmp);
      f_ptr->value(tmp, y);
    }
    
    size_t size_out() const {
      return f_ptr->size_out();
    }
    
    size_t size_in() const {
      return g_ptr->size_in();
    }
    
  }; // class vector_function_composition

  //! Composes two vector functions
  //! \relates vector_function_composition
  vector_function_composition 
  compose(const vector_function& f, const vector_function& g) {
    return vector_function_composition(f, g);
  }

} // namespace sill

#endif
