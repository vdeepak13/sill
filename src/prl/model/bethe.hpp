#ifndef SILL_BETHE_HPP
#define SILL_BETHE_HPP

#include <boost/unordered_map.hpp>

#include <sill/model/interfaces.hpp>
#include <sill/model/region_graph.hpp>

#include <sill/macros_def.hpp>

namespace sill {
  
  /**
   * Implements the Bethe construction for a factorized model.
   * \ingroup model
   */
  template <typename F>
  void make_bethe(const factorized_model<F>& model,
                  region_graph<typename F::variable_type*, F>& rg) {
    typedef typename F::variable_type variable_type;

    boost::unordered_map<variable_type*, size_t> vertices;
    rg.clear();
    
    foreach(variable_type* v, model.arguments())
      vertices[v] = rg.add_region(v, F(1));
    
    foreach(const F& f, model.factors()) {
      size_t u = rg.add_region(f.arguments(), f);
      foreach(variable_type* var, f.arguments())
        rg.add_edge(u, vertices[var]);
    }

    rg.recompute_counting_numbers();
  }
  
} // namespace sill

#include <sill/macros_undef.hpp>

#endif
