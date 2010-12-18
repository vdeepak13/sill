#ifndef SILL_MOOIJ_KAPPEN_DERIVATIVES_HPP
#define SILL_MOOIJ_KAPPEN_DERIVATIVES_HPP

#include <sill/factor/table_factor.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/macros_def.hpp>
#include <cmath>
namespace sill{

  /* Computes the msg derivative between v1 and f where
    f is a factor. it will take the max of v1 to all other vertices through f*/
  template<typename F>
  double bp_msg_max_derivative_ub(factor_graph_model<F> &fg, 
                          typename factor_graph_model<F>::vertex_type f,
                          typename factor_graph_model<F>::vertex_type v1) {
    double mx = 0;
    foreach(typename factor_graph_model<F>::vertex_type v, fg.neighbors(f)) {
      if (v != v1) {
        //mx = std::max(mx, bp_msg_derivative_ub(fg,f,v1,v));
        mx = std::max(mx, f.factor().bp_msg_derivative_ub(&(v1.variable()),&(v.variable())));
      }
    }
    return mx;
  }

}
#include <sill/macros_undef.hpp>
#endif
