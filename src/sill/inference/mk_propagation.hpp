#ifndef SILL_INFERENCE_MK_PROPAGATION_HPP
#define SILL_INFERENCE_MK_PROPAGATION_HPP

#include <sill/model/factor_graph_model.hpp>
#include <map>

#include <sill/macros_def.hpp>

namespace sill{ 
//! Propagates the Mooij and Kappen derivative values
template <typename F>
class mk_propagation {
public:
	typedef typename factor_graph_model<F>::vertex_type        vertex_type; 
	typedef typename factor_graph_model<F>::factor_type        factor_type;
	typedef typename factor_graph_model<F>::variable_type      variable_type;
	
private:
  typedef std::map<vertex_type, std::map<vertex_type, double> > message_map_type;
  const factor_graph_model<F> &model_;
  message_map_type initiald_;  // log of original computed mk values
  message_map_type messages_; // log of propagated messages

  void propagate_once(void) {
    message_map_type newmessages; // essentially the derivatives

    foreach(const vertex_type &u, model_.vertices()) {
      // we do the equivalent of the fast update rule here
      // take the sum of everything entering u

      foreach(const vertex_type &v, model_.neighbors(u)) {
        double s = 0;
        foreach(const vertex_type &vp, model_.neighbors(u)) {
          if (vp != v) {
            s = (s +  messages_[vp][u]);
          }
        }
        newmessages[u][v] = s;
      }
    }
    messages_ = newmessages;
  }
public:

  mk_propagation(const factor_graph_model<F> &model):model_(model) {}

  void compute_initial_values() {
    foreach(const vertex_type &u, model_.vertices()) {
      foreach(const vertex_type &v, model_.neighbors(u)) {
        double d = mooij_kappen_w_ub(model_, u, v);
        // d can and will be 0 for some graphs...
        d = std::log(d+0.00001);
        initiald_[u][v] = d;
        initiald_[v][u] = d;
      }
    }
    messages_ = initiald_;
  }

  // propagate the values by n steps
  void propagate(size_t n) {
    for (size_t i = 0; i < n; ++i) {
      std::cout << ".";
      std::cout.flush();
      propagate_once();
    }
    std::cout << "\n";
  }
  
  double get_value(const vertex_type &u, const vertex_type &v) const{
    if (u.is_factor()) return get_value(v,u);
		typedef typename message_map_type::const_iterator iterator1;
		iterator1 i = messages_.find(u);
		assert(i != messages_.end());

		typedef typename message_map_type::mapped_type::const_iterator iterator2;
		iterator2 j = i->second.find(v);
		assert(j != i->second.end());
		
		return j->second;
  }

  
  //! This will output the graph to the provided ostream as a list of edges
  //! together with the computed mk values attached.
  void output_graph(std::ostream& out) const {
		typedef typename factor_graph_model<F>::vertex_id_type  vertex_id_type;

		foreach(const vertex_type& u, model_.vertices()) {
			vertex_id_type u_id = model_.vertex2id(u);
			foreach(const vertex_type& v, model_.neighbors(u)) {
				vertex_id_type v_id = model_.vertex2id(v);
				if(u_id < v_id) {
					double weight = get_value(u, v) ;
					out << u_id << ", " << v_id << ", " << weight << std::endl;
				}
			}
		}
  } // end of operator<<

};


}
#include <sill/macros_undef.hpp>
#endif
