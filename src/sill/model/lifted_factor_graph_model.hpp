#ifndef SILL_LIFTED_FACTOR_GRAPH_MODEL_HPP
#define SILL_LIFTED_FACTOR_GRAPH_MODEL_HPP

#include <vector>
#include <map>
#include <iostream>

#include <sill/factor/concepts.hpp>
#include <sill/model/interfaces.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/range/forward_range.hpp>

#include <sill/serialization/serialize.hpp>
#include <sill/serialization/list.hpp>
#include <sill/macros_def.hpp>
namespace sill {

  /**
   * This represents a factor graph graphical models.  A factor graph
   * graphical models is a bipartite graphical model where the two
   * sets of vertices correspond to variables and factors and there is
   * an undirected edge between a variable and a factor if the
   * variable is in the domain of the factor.
   *
   * \ingroup model
   */
  template <typename F>
  class lifted_factor_graph_model : public factor_graph_model<F> {
    concept_assert((Factor<F>));
  public:
    
    typedef factor_graph_model<F>  base;
    typedef typename factor_graph_model<F>::vertex_type vertex_type;
    typedef typename base::factor_type        factor_type;
    typedef typename factor_type::result_type        result_type;
    typedef typename base::variable_type      variable_type;
    typedef typename base::domain_type        domain_type;
    typedef typename base::assignment_type    assignment_type;
    //! The set of neighbors type
    typedef std::vector<vertex_type>          neighbors_type;
    
    //! type of vertex id
    typedef typename base::vertex_id_type vertex_id_type;

  private:
    std::vector<std::map<size_t, size_t> > weight_;
  public:
    /**
     * The save routine saves the:
     *   factors_
     *   neighbors_ map
     *   vertices_ (ordered)
     *   args_ (domain of this model)
     *   edge weights
     */
    void save(oarchive& ar) const {
      ar << base::factors();
      ar << weight_;
    }
    
    /**
     * This deserialization routien reconstructs the factor graph
     * including the reverse map from vertex to global ID.
     */
    void load(iarchive& ar) {
      clear();
      std::list<factor_type> factorstmp;
      ar >> factorstmp;
      foreach (factor_type &f, factorstmp) {
        base::add_factor(f);
      }
      base::refill_vertex2id();
      ar >> weight_;
    }

    void clear() {
      base::clear();
      weight_.clear();
    }
    /**
     * Creates a factor graph model with no factors and no variables
     * Use the add_factor method to add factors to this factor graph.
     */
    lifted_factor_graph_model():base() { }

    vertex_id_type add_factor(const factor_type& factor) {
      vertex_id_type fid = base::add_factor(factor);
      
      // set default weights to 1
      foreach (vertex_type v, neighbors(id2vertex(fid))) {
        vertex_id_type vid = vertex2id(v);
        if (weight_.size() < std::max(vid + 1, fid + 1)) {
          weight_.resize(std::max(vid + 1, fid + 1));
        }
        
        weight_[vid][fid] = 1;
        weight_[fid][vid] = 1;
      }
      return fid;
    }
    
    size_t weight(vertex_type u, vertex_type v) const{
      return weight_[vertex2id(u)].find(vertex2id(v))->second;
    }
    size_t weight(vertex_id_type uid, vertex_id_type vid) const{
      return weight_[uid].find(vid)->second;
    }
    
    void set_weight(vertex_type u, vertex_type v, size_t w) {
      weight_[vertex2id(u)][vertex2id(v)] = w;
      weight_[vertex2id(v)][vertex2id(u)] = w;
    }

    void set_weight(vertex_id_type uid, vertex_id_type vid, size_t w){
      weight_[uid][vid] = w;
      weight_[vid][uid] = w;
    }

    std::vector<std::map<size_t, size_t> > & weights() {
      return weight_;
    }
    

    /**
     * Simplify the model by merging factors. This may not be so
     * well defined for lifted models. So lets ignore it for now.
     */
     void simplify() {}
     
     void simplify_stable() {}


    /**
     * Check the consistency of the internal data structures.
     * Useful for making sure the modifications such as simplify()
     * resulted in a correct state.
     *
     */
    bool is_consistent(){
      // check if weights are consistent
      foreach(vertex_type u, base::vertices()) {
        vertex_id_type uid = vertex2id(u);
        foreach(vertex_type v, base::neighbors(u)) {
          vertex_id_type vid = vertex2id(v);
          if (! (weight_[uid][vid] > 0 && 
                weight_[uid][vid] == weight_[vid][uid])) {
            return false;
          }
        }
      }
      return factor_graph_model<F>::is_consistent();
    } 

    //
    double log_likelihood(const assignment_type& a) const {
      std::cerr << "TODO: Lifted log likelihood not defined. Currently using standard log likelihood\n";
      return base::log_likelihood(a);
    }

    logarithmic<double> operator()(const assignment_type& a) const {
      return logarithmic<double>(log_likelihood(a), log_tag());
    }

    //! Prints the arguments and factors of the model.
    template <typename OutputStream>
    void print(OutputStream& out) const {
      out << "Arguments: " << base::arguments() << "\n"
          << "Factors:\n";
      foreach(F f, base::factors()) out << f;
    }


    /**
     * Print the adjacency structure in the form of comma separated
     * edge pairs
     */
    void print_adjacency(std::ostream& out) const {
      foreach(const vertex_type& u, base::vertices()) {
        vertex_id_type u_id = base::vertex2id(u);
        foreach(const vertex_type& v, base::neighbors(u)) {
          vertex_id_type v_id = base::vertex2id(v);
          if(u_id < v_id) {
            out << u_id << ", " << v_id << ": " << weight(u_id, v_id) << std::endl;
          }
        }
      } 
    } // end of print adjacency

    double bethe(std::map<vertex_type, factor_type> &beliefs) {
      std::cerr << "TODO: Lifted Bethe not defined. Currently using standard Bethe\n";
      return base::bethe(beliefs);
    }
  };  
};


#include <sill/macros_undef.hpp>

#endif // SILL_FACTOR_GRAPH_MODEL_HPP
