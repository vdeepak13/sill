#ifndef SILL_FACTOR_GRAPH_MODEL_HPP
#define SILL_FACTOR_GRAPH_MODEL_HPP

#include <vector>
#include <map>

#include <sill/factor/concepts.hpp>
#include <sill/model/interfaces.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/log_table_factor.hpp>
#include <sill/range/forward_range.hpp>

#include <sill/serialization/serialize.hpp>
#include <sill/serialization/list.hpp>
#include <sill/base/stl_util.hpp>
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
  class factor_graph_model : public factorized_model<F> {
    concept_assert((Factor<F>));
  public:
    class vertex_type;     // predeclaration
    typedef graphical_model<F>  base;
    typedef typename base::factor_type        factor_type;
    typedef typename factor_type::result_type        result_type;
    typedef typename base::variable_type      variable_type;
    typedef typename base::domain_type        domain_type;
    typedef typename base::assignment_type    assignment_type;
    typedef typename std::vector<size_t> neighbors_type;

    /**
     * The class used to represent vertices in the factor graph model.
     * Each vertex_type is the "union" of either a variable or a factor.
     * The underlying factor or variable can be accessed by reference
     * through the vertex_type.
     */
    class vertex_type {
    private:
      const factor_type* factor_;
      variable_type* variable_;
      size_t id_;

      vertex_type(const factor_type* f, size_t id) :
                                        factor_(f), variable_(NULL), id_(id) { }
      vertex_type(variable_type* v, size_t id) :
                                        factor_(NULL), variable_(v), id_(id) { }

      friend class factor_graph_model<F>;
    public:
      vertex_type() : factor_(NULL), variable_(NULL) { }
      inline bool is_factor() const { return factor_ != NULL; }
      inline bool is_variable() const { return variable_ != NULL; }
      inline const factor_type& factor() const {
        assert(is_factor()); return *factor_;
      }
      inline size_t id() const{
        return id_;
      }
      //! Returns a pointer to the variable
      inline variable_type& variable() const {
        assert(is_variable()); return *variable_;
      }
      bool operator<(const vertex_type& other) const {
        return std::make_pair(factor_, variable_) <
          std::make_pair(other.factor_, other.variable_);
      }
      bool operator==(const vertex_type& other) const {
        return (factor_ == other.factor_) && (variable_ == other.variable_);
      }
      bool operator!=(const vertex_type& other) const {
        return !(*this == other);
      }
      void print(std::ostream& out) const {
        if (is_variable()) {
          out << variable_;
        }
        else {
          out<<factor_;
        }
      }
    };
    
  private:

    //! A map from a vertex to its neighbors
    typedef std::vector<neighbors_type> neighbors_map_type;

    //! All the factors
    std::list<factor_type> factors_;

    std::map<const factor_type*, size_t> factor2id_;
    std::map<variable_type*, size_t> variable2id_;
    
    //! A map to the neighbors of a vertex
    neighbors_map_type neighbors_;

    /**
     * The set of all vertices. This also provides the conversion from
     * the global vertex id to the vertex_type itself. It is important
     * to note that ordering matters as seralization based routines
     * (like those used with MPI) rely on the ordering as a conversion
     * from vertex to ID.
     */
    std::vector<vertex_type> vertices_;
    mutable std::vector<size_t> work_per_update_cache_;
    /**
     * The set of variables which this model represents
     */
    domain_type args_;

public:
    /**
     * The save routine saves the:
     *   factors_
     *   neighbors_ map
     *   vertices_ (ordered)
     *   args_ (domain of this model)
     */
    void save(oarchive& ar) const {
      ar << factors_;
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
        add_factor(f);
      }
      rebuild_neighbors_and_indexes();
    }

    // TODO: we should check the indices, too
    //       but these are not preserved in serialization
    bool operator==(const factor_graph_model& other) const {
      return factors_ == other.factors_;
    }

    bool operator!=(const factor_graph_model& other) const {
      return factors_ != other.factors_;
    }

    void clear() {
      factors_.clear();
      neighbors_.clear();
      vertices_.clear();
      args_.clear();
      work_per_update_cache_.clear();
    }
    
    void build_work_per_update_cache() const {
      work_per_update_cache_.resize(vertices_.size());
      for (size_t i = 0; i< vertices_.size(); ++i) {
        if(vertices_[i].is_variable()) {
          size_t w = vertices_[i].variable().size() * num_neighbors(i);
          foreach(size_t j, neighbor_ids(i)) {
            w += vertices_[j].factor().size();
          }
          work_per_update_cache_[i] = w;
        }
        else {
          size_t w = vertices_[i].factor().size() * num_neighbors(i);
          foreach(size_t j, neighbor_ids(i)) {
            w += vertices_[j].variable().size();
          }
          work_per_update_cache_[i] = w;
        }
      }
    }
    /**
     * Creates a factor graph model with no factors and no variables
     * Use the add_factor method to add factors to this factor graph.
     */
    factor_graph_model() { }


    /**
     * Add a factor to this factor graph. All the variables are added
     * to this graphical model potentially changing the domain.
     */
    virtual size_t add_factor(const factor_type& factor) {
      factors_.push_back(factor);

      // Add the factor to the set of vertices
      // and update the vertices and neighbors map
      size_t factorid = add_factor_vertex_and_neighbors(&factors_.back());

      // Add the arguments to the domain of this graphical model
      foreach (variable_type* v, factor.arguments()) {
        args_.insert(v);
      }
//      refill_vertex2id();
      return factorid;
    }


    class VariablePtrComparator{
    public:
      bool operator()(variable_type* v1, variable_type* v2) const {
        return std::string(*v1) < std::string(*v2);
      }
    };

    /**
     * Randomly permutes neighbor lists to make graph traversal order
     * independent of the initial model construction order
     */
    void randomly_shuffle_neighbors(){
      for (size_t i = 0;i < vertices_.size(); ++i) 
        std::random_shuffle(neighbors_[i].begin(), neighbors_[i].end());
    }


    /**
     * Simplify the model by merging factors. For a factor A
     * such that a factor B exists with domain A in domain B,
     * remove A from the model and replace B with combine(B,A)
     */
     virtual void simplify(bool randomize = false) {

       // this is essentially the neighbor map, but we need sets for
       // efficient intersections
       std::map<variable_type*, std::set<factor_type*> > container_map;

       // Here we populate the new container_map.  TODO: This can be
       // done more efficiently by looping over the variables and then
       // using the range add operation on the neighbors of each
       // variable?
       //
       std::vector<factor_type*> factorpermute;
       foreach(factor_type& factor, factors_) {
         factorpermute.push_back(&factor);
         factor.normalize();
         foreach(variable_type* v, factor.arguments()) {
           container_map[v].insert(&factor);
         }
       }

       std::set<factor_type*> redundant_factors;
       size_t factors_processed = 0;
       if (randomize) {
         std::random_shuffle(factorpermute.begin(), factorpermute.end());
       }
       foreach(factor_type* f, factorpermute){
         factor_type &factor = *f;
         //here we need to find a non-redundant factor that includes
         //all the variables of factor in the old simplify(), we
         //actually find all such factors which can take a lot of time
         //here we only find the first one
         
         typedef typename std::set<factor_type*>::iterator iterator;
         std::vector<iterator> candidate_iterators;
         std::vector<iterator> candidate_ends;
         foreach(variable_type* v, factor.arguments()){
           candidate_iterators.push_back(container_map[v].begin());
           candidate_ends.push_back(container_map[v].end());
         }

         //the elements of std::set are sorted in the increasing order
         //so we will increment the smallest iterator of
         //candidate_iterators until either all of the iterators point
         //to the same factor (success) or some iterator points to the
         //end (failure)

         factor_type* intersection_result = NULL;
         int min_iterator_index, max_iterator_index;
         do{
           min_iterator_index = 0, max_iterator_index = 0;
           //looking for min and max iterator positions
           for(size_t i=0 ; i<candidate_iterators.size(); i++){
             if(*candidate_iterators[i] > 
                *candidate_iterators[max_iterator_index])
               max_iterator_index = i;
             if(*candidate_iterators[i] < 
                *candidate_iterators[min_iterator_index])
               min_iterator_index = i;
           }

           //check if everybody points to the same factor
           if(*candidate_iterators[max_iterator_index] == 
              *candidate_iterators[min_iterator_index])
             //check if that same factor is not the factor we are
             //trying to eliminate
             if(*candidate_iterators[min_iterator_index] != &factor ){
               intersection_result = 
                 *candidate_iterators[min_iterator_index];
               break;
             }

           //the break above did not fire -> not everybody points to
           //the same factor, or everybody points to the factrs we're
           //trying to remove increment the smallest iterator TODO
           //sometimes it is possible to do several increments here
           //and avoid recomputing the min/max iterators but boundary
           //checks are more complex
           candidate_iterators[min_iterator_index]++;

         }while(candidate_iterators[min_iterator_index] != 
                candidate_ends[min_iterator_index] );
         //while we are not out of factors along this dimension

         if(intersection_result != NULL){
           intersection_result->combine_in(factor, product_op);
           intersection_result->normalize();
           redundant_factors.insert(&factor);
           foreach(variable_type* vi, factor.arguments())
             container_map[vi].erase(&factor);
         }

         factors_processed++;
         if(factors_processed % 10000 == 0)
           std::cout << "processed " 
                     << factors_processed 
                     << " factors" << std::endl;
       } // End of outer foreach(factor_type& ...)

       // remove redundant factors.  TODO, I believe their is an
       // optimized remove operation.  We should consider running that
       // isntead
       typedef typename std::list<factor_type>::iterator iterator;
       for(iterator i = factors_.begin(); i != factors_.end();) {
         if(redundant_factors.find(&(*i)) != redundant_factors.end()) {
           i = factors_.erase(i);
         } else {
           i++;
         }
       }

       // rebuild vertices, neighbors etc. since vertices are passed
       // by value anyway, just clear all the old ones and recreate
       // everything for the new factors.
       // TODO maybe this can be made more efficient, but does not
       // seem like a huge deal right now
       neighbors_.clear();
       vertices_.clear();
       factor2id_.clear();
       variable2id_.clear();
       foreach(factor_type& f, factors_) {
         add_factor_vertex_and_neighbors(&f);
       }
       //re-fill the vertex->id map
       rebuild_neighbors_and_indexes();
     };


    /**
    * Simplify the model by merging factors. For a factor A
    * such that a factor B exists with domain A in domain B,
    * remove A from the model and replace B with combine(B,A).
    * This version compares variable names by value, so it gives
    * consistent results across serialize/deserialize cycles,
    * but is slower than simplify()
    */
    virtual void simplify_stable() {

      // this is essentially the neighbor map, but we need sets for
      // efficient intersections
      std::map<variable_type*, std::set<int> > container_map;
      std::map<factor_type*, std::vector<variable_type*> > 
        factor_args_ordering;
      std::map<factor_type*, int > factornum;
      std::vector<factor_type*> num2factor;
      // Here we populate the new container_map.  TODO: This can be
      // done more efficiently by looping over the variables and then
      // using the range add operation on the neighbors of each
      // variable?
      int count =0;
      foreach(factor_type& factor, factors_) {
        factor.normalize();
        factornum[&factor] = count;
        num2factor.push_back(&factor);
        count++;
        std::vector<variable_type*> ordering;
        foreach(variable_type* v, factor.arguments()){ 
          ordering.push_back(v);
        }
        std::sort(ordering.begin(),ordering.end(),
                  VariablePtrComparator());
        factor_args_ordering[&factor] = ordering;

        foreach(variable_type* v, factor_args_ordering[&factor]) {
          container_map[v].insert(factornum[&factor]);
        }
      }

      std::set<factor_type*> redundant_factors;
      size_t factors_processed = 0;
      foreach(factor_type& factor, factors_){
        //here we need to find a non-redundant factor
        //that includes all the variables of factor
        //in the old simplify(), we actually find all such factors
        //which can take a lot of time
        //here we only find the first one

        typedef typename std::set<int>::iterator iterator;
        std::vector<iterator> candidate_iterators;
        std::vector<iterator> candidate_ends;
        foreach(variable_type* v, factor_args_ordering[&factor]){
          candidate_iterators.push_back(container_map[v].begin());
          candidate_ends.push_back(container_map[v].end());
        }

        //the elements of std::set are sorted in the increasing order
        //so we will increment the smallest iterator of
        //candidate_iterators until either all of the iterators point
        //to the same factor (success) or some iterator points to the
        //end (failure)

        int intersection_result = -1;
        int min_iterator_index, max_iterator_index;
        do{
          min_iterator_index = 0, max_iterator_index = 0;
          //looking for min and max iterator positions
          for(size_t i=0 ; i<candidate_iterators.size(); i++){
            if(*candidate_iterators[i] > 
               *candidate_iterators[max_iterator_index]) {
              max_iterator_index = i;
            }
            if(*candidate_iterators[i] < 
               *candidate_iterators[min_iterator_index]) {
              min_iterator_index = i;
            }
          }

          //check if everybody points to the same factor
          if(*candidate_iterators[max_iterator_index] == 
             *candidate_iterators[min_iterator_index])
            //check if that same factor is not the factor we are
            //trying to eliminate
            if(num2factor[*candidate_iterators[min_iterator_index]] != 
               &factor ){
              intersection_result = 
                *candidate_iterators[min_iterator_index];
              break;
            }

          //the break above did not fire -> not everybody points to
          //the same factor, or everybody points to the factrs we're
          //trying to remove increment the smallest iterator TODO
          //sometimes it is possible to do several increments here and
          //avoid recomputing the min/max iterators but boundary
          //checks are more complex
          candidate_iterators[min_iterator_index]++;

        }while(candidate_iterators[min_iterator_index] != 
               candidate_ends[min_iterator_index]);
        //while we are not out of factors along this dimension

        if(intersection_result >=0){
          num2factor[intersection_result]->combine_in(factor, product_op);
          num2factor[intersection_result]->normalize();
          redundant_factors.insert(&factor);

          foreach(variable_type* vi, factor_args_ordering[&factor]) {
            container_map[vi].erase(factornum[&factor]);
          }
        }

        factors_processed++;
        if(factors_processed % 10000 == 0)
          std::cout << "processed " 
                    << factors_processed 
                    << " factors" << std::endl;
      } // End of outer foreach(factor_type& ...)

      // remove redundant factors.  TODO, I believe their is an
      // optimized remove operation.  We should consider running that
      // isntead
      typedef typename std::list<factor_type>::iterator iterator;
      for(iterator i = factors_.begin(); i != factors_.end();) {
        if(redundant_factors.find(&(*i)) != redundant_factors.end()) {
          i = factors_.erase(i);
        } else {
          i++;
        }
      }

      // rebuild vertices, neighbors etc. since vertices are passed
      // by value anyway, just clear all the old ones and recreate
      // everything for the new factors.
      // TODO maybe this can be made more efficient, but does not
      // seem like a huge deal right now
      neighbors_.clear();
      vertices_.clear();
      factor2id_.clear();
      variable2id_.clear();

      foreach(factor_type& f, factors_) {
        add_factor_vertex_and_neighbors(&f);
      }
      //re-fill the vertex->id map
      rebuild_neighbors_and_indexes();
    }

    /**
     * Normalize all factors
     */
    void normalize() {
      result_type maxval = -std::numeric_limits<double>::max();
      result_type minval = std::numeric_limits<double>::max();
      bool start = true;
      foreach(factor_type& f, factors_) {
        f.normalize();
        if (start) {
          maxval = f.maximum();
          minval = f.minimum();
          start = false;
        } else {
	        maxval = std::max(maxval,f.maximum());
          minval = std::min(minval,f.minimum());
        }
      }
      std::cout << "Factor max val: " << maxval << std::endl;
      std::cout << "Factor min val: " << minval << std::endl;
    } // end of normalize


    /**
     * Returns the number of neighbors of a vertex.  This will also
     * return zero if the vertex is not in the domain of this model.
     * \todo: should probably throw an exception
     */
    inline size_t num_neighbors(const vertex_type& v) const {
      return neighbors_[v.id()].size();
    }
    
    inline size_t num_neighbors(size_t vid) const {
      return neighbors_[vid].size();
    }


    /**
     * Returns the neighbors of a vertex.  This returns the set of
     * neighbors of a vertex.
     *
     * \todo We probably don't need to use the forward_range type?
     */
    const neighbors_type&
    neighbor_ids(const vertex_type& v) const {
      return neighbors_[v.id()];
    }

    const neighbors_type&
    neighbor_ids(size_t vid) const {
      return neighbors_[vid];
    }

    std::vector<vertex_type>
    neighbors(const vertex_type& v) const {
      return neighbors(v.id());
    }

    std::vector<vertex_type>
    neighbors(size_t vid) const {
      std::vector<vertex_type> ret;
      foreach(size_t nid, neighbor_ids(vid)) {
        ret.push_back(id2vertex(nid));
      }
      return ret;
    }

    /**
     * Compute the ammount of work required to update the vertex.
     * Effectively this is the weight of the vertex when partitioning
     * the graph for inference.
     */
    inline size_t work_per_update(const vertex_type& v) const {
      return work_per_update(v.id());
    } // end of work per update

    inline size_t work_per_update(const size_t vid) const {
      if (work_per_update_cache_.size() == 0) {
        build_work_per_update_cache();
      }
      return work_per_update_cache_[vid];
    } // end of work per update


    /**
     * Returns all the vertices associated with this factor graph.
     * Currently certain external libraries rely on the ordering of
     * these vertices.  Contact Joey or Yucheng before modifying.
     */
    const std::vector<vertex_type>& vertices() const {
      return vertices_;
    }
//     forward_range<const vertex_type&> vertices() const {
//       return vertices_;
//     }


    /**
     * Returns the number of vertices
     */
    size_t num_vertices() const {
      return vertices_.size();
    }

    //! The number of factors in this graphical model
    size_t size() const { return factors_.size(); }

    /**
     * Check the consistency of the internal data structures.
     * Useful for making sure the modifications such as simplify()
     * resulted in a correct state.
     *
     */
    virtual bool is_consistent(){
      if(args_.size() + factors_.size() != vertices_.size() )
        return false;

      //make sure that the vertices contain every variable exacty once
      foreach(variable_type* v, args_) {
        if(!check_unique_vertex(v)) {
          return false;
        }
      }

      //make sure the vertices contain every factor exacty once
      foreach(factor_type& f, factors_) {
        if(!check_unique_vertex(&f)) {
          return false;
        }
      }

      std::map<variable_type*, size_t> variable_neighbors_num;
      foreach(variable_type* v, args_) {
        variable_neighbors_num[v]=0;
      }

      //make sure every factor has its variables as neighbors
      //and nothing else
      foreach(factor_type& f, factors_){
        if(neighbors_[factor2id(&f)].size() != f.arguments().size()) {
          return false;
        }

        foreach(variable_type* v, f.arguments()){
          variable_neighbors_num[v]++;
          if(std::find(neighbors_[factor2id(&f)].begin(),
                       neighbors_[factor2id(&f)].end(),
                       variable2id(v)) == neighbors_[factor2id(&f)].end()) {
            return false;
          }
        }
      }

      //make sure every variable has its factors as neighbors
      //and nothing else
      foreach(variable_type* v, args_){
        if(neighbors_[variable2id(v)].size() != variable_neighbors_num[v]) {
          return false;
        }
        foreach(const vertex_type& f, neighbors(to_vertex(v))) {
          if(!f.factor().arguments().count(v)) {
            return false;
          }
        }
      }

      return true;
    } // end of bool is consistent()


    /////////////////////////////////////////////////////////////////
    // factorized_model<F> interface
    /////////////////////////////////////////////////////////////////

    domain_type arguments() const { return args_; }

    forward_range<const factor_type&> factors() const {
      return factors_;
    }

    virtual void integrate_evidence(const finite_assignment &asg) {
      neighbors_.clear();
      vertices_.clear();
      factor2id_.clear();
      variable2id_.clear();
      std::set<finite_variable*> asgkeys = keys(asg);
      args_ = set_difference(args_,asgkeys);
      foreach(factor_type& f, factors_) {
        if (is_subset(f.arguments(), asgkeys)) continue;
        f = f.restrict(asg);
        add_factor_vertex_and_neighbors(&f);
      }
      //re-fill the vertex->id map
      rebuild_neighbors_and_indexes();
    }
    
    // a must be complete! it must cover all the variables!
    virtual double log_likelihood(const assignment_type& a) const {
      double result = 0;
      foreach(const F& factor, factors_) {
        assignment_type localassg;
        foreach(variable_type* v, factor.arguments()) {
          localassg[v] = safe_get(a, v);
        }
        result += factor.logv(localassg);
      }
      return result;
    }

    virtual logarithmic<double> operator()(const assignment_type& a) const {
      return logarithmic<double>(log_likelihood(a), log_tag());
    }

    //! Prints the arguments and factors of the model.
    template <typename OutputStream>
    void print(OutputStream& out) const {
      out << "Arguments: " << arguments() << "\n"
          << "Factors:\n";
      foreach(F f, factors()) out << f;
    }

    virtual operator std::string() const {
      assert(false); // TODO
      // std::ostringstream out; out << *this; return out.str();
      return std::string();
    }

//     /////////////////////////////////////////////////////////////////
//     // graphical_model<F> interface
//     /////////////////////////////////////////////////////////////////

//     /**
//      * Returns a minimal markov graph that captures the dependencies
//      * in this model
//      * \todo implement this
//      */
//      sill::markov_graph<variable_type*> markov_graph() {
//       assert(false); // TODO
//       return sill::markov_graph<variable_type*>();
//     }

//     /**
//      * Determines whether x and y are separated, given z
//      * \todo implement this
//      */
//     bool d_separated(const domain_type& x, const domain_type& y,
//                      const domain_type& z = domain_type::empty_set) {
//       assert(false); // TODO
//       return false;
//     }


    /**
     * Prints the degree distribution to standard out.
     */
    void print_degree_distribution() {
      std::map<int,int> count;
      foreach(vertex_type v, vertices_) {
        count[num_neighbors(v)]++;
      }
      typename std::map<int,int>::iterator i = count.begin();
      while (i != count.end()) {
        std::cout << i->first << " " << i->second << std::endl;
        ++i;
      }
    }

    /**
     * Print the adjacency structure in the form of comma separated
     * edge pairs
     */
    virtual void print_adjacency(std::ostream& out) const {
      foreach(const vertex_type& u, vertices()) {
        size_t u_id = u.id();
        foreach(const vertex_type& v, neighbors(u)) {
          size_t v_id = v.id();
          if(u_id < v_id) {
            out << u_id << ", " << v_id << ", "  << std::endl;
          }
        }
      } 
    } // end of print adjacency


    /**
     * Prints the vertex info for each vertex
     *  vertexid, arity1, arity2, ... , arityn
     *  for all n variables in the domain.  
     * Vertices corresponding to variables only have one arity.
     */
    virtual void print_vertex_info(std::ostream& out) const {
      foreach(const vertex_type& u, vertices()) {
        size_t u_id = u.id();
        out << u_id;
        if(u.is_variable()) {
          out << ", " << u.variable().size() << std::endl;
        } else {
          const factor_type& factor(u.factor());
          domain_type args(factor.arguments());
          foreach(variable_type* var, args) {
            out << ", " << var->size();
          }
          out << std::endl;
        }
      } 
    } // end of print vertex info


    /**
     * Returns the unique id for a vertex
     */
    size_t vertex2id(const vertex_type& v) const{
      return v.id();
    }

    size_t factor2id(const factor_type* f) const {
      return safe_get(factor2id_, f);
    }
    
    size_t variable2id(variable_type* f) const {
      return safe_get(variable2id_, f);
    }

    vertex_type to_vertex(const factor_type* f) const {
      return vertices_[factor2id(f)];
    }

    vertex_type to_vertex(variable_type* v) const {
      return vertices_[variable2id(v)];
    }
    
    /**
     * Reteruns the vertex associated with a unique id.
     */
    vertex_type id2vertex(size_t id) const{
      return vertices_[id];
    }

    /**
     * fill in the vertex -> integer id map.  We are currently using
     * this for graph serialization but this mapping really belongs in
     * the fundamental variable type.
     *
     * \todo move id mapping into variable.h
     */
    void rebuild_neighbors_and_indexes(){
      factor2id_.clear();
      variable2id_.clear();
      work_per_update_cache_.clear();
      // first pass. Fill out for the factor2id and variable2id tables
      for (size_t i = 0; i< vertices_.size(); ++i) {
        vertices_[i].id_ = i;
        if (vertices_[i].is_variable()) {
          variable2id_[&(vertices_[i].variable())] = vertices_[i].id();
        }
        else {
          factor2id_[&(vertices_[i].factor())] = vertices_[i].id();
        }
      }

      // second pass. Rebuild the neighbors table
      neighbors_.clear();
      neighbors_.resize(vertices_.size());
      for (size_t i = 0; i< vertices_.size(); ++i) {
        vertices_[i].id_ = i;
        if (vertices_[i].is_factor()) {
          foreach(variable_type* v, vertices_[i].factor().arguments()) {
            neighbors_[i].push_back(variable2id_[v]);
            neighbors_[variable2id_[v]].push_back(i);
          }
        }
      }

    }

    virtual double bethe(const std::map<vertex_type, factor_type> &beliefs) {
      double U = 0;
      double H = 0;
      foreach (factor_type &f, factors_) {
        // crossentropy(b,f) = entropy(b) + KL(b||f)
        // U+=beliefs[vertex_type(&f)].entropy() +
        // beliefs[vertex_type(&f)].relative_entropy(f);
        U += safe_get(beliefs, to_vertex(&f)).cross_entropy(f);
      }

      foreach (factor_type &f, factors_) {
        H += safe_get(beliefs, to_vertex(&f)).entropy();
      }

      foreach (variable_type *v, arguments()) {
        vertex_type vert = to_vertex(v);
        H -= (num_neighbors(vert) - 1) * safe_get(beliefs, vert).entropy();
      }
      return U - H;
    }
  private:
    ///////////////////////////////////////////////////////////////
    // Private helper routines

    /**
     * Add a factor vertex and link it to the vertices of the arguments.
     * IMPORTANT: does not check whether the vertex for a given factor
     * already exists. Returns the vertex id of the factor
     */
    size_t add_factor_vertex_and_neighbors(factor_type *f){
      int factorid = vertices_.size();
      vertices_.push_back(vertex_type(f, vertices_.size()));
      factor2id_[f] = factorid;
      if (neighbors_.size() < vertices_.size()) neighbors_.resize(vertices_.size());
      foreach(variable_type* v, f->arguments()) {
        if(variable2id_.find(v) == variable2id_.end()) {
          variable2id_[v] = vertices_.size();
          vertices_.push_back( vertex_type(v, vertices_.size()) );
        }
        if (neighbors_.size() < vertices_.size()) neighbors_.resize(vertices_.size());
        neighbors_[variable2id(v)].push_back(factor2id(f));
        neighbors_[factor2id(f)].push_back(variable2id(v));
      }
      return factorid;
    }


    template<typename T>
    bool check_unique_vertex(T* t)
    {
      typename std::vector<vertex_type>::iterator i =
        std::find(vertices_.begin(), vertices_.end(), to_vertex(t));
      if(i == vertices_.end())
        return false;

      i = std::find(i+1, vertices_.end(), to_vertex(t));
      if(i != vertices_.end())
        return false;

      return true;
    }


  }; // factor_graph_model






  template<typename F> double
  mooij_kappen_w_ub(const factor_graph_model<F>& fg,
		    typename factor_graph_model<F>::vertex_type v1,
		    typename factor_graph_model<F>::vertex_type v2) {
    typedef typename factor_graph_model<F>::vertex_type vertex_type;
    typedef typename factor_graph_model<F>::variable_type variable_type;
    assert( (v1.is_factor() && v2.is_variable()) ||
	    (v2.is_factor() && v1.is_variable()));
    if(v2.is_factor()) std::swap(v1,v2);
    assert(v1.is_factor());
    assert(v2.is_variable());

    vertex_type v_factor = v1;
    const F& factor = v_factor.factor();
    vertex_type v_variable = v2;
    variable_type* variable = &(v_variable.variable());

    double mx = 0;
    foreach(vertex_type u, fg.neighbors(v_factor)) {
      if (u != v_variable) {
        mx = (mx + factor.bp_msg_derivative_ub(&(u.variable()), variable));
      }
    }
    return mx;
  } // end of bp_msg_max_derivative_ub








  /**
   * Write a factor graph to an output stream.  The display format is:
   *   <factor1>
   *   <factor2>
   *      ...
   *   <factorM>
   *
   * \todo we should improve the quality of the output of this
   * function.
   */
  template<typename F>
  inline std::ostream&
  operator<<(std::ostream& out,
             const factor_graph_model<F> fg) {
    foreach(const F& f, fg.factors()) {
      out << f << std::endl;
    }
    return out;
  } // end of operator<<

//   template<typename F>
//   inline std::ostream&
//   operator<<(std::ostream& out, const factor_graph_model<F>& fg) {
//     typedef factor_graph_model<F> factor_graph_model_type;
//     typedef typename factor_graph_model_type::vertex_type
//       vertex_type;
//     typedef typename factor_graph_model_type::vertex_id_type
//       vertex_id_type;
//     foreach(const vertex_type& u, fg.vertices()) {
//       vertex_id_type u_id = fg.vertex2id(u);
//       foreach(const vertex_type& v, fg.neighbors(u)) {
// 				vertex_id_type v_id = fg.vertex2id(v);
// 				if(u_id < v_id) {
// 					out << u_id << ", " << v_id << ", "  << std::endl;
// 				}
//       }
//     }
//     return out;
//   } // end of operator<<



  inline std::ostream&
  operator<<(std::ostream& out,
             const factor_graph_model<table_factor>::vertex_type& v) {
    if(v.is_variable()) {
      // return out << v.variable();
      return out << "variable";
    } else {
      // return out << "Factor:(" << v.factor().arguments() << ")";
      return out << "Factor";
    }
  } // end of operator<<

  inline std::ostream&
  operator<<(std::ostream& out,
             const factor_graph_model<log_table_factor>::vertex_type& v) {
    if(v.is_variable()) {
      // return out << v.variable();
      return out << "variable";
    } else {
      // return out << "Factor:(" << v.factor().arguments() << ")";
      return out << "Factor";
    }
  } // end of operator<<


    

}

#include <sill/macros_undef.hpp>

#endif // SILL_FACTOR_GRAPH_MODEL_HPP
