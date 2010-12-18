#ifndef SILL_DISTRIBUTED_FACTOR_GRAPH_HPP
#define SILL_DISTRIBUTED_FACTOR_GRAPH_HPP

#error "Incomplete and do not use!"

#include <sill/factor/concepts.hpp>
#include <sill/model/interfaces.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/macros_def.hpp>
#include <sill/mpi/mpi_wrapper.hpp>
#include <sill/range/forward_range.hpp>
#include <sill/parallel/pthread_tools.hpp>

#include <unistd.h> // for usleep
namespace sill {

  // protocol headers
  const char DF_TOTAL_BLOCKS_QUERY[5] = "tbk?";     // tbk?
  const char DF_TOTAL_BLOCKS_RESPONSE[5] = "tbk=";  // tbk=[int32]
  const char BLOCK_MOVE_UPDATE[5] = "bkm.";         // bkm.[int32block][int32newowner]
  const int DF_HEADER_LEN = 4;
  /// distributed factor graph protocol handler
  template <class F>
  class distributed_factor_graph_handler: 
                                    public mpi_post_office::po_box_callback {
  private:
    distributed_factor_graph<F> &g;
    map<string, const char*> command_map;
  public:
    distributed_factor_graph_handler(distributed_factor_graph<F> &graph)
                                                            :g(graph) {
      command_map[DF_TOTAL_BLOCKS_QUERY]=DF_TOTAL_BLOCKS_QUERY;
      command_map[DF_TOTAL_BLOCKS_RESPONSE]=DF_TOTAL_BLOCKS_RESPONSE;
    }
    
    
    void recv_message(const mpi_post_office::message& msg) {
      // grab the first 4 bytes
      char* body = msg.body;
      char* header = command_map[string(body,DF_HEADER_LEN)];
      body = header + DF_HEADER_LEN;
      if (header == NULL) return;
      
      // remote asked for the number of blocks
      if(header == DF_TOTAL_BLOCKS_QUERY) {
        // construct a response
        string response = DF_TOTAL_BLOCKS_RESPONSE;
        int32_t nblocks = g.total_block_count();
        response = response + string(reinterpret_cast<char*>(nblocks), 
                                      sizeof(int32_t));
        // send it back to originator
        g.send_message(msg.orig, MPI_FACTOR_GRAPH, 
                          response.length(), response.c_str());
      }
      
      
      // remote told me the number of blocks
      if(header == DF_TOTAL_BLOCKS_RESPONSE) {
        int32_t nblocks = g.total_block_count();
        g.totalnumblocks = *reinterpret_cast<int32_t*>(body);
      }
      
      
      // remote told me that a block moved
      if(header == BLOCK_MOVE_UPDATE) {
        int32_t block = *reinterpret_cast<int32_t*>(body);
        int32_t newowner = *reinterpret_cast<int32_t*>(body + 4);
        g.block2owner[block] = newowner;
      }
    }
  }
};





  /**
   * This represents a distributed factor graph graphical models.  A factor
   * graphical model is a bipartite graphical model where the two
   * sets of vertices correspond to variables and factors and there is
   * an undirected edge between a variable and a factor if the
   * variable is in the domain of the factor.
   *
   *
   * \ingroup model
   */
  template <typename F>
  class distributed_factor_graph : public factorized_model<F> {
    concept_assert((Factor<F>));
  public:
    friend distributed_factor_graph_handler<F>;
    
    typedef factor_graph_model::vertex_type   vertex_type;     // predecleration
    typedef graphical_model<F>  base;
    typedef typename base::factor_type        factor_type;
    typedef typename base::variable_type      variable_type;
    typedef typename base::domain_type        domain_type;
    typedef typename base::assignment_type    assignment_type;
    //! The set of neighbors type
    typedef std::vector<vertex_type>          neighbors_type;


    
    /**
     * The class used to represent vertices in the factor graph model.
     * Each vertex_type is the "union" of either a variable or a factor.
     * The underlying factor or variable can be accessed by reference
     * through the vertex_type.
     *
     * NOTE:
     * This is a clone of factor_graph_model::vertex_type;
     * I could inherit, but I worry about factor_type and variable_type
     * changing
     *
     */
    class vertex_type {
      const factor_type* factor_;
      variable_type* variable_;
    public:
      vertex_type() : factor_(NULL), variable_(NULL), local(true) { }
      vertex_type(const factor_type* f):factor_(f),variable_(NULL),local(true){}
      vertex_type(variable_type* v) : factor_(NULL), variable_(v), local(true){}
      inline bool is_factor() const { return factor_ != NULL; }
      inline bool is_variable() const { return variable_ != NULL; }
      inline const factor_type& factor() const { 
        assert(is_factor()); return *factor_;
      }
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
      
      //! whether this vertex is define
      bool local;
    };




  private:
  
    //! converts a  block number to an owner number (MPI rank)
    vector<int> block2owner;

    //! a list of all the factors in a block
    vector<vector<factor_type*> > block2factors;
    
    //! a list of all the variables in a block
    vector<vector<variable_type*> > block2vars;
    
    //! map from variable pointers to globally unique variable id 
    map<vertex_type, int32_t> vert2vertid;
    //! map from variable id to local variable pointers
    map<int32_t, vertex_type> vertid2vert;
    
    map<vertex_type, neighbors_type> neighbors_;
    
    int myblocks; ///< Number of blocks I own 
    int maxblocks;  ///< Maximum number of blocks this node can take
    int totalnumblocks; ///< Total number of blocks in the system
    
   
    // these variables mainly exist to provide fast access
    vector<vertex_type> vertices_;
    map<variable_type*, factor_type*> node_factor_;
    
    
    mpi_post_office &mpi;
    
    
    // creates the globally unique vertex numbering
    void createvertices() {
      int vid = 0;
      vertices.clear();
      for (int i = 0;i < totalnumblocks; ++i) {
        foreach(factor_type* f, block2factors[i]) {
          vert2vertid[vertex_type(f)] = vid;
          vertid2vert[vid] = vertex_type(f);
          vertices_.push_back(vertex_type(f));
          ++vid;
        }
        foreach(variable_type* v, block2vars[i]) {
          vert2vertid[vertex_type(v)] = vid;
          vertid2vert[vid] = vertex_type(v);
          vertices_.push_back(vertex_type(v));
          ++vid;
        }
      }
    }

    void createneighbors() {
      node_factor_.clear();
      neighbors_.clear();
      for (int i = 0;i < totalnumblocks; ++i) {
        foreach(factor_type* f, block2factors[i]) {
          if (f->arguments().size() > 1) {
            foreach(variable_type* v, f->arguments()) {
              neighbors_[vertex_type(f)].push_back(neighbors[vertex_type(v)];
              neighbors_[vertex_type(v)].push_back(neighbors[vertex_type(f)];
            }
          }
          else {
            node_factor_[*(f->arguments())] = f;
          }
        }
      }
    }
    
    // mostly copied from task_manager in old paraml code
    void createblocks(factor_graph_model<F> &f, int maxblocksize) {
      // create a list of all unassigned variables
      std::set<vertex_type> unassigned;
      copy(inserter(unassigned, unassigned.begin()), 
           f.vertices().begin(), f.vertices().end());
      
      /*
        NOTE:
        This is mildly annoying because the factors know its arguments
        through pointers to variable_type. Therefore, I can't just copy
        the factor directly. I will have to copy it, then perform
        a substitution later to totally seperate myself from 'f'
      */
      map<variable_type*, variable_type*> old2new_varmap;
      while(!unassigned.empty()) {  
        std::list<vertex_type> queue;    // Breadth first queue 
        std::set<vertex_type>  visited;  // Set of visited vertices
        vector<factor_type*> factorsinthisblock;
        vector<variable_type*> variablesinthisblock;
        // While the task is still too small and their remains unassigned
        // vertices
        int curblocksize = 0;
        while(curblocksize < maxblocksize && !unassigned.empty()) {
          if(queue.empty()) { 
            queue.push_front(*unassigned.begin());
            visited.insert(*unassigned.begin());
          }
          assert(!queue.empty());
          // Pop the first element off the queue 
          vertex_type v = queue.front(); queue.pop_front();
          // Add the element to the task
          if (v.is_factor()) {
            factorsinthisblock.push_back(new F(v.factor()));
          }
          else if (v.is_variable()) {
            variable_type* newvar = new variable_type(v.variable());
            old2new_varmap[v.variable()] = newvar;
            variablesinthisblock.push_back(newvar);
          }
          curblocksize++;
          
          // Remove the vertex from the set of unassigned vertices
          unassigned.erase(v); 
          // Add all its unassigned and unvisited neighbors to the queue
          foreach(vertex u, f.neighbors(v)) {
            if(unassigned.find(u) != unassigned.end() &&
              visited.find(u) == visited.end()) {
              queue.push_back(u);
              visited.insert(u);
            }
          } // end of add neighbors for loop
        } // End of block build foor loop
        block2factors.push_back(factorsinthisblock);
        block2vars.push_back(variablesinthisblock);
      }
      // update all the factors to use my allocation of variables
      for (int i = 0;i < totalnumblocks; ++i) {
        foreach(factor_type* f, block2factors[i]) {
          f->subst_args(old2new_varmap);
        }
      }

      // set the block owner (to myself)
      totalnumblocks = block2factors.size();
      myblocks = totalnumblocks;
      block2owner.resize(totalnumblocks);
      for(size_t i = 0;i < block2owner.size(); ++i) block2owner[i] = mpi.id();
    }
    
    
    /**
      Disconnects a block from the local graph and stores it in the 
      Block datastructure
    */
    void disconnect_block(int blockid, Block &b) {
      
    }
    /**
      Connects a block from the Block datastructure 
      to the local graph
    */
    void connect_block(int blockid, Block &b) {
      
    }
  public:
    /**
     * Creates a distributed factor graph model using a local factor graph
     * Use the add_factor method to add factors to this factor graph.
     */
    distributed_factor_graph(mpi_post_office &po, 
                             factor_graph_model<F> &f,
                             int maxblocksize,
                             int maxblockownership):
                                      mpi(po), maxblocks(maxblockownership) {
      // Create the blocks
      createblocks(f, maxblocksize);
      // Create vertex ids
      createvertices();
      createneighbors();
      //register callback
      mpi.register_handler(MPI_FACTOR_GRAPH, 
                           new distributed_factor_graph_handler(this));
    }
    
    /**
     * Creates a distributed factor graph object attached to the
     * "master" distributed factor graph at node 0
     */
    distributed_factor_graph(mpi_post_office &po,
                             int maxblockownership):
                                      mpi(po), maxblocks(maxblockownership) {

      totalnumblocks = -1;
      mpi.register_handler(MPI_FACTOR_GRAPH, 
                          new distributed_factor_graph_handler(this));
      
      // 1: Ask the master about the number of blocks
      mpi.send_message(0, MPI_FACTOR_GRAPH, 
                       DF_HEADER_LEN, DF_TOTAL_BLOCKS_QUERY);
      // then wait
      while(totalnumblocks < 0) {usleep(100000);}
      
      /// TODO: Otherstuff
      
    }
    
    int total_block_count() const{
      return totalnumblocks;
    }
    
    int my_block_count() const{
      return myblocks;
    }
    
    
    /**
     * Returns the neighbors of a vertex
     */
    size_t num_neighbors(const vertex_type& v) const {
      typedef typename neighbors_map_type::const_iterator iterator;
      iterator i = neighbors_.find(v);
      if (i == neighbors_.end()) return 0;
      else return i->second.size();
    }

    /**
     * Returns the neighbors of a vertex
     */
    forward_range<const vertex_type&> neighbors(const vertex_type& v) const {
      typedef typename neighbors_map_type::const_iterator iterator;
      iterator i = neighbors_.find(v);
      assert( i != neighbors_.end() );
      return i->second;
    }

    /**
     * Returns all the unary factors associated with a particular
     * variable. If none is associated this will return a NULL pointer
     */
    const factor_type* node_factor(variable_type* v) const {
      typename std::map<variable_type*, factor_type>::const_iterator
                                              i = node_factor_.find(v);
      if(i == node_factor_.end()) {
        return NULL;
      } else {
        return &(i->second);
      }
    }

    /**
     * Returns all the vertices associated with this factor graph
     */
    forward_range<const vertex_type&> vertices() const {
      return vertices_;
    }

    size_t size() const { return factors_.size(); }

    /////////////////////////////////////////////////////////////////
    // factorized_model<F> interface
    /////////////////////////////////////////////////////////////////

    domain_type arguments() const { return args_; }


    //! Prints the arguments and factors of the model.
    template <typename OutputStream>
    void print(OutputStream& out) const {
      out << "Arguments: " << arguments() << "\n"
          << "Factors:\n";
      foreach(F f, factors()) out << f;
    }

    operator std::string() const {
      assert(false); // TODO
      // std::ostringstream out; out << *this; return out.str(); 
      return std::string();
    }
  }; // factor_graph_model

  std::ostream&
  operator<<(std::ostream& out,
             const distributed_factor_graph<tablef>::vertex_type& v) {
    if(v.is_variable()) {
      // return out << v.variable();
      return out << "variable";
    } else {
      // return out << "Factor:(" << v.factor().arguments() << ")";
      return out << "Factor";
    }
  }

}





#include <sill/macros_undef.hpp>

#endif // SILL_DISTRIBUTED_FACTOR_GRAPH_HPP
