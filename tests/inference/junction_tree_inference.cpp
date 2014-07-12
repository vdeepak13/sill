/*#include <sill/factor/gaussian_factors.hpp>
#include <sill/math/bindings/lapack.hpp>
#include <sill/model/junction_tree.hpp>
*/

#include <iostream>

#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/factor/table_factor.hpp>
#include <sill/graph/grid_graph.hpp>
#include <sill/model/decomposable.hpp>
#include <sill/model/markov_network.hpp>
#include <sill/model/random.hpp>
#include <sill/inference/junction_tree_inference.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

int main(int argc, char** argv) {
  using namespace boost;
  using namespace sill;
  using namespace std;

  boost::mt19937 rng;

  assert(argc==3);
  // Type definitions
  typedef pairwise_markov_network<table_factor> mn_type;

  // Loads a small treewidth model
  size_t m = lexical_cast<size_t>(argv[1]);
  size_t n = lexical_cast<size_t>(argv[2]);

  universe u;
  finite_var_vector variables = u.new_finite_variables(m*n, 2);

  cout << "Generating random model" << endl;
  mn_type mn;
  make_grid_graph(variables, m, n, mn);
  random_ising_model(mn, rng);
  if (m<10) cout << mn;

  cout << "Marginals using the multiplicative algorithm: " << endl;
  shafer_shenoy<table_factor> ss(mn);
  ss.calibrate();
  ss.normalize();
  cout << ss.clique_beliefs() << endl;
  
  cout << "Marginals using the division algorithm: " << endl;
  hugin<table_factor> hugin(mn);
  hugin.calibrate();
  hugin.normalize();
  cout << hugin.clique_beliefs() << endl;
  
  cout << "Clique marginals of a decomposable model: " << endl;
  decomposable<table_factor> dm;
  dm *= mn.factors();
  cout << dm.clique_marginals() << endl;
  
  //
  // cout << mn[0u].debug() << endl; // causes a compilation error
  //if (m<10) cout << mn;
}


// using namespace sill::math::bindings::lapack;
/*

// singleton constructor
double_vector vec(double x) {
  double_vector v(1);
  v[0] = x;
  return v;
}

// could change this to variable-argument style constructors
double_vector vec(double x0, double x1) {
  double_vector v(2);
  v[0] = x0; v[1] = x1;
  return v;
}

double_matrix mat(double x) {
  double_matrix m(1,1);
  m(0,0) = x;
  return m;
}

double_matrix mat(double x00, double x01, double x10, double x11) {
  double_matrix m(2,2);
  m(0,0) = x00; m(0,1) = x01; m(1,0) = x10; m(1,1) = x11;
  return m;
}

typedef sill::canonical_gaussian<double_matrix,double_vector> canonical_gaussian;
typedef sill::moment_gaussian<double_matrix,double_vector> moment_gaussian;
typedef sill::junction_tree<sill::variable_h,
                           canonical_gaussian,
                           sill::void_,
                           canonical_gaussian> jt_type;

std::map<std::size_t, jt_type::vertex_t> vertex;
std::map<jt_type::vertex_t, std::size_t> ind;

struct pass_flow_visitor
{
  typedef jt_type::edge_t edge_descriptor;
  typedef jt_type::vertex_t vertex_descriptor;
  jt_type* jt_ptr;

  pass_flow_visitor(jt_type& jt) : jt_ptr(&jt) { }

  template <typename G>
  void operator()(edge_descriptor e, G& g) const {
    using namespace std;
    vertex_descriptor u = source(e,g), v = target(e,g);
    canonical_gaussian f = jt_ptr->get_info(u);
    foreach(edge_descriptor d, in_edges(u, g)) {
      if (source(d,g) != v) f *= jt_ptr->get_dir_info(d);
    }
    f = f.marginal(jt_ptr->separator(e));
    cout << ind[u] << "->" << ind[v] << ": " << f << endl;
    jt_ptr->get_dir_info(e) = f;
  }
};

template <typename T>
std::vector<T> seq(T a) {
  return std::vector<T>(&a, &a + 1);
}

template <typename T>
std::vector<T> seq(T a, T b) {
  std::vector<T> result;
  result.push_back(a); result.push_back(b);
  return result;
}

int main(int argc, char* argv[])
{
  using namespace sill;
  using namespace std;
  using boost::lexical_cast;
  typedef jt_type::edge_t edge_descriptor;
  typedef jt_type::vertex_t vertex_descriptor;
  typedef ::canonical_gaussian canonical_gaussian;
  typedef ::moment_gaussian moment_gaussian;

  const size_t n = 4;
  universe u;

  // Create the variables
  std::map<size_t, variable_h> t;
  for(size_t i = 1; i <= n; i++)
    t[i] = u.new_vector_variable(lexical_cast<string>(i), 1);

  // Create the junction tree
  jt_type jt;
  for(size_t i = 1; i <= n; i++) {
    vertex[i] = jt.add_vertex(domain());
    ind[vertex[i]] = i;
  }

  // Load the model
  std::map< size_t, std::list<canonical_gaussian> > factors;
  #include "../../distinf/p2/data/temperature-estimation/model-cliquepot-4.cpp"
  for(size_t i = 1; i <= n; i++) {
    canonical_gaussian f = prod(factors[i]);
    jt.get_info(vertex[i]) = f;
    jt.update_clique(vertex[i], f.arguments());
    cout << i << ":" << f << endl;
  }
  for(size_t i = 2; i<=n; i++) jt.add_edge(vertex[1], vertex[i]);
  jt.triangulate();

  cout << "JT: " << jt << endl;

  // Do the message passing
  cout << "Messages:" << endl;
  mpp_traversal(jt.graph(), pass_flow_visitor(jt));

  // Compute the beliefs, just to be sure
  cout << "Beliefs:" << endl;
  foreach(vertex_descriptor v, jt.vertices()) {
    canonical_gaussian f = jt.get_info(v);
    foreach(edge_descriptor e, in_edges(v, jt.graph()))
      f *= jt.get_dir_info(e);
    cout << "Node " << ind[v] << ": " << moment_gaussian(f) << endl;
  }

  return 0;
}
*/
