#include <prl/math/math.hpp>

#include <prl/variable.hpp>
#include <prl/model/markov_network.hpp>
#include <prl/model/random.hpp>
#include <prl/inference/belief_propagation.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/datastructure/dense_table.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/multi_array.hpp>
#include <boost/range.hpp>
#include <boost/tokenizer.hpp>

#include <vector>
#include <iostream>
#include <fstream>

#include <prl/macros_def.hpp>

enum { MAX_BUF = 60000 };

typedef boost::multi_array<double, 2> image_type;
typedef boost::multi_array<size_t, 2> vertex_array;
typedef prl::pairwise_markov_network<prl::tablef> markov_network_type;

image_type load_image(const char* filename) {
  typedef boost::tokenizer< boost::char_separator<char> > tokenizer;
  boost::char_separator<char> sep("\t ");

  using namespace std;
  char buf[MAX_BUF];

  // count the number of rows & columns
  size_t m = 0, n = -1;
  ifstream fin(filename);
  while (fin.good()) {
    fin.getline(buf, MAX_BUF); m++;
    if (n == size_t(-1)) {
      string str(buf);
      tokenizer tokens(str, sep);
      n = 0;
      foreach(const string& token, tokens) { token.size(); n++; }
    }
  }
  cout << m << " " << n << endl;
  assert(m>0 && n>0);

  // read in the image data
  image_type a(boost::extents[m-1][n]);
  fin.clear();
  fin.seekg(0);
  size_t i = 0;
  while (fin.good()) {
    fin.getline(buf, 60000);
    if (fin.good()) {
      string str(buf);
      tokenizer tokens(str, sep);
      size_t j = 0;
      foreach(const string& token, tokens) {
        assert(j < n);
        cout << token << endl;
        a[i][j++] = boost::lexical_cast<double>(token);
      }
      i++;
    }
  }

  return a;
}

void save_image(const image_type& image, const char* filename) {
  using namespace std;
  ofstream os(filename);
  for(size_t i = 0; i < image.shape()[0]; i++) {
    for(size_t j = 0; j < image.shape()[1]; j++)
      os << image[i][j] << " ";
    os << endl;
  }
}

template <typename Engine>
void save_belief(const char* filename,
                 const Engine& engine, const vertex_array& vertex) {
  using namespace std;
  using namespace prl;
  ofstream os(filename);
  for(size_t i = 0; i < vertex.shape()[0]; i++) {
    for(size_t j = 0; j < vertex.shape()[1]; j++) {
      variable_h v = engine.graphical_model().node(vertex[i][j]);
      os << engine.belief(vertex[i][j])(assignment(v, 1)) << " ";
    }
    os << endl;
  }
  os.flush(); os.close();
}

std::pair<markov_network_type, vertex_array>
make_segmentation_model(const image_type& image,
                        prl::universe& u,
                        double mean0, double var0,
                        double mean1, double var1,
                        double lambda) {
  using namespace std;
  using namespace prl;
  size_t m = image.shape()[0], n = image.shape()[1];

  adjacency_list<vecS,vecS,undirectedS> g;
  vertex_array vertex(boost::extents[m][n]);
  boost::tie(g, vertex) = make_grid_graph(m, n);
  markov_network_type mn(g, u.new_finite_variables(m*n, 2));

  // construct the vertex potentials
  for(size_t i = 0; i < m; i++)
    for(size_t j = 0; j < n; j++) {
      size_t v = vertex[i][j];
      variable_h var = mn.node(v);
      mn[v] = tablef(domain(var), 0);
      mn[v](assignment(var, 0)) = exp(-sqr((image[i][j]-mean0))/var0);
      mn[v](assignment(var, 1)) = exp(-sqr((image[i][j]-mean1))/var1);
    }

  // construct the edge potentials
  foreach(markov_network_type::edge_descriptor e, mn.edges())
    mn[e] = make_ising_factor<tablef>(mn.source_node(e),
                                           mn.target_node(e), lambda);
  return make_pair(mn, vertex);
}

int main(size_t argc, char* argv[])
{
  using namespace prl;
  using std::cout;
  using std::endl;
  using boost::lexical_cast;
  cout << "Arguments given:" << endl;
  for(size_t i = 0; i < argc; i++) cout << argv[i] << endl;

  assert(argc == 3);
  universe u;

  cout << "Loading the image: " << argv[1] << endl;
  image_type im = load_image(argv[1]);
  size_t m = im.shape()[0], n = im.shape()[1];

  cout << "Creating the network" << endl;
  markov_network_type mn;
  vertex_array vertex(boost::extents[m][n]);
  tie(mn, vertex) = make_segmentation_model(im, u, 147, 2, 150, 1, 2);
  if (m < 10 && n < 10) cout << mn << endl;

  cout << "Running LBP" << endl;
  residual_loopy_bp<markov_network_type> engine(mn);
  size_t niters = lexical_cast<size_t>(argv[2]);
  for(size_t i = 0; i < niters; i++) {
    engine.iterate(1);
    std::string fname = "output/para_result" + boost::lexical_cast<std::string>(i) + ".txt";
    save_belief(fname.c_str(), engine, vertex);
  }

//
//  // cout << mn << engine.node_beliefs() << endl;
//  save_belief("result.txt", engine, vertex);
//
  cout << "Average norm1: " << engine.average_residual() << endl;
}
