#include <sill/base/universe.hpp>
#include <sill/model/region_graph.hpp>
#include <sill/inference/kikuchi.hpp>

int main() {
  using namespace sill;
  using namespace std;

  universe u;
  std::vector<finite_variable*> x = u.new_finite_variables(6, 2);
  std::vector<finite_domain> root_clusters;

  root_clusters.push_back(make_domain(x[1], x[2], x[5]));
  root_clusters.push_back(make_domain(x[2], x[3], x[5]));
  root_clusters.push_back(make_domain(x[3], x[4], x[5]));
  root_clusters.push_back(make_domain(x[4], x[1], x[5]));

  region_graph<finite_variable*> rg;
  kikuchi(root_clusters, rg);

  cout << rg << endl;
}
