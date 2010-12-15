#include <prl/factor/table_factor.hpp>
#include <prl/base/universe.hpp>
#include <prl/range/numeric.hpp>
#include <boost/lexical_cast.hpp>

using prl::universe;
using prl::finite_variable;

universe u;
finite_variable* x;
finite_variable* y;

prl::table_factor compute_message(const std::vector<prl::table_factor>& factors) {
  return prl::combine(factors,prl::product_op).marginal(make_domain(y));
}

int main(int argc, char* argv[])
{
  using namespace prl;
  using namespace std;
  assert(argc==3);
  
  size_t arity = boost::lexical_cast<size_t>(argv[1]);
  assert(arity>1);
  x = u.new_finite_variable("y",arity);
  y = u.new_finite_variable("y",arity);

  
  finite_domain doma = make_domain(x);
  finite_domain domb = make_domain(x, y);

  table_factor fa(doma, 0);
  table_factor fb(domb, 0);

  std::vector<table_factor> factors;
  factors.push_back(fa);
  factors.push_back(fa);
  factors.push_back(fa);
  factors.push_back(fb);

  size_t n = boost::lexical_cast<size_t>(argv[2]);
  for(size_t i = 0; i<n; i++) compute_message(factors);
}
