#define BOOST_DISABLE_ASSERTS
#define NDEBUG

// This file does not compile at the moment

#include <boost/timer.hpp>
#include <boost/lexical_cast.hpp>

#include <prl/factor/table_factor.hpp>
#include <prl/factor/fixed_factors.hpp>
#include <prl/factor/random.hpp>

#include <boost/random/mersenne_twister.hpp>

const size_t n = 10;
std::vector< prl::tablef > tf(n);
std::vector< prl::binary_factor<10> > bf(n);
std::vector< prl::unary_factor<10> > uf(n);


template <typename Factor, typename Tag>
double test(size_t m, const std::vector<Factor>& f, Tag tag) {
  boost::timer t;
  Factor result(f[0].arguments(), 0);
  //double result = 0;
  for(size_t j = 0; j < m; j++) {
    for(size_t i = 0; i < n; i++) {
      result.combine_in(f[i], tag);
    }
  }
  std::cout << (m*n/t.elapsed())/1000 << "KOPS" << std::endl;
  return result.norm_constant();
  //return result;
}

template <typename Tag>
double testall(size_t m, Tag tag) {
  double result = 0;
  std::cout << "Table factor: "; result += test(m, tf, tag);
  std::cout << "Unary factor: "; result += test(m*1000, uf, tag);
  std::cout << "Binary factor: "; result += test(m*100, bf, tag);
  return result;
}

int main(int argc, char** argv) {

  using namespace prl;
  using namespace std;

  boost::mt19937 rng;

  size_t m = (argc < 2) ? 1000 : boost::lexical_cast<size_t>(argv[1]);
  universe u;

  finite_var_vector v = u.new_finite_variables(3, 10);

  finite_domain abc = finite_domain(v);
  finite_domain ab = make_domain(v[0],v[1]);
  finite_domain a = finite_domain(v[0]);

  for(size_t i = 0; i < n; i++) {
    tf[i] = random_discrete_factor< tablef >(abc, rng);
    bf[i] = random_discrete_factor< binary_factor<10> >(ab, rng);
    uf[i] = random_discrete_factor< unary_factor<10> >(a, rng);
  }

  boost::timer t;
  double result = 0;

  cout << "Timings with op<plus>" << endl;
  result += testall(m, sum_op);

  cout << "Timings with op<times>" << endl;
  result += testall(m, product_op);

  /*
  cout << "Timings with op<custom>" << endl;
  result += testall(m, custom_tag(2));

  cout << "Timings with op<custom>" << endl;
  result += testall(m, custom_tag(1));

  cout << "Timings with op<virtual>" << endl;
  result += testall(m, virtual_tag());
  */

  cout << result << endl;
}

/*
Timings with op<plus>
Table factor: 18.5185KOPS
Unary factor: 21052.6KOPS
Binary factor: 2857.14KOPS

Timings with op<custom>
Table factor: 18.8679KOPS
Unary factor: 40000KOPS
Binary factor: 1242.24KOPS
\
Timings with op<custom>
Table factor: 19.2308KOPS
Unary factor: 40000KOPS
Binary factor: 1257.86KOPS

Timings with op<virtual>
Table factor: 15.0376KOPS
Unary factor: 4555.81KOPS
Binary factor: 952.381KOPS
*/

