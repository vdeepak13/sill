#include <iostream>

#include <boost/random/mersenne_twister.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/gaussian_crf_factor.hpp>
#include <sill/model/random.hpp>

#include <sill/macros_def.hpp>

/**
 * \file gaussian_crf_factor_test.cpp  Test gaussian_crf_factor.
 *
 */

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  unsigned random_seed = time(NULL);

  universe u;
  boost::mt11213b rng(random_seed);

  vector_variable* Y1 = u.new_vector_variable(1);
  vector_variable* X1 = u.new_vector_variable(1);

  moment_gaussian
    mg_Y1X1(make_marginal_gaussian_factor(make_vector(Y1,X1), .3, .3, .2, rng));
  moment_gaussian mg_Y1_given_X1(mg_Y1X1.conditional(make_domain(X1)));
  gaussian_crf_factor gcf_Y1_given_X1(mg_Y1_given_X1);
  canonical_gaussian
    cg_Y1_given_X1(gcf_Y1_given_X1.get_gaussian<canonical_gaussian>());

  {
    std::cout << "Test:\tSample X1 ~ mg_X1.\n"
              << "\tCompare log P(Y1|X1), where P is mg / gcf / cg.\n"
              << "\t (with gcf built from mg, cg built from gcf)\n\n"
              << "nsamples\tmg\tgcf\tcg\tdiff(mg,gcf)\tdiff(gcf,cg)\n"
              << "-----------------------------------------" << std::endl;
    size_t n = 10000;
    size_t print_next = 1;
    double mg_ll = 0;
    double gcf_ll = 0;
    double cg_ll = 0;
    for (size_t i = 0; i < n; ++i) {
      vector_assignment a(mg_Y1X1.sample(rng));
      vector_assignment a_X1;
      a_X1[X1] = a[X1];
      moment_gaussian tmp_mg(mg_Y1_given_X1.restrict(a_X1));
      mg_ll += tmp_mg.logv(a);
//      mg_ll += mg_Y1_given_X1.logv(a);
      canonical_gaussian tmp_cg(gcf_Y1_given_X1.condition(a));
      tmp_cg.normalize();
      gcf_ll += tmp_cg.logv(a);
      tmp_cg = cg_Y1_given_X1.restrict(a_X1);
      tmp_cg.normalize();
      cg_ll += tmp_cg.logv(a);
      if (i == print_next) {
        std::cout << i << "\t" << (mg_ll / i) << "\t" << (gcf_ll / i)
                  << "\t" << (cg_ll / i) << "\t" << (abs(mg_ll - gcf_ll) / i)
                  << "\t" << (abs(cg_ll - gcf_ll) / i) << std::endl;
        print_next *= 2;
      }
    }
    std::cout << n << "\t" << (mg_ll / n) << "\t" << (gcf_ll / n)
              << "\t" << (cg_ll / n) << "\t" << (abs(mg_ll - gcf_ll) / n)
              << "\t" << (abs(cg_ll - gcf_ll) / n) << "\n\n"
              << "=================================================\n"
              << std::endl;
  }

} // main
