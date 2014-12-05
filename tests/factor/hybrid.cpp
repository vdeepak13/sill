#define BOOST_TEST_MODULE hybrid
#include <boost/test/unit_test.hpp>

#include <sill/factor/hybrid.hpp>
#include <sill/factor/canonical_gaussian.hpp>
#include <sill/factor/moment_gaussian.hpp>

#include <boost/range/algorithm.hpp>

#include "predicates.hpp"

using namespace sill;

template class hybrid<canonical_gaussian>;
template class hybrid<moment_gaussian>;

typedef hybrid<moment_gaussian> hybrid_moment;
typedef hybrid<canonical_gaussian> hybrid_canonical;
typedef hybrid_values<double> hybrid_index;

template std::ostream& operator<<(std::ostream&, const hybrid_moment&);
template std::ostream& operator<<(std::ostream&, const hybrid_canonical&);


BOOST_AUTO_TEST_CASE(test_construct) {
  universe u;
  finite_var_vector finite_vars = u.new_finite_variables(2, 2);
  vector_var_vector vector_vars = u.new_vector_variables(3, 1);
  finite_domain finite_dom = make_domain(finite_vars);
  vector_domain vector_dom = make_domain(vector_vars);
  finite_var_vector finite_sorted = finite_vars;
  vector_var_vector vector_sorted = vector_vars;
  boost::sort(finite_sorted);
  boost::sort(vector_sorted);

  // test constant constructor
  hybrid_moment h1(2.0);
  BOOST_CHECK_EQUAL(h1.num_finite(), 0);
  BOOST_CHECK_EQUAL(h1.num_vector(), 0);
  BOOST_CHECK_EQUAL(h1.finite_args(), finite_var_vector());
  BOOST_CHECK_EQUAL(h1.vector_args(), vector_var_vector());
  BOOST_CHECK_EQUAL(h1.arguments(), domain());
  BOOST_CHECK_EQUAL(h1.size(), 1);
  BOOST_CHECK_CLOSE(double(h1[0].norm_constant()), 2.0, 1e-2 /* percent */);
  BOOST_CHECK_CLOSE(double(h1(assignment())), 2.0, 1.0 /* percent */);
  BOOST_CHECK_CLOSE(double(h1(hybrid_index())), 2.0, 1.0 /* percent */);
  
  // test constructor with finite and vector argument sequences
  hybrid_moment h2(finite_vars, vector_vars, 0.5);
  BOOST_CHECK_EQUAL(h2.num_finite(), 2);
  BOOST_CHECK_EQUAL(h2.num_vector(), 3);
  BOOST_CHECK_EQUAL(h2.finite_args(), finite_vars);
  BOOST_CHECK_EQUAL(h2.vector_args(), vector_vars);
  BOOST_CHECK_EQUAL(h2.arguments(), set_union(finite_dom, vector_dom));
  BOOST_CHECK_EQUAL(h2.size(), 4);
  for (size_t i = 0; i < h2.size(); ++i) {
    BOOST_CHECK_CLOSE(double(h2[i].norm_constant()), 0.5, 1e-2 /* percent */);
  }

  // test constructor with finite and vector domains
  hybrid_moment h3(finite_dom, vector_dom, 3.0);
  BOOST_CHECK_EQUAL(h3.num_finite(), 2);
  BOOST_CHECK_EQUAL(h3.num_vector(), 3);
  BOOST_CHECK_EQUAL(h3.finite_args(), finite_sorted);
  BOOST_CHECK_EQUAL(h3.vector_args(), vector_sorted);
  BOOST_CHECK_EQUAL(h3.arguments(), set_union(finite_dom, vector_dom));
  BOOST_CHECK_EQUAL(h3.size(), 4);
  for (size_t i = 0; i < h3.size(); ++i) {
    BOOST_CHECK_CLOSE(double(h3[i].norm_constant()), 3.0, 1e-2 /* percent */);
  }

  // conversion constructor from table factor
  boost::array<double, 4> val = {{ 1, 0.5, 2, 3 }};
  hybrid_moment h4(make_dense_table_factor(finite_vars, val));
  BOOST_CHECK_EQUAL(h4.num_finite(), 2);
  BOOST_CHECK_EQUAL(h4.num_vector(), 0);
  BOOST_CHECK_EQUAL(h4.finite_args(), finite_vars);
  BOOST_CHECK_EQUAL(h4.vector_args(), vector_var_vector());
  BOOST_CHECK_EQUAL(h4.arguments(), domain(finite_vars.begin(), finite_vars.end()));
  BOOST_CHECK_EQUAL(h4.size(), 4);
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(double(h4[i].norm_constant()), val[i], 1e-2 /* percent */);
  }

  // conversion constructor from the component factor
  moment_gaussian mg(vector_vars, "1 2 3", arma::eye(3,3));
  hybrid_moment h5(mg);
  BOOST_CHECK_EQUAL(h5.num_finite(), 0);
  BOOST_CHECK_EQUAL(h5.num_vector(), 3);
  BOOST_CHECK_EQUAL(h5.finite_args(), finite_var_vector());
  BOOST_CHECK_EQUAL(h5.vector_args(), vector_vars);
  BOOST_CHECK_EQUAL(h5.arguments(), domain(vector_vars.begin(), vector_vars.end()));
  BOOST_CHECK_EQUAL(h5.size(), 1);
  BOOST_CHECK_SMALL(norm_inf(h5[0], mg), 1e-2);
}


BOOST_AUTO_TEST_CASE(test_assign) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 2);
  vector_variable* v = u.new_vector_variable("v", 1);
  vector_variable* w = u.new_vector_variable("w", 1);
  vector_variable* q = u.new_vector_variable("q", 1);
  vector_variable* r = u.new_vector_variable("r", 2);
  finite_var_vector finite_vars = make_vector(x, y);
  vector_var_vector vector_vars = make_vector(v, w, q);
  finite_domain finite_dom = make_domain(finite_vars);
  vector_domain vector_dom = make_domain(vector_vars);

  // assignment from another hybrid factor with same finite args
  boost::array<double, 4> val = {{ 1, 0.5, 2, 3 }};
  hybrid_moment h(make_dense_table_factor(finite_vars, val));
  hybrid_moment h1(finite_vars, vector_vars, 2.0);
  h = h1;
  BOOST_CHECK_EQUAL(h.num_finite(), 2);
  BOOST_CHECK_EQUAL(h.num_vector(), 3);
  BOOST_CHECK_EQUAL(h.finite_args(), finite_vars);
  BOOST_CHECK_EQUAL(h.vector_args(), vector_vars);
  BOOST_CHECK_EQUAL(h.arguments(), set_union(finite_dom, vector_dom));
  BOOST_CHECK_EQUAL(h.size(), 4);
  for (size_t i = 0; i < h.size(); ++i) {
    BOOST_CHECK_CLOSE(double(h[i].norm_constant()), 2.0, 1e-2 /* percent */);
  }

  // assignment from another hybrid factor with different finite args
  moment_gaussian mg(make_vector(r));
  hybrid_moment h2(mg);
  h = h2;
  BOOST_CHECK_EQUAL(h.num_finite(), 0);
  BOOST_CHECK_EQUAL(h.num_vector(), 1);
  BOOST_CHECK_EQUAL(h.finite_args(), finite_var_vector());
  BOOST_CHECK_EQUAL(h.vector_args(), make_vector(r));
  BOOST_CHECK_EQUAL(h.arguments(), make_domain<variable>(r));
  BOOST_CHECK_EQUAL(h.size(), 1);
  BOOST_CHECK_CLOSE(double(h[0].norm_constant()), 1.0, 1e-2 /* percent */);
  BOOST_CHECK(equal(h[0].mean(), vec(arma::zeros(2))));
  BOOST_CHECK(equal(h[0].covariance(), mat(arma::eye(2,2))));
  
  // assignment from a table factor with same finite variables
  h = h1;
  h = make_dense_table_factor(finite_vars, val);
  BOOST_CHECK_EQUAL(h.num_finite(), 2);
  BOOST_CHECK_EQUAL(h.num_vector(), 0);
  BOOST_CHECK_EQUAL(h.finite_args(), finite_vars);
  BOOST_CHECK_EQUAL(h.vector_args(), vector_var_vector());
  BOOST_CHECK_EQUAL(h.arguments(), domain(finite_vars.begin(), finite_vars.end()));
  BOOST_CHECK_EQUAL(h.size(), 4);
  for (size_t i = 0; i < 4; ++i) {
    BOOST_CHECK_CLOSE(double(h[i].norm_constant()), val[i], 1e-2 /* percent */);
  }

  // assignment from a constant
  h = 3.0;
  BOOST_CHECK_EQUAL(h.num_finite(), 0);
  BOOST_CHECK_EQUAL(h.num_vector(), 0);
  BOOST_CHECK_EQUAL(h.finite_args(), finite_var_vector());
  BOOST_CHECK_EQUAL(h.vector_args(), vector_var_vector());
  BOOST_CHECK_EQUAL(h.arguments(), domain());
  BOOST_CHECK_EQUAL(h.size(), 1);
  BOOST_CHECK_CLOSE(double(h[0].norm_constant()), 3.0, 1e-2 /* percent */);

  // assignment from a component factor
  h = h1;
  h = moment_gaussian(make_vector(r), 2.0);
  BOOST_CHECK_EQUAL(h.num_finite(), 0);
  BOOST_CHECK_EQUAL(h.num_vector(), 1);
  BOOST_CHECK_EQUAL(h.finite_args(), finite_var_vector());
  BOOST_CHECK_EQUAL(h.vector_args(), make_vector(r));
  BOOST_CHECK_EQUAL(h.arguments(), make_domain<variable>(r));
  BOOST_CHECK_EQUAL(h.size(), 1);
  BOOST_CHECK_CLOSE(double(h[0].norm_constant()), 2.0, 1e-2 /* percent */);
  BOOST_CHECK(equal(h[0].mean(), vec(zeros(2))));
  BOOST_CHECK(equal(h[0].covariance(), mat(eye(2, 2))));
}


BOOST_AUTO_TEST_CASE(test_value) {
  universe u;
  finite_var_vector finite_vars = u.new_finite_variables(2, 2);
  vector_var_vector vector_vars = u.new_vector_variables(1, 1);
  finite_variable* x = finite_vars[0];
  finite_variable* y = finite_vars[1];
  vector_variable* w = vector_vars[0];
  
  // create the instance
  hybrid_canonical h(finite_vars, vector_vars, 5.0);
  h[0] = canonical_gaussian(vector_vars, "1", "2");
  h[1] = canonical_gaussian(vector_vars, "2", "1");
  h[3] = canonical_gaussian(vector_vars, "0", "1", 10.0);
  
  // test operator()(assignment)
  assignment a;
  a[x] = 0;
  a[y] = 0;
  a[w] = "0.5";
  BOOST_CHECK_CLOSE(log(h(a)), -0.5 * (0.5 * 0.5) + 2 * 0.5, 1e-2 /* percent */);
  a[x] = 1;
  BOOST_CHECK_CLOSE(log(h(a)), -1.0 * (0.5 * 0.5) + 1 * 0.5, 1e-2 /* percent */);
  
  // test operator()(index)
  hybrid_index index(2, 1);
  index.finite[0] = 0;
  index.finite[1] = 1;
  index.vector[0] = 0.5;
  BOOST_CHECK_CLOSE(log(h(index)), log(5.0), 1e-2 /* percent */);
  index.finite[0] = 1;
  BOOST_CHECK_CLOSE(log(h(index)), 10.5, 1e-2 /* percent */);
}


BOOST_AUTO_TEST_CASE(test_combine) {
  // create the variables
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 2);
  finite_variable* z = u.new_finite_variable("z", 2);
  vector_variable* v = u.new_vector_variable("v", 1);
  vector_variable* w = u.new_vector_variable("w", 1);
  finite_var_vector finite_a = make_vector(x, y);
  finite_var_vector finite_b = make_vector(y, z);
  vector_var_vector vector_a = make_vector(v);
  vector_var_vector vector_b = make_vector(w);
  
  // create some hybrid factors
  boost::array<double, 4> val_a = {{ 1, 2, 3, 4 }};
  boost::array<double, 4> val_b = {{ 1, 0.5, 2, 3 }};
  hybrid_canonical ha(finite_a, vector_a);
  hybrid_canonical hb(finite_b, vector_b);
  for (size_t i = 0; i < 4;  ++i) {
    ha[i] = canonical_gaussian(vector_a, val_a[i]);
    hb[i] = canonical_gaussian(vector_b, val_b[i]);
  }

  // test the combination of two hybrids
  hybrid_canonical h1 = ha * hb;
  hybrid_canonical h2 = ha / hb;
  BOOST_CHECK_EQUAL(h1.arguments(), make_domain<variable>(x, y, z, v, w));
  BOOST_CHECK_EQUAL(h2.arguments(), make_domain<variable>(x, y, z, v, w));
  finite_assignment a;
  for (size_t i = 0; i < 2; ++i) {
    a[x] = i;
    for (size_t j = 0; j < 2; ++j) {
      a[y] = j;
      for (size_t k = 0; k < 2; ++k) {
        a[z] = k;
        double lm1 = log(val_a[i + 2*j] * val_b[j + 2*k]);
        double lm2 = log(val_a[i + 2*j] / val_b[j + 2*k]);
        BOOST_CHECK_EQUAL(h1(a).arguments(), make_domain(v, w));
        BOOST_CHECK_EQUAL(h2(a).arguments(), make_domain(v, w));
        BOOST_CHECK_CLOSE(h1(a).log_multiplier(), lm1, 1e-2 /* percent */);
        BOOST_CHECK_CLOSE(h2(a).log_multiplier(), lm2, 1e-2 /* percent */);
      }
    }
  }

  // test the combination of a hybrid and table factor
  table_factor f = make_dense_table_factor(finite_b, val_b);
  hybrid_canonical h3 = ha * f;
  hybrid_canonical h4 = ha / f;
  BOOST_CHECK_EQUAL(h3.arguments(), make_domain<variable>(x, y, z, v));
  BOOST_CHECK_EQUAL(h4.arguments(), make_domain<variable>(x, y, z, v));
  for (size_t i = 0; i < 2; ++i) {
    a[x] = i;
    for (size_t j = 0; j < 2; ++j) {
      a[y] = j;
      for (size_t k = 0; k < 2; ++k) {
        a[z] = k;
        double lm1 = log(val_a[i + 2*j] * val_b[j + 2*k]);
        double lm2 = log(val_a[i + 2*j] / val_b[j + 2*k]);
        BOOST_CHECK_EQUAL(h3(a).arguments(), make_domain(v));
        BOOST_CHECK_EQUAL(h4(a).arguments(), make_domain(v));
        BOOST_CHECK_CLOSE(h3(a).log_multiplier(), lm1, 1e-2 /* percent */);
        BOOST_CHECK_CLOSE(h4(a).log_multiplier(), lm2, 1e-2 /* percent */);
      }
    }
  }
  
  // test the combination of a hybrid and component factor
  hybrid_canonical h5 = ha * canonical_gaussian(vector_b, 2.0);
  hybrid_canonical h6 = ha / canonical_gaussian(vector_b, 2.0);
  BOOST_CHECK_EQUAL(h5.arguments(), make_domain<variable>(x, y, v, w));
  BOOST_CHECK_EQUAL(h6.arguments(), make_domain<variable>(x, y, v, w));
  std::vector<size_t> index(2);
  for (size_t i = 0; i < 2; ++i) {
    index[0] = i;
    for (size_t j = 0; j < 2; ++j) {
      index[1] = j;
      double lm1 = log(val_a[i + 2*j] * 2.0);
      double lm2 = log(val_a[i + 2*j] / 2.0);
      BOOST_CHECK_EQUAL(h5(index).arguments(), make_domain(v, w));
      BOOST_CHECK_EQUAL(h6(index).arguments(), make_domain(v, w));
      BOOST_CHECK_CLOSE(h5(index).log_multiplier(), lm1, 1e-2 /* percent */);
      BOOST_CHECK_CLOSE(h6(index).log_multiplier(), lm2, 1e-2 /* percent */);
    }
  }

  // test the combination of a hybrid and a constant
  hybrid_canonical h7 = ha * 2.0;
  hybrid_canonical h8 = ha / 2.0;
  BOOST_CHECK_EQUAL(h7.arguments(), make_domain<variable>(x, y, v));
  BOOST_CHECK_EQUAL(h8.arguments(), make_domain<variable>(x, y, v));
  for (size_t i = 0; i < 2; ++i) {
    index[0] = i;
    for (size_t j = 0; j < 2; ++j) {
      index[1] = j;
      double lm1 = log(val_a[i + 2*j] * 2.0);
      double lm2 = log(val_a[i + 2*j] / 2.0);
      BOOST_CHECK_EQUAL(h7(index).arguments(), make_domain(v));
      BOOST_CHECK_EQUAL(h8(index).arguments(), make_domain(v));
      BOOST_CHECK_CLOSE(h7(index).log_multiplier(), lm1, 1e-2 /* percent */);
      BOOST_CHECK_CLOSE(h8(index).log_multiplier(), lm2, 1e-2 /* percent */);
    }
  }
}


BOOST_AUTO_TEST_CASE(test_combine_in) {
  // create the variables
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 2);
  vector_variable* v = u.new_vector_variable("v", 1);
  vector_variable* w = u.new_vector_variable("w", 1);
  finite_var_vector finite_a = make_vector(x, y);
  finite_var_vector finite_b = make_vector(y);
  vector_var_vector vector_a = make_vector(v);
  vector_var_vector vector_b = make_vector(w);

  // create some hybrid factors
  boost::array<double, 4> val_a = {{ 1, 2, 3, 4 }};
  boost::array<double, 2> val_b = {{ 1, 0.5 }};
  hybrid_canonical ha(finite_a, vector_a);
  hybrid_canonical hb(finite_b, vector_b);
  for (size_t i = 0; i < 4;  ++i) {
    ha[i] = canonical_gaussian(vector_a, val_a[i]);
  }
  for (size_t i = 0; i < 2; ++i) {
    hb[i] = canonical_gaussian(vector_b, val_b[i]);
  }
  
  // test the combination of two hybrids
  hybrid_canonical h1 = ha; h1 *= hb;
  hybrid_canonical h2 = ha; h2 /= hb;
  BOOST_CHECK_EQUAL(h1.arguments(), make_domain<variable>(x, y, v, w));
  BOOST_CHECK_EQUAL(h2.arguments(), make_domain<variable>(x, y, v, w));
  std::vector<size_t> index(2);
  for (size_t i = 0; i < 2; ++i) {
    index[0] = i;
    for (size_t j = 0; j < 2; ++j) {
      index[1] = j;
      double lm1 = log(val_a[i + 2*j] * val_b[j]);
      double lm2 = log(val_a[i + 2*j] / val_b[j]);
      BOOST_CHECK_EQUAL(h1(index).arguments(), make_domain(v, w));
      BOOST_CHECK_EQUAL(h2(index).arguments(), make_domain(v, w));
      BOOST_CHECK_CLOSE(h1(index).log_multiplier(), lm1, 1e-2 /* percent */);
      BOOST_CHECK_CLOSE(h2(index).log_multiplier(), lm2, 1e-2 /* percent */);
    }
  }

  // test the combination of a hybrid and a table factor
  table_factor f = make_dense_table_factor(finite_b, val_b);
  hybrid_canonical h3 = ha; h3 *= f;
  hybrid_canonical h4 = ha; h4 /= f;
  BOOST_CHECK_EQUAL(h3.arguments(), make_domain<variable>(x, y, v));
  BOOST_CHECK_EQUAL(h4.arguments(), make_domain<variable>(x, y, v));
  for (size_t i = 0; i < 2; ++i) {
    index[0] = i;
    for (size_t j = 0; j < 2; ++j) {
      index[1] = j;
      double lm1 = log(val_a[i + 2*j] * val_b[j]);
      double lm2 = log(val_a[i + 2*j] / val_b[j]);
      BOOST_CHECK_EQUAL(h3(index).arguments(), make_domain(v));
      BOOST_CHECK_EQUAL(h4(index).arguments(), make_domain(v));
      BOOST_CHECK_CLOSE(h3(index).log_multiplier(), lm1, 1e-2 /* percent */);
      BOOST_CHECK_CLOSE(h4(index).log_multiplier(), lm2, 1e-2 /* percent */);
    }
  }

  // test the combination of a hybrid and a component factor
  canonical_gaussian cg(vector_b, 2.0);
  hybrid_canonical h5 = ha; h5 *= cg;
  hybrid_canonical h6 = ha; h6 /= cg;
  BOOST_CHECK_EQUAL(h5.arguments(), make_domain<variable>(x, y, v, w));
  BOOST_CHECK_EQUAL(h6.arguments(), make_domain<variable>(x, y, v, w));
  for (size_t i = 0; i < 2; ++i) {
    index[0] = i;
    for (size_t j = 0; j < 2; ++j) {
      index[1] = j;
      double lm1 = log(val_a[i + 2*j] * 2.0);
      double lm2 = log(val_a[i + 2*j] / 2.0);
      BOOST_CHECK_EQUAL(h5(index).arguments(), make_domain(v, w));
      BOOST_CHECK_EQUAL(h6(index).arguments(), make_domain(v, w));
      BOOST_CHECK_CLOSE(h5(index).log_multiplier(), lm1, 1e-2 /* percent */);
      BOOST_CHECK_CLOSE(h6(index).log_multiplier(), lm2, 1e-2 /* percent */);
    }
  }
  
  // test the combination of a hybrid and a constant
  hybrid_canonical h7 = ha; h7 *= 2.0;
  hybrid_canonical h8 = ha; h8 /= 2.0;
  BOOST_CHECK_EQUAL(h7.arguments(), make_domain<variable>(x, y, v));
  BOOST_CHECK_EQUAL(h8.arguments(), make_domain<variable>(x, y, v));
  for (size_t i = 0; i < 2; ++i) {
    index[0] = i;
    for (size_t j = 0; j < 2; ++j) {
      index[1] = j;
      double lm1 = log(val_a[i + 2*j] * 2.0);
      double lm2 = log(val_a[i + 2*j] / 2.0);
      BOOST_CHECK_EQUAL(h7(index).arguments(), make_domain(v));
      BOOST_CHECK_EQUAL(h8(index).arguments(), make_domain(v));
      BOOST_CHECK_CLOSE(h7(index).log_multiplier(), lm1, 1e-2 /* percent */);
      BOOST_CHECK_CLOSE(h8(index).log_multiplier(), lm2, 1e-2 /* percent */);
    }
  }
}


BOOST_AUTO_TEST_CASE(test_marginal) {
  // create some variables
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 2);
  finite_variable* z = u.new_finite_variable("z", 2);
  vector_variable* v = u.new_vector_variable("v", 1);
  vector_variable* w = u.new_vector_variable("w", 1);
  finite_var_vector finite_vars = make_vector(x, y, z);
  vector_var_vector vector_vars = make_vector(v, w);

  // marginal over a finite domain
  boost::array<double, 8> vals = {{1, 0.1, 3.2, 4.8, 0.5, 2.1, 2.0, 1.5}};
  hybrid_moment ha(finite_vars, vector_vars);
  for (size_t i = 0; i < 8; ++i) {
    ha[i] = moment_gaussian(vector_vars, vals[i]);
  }
  table_factor f1 = ha.marginal(make_domain(x, z));
  BOOST_CHECK_EQUAL(f1.arguments(), make_domain(x, z));
  std::vector<size_t> index(2);
  for (size_t i = 0; i < 2; ++i) {
    index[0] = i;
    for (size_t j = 0; j < 2; ++j) {
      index[1] = j;
      BOOST_CHECK_CLOSE(f1(index), vals[i + 4*j] + vals[i + 2 + 4*j], 1e-2);
    }
  }

  // marginalizing out vector variables
  hybrid_moment hb(make_vector(x), vector_vars);
  hb[0] = moment_gaussian(vector_vars, "1 2", eye(2, 2) * 2.0, 1.5);
  hb[1] = moment_gaussian(vector_vars, "2 4", eye(2, 2) * 0.5, 2.5);
  hybrid_moment h2 = hb.marginal(make_domain<variable>(x, v));
  BOOST_CHECK_EQUAL(h2.arguments(), make_domain<variable>(x, v));
  BOOST_CHECK_EQUAL(h2.num_finite(), 1);
  BOOST_CHECK_EQUAL(h2.num_vector(), 1);
  BOOST_CHECK_EQUAL(h2.finite_args(), make_vector(x));
  BOOST_CHECK_EQUAL(h2.vector_args(), make_vector(v));
  BOOST_CHECK_EQUAL(h2.size(), 2);
  BOOST_CHECK(equal(h2[0].mean(), vec("1")));
  BOOST_CHECK(equal(h2[0].covariance(), arma::mat("2")));
  BOOST_CHECK_CLOSE(double(h2[0].norm_constant()), 1.5, 1e-2 /* percent */);
  BOOST_CHECK(equal(h2[1].mean(), vec("2")));
  BOOST_CHECK(equal(h2[1].covariance(), mat("0.5")));
  BOOST_CHECK_CLOSE(double(h2[1].norm_constant()), 2.5, 1e-2 /* percent */);

  // unsupported marginals
  BOOST_CHECK_THROW(ha.marginal(make_domain<variable>(x, v)),
                    std::invalid_argument);
}


BOOST_AUTO_TEST_CASE(test_restrict) {
  // create some variables
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 2);
  vector_variable* v = u.new_vector_variable("v", 1);
  vector_variable* w = u.new_vector_variable("w", 1);
  finite_var_vector finite_vars = make_vector(x, y);
  vector_var_vector vector_vars = make_vector(v, w);

  // create the factor to be restricted
  hybrid_canonical h(finite_vars, vector_vars);
  std::vector<size_t> index(2);
  h(index) = canonical_gaussian(vector_vars, eye(2, 2)*1.2, "3 4", 3.0);
  index[0] = 1;
  h(index) = canonical_gaussian(vector_vars, eye(2, 2)*1.3, "4 5", 4.0);
  index[0] = 0; index[1] = 1;
  h(index) = canonical_gaussian(vector_vars, eye(2, 2), "1 2", 1.0);
  index[0] = 1;
  h(index) = canonical_gaussian(vector_vars, eye(2, 2)*1.1, "2 3", 2.0);
  
  // restrict some finite variables and some vector variables
  assignment a;
  a[y] = 1;
  a[w] = 2.0;
  hybrid_canonical h1 = h.restrict(a);
  BOOST_CHECK_EQUAL(h1.num_finite(), 1);
  BOOST_CHECK_EQUAL(h1.num_vector(), 1);
  BOOST_CHECK_EQUAL(h1.finite_args(), make_vector(x));
  BOOST_CHECK_EQUAL(h1.vector_args(), make_vector(v));
  BOOST_CHECK_EQUAL(h1.arguments(), make_domain<variable>(x, v));
  BOOST_CHECK_EQUAL(h1.size(), 2);
  BOOST_CHECK(equal(h1[0].inf_vector(), vec("1")));
  BOOST_CHECK(equal(h1[0].inf_matrix(), mat(eye(1,1))));
  BOOST_CHECK_CLOSE(h1[0].log_multiplier(), 1.0 + 4.0 - 2.0, 1e-2 /* percent */);
  BOOST_CHECK(equal(h1[1].inf_vector(), vec("2")));
  BOOST_CHECK(equal(h1[1].inf_matrix(), mat(eye(1,1)*1.1)));
  BOOST_CHECK_CLOSE(h1[1].log_multiplier(), 2.0 + 6.0 - 2.2, 1e-2 /* percent */);

  // restrict some finite variables and all vector variables
  a[v] = 0.5;
  table_factor f2;
  h.restrict(a, f2);
  BOOST_CHECK_EQUAL(f2.arguments(), make_domain(x));
  BOOST_CHECK_CLOSE(log(f2(0)), 1.0 + 4.5 - 0.5 * 4.25, 1e-2 /* percent */);
  BOOST_CHECK_CLOSE(log(f2(1)), 2.0 + 7.0 - 0.55 * 4.25, 1e-2 /* percent */);
}


BOOST_AUTO_TEST_CASE(test_normalize) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 3);
  vector_variable* y = u.new_vector_variable("y", 2);
  finite_var_vector finite_vars = make_vector(x);
  vector_var_vector vector_vars = make_vector(y);

  hybrid_moment h(finite_vars, vector_vars);
  h[0] = moment_gaussian(vector_vars, "0 1", eye(2,2), 1.0);
  h[1] = moment_gaussian(vector_vars, "1 2", eye(2,2) * 1.1, 2.0);
  h[2] = moment_gaussian(vector_vars, "2 3", eye(2,2) * 1.2, 2.0);
  BOOST_CHECK_CLOSE(double(h.norm_constant()), 5.0, 1e-2 /* percent */);
  h.normalize();

  BOOST_CHECK_EQUAL(h.finite_args(), finite_vars);
  BOOST_CHECK_EQUAL(h.vector_args(), vector_vars);
  BOOST_CHECK_EQUAL(h.arguments(), make_domain<variable>(x, y));
  BOOST_CHECK_CLOSE(double(h.norm_constant()), 1.0, 1e-2 /* percent */);
  BOOST_CHECK_CLOSE(double(h[0].norm_constant()), 0.2, 1e-2 /* percent */);
  BOOST_CHECK_CLOSE(double(h[1].norm_constant()), 0.4, 1e-2 /* percent */);
  BOOST_CHECK_CLOSE(double(h[2].norm_constant()), 0.4, 1e-2 /* precent */);
  BOOST_CHECK(equal(h[0].mean(), vec("0 1")));
  BOOST_CHECK(equal(h[1].mean(), vec("1 2")));
  BOOST_CHECK(equal(h[2].mean(), vec("2 3")));
  BOOST_CHECK(equal(h[0].covariance(), mat(eye(2,2))));
  BOOST_CHECK(equal(h[1].covariance(), mat(eye(2,2) * 1.1)));
  BOOST_CHECK(equal(h[2].covariance(), mat(eye(2,2) * 1.2)));
}


BOOST_AUTO_TEST_CASE(test_evaluator) {
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);
  finite_var_vector finite_vars = make_vector(x, y);
  
  boost::array<double, 6> vals = {{ 0.0, 0.5, 1.0, 1.5, 2.0, 2.5 }};
  hybrid_moment h(finite_vars);
  boost::copy(vals, h.begin());
  factor_evaluator<hybrid_moment> evaluator(h);
  
  hybrid_values<double> index(2, 0);
  for (size_t i = 0; i < 2; ++i) {
    index.finite[0] = i;
    for (size_t j = 0; j < 3; ++j) {
      index.finite[1] = j;
      BOOST_CHECK_CLOSE(double(evaluator(index)), vals[i + j*2], 1e-2 /* percent */);
    }
  }
}

BOOST_AUTO_TEST_CASE(test_marginal_sampler_mle) {
  // experiment parameters
  size_t nsamples = 100000;
  double tol = 5.0; /* percent */

  // create a few variables
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);
  vector_variable* z = u.new_vector_variable("z", 1);
  finite_var_vector finite_vars = make_vector(x, y);
  vector_var_vector vector_vars = make_vector(z);

  // create the hybrid factor
  hybrid_moment h(finite_vars, vector_vars);
  for (size_t i = 0; i < h.size(); ++i) {
    h[i] = moment_gaussian(vector_vars, ones(1,1) * (i+0.5), ones(1,1) * (i+1), i+1);
  }
  h.normalize();

  // draw samples and simultaneously train a model
  factor_sampler<hybrid_moment> sampler(h);
  factor_mle_incremental<hybrid_moment> mle(concat(finite_vars, vector_vars));
  hybrid_values<double> sample;
  boost::lagged_fibonacci607 rng;
  for (size_t i = 0; i < nsamples; ++i) {
    sampler(sample, rng);
    mle.process(sample, 1.0);
    //std::cout << sample << std::endl;
  }

  // compare the trained and the original model
  hybrid_moment h2 = mle.estimate();
  BOOST_CHECK_EQUAL(h2.arguments(), make_domain<variable>(x, y, z));
  BOOST_CHECK_EQUAL(h2.size(), 6);
  BOOST_CHECK_EQUAL(h2.num_finite(), 2);
  BOOST_CHECK_EQUAL(h2.num_vector(), 1);
  for (size_t i = 0; i < 6; ++i) {
    BOOST_CHECK_CLOSE(double(h2[i].norm_constant()), (i+1) / double(1+2+3+4+5+6), tol);
    BOOST_CHECK_CLOSE(h2[i].mean()[0], i+0.5, tol);
    BOOST_CHECK_CLOSE(h2[i].covariance()(0,0), i+1, tol);
  }
}


BOOST_AUTO_TEST_CASE(test_conditional_sampler_mle) {
  // experiment parameters
  size_t nsamples = 50000; // samples per assignment to tail
  double tol = 5.0; /* percent */

  // create a few variables
  universe u;
  finite_variable* x = u.new_finite_variable("x", 2);
  finite_variable* y = u.new_finite_variable("y", 3);
  vector_variable* z = u.new_vector_variable("z", 1);
  var_vector head_vars = make_vector<variable>(y, z);
  var_vector tail_vars = make_vector<variable>(x);

  // create the hybrid factor
  hybrid_moment h(make_vector(y, x), make_vector(z));
  std::vector<double> sum(2);
  for (size_t i = 0; i < h.size(); ++i) {
    h[i] = moment_gaussian(make_vector(z),
                           ones(1,1) * (i+0.5),
                           ones(1,1) * (i+1),
                           i+1);
    sum[i / 3] += i + 1;
  }
  h /= h.marginal(make_domain(x)); // conditional p(z, y | x)

  // draw samples and simultaneously train a model
  factor_sampler<hybrid_moment> sampler(h, head_vars);
  factor_mle_incremental<hybrid_moment> mle(head_vars, tail_vars);
  hybrid_values<double> sample(2,1);
  boost::lagged_fibonacci607 rng;
  for (size_t k = 0; k < 2; ++k) {
    hybrid_values<double> tail(1, 0);
    hybrid_values<double> head(1, 1);
    tail.finite[0] = k;
    sample.finite[1] = k;
    for (size_t i = 0; i < nsamples; ++i) {
      sampler(head, tail, rng);
      sample.finite[0] = head.finite[0];
      sample.vector = head.vector;
      mle.process(sample, 1.0);
    }
  }

  // compare the trained and the original model
  hybrid_moment h2 = mle.estimate();
  BOOST_CHECK_EQUAL(h2.arguments(), make_domain<variable>(x, y, z));
  BOOST_CHECK_EQUAL(h2.size(), 6);
  BOOST_CHECK_EQUAL(h2.num_finite(), 2);
  BOOST_CHECK_EQUAL(h2.num_vector(), 1);
  BOOST_CHECK_EQUAL(h2.finite_args(), make_vector(y, x));
  BOOST_CHECK_EQUAL(h2.vector_args(), make_vector(z));
  for (size_t i = 0; i < 6; ++i) {
    BOOST_CHECK_CLOSE(double(h2[i].norm_constant()), (i+1) / sum[i / 3], tol);
    BOOST_CHECK_CLOSE(h2[i].mean()[0], i+0.5, tol);
    BOOST_CHECK_CLOSE(h2[i].covariance()(0,0), i+1, tol);
  }
}
