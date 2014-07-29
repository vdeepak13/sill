#define BOOST_TEST_MODULE vector_dataset
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset2/vector_dataset.hpp>
#include <sill/learning/parameter/moment_gaussian_mle.hpp>

#include <boost/math/special_functions.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

template class vector_dataset<double>;
// template class vector_dataset<float>;
// unsupported until we parameterize the vector_assignment
template class basic_record_iterator<vector_dataset<> >;
template class basic_const_record_iterator<vector_dataset<> >;
template class basic_sample_iterator<vector_dataset<> >;

BOOST_TEST_DONT_PRINT_LOG_VALUE(std::vector<double>);

BOOST_AUTO_TEST_CASE(test_insert) {
  universe u;
  vector_var_vector v = u.new_vector_variables(2, 1);
  v.push_back(u.new_vector_variable(2));
  
  vector_dataset<> ds;
  ds.initialize(v);
  
  // insert a record
  vector_record2<> r(4);
  r.values[0] = 2.0;
  r.values[1] = 0.0;
  r.values[2] = 1.0;
  r.values[3] = 1.5;
  r.weight = 0.5;
  ds.insert(r);

  // insert a vector assignment
  vector_assignment a;
  a[v[0]] = 1.0;
  a[v[1]] = 2.0;
  a[v[2]] = vec("0.0 0.5");
  ds.insert(a, 0.7);

  // insert a bunch of empty records
  ds.insert(10);

  // basic checks
  BOOST_CHECK_EQUAL(ds.size(), 12);
  vector_dataset<>::const_record_iterator it, end;
  boost::tie(it, end) = ds.records(v);

  // check if the first record is correct
  BOOST_CHECK(equal(it->values, vec("2.0 0.0 1.0 1.5")));
  BOOST_CHECK_EQUAL(it->weight, 0.5);
  ++it;

  // check if the second record is correct
  BOOST_CHECK(equal(it->values, vec("1.0 2.0 0.0 0.5")));
  BOOST_CHECK_EQUAL(it->weight, 0.7);
  ++it;

  // check the remaining records
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->values.size(), 4);
    for (size_t j = 0; j < 4; ++j) {
      BOOST_CHECK(boost::math::isnan(it->values[j]));
    }
    BOOST_CHECK_EQUAL(it->weight, 1.0);
    ++it;
  }
  
  // check that we covered all the records
  BOOST_CHECK(it == end);
}

BOOST_AUTO_TEST_CASE(test_iterator_comparisons) {
  universe u;
  vector_var_vector v = u.new_vector_variables(3, 1);
  
  vector_dataset<> ds;
  ds.initialize(v);
  ds.insert(vector_record2<>(3));

  vector_dataset<>::record_iterator it1, end1;
  boost::tie(it1, end1) = ds.records(v);

  vector_dataset<>::const_record_iterator it2, end2;
  boost::tie(it2, end2) = ds.records(v);

  BOOST_CHECK(it1 == it2);
  BOOST_CHECK(it2 == it1);

  BOOST_CHECK(end1 == end2);
  BOOST_CHECK(end2 == end1);

  BOOST_CHECK(it1 != end1);
  BOOST_CHECK(it1 != end2);
  BOOST_CHECK(it2 != end1);
  BOOST_CHECK(it2 != end2);

  BOOST_CHECK(++it1 == end1);
  BOOST_CHECK(++it2 == end2);
}

struct fixture {
  fixture()
    : v(u.new_vector_variables(3, 1)),
      f(v, "0.5 1 2", "3 2 1; 2 2 1; 1 1 2") {
    ds.initialize(v);
    f.normalize();
    for (size_t i = 0; i < 1000; ++i) {
      ds.insert(f.sample(rng));
    }
  }
  universe u;
  vector_var_vector v;
  vector_dataset<> ds;
  moment_gaussian f;
  boost::mt19937 rng;
};

BOOST_FIXTURE_TEST_CASE(test_records, fixture) {
  // verify that the distribution retrieved by immutable iterators
  // matches the factor for every variable or every pair of variables
  moment_gaussian_mle<> estim(&ds);
  for (size_t i = 0; i < v.size(); ++i) {
    for (size_t j = i; j < v.size(); ++j) {
      vector_domain dom = make_domain(v[i], v[j]);
      moment_gaussian mle = estim(dom);
      double kl = f.marginal(dom).relative_entropy(mle);
      std::cout << dom << ": " << kl << std::endl;
      BOOST_CHECK_SMALL(kl, 1e-2);
    }
  }

  // fill the content of the dataset using mutable iteration
  vector_var_vector v01 = make_vector(v[0], v[1]);
  foreach(vector_record2<>& r, ds.records(v01)) {
    r.values[0] = std::numeric_limits<double>::quiet_NaN();
    r.values[1] = std::numeric_limits<double>::quiet_NaN();
  }

  // verify that we get the mutated version back
  foreach(const vector_record2<>& r, ds.records(v01)) {
    BOOST_CHECK_EQUAL(r.values.size(), 2);
    BOOST_CHECK(boost::math::isnan(r.values[0]));
    BOOST_CHECK(boost::math::isnan(r.values[1]));
  }

  // verify that the marginal over v[2] is still good
  vector_domain dom2 = make_domain(v[2]);
  moment_gaussian mle = estim(dom2);
  double kl = f.marginal(dom2).relative_entropy(mle);
  std::cout << "Rest: " << kl << std::endl;
  BOOST_CHECK_SMALL(kl, 1e-2);
}

BOOST_FIXTURE_TEST_CASE(test_samples, fixture) {
  // draw samples from the dataset and attempt to recover the mean
  vec mean = arma::zeros(3);
  vector_dataset<>::sample_iterator it = ds.samples(v);
  for (size_t i = 0; i < 600; ++i, ++it) {
    mean += it->values * it->weight;
  }
  mean /= 600;
  double diff = norm(mean - f.mean(), 2);
  std::cout << "Samples: " << diff << std::endl;
  BOOST_CHECK_SMALL(diff, 0.1);
}

BOOST_FIXTURE_TEST_CASE(test_subset, fixture) {
  // test a range of the rows
  vector_dataset<> ds1 = ds.subset(0, 500);
  moment_gaussian_mle<> estim1(&ds1);
  moment_gaussian mle1 = estim1(v);
  double kl1 = f.relative_entropy(mle1);
  std::cout << "Range: " << kl1 << std::endl;
  BOOST_CHECK_SMALL(kl1, 1e-2);
}

BOOST_FIXTURE_TEST_CASE(test_shuffle, fixture) {
  moment_gaussian_mle<> estim(&ds);
  moment_gaussian mle1 = estim(v);
  ds.shuffle(rng);
  moment_gaussian mle2 = estim(v);
  double kl = mle1.relative_entropy(mle2);
  std::cout << "Shuffle: " << kl << std::endl;
  BOOST_CHECK_SMALL(kl, 1e-10);
}
