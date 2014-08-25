#define BOOST_TEST_MODULE vector_dataset
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/vector_dataset_io.hpp>
#include <sill/learning/dataset/vector_memory_dataset.hpp>
#include <sill/learning/factor_mle/moment_gaussian.hpp>

#include <boost/math/special_functions.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

template class vector_dataset<double>;
// template class vector_dataset<float>;
// unsupported until we parameterize the vector_assignment
template class raw_record_iterator<vector_dataset<> >;
template class raw_const_record_iterator<vector_dataset<> >;
template class slice_view<vector_dataset<> >;

BOOST_AUTO_TEST_CASE(test_insert) {
  universe u;
  vector_var_vector v = u.new_vector_variables(2, 1);
  v.push_back(u.new_vector_variable(2));
  
  vector_memory_dataset<> ds;
  ds.initialize(v);
  
  // insert a record
  vector_record<> r(4);
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

  // print the dataset
  std::cout << ds << std::endl;
  size_t i = 0;
  foreach (const vector_record<>& r, ds.records(v)) {
    std::cout << i << ": " << r << std::endl;
    ++i;
  }

  // basic checks
  BOOST_CHECK_EQUAL(ds.size(), 12);
  vector_dataset<>::const_record_iterator it, end;
  boost::tie(it, end) = ds.records(v);

  // check if the first record is correct
  BOOST_CHECK(equal(it->values, vec("2.0 0.0 1.0 1.5")));
  BOOST_CHECK_EQUAL(it->weight, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.record(0));
  ++it;

  // check if the second record is correct
  BOOST_CHECK(equal(it->values, vec("1.0 2.0 0.0 0.5")));
  BOOST_CHECK_EQUAL(it->weight, 0.7);
  BOOST_CHECK_EQUAL(*it, ds.record(1));
  ++it;

  // check the remaining records
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->values.size(), 4);
    for (size_t j = 0; j < 4; ++j) {
      BOOST_CHECK(boost::math::isnan(it->values[j]));
    }
    BOOST_CHECK_EQUAL(it->weight, 1.0);
    BOOST_CHECK_EQUAL(it->weight, ds.record(i + 2).weight);
    ++it;
  }
  
  // check that we covered all the records
  BOOST_CHECK(it == end);
}

BOOST_AUTO_TEST_CASE(test_iterator_comparisons) {
  universe u;
  vector_var_vector v = u.new_vector_variables(3, 1);
  
  vector_memory_dataset<> ds;
  ds.initialize(v);
  ds.insert(vector_record<>(3));

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
    ds.initialize(v, 1000);
    f.normalize();
    for (size_t i = 0; i < 1000; ++i) {
      ds.insert(f.sample(rng));
    }
  }
  universe u;
  vector_var_vector v;
  vector_memory_dataset<> ds;
  moment_gaussian f;
  boost::mt19937 rng;
};

BOOST_FIXTURE_TEST_CASE(test_records, fixture) {
  // verify that the distribution retrieved by immutable iterators
  // matches the factor for every variable or every pair of variables
  factor_mle<moment_gaussian> estim(&ds);
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
  foreach(vector_record<>& r, ds.records(v01)) {
    r.values[0] = std::numeric_limits<double>::quiet_NaN();
    r.values[1] = std::numeric_limits<double>::quiet_NaN();
  }

  // verify that we get the mutated version back
  foreach(const vector_record<>& r, ds.records(v01)) {
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

BOOST_FIXTURE_TEST_CASE(test_subset, fixture) {
  // contiguous range
  slice_view<vector_dataset<> > ds1 = ds.subset(0, 500);
  BOOST_CHECK_EQUAL(ds.record(48), ds1.record(48));
  BOOST_CHECK_EQUAL(ds.record(99), ds1.record(99, v));
  factor_mle<moment_gaussian> estim1(&ds1);
  moment_gaussian mle1 = estim1(v);
  double kl1 = f.relative_entropy(mle1);
  std::cout << "Single slice: " << kl1 << std::endl;
  BOOST_CHECK_SMALL(kl1, 1e-2);

  // multiple slices
  std::vector<slice> slices;
  slices.push_back(slice(900, 1000));
  slices.push_back(slice(200, 600));
  slice_view<vector_dataset<> > ds2 = ds.subset(slices);
  BOOST_CHECK_EQUAL(ds.record(940), ds2.record(40, v));
  BOOST_CHECK_EQUAL(ds.record(250), ds2.record(150));
  factor_mle<moment_gaussian> estim2(&ds2);
  moment_gaussian mle2 = estim2(v);
  double kl2 = f.relative_entropy(mle2);
  std::cout << "Two slices: " << kl2 << std::endl;
  BOOST_CHECK_SMALL(kl1, 1e-2);

  // verify the number of records
  size_t n_mutable = 0;
  foreach(const vector_record<>& r, ds2.records(v)) {
    ++n_mutable;
  }
  BOOST_CHECK_EQUAL(n_mutable, 500);

  const vector_dataset<>& dsc = ds2;
  size_t n_const = 0;
  foreach(const vector_record<>& r, dsc.records(v)) {
    ++n_const;
  }
  BOOST_CHECK_EQUAL(n_const, 500);
}

BOOST_FIXTURE_TEST_CASE(test_sample, fixture) {
  // draw samples from the dataset and attempt to recover the mean
  boost::mt19937 rng;
  vec mean = arma::zeros(3);
  for (size_t i = 0; i < 600; ++i) {
    vector_record<> r = ds.sample(v, rng);
    mean += r.values * r.weight;
    //std::cout << r.weight << ": " << r.values.t() << std::endl;
  }
  mean /= 600;
  double diff = norm(mean - f.mean(), 2);
  std::cout << "Samples: " << diff << std::endl;
  BOOST_CHECK_SMALL(diff, 0.2);
}

// BOOST_FIXTURE_TEST_CASE(test_shuffle, fixture) {
//   factor_mle<moment_gaussian> estim(&ds);
//   moment_gaussian mle1 = estim(v);
//   ds.shuffle(rng);
//   moment_gaussian mle2 = estim(v);
//   double kl = mle1.relative_entropy(mle2);
//   std::cout << "Shuffle: " << kl << std::endl;
//   BOOST_CHECK_SMALL(kl, 1e-10);
// }

BOOST_AUTO_TEST_CASE(test_load) {
  int argc = boost::unit_test::framework::master_test_suite().argc;
  BOOST_REQUIRE(argc > 1);
  std::string dir = boost::unit_test::framework::master_test_suite().argv[1];
  
  universe u;
  symbolic_format format;
  vector_memory_dataset<> ds;
  format.load_config(dir + "/vector_format.cfg", u);
  load(dir + "/vector_data.txt", format, ds);

  double values[][3] = { {180, 0, 0}, {178.2, 1, 0}, {150.4, 2, 2} };
  double weights[] = {1.0, 2.0, 0.5};
  BOOST_CHECK_EQUAL(ds.size(), 3);
  size_t i = 0;
  foreach(const vector_record<>& r, ds.records(format.vector_vars())) {
    BOOST_CHECK_CLOSE(r.values[0], values[i][0], 1e-10);
    BOOST_CHECK_CLOSE(r.values[1], values[i][1], 1e-10);
    BOOST_CHECK_CLOSE(r.values[2], values[i][2], 1e-10);
    BOOST_CHECK_EQUAL(r.weight, weights[i]);
    ++i;
  }

  save("vector_data.tmp", format, ds);
}
