#define BOOST_TEST_MODULE finite_dataset
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/random/random_table_factor_functor.hpp>
#include <sill/learning/dataset2/finite_dataset.hpp>
#include <sill/learning/parameter/table_factor_mle.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

template class finite_dataset<uint32_t>;
template class finite_dataset<uint8_t>;
template class basic_record_iterator<finite_dataset<> >;
template class basic_const_record_iterator<finite_dataset<> >;
template class basic_sample_iterator<finite_dataset<> >;

BOOST_TEST_DONT_PRINT_LOG_VALUE(std::vector<size_t>);

BOOST_AUTO_TEST_CASE(test_insert) {
  universe u;
  finite_var_vector v = u.new_finite_variables(3, 3);
  
  finite_dataset<> ds;
  ds.initialize(v);
  
  // insert a record
  finite_record2 r(3);
  r.values[0] = 2;
  r.values[1] = 0;
  r.values[2] = 1;
  r.weight = 0.5;
  ds.insert(r);

  // insert a finite assignment
  finite_assignment a;
  a[v[0]] = 1;
  a[v[1]] = 2;
  a[v[2]] = 0;
  ds.insert(a, 0.7);

  // insert a bunch of empty records
  ds.insert(10);

  // basic checks
  BOOST_CHECK_EQUAL(ds.size(), 12);
  finite_dataset<>::const_record_iterator it, end;
  boost::tie(it, end) = ds.records(v);

  // check if the first record is correct
  size_t first[] = {2, 0, 1};
  BOOST_CHECK_EQUAL(it->values, std::vector<size_t>(first, first+3));
  BOOST_CHECK_EQUAL(it->weight, 0.5);
  ++it;

  // check if the second record is correct
  size_t second[] = {1, 2, 0};
  BOOST_CHECK_EQUAL(it->values, std::vector<size_t>(second, second+3));
  BOOST_CHECK_EQUAL(it->weight, 0.7);
  ++it;

  // check the remaining records
  size_t rest[] = {3, 3, 3};
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->values, std::vector<size_t>(rest, rest+3));
    BOOST_CHECK_EQUAL(it->weight, 1.0);
    ++it;
  }
  
  // check that we covered all the records
  BOOST_CHECK(it == end);
}

BOOST_AUTO_TEST_CASE(test_iterator_comparisons) {
  universe u;
  finite_var_vector v = u.new_finite_variables(3, 3);
  
  finite_dataset<> ds;
  ds.initialize(v);
  ds.insert(finite_record2(3));

  finite_dataset<>::record_iterator it1, end1;
  boost::tie(it1, end1) = ds.records(v);

  finite_dataset<>::const_record_iterator it2, end2;
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
    : v(u.new_finite_variables(3, 2)),
      f(random_table_factor_functor(0).generate_marginal(make_domain(v))) {
    ds.initialize(v);
    f.normalize();
    for (size_t i = 0; i < 1000; ++i) {
      ds.insert(f.sample(rng));
    }
  }
  universe u;
  finite_var_vector v;
  finite_dataset<> ds;
  table_factor f;
  boost::mt19937 rng;
};

BOOST_FIXTURE_TEST_CASE(test_records, fixture) {
  // verify that the distribution retrieved by immutable iterators
  // matches the factor for every variable or every pair of variables
  table_factor_mle<> estim(&ds);
  for (size_t i = 0; i < v.size(); ++i) {
    for (size_t j = i; j < v.size(); ++j) {
      finite_domain dom = make_domain(v[i], v[j]);
      table_factor mle = estim(dom);
      double kl = f.marginal(dom).relative_entropy(mle);
      std::cout << dom << ": " << kl << std::endl;
      BOOST_CHECK_SMALL(kl, 1e-2);
    }
  }

  // fill the content of the dataset using mutable iteration
  finite_var_vector v01 = make_vector(v[0], v[1]);
  foreach(finite_record2& r, ds.records(v01)) {
    r.values[0] = 1;
    r.values[1] = 0;
  }

  // verify that we get the mutated version back
  foreach(const finite_record2& r, ds.records(v01)) {
    BOOST_CHECK_EQUAL(r.values.size(), 2);
    BOOST_CHECK_EQUAL(r.values[0], 1);
    BOOST_CHECK_EQUAL(r.values[1], 0);
  }

  // verify that the marginal over v[2] is still good
  finite_domain dom2 = make_domain(v[2]);
  table_factor mle = estim(dom2);
  double kl = f.marginal(dom2).relative_entropy(mle);
  std::cout << "Rest: " << kl << std::endl;
  BOOST_CHECK_SMALL(kl, 1e-2);
}

BOOST_FIXTURE_TEST_CASE(test_samples, fixture) {
  // draw samples from the dataset and attempt to recover f
  table_factor mle(v, 0.0);
  finite_dataset<>::sample_iterator it = ds.samples(v);
  for (size_t i = 0; i < 300; ++i, ++it) {
    mle.table()(it->values) += it->weight;
  }
  mle.normalize();
  double kl = f.relative_entropy(mle);
  std::cout << "Samples: " << kl << std::endl;
  BOOST_CHECK_SMALL(kl, 1e-2);
}

BOOST_FIXTURE_TEST_CASE(test_subset, fixture) {
  // test a range of the rows
  finite_dataset<> ds1 = ds.subset(0, 500);
  table_factor_mle<> estim1(&ds1);
  table_factor mle1 = estim1(v);
  double kl1 = f.relative_entropy(mle1);
  std::cout << "Range: " << kl1 << std::endl;
  BOOST_CHECK_SMALL(kl1, 1e-2);

  // test rows matching an assignment
  finite_assignment a;
  a[v[0]] = 1;
  a[v[1]] = 0;
  finite_dataset<> ds2 = ds.subset(a);
  table_factor_mle<> estim2(&ds2);
  table_factor mle2 = estim2(make_domain(v[2]));
  double kl2 = f.restrict(a).normalize().relative_entropy(mle2);
  std::cout << "Assignment: " << kl2 << std::endl;
  BOOST_CHECK_SMALL(kl2, 1e-2);
}

BOOST_FIXTURE_TEST_CASE(test_shuffle, fixture) {
  table_factor_mle<> estim(&ds);
  table_factor mle1 = estim(v);
  ds.shuffle(rng);
  table_factor mle2 = estim(v);
  double kl = mle1.relative_entropy(mle2);
  std::cout << "Shuffle: " << kl << std::endl;
  BOOST_CHECK_SMALL(kl, 1e-10);
}
