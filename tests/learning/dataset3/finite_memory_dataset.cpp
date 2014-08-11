#define BOOST_TEST_MODULE finite_dataset
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/random/random_table_factor_functor.hpp>
#include <sill/learning/dataset3/finite_memory_dataset.hpp>
#include <sill/learning/mle/table_factor.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

template class raw_record_iterator<finite_dataset>;
template class raw_const_record_iterator<finite_dataset>;
template class slice_view<finite_dataset>;

BOOST_TEST_DONT_PRINT_LOG_VALUE(std::vector<size_t>);

BOOST_AUTO_TEST_CASE(test_insert) {
  universe u;
  finite_var_vector v = u.new_finite_variables(3, 3);
  
  finite_memory_dataset ds;
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

  // print the records
  size_t i = 0;
  foreach (const finite_record2& r, ds.records(v)) {
    std::cout << i << " " << r.values << " " << r.weight << std::endl;
    ++i;
  }

  // basic checks
  BOOST_CHECK_EQUAL(ds.size(), 12);
  finite_dataset::const_record_iterator it, end;
  boost::tie(it, end) = ds.records(v);

  // check if the first record is correct
  size_t first[] = {2, 0, 1};
  BOOST_CHECK_EQUAL(it->values, std::vector<size_t>(first, first+3));
  BOOST_CHECK_EQUAL(it->weight, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.record(0));
  ++it;

  // check if the second record is correct
  size_t second[] = {1, 2, 0};
  BOOST_CHECK_EQUAL(it->values, std::vector<size_t>(second, second+3));
  BOOST_CHECK_EQUAL(it->weight, 0.7);
  BOOST_CHECK_EQUAL(*it, ds.record(1));
  ++it;

  // check the remaining records
  size_t rest[] = {3, 3, 3};
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->values, std::vector<size_t>(rest, rest+3));
    BOOST_CHECK_EQUAL(it->weight, 1.0);
    BOOST_CHECK_EQUAL(*it, ds.record(i + 2));
    ++it;
  }
  
  // check that we covered all the records
  BOOST_CHECK(it == end);
}

BOOST_AUTO_TEST_CASE(test_iterator_comparisons) {
  universe u;
  finite_var_vector v = u.new_finite_variables(3, 3);
  
  finite_memory_dataset ds;
  ds.initialize(v);
  ds.insert(finite_record2(3));

  finite_dataset::record_iterator it1, end1;
  boost::tie(it1, end1) = ds.records(v);

  finite_dataset::const_record_iterator it2, end2;
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
    ds.initialize(v, 1000);
    f.normalize();
    for (size_t i = 0; i < 1000; ++i) {
      ds.insert(f.sample(rng));
    }
  }
  universe u;
  finite_var_vector v;
  finite_memory_dataset ds;
  table_factor f;
  boost::mt19937 rng;
};

BOOST_FIXTURE_TEST_CASE(test_records, fixture) {
  // verify that the distribution retrieved by immutable iterators
  // matches the factor for every variable or every pair of variables
  mle<table_factor> estim(&ds);
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

BOOST_FIXTURE_TEST_CASE(test_subset, fixture) {
  // contiguous range
  slice_view<finite_dataset> ds1 = ds.subset(0, 500);
  BOOST_CHECK_EQUAL(ds.record(48), ds1.record(48));
  BOOST_CHECK_EQUAL(ds.record(99), ds1.record(99, v));
  mle<table_factor> estim1(&ds1);
  table_factor mle1 = estim1(v);
  double kl1 = f.relative_entropy(mle1);
  std::cout << "Single slice: " << kl1 << std::endl;
  BOOST_CHECK_SMALL(kl1, 1e-2);

  // multiple slices
  std::vector<slice> slices;
  slices.push_back(slice(100, 200));
  slices.push_back(slice(300, 700));
  slice_view<finite_dataset> ds2 = ds.subset(slices);
  BOOST_CHECK_EQUAL(ds.record(140), ds2.record(40, v));
  BOOST_CHECK_EQUAL(ds.record(350), ds2.record(150));
  mle<table_factor> estim2(&ds2);
  table_factor mle2 = estim2(v);
  double kl2 = f.relative_entropy(mle2);
  std::cout << "Two slices: " << kl2 << std::endl;
  BOOST_CHECK_SMALL(kl2, 1e-2);

  // verify the number of records
  size_t n_mutable = 0;
  foreach(const finite_record2& r, ds2.records(v)) {
    ++n_mutable;
  }
  BOOST_CHECK_EQUAL(n_mutable, 500);

  const finite_dataset& dsc = ds2;
  size_t n_const = 0;
  foreach(const finite_record2& r, dsc.records(v)) {
    ++n_const;
  }
  BOOST_CHECK_EQUAL(n_const, 500);
}

BOOST_FIXTURE_TEST_CASE(test_sample, fixture) {
  // draw samples from the dataset and attempt to recover f
  boost::mt19937 rng;
  table_factor mle(v, 0.0);
  for (size_t i = 0; i < 500; ++i) {
    finite_record2 r = ds.sample(v, rng);
    mle.table()(r.values) += r.weight;
  }
  mle.normalize();
  double kl = f.relative_entropy(mle);
  std::cout << "Samples: " << kl << std::endl;
  BOOST_CHECK_SMALL(kl, 2e-2);
}

// BOOST_FIXTURE_TEST_CASE(test_shuffle, fixture) {
//   mle<table_factor> estim(&ds);
//   table_factor mle1 = estim(v);
//   ds.shuffle(rng);
//   table_factor mle2 = estim(v);
//   double kl = mle1.relative_entropy(mle2);
//   std::cout << "Shuffle: " << kl << std::endl;
//   BOOST_CHECK_SMALL(kl, 1e-10);
// }
