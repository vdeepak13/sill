#define BOOST_TEST_MODULE hybrid_dataset
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/random/uniform_factor_generator.hpp>
#include <sill/learning/dataset/hybrid_dataset_io.hpp>
#include <sill/learning/dataset/hybrid_memory_dataset.hpp>
#include <sill/learning/factor_mle/table_factor.hpp>

#include <boost/math/special_functions.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/macros_def.hpp>

using namespace sill;

template class hybrid_dataset<double>;
template class hybrid_memory_dataset<double>;
template class hybrid_record<double>;
template class slice_view<hybrid_dataset<double> >;

BOOST_TEST_DONT_PRINT_LOG_VALUE(std::vector<size_t>);

BOOST_AUTO_TEST_CASE(test_insert) {
  universe u;
  finite_var_vector fv = u.new_finite_variables(3, 3);
  vector_var_vector vv = u.new_vector_variables(1, 2);
  
  hybrid_memory_dataset<> ds;
  ds.initialize(fv, vv);
  
  // insert a record
  hybrid_record<> r(fv, vv);
  r.values.finite[0] = 2;
  r.values.finite[1] = 0;
  r.values.finite[2] = 1;
  r.values.vector[0] = 2.0;
  r.values.vector[1] = 0.0;
  r.weight = 0.5;
  ds.insert(r);

  // insert an assignment
  assignment a;
  a[fv[0]] = 1;
  a[fv[1]] = 2;
  a[fv[2]] = 0;
  a[vv[0]] = "0.0 0.5";
  ds.insert(a, 0.7);

  // insert a bunch of empty records
  ds.insert(10);

  // print the dataset
  std::cout << ds << std::endl;
  size_t i = 0;
  foreach (const hybrid_record<>& r, ds.records(fv, vv)) {
    std::cout << i << ": " << r << std::endl;
    ++i;
  }

  // basic checks
  BOOST_CHECK_EQUAL(ds.size(), 12);
  hybrid_dataset<>::const_record_iterator it, end;
  boost::tie(it, end) = ds.records(fv, vv);
  finite_dataset::const_record_iterator fit, fend;
  boost::tie(fit, fend) = ds.records(fv);
  vector_dataset<>::const_record_iterator vit, vend;
  boost::tie(vit, vend) = ds.records(vv);

  // check if the first record is correct
  size_t first[] = {2, 0, 1};
  BOOST_CHECK_EQUAL(it->values.finite, std::vector<size_t>(first, first+3));
  BOOST_CHECK_EQUAL(it->values.finite, fit->values);
  BOOST_CHECK(equal(it->values.vector, vec("2.0 0.0")));
  BOOST_CHECK(equal(it->values.vector, vit->values));
  BOOST_CHECK_EQUAL(it->weight, 0.5);
  BOOST_CHECK_EQUAL(fit->weight, 0.5);
  BOOST_CHECK_EQUAL(vit->weight, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.record(0));
  ++it;
  ++fit;
  ++vit;

  // check if the second record is correct
  size_t second[] = {1, 2, 0};
  BOOST_CHECK_EQUAL(it->values.finite, std::vector<size_t>(second, second+3));
  BOOST_CHECK_EQUAL(it->values.finite, fit->values);
  BOOST_CHECK(equal(it->values.vector, vec("0.0 0.5")));
  BOOST_CHECK(equal(it->values.vector, vit->values));
  BOOST_CHECK_EQUAL(it->weight, 0.7);
  BOOST_CHECK_EQUAL(fit->weight, 0.7);
  BOOST_CHECK_EQUAL(vit->weight, 0.7);
  BOOST_CHECK_EQUAL(*it, ds.record(1));
  ++it;
  ++fit;
  ++vit;

  // check the remaining records
  size_t rest[] = {3, 3, 3};
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->values.finite, std::vector<size_t>(rest, rest+3));
    BOOST_CHECK_EQUAL(it->weight, 1.0);
    BOOST_CHECK_EQUAL(fit->weight, 1.0);
    BOOST_CHECK_EQUAL(vit->weight, 1.0);
    for (size_t j = 0; j < 2; ++j) {
      BOOST_CHECK(boost::math::isnan(it->values.vector[j]));
      BOOST_CHECK(boost::math::isnan(vit->values[j]));
    }
    ++it;
    ++fit;
    ++vit;
  }
  
  // check that we covered all the records
  BOOST_CHECK(it == end);
  BOOST_CHECK(fit == fend);
  BOOST_CHECK(vit == vend);
}

BOOST_AUTO_TEST_CASE(test_iterator_comparisons) {
  universe u;
  finite_var_vector fv = u.new_finite_variables(3, 3);
  vector_var_vector vv = u.new_vector_variables(3, 2);
  var_vector v = concat(fv, vv);
  
  hybrid_memory_dataset<> ds;
  ds.initialize(v);
  ds.insert(hybrid_record<>(fv, vv));

  hybrid_dataset<>::record_iterator it1, end1;
  boost::tie(it1, end1) = ds.records(v);

  hybrid_dataset<>::const_record_iterator it2, end2;
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


BOOST_AUTO_TEST_CASE(test_sample) {
  universe u;
  finite_var_vector fv = u.new_finite_variables(1, 2);
  vector_var_vector vv = u.new_vector_variables(1, 1);

  hybrid_memory_dataset<> ds;
  ds.initialize(fv, vv);

  // insert two records, one (0, 1), and the other (1, -1)
  assignment a;
  a[fv[0]] = 0;
  a[vv[0]] = "1";
  ds.insert(a);
  a[fv[0]] = 1;
  a[vv[0]] = "-1";
  ds.insert(a);

  // check that the sample average is approximately (0.5, 0.0)
  size_t nsamples = 500;
  double fsum = 0.0;
  double vsum = 0.0;
  boost::mt19937 rng;
  for (size_t i = 0; i < nsamples; ++i) {
    hybrid_record<> r = ds.sample(fv, vv, rng);
    fsum += r.values.finite[0];
    vsum += r.values.vector[0];
  }

  BOOST_CHECK_SMALL(fsum / nsamples - 0.5, 0.05);
  BOOST_CHECK_SMALL(vsum / nsamples, 0.05);
}

// BOOST_FIXTURE_TEST_CASE(test_shuffle, fixture) {
//   factor_mle<table_factor> estim(&ds);
//   table_factor mle1 = estim(v);
//   ds.shuffle(rng);
//   table_factor mle2 = estim(v);
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
  hybrid_memory_dataset<> ds;
  format.load_config(dir + "/hybrid_format.cfg", u);
  load(dir + "/hybrid_data.txt", format, ds);

  size_t fvalues[][2] = { {0, 2}, {1, 3}, {0, 0} };
  double vvalues[][2] = { {33.0, 180.0}, {22.0, 178.0}, {11.0, 150.0} };
  double weights[] = {1.0, 2.0, 0.5};
  BOOST_CHECK_EQUAL(ds.size(), 3);
  size_t i = 0;
  foreach(const hybrid_record<>& r, ds.records(ds.arg_vector())) {
    BOOST_CHECK_EQUAL(r.values.finite[0], fvalues[i][0]);
    BOOST_CHECK_EQUAL(r.values.finite[1], fvalues[i][1]);
    BOOST_CHECK_CLOSE(r.values.vector[0], vvalues[i][0], 1e-10);
    BOOST_CHECK_CLOSE(r.values.vector[1], vvalues[i][1], 1e-10);
    BOOST_CHECK_EQUAL(r.weight, weights[i]);
    ++i;
  }

  save("hybrid_data.tmp", format, ds);
}
