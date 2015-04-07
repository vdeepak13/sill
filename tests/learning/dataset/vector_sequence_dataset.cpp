#define BOOST_TEST_MODULE vector_sequence_dataset
#include <boost/test/unit_test.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/vector_sequence_dataset.hpp>
#include <sill/learning/dataset/vector_sequence_dataset_io.hpp>

#include <random>

namespace sill {
  template class basic_sequence_dataset<vector_sequence_traits<double> >;
  template class basic_sequence_dataset<vector_sequence_traits<float> >;
}

using namespace sill;

typedef domain<vector_discrete_process*> domain_type;
typedef dynamic_matrix<double> data_type;
typedef std::pair<data_type, double> sample_type;
typedef std::pair<vector_assignment<double>, double> sample_assignment_type;
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_type);
BOOST_TEST_DONT_PRINT_LOG_VALUE(sample_assignment_type);

struct fixture {
  domain_type p;
  data_type seq0, seq1;
  vector_sequence_dataset<> ds;

  fixture()
    : p(2), seq0(3, 2), seq1(3, 1) {
    p[0] = new vector_discrete_process("a", 1);
    p[1] = new vector_discrete_process("b", 2);
    seq0 << 0, 1, 1, 2, 2, 3;
    seq1 << 1, 2, 3;
    ds.initialize(p);
    BOOST_CHECK(ds.empty());
    ds.insert(seq0, 0.5);
    ds.insert(seq1, 1.0);
  }
};

BOOST_FIXTURE_TEST_CASE(test_insert, fixture) {
  ds.insert(10);
  
  // print the datset
  std::cout << ds << std::endl;
  
  // check the size of the dataset
  BOOST_CHECK_EQUAL(ds.size(), 12);
  BOOST_CHECK(!ds.empty());

  // direct iteration
  vector_sequence_dataset<>::const_iterator it = ds.begin();
  vector_sequence_dataset<>::const_iterator end = ds.end();
  BOOST_CHECK_EQUAL(it->first, seq0);
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds[0]);
  ++it;
  BOOST_CHECK_EQUAL(it->first, seq1);
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK_EQUAL(*it, ds[1]);
  ++it;
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->first.rows(), 3);
    BOOST_CHECK_EQUAL(it->first.cols(), 0);
    BOOST_CHECK_EQUAL(it->second, 1.0);
    ++it;
  }
  BOOST_CHECK(it == end);

  // indirect iteration
  // direct iteration
  domain_type p1 = {p[1]};
  const auto& cds = ds;
  std::tie(it, end) = cds(p1);
  BOOST_CHECK_EQUAL(it->first, seq0.block(1, 0, 2, 2));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds(0, p1));
  ++it;
  BOOST_CHECK_EQUAL(it->first, seq1.block(1, 0, 2, 1));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK_EQUAL(*it, ds(1, p1));
  ++it;
  for (size_t i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it->first.rows(), 2);
    BOOST_CHECK_EQUAL(it->first.cols(), 0);
    BOOST_CHECK_EQUAL(it->second, 1.0);
    ++it;
  }
  BOOST_CHECK(it == end);
}

BOOST_FIXTURE_TEST_CASE(test_value_iterators, fixture) {
  vector_sequence_dataset<>::iterator it1, end1;
  std::tie(it1, end1) = ds(p);

  vector_sequence_dataset<>::const_iterator it2 = ds.begin();
  vector_sequence_dataset<>::const_iterator end2 = ds.end();

  BOOST_CHECK(it1 == it2);
  BOOST_CHECK(it2 == it1);

  BOOST_CHECK(end1 == end2);
  BOOST_CHECK(end2 == end1);

  BOOST_CHECK(it1 != end1);
  BOOST_CHECK(it1 != end2);
  BOOST_CHECK(it2 != end1);
  BOOST_CHECK(it2 != end2);
  BOOST_CHECK(!it1.end());
  BOOST_CHECK(!it2.end());

  ++it1;
  ++it2;
  BOOST_CHECK(!it1.end());
  BOOST_CHECK(!it2.end());

  BOOST_CHECK(++it1 == end1);
  BOOST_CHECK(++it2 == end2);
  BOOST_CHECK(it1.end());
  BOOST_CHECK(it2.end());
}

BOOST_FIXTURE_TEST_CASE(test_asignment_iterators, fixture) {
  domain_type p01 = {p[0], p[1]};

  vector_sequence_dataset<>::assignment_iterator it, end;
  std::tie(it, end) = ds.assignments(p);
  
  // check the first sample
  BOOST_CHECK_EQUAL(it->first.size(), 4);
  BOOST_CHECK_EQUAL(it->first.at(p[0]->at(0)), seq0.block(0, 0, 1, 1));
  BOOST_CHECK_EQUAL(it->first.at(p[1]->at(0)), seq0.block(1, 0, 2, 1));
  BOOST_CHECK_EQUAL(it->first.at(p[0]->at(1)), seq0.block(0, 1, 1, 1));
  BOOST_CHECK_EQUAL(it->first.at(p[1]->at(1)), seq0.block(1, 1, 2, 1));
  BOOST_CHECK_EQUAL(it->second, 0.5);
  BOOST_CHECK_EQUAL(*it, ds.assignment(0));
  BOOST_CHECK_EQUAL(*it, ds.assignment(0, p));
  BOOST_CHECK(!it.end());
  ++it;

  // check the second sample
  BOOST_CHECK_EQUAL(it->first.size(), 2);
  BOOST_CHECK_EQUAL(it->first.at(p[0]->at(0)), seq1.block(0, 0, 1, 1));
  BOOST_CHECK_EQUAL(it->first.at(p[1]->at(0)), seq1.block(1, 0, 2, 1));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK_EQUAL(*it, ds.assignment(1));
  BOOST_CHECK_EQUAL(*it, ds.assignment(1, p));
  BOOST_CHECK(!it.end());
  ++it;

  // check if finished
  BOOST_CHECK(it == end);
  BOOST_CHECK(it.end());
}

BOOST_FIXTURE_TEST_CASE(test_mutation, fixture) {
  // replace the values for process 1 with all-zeros
  for (auto& s : ds({p[1]})) {
    s.first.row(0).fill(0.1);
    s.second = 0.3;
  }

  // check if we have stored the values
  seq0.row(1).fill(0.1);
  seq1.row(1).fill(0.1);
  BOOST_CHECK_EQUAL(ds[0].first, seq0);
  BOOST_CHECK_EQUAL(ds[0].second, 0.3);
  BOOST_CHECK_EQUAL(ds[1].first, seq1);
  BOOST_CHECK_EQUAL(ds[1].second, 0.3);
}

BOOST_FIXTURE_TEST_CASE(test_shuffle, fixture) {
  // repeatedly shuffle and check if both permutations have the same frequency
  std::mt19937 rng;
  int norig = 0;
  int nswap = 0;
  int nbad = 0;
  size_t nshuffles = 500;
  for (size_t i = 0; i < nshuffles; ++i) {
    ds.shuffle(rng);
    if (ds.size() == 2 && ds.arguments() == p) {
      if (ds[0].first.cols() == 2 && ds[0].first == seq0 &&
          ds[1].first.cols() == 1 && ds[1].first == seq1) {
        ++norig;
      } else if (ds[0].first.cols() == 1 && ds[0].first == seq1 &&
                 ds[1].first.cols() == 2 && ds[1].first == seq0) {
        ++nswap;
      } else {
        ++nbad;
      }
    } else {
      ++nbad;
    }
  }

  BOOST_CHECK_EQUAL(nbad, 0);
  BOOST_CHECK_SMALL(double(norig - nswap) / nshuffles, 0.05);  
}

BOOST_AUTO_TEST_CASE(test_load) {
  int argc = boost::unit_test::framework::master_test_suite().argc;
  BOOST_REQUIRE(argc > 1);
  std::string dir = boost::unit_test::framework::master_test_suite().argv[1];

  // load the data
  universe u;
  symbolic_format format;
  vector_sequence_dataset<> ds;
  format.load(dir + "/vector_seq.cfg", u);
  load({dir + "/vector_seq0.txt", dir + "/vector_seq1.txt"}, format, ds);

  // check the sequences
  data_type seq0(3, 3), seq1(3, 2);
  seq0 << 0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 1.0, 0.9, 0.8;
  seq1 << -0.1, -0.2, -0.2, -0.3, 0.8, 0.9;
  BOOST_CHECK_EQUAL(ds.size(), 2);
  BOOST_CHECK_EQUAL(ds[0].first, seq0);
  BOOST_CHECK_EQUAL(ds[0].second, 1.0);
  BOOST_CHECK_EQUAL(ds[1].first, seq1);
  BOOST_CHECK_EQUAL(ds[1].second, 1.0);

  save({"vector_seq0.tmp", "vector_seq1.tmp"}, format, ds);
}