#define BOOST_TEST_MODULE fixed_view
#include <boost/test/unit_test.hpp>

#include <sill/learning/dataset/fixed_view.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/finite_sequence_dataset.hpp>
#include <sill/learning/dataset/vector_sequence_dataset.hpp>
#include <sill/learning/dataset/slice_view.hpp>

#include "../../math/eigen/helpers.hpp"

namespace sill {
  template class fixed_view<finite_sequence_dataset<> >;
  template class fixed_view<vector_sequence_dataset<> >;
  template class fixed_view<slice_view<finite_sequence_dataset<> > >;
  template class fixed_view<slice_view<vector_sequence_dataset<> > >;
}

using namespace sill;

BOOST_TEST_DONT_PRINT_LOG_VALUE(finite_index);
BOOST_TEST_DONT_PRINT_LOG_VALUE(finite_assignment);
BOOST_TEST_DONT_PRINT_LOG_VALUE(vector_assignment<>);

// Finite fixed views
//============================================================================

struct finite_fixture {
  finite_discrete_process* a;
  finite_discrete_process* b;
  finite_variable* a0;
  finite_variable* a1;
  finite_variable* b0;
  finite_sequence_dataset<> ds;
  typedef fixed_view<finite_sequence_dataset<> > view_type;
  typedef domain<finite_variable*> domain_type;

  finite_fixture() {
    a = new finite_discrete_process("a", 5);
    b = new finite_discrete_process("b", 5);
    a0 = a->at(0);
    a1 = a->at(1);
    b0 = b->at(0);

    ds.initialize({a, b});

    dynamic_matrix<size_t> values;
    values.resize(2, 3);
    values << 0, 1, 2, 3, 4, 1;
    ds.insert(values, 1.0);

    values.resize(2, 1);
    values << 3, 2;
    ds.insert(values, 2.0);

    values.resize(2, 2);
    values << 2, 3, 1, 0;
    ds.insert(values, 3.0);
  
    values.resize(2, 0);
    ds.insert(values, 0.5);
  }
};

BOOST_FIXTURE_TEST_CASE(test_accessors, finite_fixture) {
  // view over a single step
  view_type view1(&ds, 1, 1);
  domain_type args = { a->at(0), b->at(0) };
  BOOST_CHECK_EQUAL(view1.arguments(), args);
  BOOST_CHECK_EQUAL(view1.arity(), 2);
  BOOST_CHECK_EQUAL(view1.size(), 2);
  BOOST_CHECK(!view1.empty());
  std::cout << view1 << std::endl;

  // view over two steps
  view_type view2(&ds, 1, 2);
  args.insert(args.end(), { a->at(1), b->at(1) });
  BOOST_CHECK_EQUAL(view2.arguments(), args);
  BOOST_CHECK_EQUAL(view2.arity(), 4);
  BOOST_CHECK_EQUAL(view2.size(), 1);
  BOOST_CHECK(!view2.empty());

  // view over three steps
  view_type view3(&ds, 1, 3);
  args.insert(args.end(), { a->at(2), b->at(2) });
  BOOST_CHECK_EQUAL(view3.arguments(), args);
  BOOST_CHECK_EQUAL(view3.arity(), 6);
  BOOST_CHECK_EQUAL(view3.size(), 0);
  BOOST_CHECK(view3.empty());
}

BOOST_FIXTURE_TEST_CASE(test_const_iterator, finite_fixture) {
  // view over a single step, iterate over all variables
  view_type view1(&ds, 1, 1);
  view_type::const_iterator it = view1.begin();
  view_type::const_iterator end = view1.end();

  BOOST_CHECK_EQUAL(it->first, finite_index({1, 4}));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK(!it.end());
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK_EQUAL(it->first, finite_index({3, 0}));
  BOOST_CHECK_EQUAL(it->second, 3.0);
  BOOST_CHECK(!it.end());
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK(it.end());
  BOOST_CHECK(it == end);

  // view over two steps, iterate over a subset of variables
  view_type view2(&ds, 1, 2);
  std::tie(it, end) = view2({ a1, a0, b0 });

  BOOST_CHECK_EQUAL(it->first, finite_index({2, 1, 4}));
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK(!it.end());
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK(it.end());
  BOOST_CHECK(it == end);
}

BOOST_FIXTURE_TEST_CASE(test_assignment_iterator, finite_fixture) {
  // view over a single step, iterate over all variables
  view_type view1(&ds, 1, 1);
  view_type::assignment_iterator it, end;
  std::tie(it, end) = view1.assignments();
  
  BOOST_CHECK_EQUAL(it->first.size(), 2);
  BOOST_CHECK_EQUAL(it->first.at(a0), 1);
  BOOST_CHECK_EQUAL(it->first.at(b0), 4);
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK(!it.end());
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK_EQUAL(it->first.size(), 2);
  BOOST_CHECK_EQUAL(it->first.at(a0), 3);
  BOOST_CHECK_EQUAL(it->first.at(b0), 0);
  BOOST_CHECK_EQUAL(it->second, 3.0);
  BOOST_CHECK(!it.end());
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK(it.end());
  BOOST_CHECK(it == end);

  // view over two steps, iterate over a subset of variables
  view_type view2(&ds, 1, 2);
  std::tie(it, end) = view2.assignments({ a1, a0, b0 });

  BOOST_CHECK_EQUAL(it->first.size(), 3);
  BOOST_CHECK_EQUAL(it->first.at(a1), 2);
  BOOST_CHECK_EQUAL(it->first.at(a0), 1);
  BOOST_CHECK_EQUAL(it->first.at(b0), 4);
  BOOST_CHECK_EQUAL(it->second, 1.0);
  BOOST_CHECK(!it.end());
  BOOST_CHECK(it != end);
  ++it;

  BOOST_CHECK(it.end());
  BOOST_CHECK(it == end);
}

BOOST_FIXTURE_TEST_CASE(test_finite_access, finite_fixture) {
  // view over a single step, access all variables
  view_type view1(&ds, 1, 1);
  BOOST_CHECK_EQUAL(view1[0].first, finite_index({1, 4}));
  BOOST_CHECK_EQUAL(view1[1].first, finite_index({3, 0}));
  BOOST_CHECK_EQUAL(view1[0].second, 1.0);
  BOOST_CHECK_EQUAL(view1[1].second, 3.0);
  BOOST_CHECK_EQUAL(view1.assignment(1).first,
                    finite_assignment({{a0, 3}, {b0, 0}}));
  BOOST_CHECK_EQUAL(view1.assignment(1).second, 3.0);

  // view over two steps, access a subset of variables
  view_type view2(&ds, 1, 2);
  domain_type dom = {a1, b0};
  BOOST_CHECK_EQUAL(view2(0, dom).first, finite_index({2, 4}));
  BOOST_CHECK_EQUAL(view2(0, dom).second, 1.0);
  BOOST_CHECK_EQUAL(view2.assignment(0, dom).first,
                    finite_assignment({{a1, 2}, {b0, 4}}));
  BOOST_CHECK_EQUAL(view2.assignment(0, dom).second, 1.0);
}

// Vector fixed views
//============================================================================

struct vector_fixture {
  vector_discrete_process* a;
  vector_discrete_process* b;
  vector_variable* a0;
  vector_variable* a1;
  vector_variable* b0;
  vector_sequence_dataset<> ds;
  typedef fixed_view<vector_sequence_dataset<> > view_type;
  typedef domain<vector_variable*> domain_type;

  vector_fixture() {
    a = new vector_discrete_process("a", 1);
    b = new vector_discrete_process("b", 2);
    a0 = a->at(0);
    a1 = a->at(1);
    b0 = b->at(0);

    ds.initialize({a, b});

    dynamic_matrix<double> values;
    values.resize(3, 3);
    values << 0, 1, 2, 3, 4, 1, 6, 7, 8;
    ds.insert(values, 1.0);

    values.resize(3, 1);
    values << 3, 2, 1;
    ds.insert(values, 2.0);

    values.resize(3, 2);
    values << 2, 3, 1, 0, 8, 9;
    ds.insert(values, 3.0);
  
    values.resize(3, 0);
    ds.insert(values, 0.5);
  }
};

BOOST_FIXTURE_TEST_CASE(test_vector_access, vector_fixture) {
  // view over a single step, access all variables
  view_type view1(&ds, 1, 1);
  BOOST_CHECK_EQUAL(view1[0].first, vec3(1, 4, 7));
  BOOST_CHECK_EQUAL(view1[1].first, vec3(3, 0, 9));
  BOOST_CHECK_EQUAL(view1[0].second, 1.0);
  BOOST_CHECK_EQUAL(view1[1].second, 3.0);
  BOOST_CHECK_EQUAL(view1.assignment(1).first,
                    vector_assignment<>({{a0, vec1(3)}, {b0, vec2(0, 9)}}));
  BOOST_CHECK_EQUAL(view1.assignment(1).second, 3.0);

  // view over two steps, access a subset of variables
  view_type view2(&ds, 1, 2);
  domain_type dom = {a1, b0};
  BOOST_CHECK_EQUAL(view2(0, dom).first, vec3(2, 4, 7));
  BOOST_CHECK_EQUAL(view2(0, dom).second, 1.0);
  BOOST_CHECK_EQUAL(view2.assignment(0, dom).first,
                    vector_assignment<>({{a1, vec1(2)}, {b0, vec2(4, 7)}}));
  BOOST_CHECK_EQUAL(view2.assignment(0, dom).second, 1.0);
}