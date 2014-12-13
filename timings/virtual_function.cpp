
#include <iostream>
#include <vector>
#include <sill/range/concepts.hpp>
#include <sill/copy_ptr.hpp>
#include <boost/timer.hpp>

#include <sill/macros_def.hpp>

/**
 * \file virtual_functions_timing.cpp
 * Test of virtual function calls for deciding how often to use them in SILL,
 * such as in views of datasets.
 *
 * Types of views:
 * - select segment of data
 * - select sets of indices
 * - select sets of variables
 * Possible ways of doing views:
 * - (if virtual functions are expensive)
 *   - base dataset class supports views, so no virtual function calls
 *   - probably want the base dataset class to contain a shared pointer
 *     to a class holding the actual data so data may be shared between views
 * - (if virtual functions are cheap)
 *   - base dataset class with common functionality such as iterating over
 *     records (which will be virtual calls)
 *   - actual datasets and views inherited from it
 */

using namespace sill;

/**
 * - (if virtual functions are expensive)
 *   - base dataset class supports views, so no virtual function calls
 *   - probably want the base dataset class to contain a shared pointer
 *     to a class holding the actual data so data may be shared between views
 *   - Though we could allow views to be reset with this implementation, it
 *     will be safest not to allow it.
 *   - For now, this will not allow construction of views from other views;
 *     however, it would in practice by handling the ranges/sets appropriately.
 */
class dataset1 {
private:
  std::size_t nrecords;
  boost::shared_ptr<std::vector<double> > data_ptr;
  //! No view, range of indices (efficient), set of indices (less efficient).
  enum view_type {NOVIEW, RANGE, SET};
  view_type view;
  //! Range of indices
  std::size_t range_begin;
  std::size_t range_end;
  //! Set of indices
  std::vector<std::size_t> index_vector;
public:
  std::size_t size() const { return nrecords; }
  dataset1(std::size_t n) : nrecords(n), view(NOVIEW) {
    // Create a dataset with n records.
    data_ptr = boost::shared_ptr<std::vector<double> >(new std::vector<double>(n));
    for (std::size_t i = 0; i < n; i++)
      data_ptr->operator[](i) = i;
  }
  //! Create a dataset from datapoints [begin, end)
  dataset1(const dataset1& ds, std::size_t begin, std::size_t end) {
    assert(begin < end);
    assert(end <= ds.nrecords);
    assert(ds.view == NOVIEW);
    nrecords = end - begin;
    data_ptr = ds.data_ptr;
    view = RANGE;
    range_begin = begin;
    range_end = end;
  }
  //! Create a dataset from an ordered range of indices.
  template <typename Range>
  dataset1(const dataset1& ds, const Range& indices) {
    concept_assert((boost::Convertible<typename boost::range_value<Range>::type,
                                       std::size_t>));
    assert(ds.view == NOVIEW);
    foreach(std::size_t i, indices) {
      assert(i <= ds.nrecords);
      index_vector.push_back(i);
    }
    nrecords = index_vector.size();
    data_ptr = ds.data_ptr;
    view = SET;
  }
  //! element access
  double operator[](std::size_t i) const {
    assert(i < nrecords);
    switch(view) {
    case NOVIEW:
      return data_ptr->operator[](i);
    case RANGE:
      return data_ptr->operator[](range_begin + i);
    case SET:
      return data_ptr->operator[](index_vector[i]);
    default:
      assert(false);
    }
  }
};

/**
 * - (if virtual functions are cheap)
 *   - base dataset class with common functionality such as iterating over
 *     records (which will be virtual calls)
 *   - actual datasets and views inherited from it
 */
class dataset2base {
public:
  std::size_t nrecords;
  std::size_t size() const { return nrecords; }
  //! element access
  virtual double operator[](std::size_t i) const = 0;
  virtual ~dataset2base() { }
};

class dataset2dataset : public dataset2base {
protected:
  std::vector<double> data;
public:
  dataset2dataset(std::size_t n) {
    nrecords = n;
    // Create a dataset with n records.
    data.reserve(n);
    for (std::size_t i = 0; i < n; i++)
      data[i] = i;
  }
  //! element access
  double operator[](std::size_t i) const {
    assert(i < nrecords);
    return data[i];
  }
};

class dataset2view : public dataset2base {
protected:
  const dataset2dataset& data;
  //! No view, range of indices (efficient), set of indices (less efficient).
  enum view_type {NOVIEW, RANGE, SET};
  view_type view;
  //! Range of indices
  std::size_t range_begin;
  std::size_t range_end;
  //! Set of indices
  std::vector<std::size_t> index_vector;
public:
  dataset2view(const dataset2dataset& data)
    : data(data), view(NOVIEW) {
    this->nrecords = data.nrecords;
  }
  dataset2view(const dataset2dataset& data, std::size_t begin, std::size_t end)
    : data(data) {
    assert(begin < end);
    assert(end <= data.nrecords);
    nrecords = end - begin;
    view = RANGE;
    range_begin = begin;
    range_end = end;
  }
  template <typename Range>
  dataset2view(const dataset2dataset& data, const Range& indices) : data(data) {
    concept_assert((boost::Convertible<typename boost::range_value<Range>::type,
                                       std::size_t>));
    foreach(std::size_t i, indices) {
      assert(i <= data.nrecords);
      index_vector.push_back(i);
    }
    nrecords = index_vector.size();
    view = SET;
  }
  // Note: I would also make constructors for making views from views.
  //! element access
  double operator[](std::size_t i) const {
    assert(i < nrecords);
    switch(view) {
    case NOVIEW:
      return data[i];
    case RANGE:
      return data[range_begin + i];
    case SET:
      return data[index_vector[i]];
    default:
      assert(false);
    }
  }
};

/**
 * Copying instead of using actual views.
 */
class dataset3 {
private:
  std::size_t nrecords;
  std::vector<double> data;
public:
  std::size_t size() const { return nrecords; }
  dataset3(std::size_t n) : nrecords(n) {
    // Create a dataset with n records.
    data.reserve(n);
    for (std::size_t i = 0; i < n; i++)
      data[i] = i;
  }
  //! Create a dataset from datapoints [begin, end)
  dataset3(const dataset3& ds, std::size_t begin, std::size_t end) {
    assert(begin < end);
    assert(end <= ds.nrecords);
    nrecords = end - begin;
    data.reserve(nrecords);
    for (size_t i = 0; i < nrecords; i++)
      data[i] = ds[begin + i];
  }
  //! Create a dataset from an ordered range of indices.
  template <typename Range>
  dataset3(const dataset3& ds, const Range& indices) {
    concept_assert((boost::Convertible<typename boost::range_value<Range>::type,
                                       std::size_t>));
    nrecords = 0;
    foreach(std::size_t i, indices) {
      assert(i <= ds.nrecords);
      nrecords++;
    }
    data.reserve(nrecords);
    std::size_t j = 0;
    foreach(std::size_t i, indices) {
      data[j] = ds[i];
      j++;
    }
  }
  //! element access
  double operator[](std::size_t i) const {
    assert(i < nrecords);
    return data[i];
  }
};

/**
 * Test 3 types of datasets:
 *  - Views without virtual functions (all in one class)
 *  - Views with virtual functions
 *  - No views (just copying)
 *
 * Test each type of dataset as follows (averaging each test over many runs):
 *  - Create a small range view (50 examples)
 *  - Create a large range view (10000 examples)
 *  - Create a small set view (50 examples)
 *  - Create a large set view (10000 examples)
 *  - Given a view, iterate through the records a bunch.
 */
int main(int argc, char** argv) {

  std::size_t nrecords = 15000;
  std::size_t nruns = 500000;
  dataset1 d1(nrecords);
  dataset2dataset d2(nrecords);
  dataset3 d3(nrecords);

  std::size_t tmp = 0;

  std::cout << "\nDataset type:\t1\t2\t3\n\nConstructing views:\n";

  // ranges
  std::vector<std::size_t> ranges;
  ranges.push_back(50);
  ranges.push_back(10000);

  boost::timer t;
  // *  - Create range views
  foreach(std::size_t r, ranges) {
    std::cout << "range (" << r << ")\t";
    t.restart();
    for (std::size_t i = 0; i < nruns; i++) {
      dataset1 d1t(d1, 0, r);
      tmp += d1t.size();
    }
    std::cout << t.elapsed() / nruns << "\t";
    t.restart();
    for (std::size_t i = 0; i < nruns; i++) {
      dataset2view d2t(d2, 0, r);
      tmp += d2t.size();
    }
    std::cout << t.elapsed() / nruns << "\t";
    t.restart();
    for (std::size_t i = 0; i < nruns / 100; i++) {
      dataset3 d3t(d3, 0, r);
      tmp += d3t.size();
    }
    std::cout << t.elapsed() / (nruns / 100) << std::endl;
  }

  // *  - Create set views
  foreach(std::size_t r, ranges) {
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < r; i++)
      indices.push_back(i);
    std::cout << "set (" << r << ")\t";
    t.restart();
    for (std::size_t i = 0; i < nruns / 100; i++) {
      dataset1 d1t(d1, indices);
      tmp += d1t.size();
    }
    std::cout << t.elapsed() / (nruns / 100) << "\t";
    t.restart();
    for (std::size_t i = 0; i < nruns / 100; i++) {
      dataset2view d2t(d2, indices);
      tmp += d2t.size();
    }
    std::cout << t.elapsed() / (nruns / 100) << "\t";
    t.restart();
    for (std::size_t i = 0; i < nruns / 100; i++) {
      dataset3 d3t(d3, indices);
      tmp += d3t.size();
    }
    std::cout << t.elapsed() / (nruns / 100) << std::endl;
  }

  std::cout << "\nIterating over records:\n";

  // *  - Given a range view, iterate through the records a bunch.
  std::cout << "range it\t";
  dataset1 d1t(d1, 0, nrecords / 10);
  t.restart();
  for (std::size_t i = 0; i < nruns / 10; i++) {
    for (std::size_t j = 0; j < d1t.size(); j++)
      tmp += (std::size_t)(d1t[j]);
  }
  std::cout << t.elapsed() / (nruns * d1t.size() / 10) << "\t";
  dataset2base& d2t = *(new dataset2view(d2, 0, nrecords / 10));
  t.restart();
  for (std::size_t i = 0; i < nruns / 10; i++) {
    for (std::size_t j = 0; j < d2t.size(); j++)
      tmp += (std::size_t)(d2t[j]);
  }
  std::cout << t.elapsed() / (nruns * d2t.size() / 10) << "\t";
  dataset3 d3t(d3, 0, nrecords / 10);
  t.restart();
  for (std::size_t i = 0; i < nruns / 10; i++) {
    for (std::size_t j = 0; j < d3t.size(); j++)
      tmp += (std::size_t)(d3t[j]);
  }
  std::cout << t.elapsed() / (nruns * d3t.size() / 10) << std::endl;

  // *  - Given a set view, iterate through the records a bunch.
  std::vector<std::size_t> indices;
  for (std::size_t i = 0; i < nrecords / 10; i++)
    indices.push_back(i);
  std::cout << "set it\t\t";
  dataset1 d1s(d1, indices);
  t.restart();
  for (std::size_t i = 0; i < nruns / 10; i++) {
    for (std::size_t j = 0; j < d1s.size(); j++)
      tmp += (std::size_t)(d1s[j]);
  }
  std::cout << t.elapsed() / (nruns * d1s.size() / 10) << "\t";
  dataset2base& d2s = *(new dataset2view(d2, indices));
  t.restart();
  for (std::size_t i = 0; i < nruns / 10; i++) {
    for (std::size_t j = 0; j < d2s.size(); j++)
      tmp += (std::size_t)(d2s[j]);
  }
  std::cout << t.elapsed() / (nruns * d2s.size() / 10) << "\t";
  dataset3 d3s(d3, indices);
  t.restart();
  for (std::size_t i = 0; i < nruns / 10; i++) {
    for (std::size_t j = 0; j < d3s.size(); j++)
      tmp += (std::size_t)(d3s[j]);
  }
  std::cout << t.elapsed() / (nruns * d3s.size() / 10) << std::endl << std::endl;

}
