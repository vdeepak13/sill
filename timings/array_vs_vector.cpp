#define NDEBUG // disable boundary checking in ublas::storage
//#define BOOST_VARIANT_MINIMIZE_SIZE

#include <vector>
#include <boost/array.hpp>
#include <boost/progress.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/variant.hpp>
#include <boost/any.hpp>
#include <boost/shared_ptr.hpp>
// #include <prl/allocator.hpp>

const int k = 5;
int d[k];
typedef boost::numeric::ublas::bounded_array<int,k> ba;

/*
namespace boost {
  template <>
  struct has_nothrow_copy< ba > : mpl::true_ { };
}
*/

void shared_ptr_experiment(const int count)  {
  using namespace std;
  boost::timer t;
  int c=0;
  
  cout << c << endl;
}

int main(int argc, char** argv) 
{
  using namespace std;
  using namespace boost::numeric::ublas;
  const int count = 20000000;

  boost::mt19937 rng;

  for (int j = 0; j < k; j++) 
    d[j] = rng();

  boost::timer t;
  int c=0;

  for (int i = 0; i < count * 10; i++) {
    boost::array<int, k> a;
    for (int j = 0; j < k; j++) {
      a[j] = d[j];
      c+=c+j;
    }
  }
  cout << "boost::array time = " << t.elapsed()/10 << " secs." << endl;

  t.restart();
  for (int i = 0; i < count; i++) {
    std::vector<int> a(k);
    // static vector<int> a(k); // eliminates almost all overhead
    for (int j = 0; j < k; j++)
      a[j] = d[j];
  }
  cout << "vector time = " << t.elapsed() << " secs." << endl;
  
//   t.restart();
//   for (int i = 0; i < count; i++) {
//     std::vector<int, prl::pre_allocator<int, k> > a(k);
//     for (int j = 0; j < k; j++) {
//       a[j] = d[j];
//       c+=c+j;
//     }
//   }
//   cout << "vector (using pre-allocator) time = " << t.elapsed() 
//        << " secs." << endl;

  t.restart();
  for (int i = 0; i < count; i++) {
    unbounded_array<int> a(k);
    // static vector<int> a(k); // eliminates almost all overhead
    for (int j = 0; j < k; j++) {
      a[j] = d[j];
      c+=c+j;
    }
  }
  cout << "unbonded array time = " << t.elapsed() << " secs." << endl;

//   t.restart();
//   for (int i = 0; i < count; i++) {
//     unbounded_array<int, prl::pre_allocator<int, k> > a(k);
//     for (int j = 0; j < k; j++) {
//       a[j] = d[j];
//       c+=c+j;
//     }
//   }
//   cout << "unbounded array (using pre-allocator) time = " << t.elapsed() 
//        << " secs." << endl;

  t.restart();
  for (int i = 0; i < count; i++) {
    bounded_array<int, k> a(k);
    for (int j = 0; j < k; j++) {
      a[j] = d[j];
      c+=c+j;
    }
  }
  cout << "bounded array time = " << t.elapsed() << " secs." << endl;

  t.restart();
  typedef boost::variant< ba > variant_array; 
  for (int i = 0; i < count; i++) {
    variant_array va = ba(k);
    ba& a = boost::get<ba>(va);
    for (int j = 0; j < k; j++) {
      a[j] = d[j];
      c+=c+j;
    }
  }
  cout << "variant<bounded array> time = " << t.elapsed() << " secs." << endl;

  t.restart();
  typedef boost::any any_array;
  for (int i = 0; i < count; i++) {
    any_array aa = ba(k);
    ba& a = boost::any_cast<ba&>(aa);
    for (int j = 0; j < k; j++) {
      a[j] = d[j];
      c+=c+j;
    }
  }
  cout << "any<bounded array> time = " << t.elapsed() << " secs." << endl;
  
  typedef boost::shared_ptr< ba > shared_array;
  t.restart();
  for (int i = 0; i < count; i++) {
    shared_array sa(new ba(k));
    ba& a = *sa;
    for (int j = 0; j < k; j++) {
      a[j] = d[j];
      c+=c+j;
    }
  }
  cout << "shared_ptr<bounded array> time = " << t.elapsed() << " secs." << endl;


  cout << c << endl;
  return EXIT_SUCCESS;
}
