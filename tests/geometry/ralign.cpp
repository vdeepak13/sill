#include <prl/global.hpp>
#include <prl/range/numeric.hpp>
#include <prl/range/algorithm.hpp>
#include <prl/math/bindings/lapack.hpp>
//#include <prl/math/bindings/wm4.hpp>

#include <pstade/oven/taken.hpp>

#include <prl/stl_io.hpp>
#include <prl/geometry/alignment.hpp>

#include <vector>

int main()
{
  using namespace prl;
  using namespace std;
  
  // linear algebra kernel
  typedef prl::math::bindings::lapack_kernel<double> kernel;
  // typedef prl::math::bindings::wm4_kernel<double,2> kernel;
  
  typedef kernel::vector vector_type;
  typedef kernel::symmetric_matrix matrix_type;
      
  // a small 2D dataset
  double pts[] = {0.0,0.0, 1.0,0.0, 1.0,1.0, 0.0,1.0, 0.0,0.0};
  matrix_type m(2,2);
  m(0,0)=1.0;m(1,0)=2.0;m(0,1)=-0.5;m(1,1)=4.0;
  cout << det(m) << endl;
  
  std::vector<vector_type> x, y;
  for(int i=0; i<4; i++) {
    vector_type v(2); 
    v[0] = pts[2*i]; 
    v[1] = pts[2*i+1];
    x.push_back(v);
  }
  append(y, x | prl::dropped(1));
  append(y, x | prl::taken(1));

  cout << "X: " << x << endl;
  cout << "Y: " << y << endl;

  cout << "sum(x) = " << sum(x) << endl;
  cout << "mean(x) = " << mean(x) << endl;

  matrix_type cov_xy = cov<matrix_type>(x,y,false);
  cout << "cov(x,false) = " << cov<matrix_type>(x,false) << endl;
  cout << "cov(x,y,false) = " << cov_xy << endl;
  cout << "det(covxy) = " << det(cov_xy) << endl;

  cout << "ralign(x,y,false) = " 
       << geometry::ralign<kernel>(cov_xy) << endl;
}
