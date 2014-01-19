#include <sill/factor/approx/integration_points.hpp>

#include <sill/factor/moment_gaussian.hpp>
#include <sill/factor/nonlinear_gaussian.hpp>

namespace sill {

  moment_gaussian integration_points_approximator::
  operator()(const nonlinear_gaussian& ng, const moment_gaussian& prior) const {
    assert(prior.marginal() && prior.size() == ng.size_tail());
    std::size_t nh = ng.size_head(), nt = ng.size_tail();

    // compute the integration points (x)
    mat p;
    vec w;
    boost::tie(p, w) = points(nt);
    size_t np = w.size();
    
    vec eigval;
    mat eigvec;
    eig_sym(eigval, eigvec, prior.covariance(ng.tail_list()));
    mat a = eigvec * diagmat(sqrt(eigval));

    // Alternatively: _lower_ triangular Cholesky decomposition
    mat xc = a * p;
    mat x  = xc + repmat(prior.mean(ng.tail_list()), 1, np);

    // compute the function output (y) on the integration points
    // and some useful statistics
    mat y(nh, np);
    for(size_t i = 0; i < np; i++) {
      y.col(i) = ng.mean(x.col(i));
    }
    //vec meany = sum(elem_mult(y, repmat(w, nh, 1, true)), 1);
    vec meany = y * w;
    mat yc = y - repmat(meany, 1, np); // y centered
    mat ycwt = yc.t() % repmat(w, 1, nh);
      
    // compute the moments of the joint distribution
    span rx(0, nt-1);
    span ry(nt, nh+nt-1);
    vec mean = join_cols(prior.mean(ng.tail_list()), meany);
    mat cov(ng.size(), ng.size());
    cov(rx, rx) = prior.covariance(ng.tail_list());
    cov(rx, ry) = xc * ycwt;
    cov(ry, rx) = cov(rx, ry).t();
    cov(ry, ry) = yc * ycwt + ng.covariance();

    /*
      using namespace std;
      cout << "cov: " << prior.covariance(ng.tail_list()) << endl;
      cout << "a: " << a << endl;
      cout << "xc: " << xc << endl;
      cout << "x: " << x << endl;
      cout << "y: " << y << endl;
      cout << "yc: " << yc << endl;
      cout << "ycwt: " << ycwt << endl;
      cout << "mean: " << mean << endl;
      cout << "cov: " << cov << endl;
    */
      
    vector_var_vector args = concat(ng.tail_list(), ng.head_list());
    moment_gaussian joint(args, mean, cov, prior.norm_constant());
    return joint.restrict(ng.assignment());
  }
    
  std::pair<mat, vec> integration_points_approximator::points(int d) {
    assert(d >= 0);
    using std::sqrt;
    double u = sqrt(3.0);
    vec w0(1);         w0.fill(1+(d*d - 7*d)/18.0);
    vec w1(2*d);       w1.fill((4-d)/18.0);
    vec w2(2*(d-1)*d); w2.fill(1/36.0);
    mat p0 = zeros<mat>(d, 1);
    mat p1 = join_horiz(u*eye<mat>(d, d), -u*eye<mat>(d, d));
    mat p2 = zeros<mat>(d, 2*(d-1)*d);
    // fill out p2
    std::size_t k = 0;
    for(int i = 0; i < d-1; i++) {
      for(int j = i+1; j < d; j++) {
        p2(i, k+0) = +u;
        p2(j, k+0) = +u;
        p2(i, k+1) = +u;
        p2(j, k+1) = -u;
        p2(i, k+2) = -u;
        p2(j, k+2) = +u;
        p2(i, k+3) = -u;
        p2(j, k+3) = -u;
        k += 4;
      }
    }
    return std::make_pair(join_horiz(join_horiz(p0, p1), p2),
                          join_vert(join_vert(w0, w1), w2));
  }

} // namespace sill

