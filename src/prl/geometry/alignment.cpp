#include <prl/geometry/alignment.hpp>

namespace prl {

  mat ralign(const mat& cov_xy) {
    mat u, v;
    vec d;
    bool result = svd(cov_xy, v, d, u);
    assert(result);

    int r = rank(d), m = cov_xy.size1();
    // cout << u << " " << d << " " << v << endl;
    // cout << r << endl;
    // cout << m << endl;
    // cout << det(cov_xy) << endl;
    // cout << prod(trans(v), v) << "," << prod(trans(u), u) << det(cov_xy) << endl;
    mat s = identity(m);
    if (r > m-1) {
      if (det(cov_xy)<0) s(m-1, m-1) = -1;
    }
    else if (r == m-1) {
      if (det(u) * det(v) < 0) s(m-1, m-1) = -1;
    }
    else {
      throw std::invalid_argument
        ("Insufficient rank in covariance to determine the rigid transform");
    }
    return u * s * trans(v);
  }

  std::pair<mat, vec> ralign(const forward_range<const vec&>& src,
                             const forward_range<const vec&>& dest) {
    vec mx   = mean(src), my = mean(dest);
    mat covm = cov(src, dest, false);
    mat rot  = ralign(covm);
    return std::make_pair(rot, my - rot * mx);
  }

} // namespace prl
