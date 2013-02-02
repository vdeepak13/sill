
/**
 * Joseph B.:
 * This file has methods which I implemented around 2/24/2011
 * for moment_gaussian.  I decided they were too hastily implemented and
 * reverted to the old versions, but I wanted to keep them around for a while.
 */

/* // ORIGINAL RESTRICT

  moment_gaussian
  moment_gaussian::restrict(const vector_assignment& a) const {
    vector_var_vector x, y; // x = kept, y = restricted (in head_list)
    foreach(vector_variable* v, head_list) {
      if (a.count(v) == 0)
        x.push_back(v);
      else
        y.push_back(v);
    }
    vec new_cmean(cmean);
    if (marginal()) {
      if (y.size() == 0)
        return *this;
    } else {
      size_t ntail(0); // number of tail_list variables with values in 'a'
      foreach(vector_variable* v, tail_list)
        ntail += a.count(v);
      if (ntail + y.size() == 0) // Then we do not restrict anything.
        return *this;
      assert(ntail == tail_list.size()); // for now
      vec v_tail(sill::concat(values(a, tail_list)));
      new_cmean += coeff * v_tail;
      if (y.size() == 0)
        return moment_gaussian(x, new_cmean, cov, likelihood);
    }

    ivec ix(indices(x));
    ivec iy(indices(y));
    vec dy(sill::concat(values(a, y)));
    dy -= new_cmean(iy);
    mat invyy_covyx;
    bool result = ls_solve_chol(cov(iy,iy), cov(iy,ix), invyy_covyx);
    if (!result) {
//       using namespace std;
//       cerr << cov(iy, iy) << endl;
//       cerr << *this << endl;
//       cerr << iy << endl;
//       assert(false);
      throw invalid_operation
        ("Cholesky decomposition failed in moment_gaussian::collapse");
    }
    double logl = 0;
    logl -= 0.5 * inner_prod(dy, ls_solve_chol(cov(iy,iy), dy));
    logl -= 0.5 * (dy.size() * std::log(2*pi()) + logdet(cov(iy,iy)));
    if (x.size() == 0)
      return moment_gaussian
        (likelihood * logarithmic<double>(logl, log_tag()));
    else
      return moment_gaussian
        (x,
         new_cmean(ix) + invyy_covyx.transpose()*dy,
         cov(ix, ix) - cov(ix,iy) * invyy_covyx,
         likelihood * logarithmic<double>(logl, log_tag()));
  } // restrict(a)

 */

  moment_gaussian
  moment_gaussian::restrict(const vector_assignment& a) const {
    vector_var_vector x, y; // x = kept, y = restricted (in head_list)
    foreach(vector_variable* v, head_list) {
      if (a.count(v) == 0)
        x.push_back(v);
      else
        y.push_back(v);
    }
    vec new_cmean(cmean);
    vector_var_vector tail_minus_a_vars; // tail vars not in 'a'
    if (marginal()) {
      if (y.size() == 0)
        return *this;
    } else {
      size_t ntail(0); // number of tail_list variables with values in 'a'
      foreach(vector_variable* v, tail_list) {
        if (a.count(v))
          ++ntail;
      }
      if (ntail + y.size() == 0) // Then we do not restrict anything.
        return *this;
      if (ntail == tail_list.size()) {
        vec ta_vals(sill::concat(values(a, tail_list)));
        new_cmean += coeff * ta_vals;
        if (y.size() == 0)
          return moment_gaussian(x, new_cmean, cov, likelihood);
      } else {
        vector_var_vector tail_intersect_a_vars;
        foreach(vector_variable* v, tail_list) {
          if (a.count(v))
            tail_intersect_a_vars.push_back(v);
          else
            tail_minus_a_vars.push_back(v);
        }
        if (tail_intersect_a_vars.size() != 0) {
          ivec ta_ind; // indices of tail vars in 'a'
          this->indices(tail_intersect_a_vars, ta_ind, true);
          vec ta_vals(sill::concat(values(a, tail_intersect_a_vars)));
          new_cmean += coeff.columns(ta_ind) * ta_vals;
        }
        if (y.size() == 0)
          return moment_gaussian(x, new_cmean, cov, likelihood);
      }
    }

    ivec ix(indices(x));
    ivec iy(indices(y));
    vec dy(sill::concat(values(a, y)));
    dy -= new_cmean(iy);
    mat invyy_covyx;
    bool result = ls_solve_chol(cov(iy,iy), cov(iy,ix), invyy_covyx);
    if (!result) {
//       using namespace std;
//       cerr << cov(iy, iy) << endl;
//       cerr << *this << endl;
//       cerr << iy << endl;
//       assert(false);
      throw invalid_operation
        ("Cholesky decomposition failed in moment_gaussian::collapse");
    }
    double logl = 0;
    logl -= 0.5 * inner_prod(dy, ls_solve_chol(cov(iy,iy), dy));
    logl -= 0.5 * (dy.size() * std::log(2*pi()) + logdet(cov(iy,iy)));
    if (tail_minus_a_vars.size() == 0) {
      return moment_gaussian
        (x,
         new_cmean(ix) + invyy_covyx.transpose()*dy,
         cov(ix, ix) - cov(ix,iy) * invyy_covyx,
         likelihood * logarithmic<double>(logl, log_tag()));
    } else {
      ivec tail_minus_a_vars_inds;
      this->indices(tail_minus_a_vars, tail_minus_a_vars_inds, true);
      return moment_gaussian
        (x,
         new_cmean(ix) + invyy_covyx.transpose() * dy,
         cov(ix, ix) - cov(ix,iy) * invyy_covyx,
         tail_minus_a_vars,
         coeff.columns(tail_minus_a_vars_inds),
         likelihood * logarithmic<double>(logl, log_tag()));
    }
  } // restrict

  moment_gaussian moment_gaussian::conditional(const vector_domain& B) const {
    // TO DO: (Joseph) I recently changed this to support computing
    //        P(A|B,C) from P(A,B|C) (to support C).  Test it more thoroughly.
    foreach(vector_variable* v, tail_list) {
      if (B.count(v) != 0) {
        throw std::runtime_error
          (std::string("moment_gaussian::conditional()") +
           " given set B which intersected with tail list.");
      }
    }
    vector_var_vector new_head;
    vector_var_vector new_tail;
    foreach(vector_variable* v, head_list) {
      if (B.count(v) == 0)
        new_head.push_back(v);
      else
        new_tail.push_back(v);
    }
    if (new_tail.size() != B.size()) {
      throw std::runtime_error
        (std::string("moment_gaussian::conditional()") +
         " given set B with variables not in the factor");
    }
    ivec ia(indices(new_head));
    ivec ib(indices(new_tail));
    mat cov_ab_cov_b_inv;
    if (ib.size() != 0) {
      bool result = ls_solve_chol(cov(ib,ib), cov(ib,ia), cov_ab_cov_b_inv);
      if (!result) {
        throw invalid_operation
          ("Cholesky decomposition failed in moment_gaussian::conditional");
      }
      cov_ab_cov_b_inv = cov_ab_cov_b_inv.transpose();
    }
    if (tail_list.size() == 0) {
      return moment_gaussian(new_head,
                             cmean(ia) - cov_ab_cov_b_inv * cmean(ib),
                             cov(ia,ia) - cov_ab_cov_b_inv * cov(ib,ia),
                             new_tail, cov_ab_cov_b_inv, 1);
    } else {
      mat new_coeff;
      new_coeff.resize(new_head.size(), tail_list.size() + new_tail.size());
      new_coeff.set_submatrix(irange(0,new_head.size()),
                              irange(0,tail_list.size()),
                              coeff.rows(ia));
      if (ib.size() != 0) {
        new_coeff.set_submatrix(irange(0,new_head.size()),
                                irange(0,new_tail.size()),
                                cov_ab_cov_b_inv);
        return moment_gaussian(new_head,
                               cmean(ia) - cov_ab_cov_b_inv * cmean(ib),
                               cov(ia,ia) - cov_ab_cov_b_inv * cov(ib,ia),
                               sill::concat(tail_list, new_tail),
                               new_coeff, 1);
      } else {
        return moment_gaussian(new_head,
                               cmean(ia),
                               cov(ia,ia),
                               sill::concat(tail_list, new_tail),
                               new_coeff, 1);
      }
    }
  } // conditional

  moment_gaussian
  moment_gaussian::direct_combination(const moment_gaussian& x,
                                      const moment_gaussian& y) {
    assert(x.marginal());
    vector_domain args = set_union(x.arguments(), y.arguments());
    moment_gaussian result(args, x.likelihood * y.likelihood);
    ivec xh = result.indices(x.head_list);
    ivec yh = result.indices(y.head_list);
    if (includes(x.arguments(), make_domain(y.tail_list))) {
      ivec x_yt  = x.indices(y.tail_list);
      ivec x_all = x.indices(x.head_list);
      result.cmean.set_subvector(xh, x.cmean);
      result.cmean.set_subvector(yh, y.coeff * x.cmean(x_yt) + y.cmean);
      mat covyx = y.coeff * x.cov(x_yt, x_all);
      mat covyy = y.coeff * x.cov(x_yt, x_yt) * y.coeff.transpose() + y.cov;
      result.cov.set_submatrix(xh, xh, x.cov);
      result.cov.set_submatrix(yh, xh, covyx);
      result.cov.set_submatrix(xh, yh, covyx.transpose());
      result.cov.set_submatrix(yh, yh, covyy);
    } else {
      // TO DO: (Joseph): This was originally implemented s.t. it failed when
      //                  x did not include all of y.tail_list.
      //                  Test more thoroughly to make sure I fixed this issue.
      std::cerr
        << "WARNING: moment_gaussian::direct_combination new part called"
        << std::endl;
      ivec x_yt;
      x.indices(y.tail_list, x_yt, false);

      // y indices of yt which are in x
      ivec y_yt_in_x(x_yt.size(), 0);
      // y indices of yt not in x
      ivec y_yt_not_in_x(y.coeff.size2() - x_yt.size(), 0);
      {
        size_t n = 0;
        size_t n2 = 0;
        foreach(vector_variable* v, y.tail_list) {
          const irange& range = safe_get(y.var_range, v);
          if (x.arguments().count(v)) {
            for (size_t i = 0; i < range.size(); ++i)
              y_yt_in_x[n++] = range(i);
          } else {
            for (size_t i = 0; i < range.size(); ++i)
              y_yt_not_in_x[n2++] = range(i);
          }
        }
        assert(n == x_yt.size());
        assert(n2 == y.coeff.size2());
      }

      ivec x_all = x.indices(x.head_list);
      result.cmean.set_subvector(xh, x.cmean);

      mat y_coeff_y_yt_in_x(y.coeff.columns(y_yt_in_x));
      if (x_yt.size() != 0)
        result.cmean.set_subvector
          (yh, y_coeff_y_yt_in_x * x.cmean(x_yt) + y.cmean);
      else
        result.cmean.set_subvector(yh, y.cmean);

      mat covyy = y.cov;
      if (x_yt.size() != 0) {
        mat covyx = y_coeff_y_yt_in_x * x.cov(x_yt, x_all);
        covyy +=
          y_coeff_y_yt_in_x * x.cov(x_yt, x_yt) * y_coeff_y_yt_in_x.transpose();
        result.cov.set_submatrix(yh, xh, covyx);
        result.cov.set_submatrix(xh, yh, covyx.transpose());
      }

      if (xh.size() != 0)
        result.cov.set_submatrix(xh, xh, x.cov);
      result.cov.set_submatrix(yh, yh, covyy);
      result.coeff = y.coeff.columns(y_yt_not_in_x);
    }
    return result;
  } // direct_combination
