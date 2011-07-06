
#include <sill/learning/validation/parameter_grid.hpp>
#include <sill/range/algorithm.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  std::vector<vec>
  create_parameter_grid(const vec& minvals, const vec& maxvals, size_t k,
                        bool log_scale, bool inclusive) {
    uvec tmpk(maxvals.size(), k);
    return create_parameter_grid(minvals, maxvals, tmpk, log_scale, inclusive);
  }

  std::vector<vec>
  create_parameter_grid(const vec& minvals, const vec& maxvals, const uvec& k,
                        bool log_scale, bool inclusive) {
    assert(minvals.size() == maxvals.size());
    assert(maxvals.size() == k.size());
    for (size_t i(0); i < k.size(); ++i) {
      assert(k[i] >= 1);
      if (k[i] == 1)
        assert(minvals[i] == maxvals[i]);
    }
    std::vector<size_t> indices(minvals.size(), 0);
    std::vector<size_t> init_indices(minvals.size(), 0);
    std::vector<vec> values;
    vec minvals_(minvals);
    vec maxvals_(maxvals);
    if (log_scale) {
      foreach(double& v, minvals_) {
        assert(v > 0);
        v = std::log(v);
      }
      foreach(double& v, maxvals_) {
        assert(v > 0);
        v = std::log(v);
      }
    }
    do {
      vec vals(minvals_.size(), 0.);
      for (size_t j(0); j < minvals_.size(); ++j) {
        if (k[j] == 1) {
          vals[j] = minvals_[j];
        } else {
          if (inclusive)
            vals[j] = minvals_[j]
              + ((maxvals_[j]-minvals_[j]) * indices[j] / (k[j]-1));
          else
            vals[j] = minvals_[j]
              + ((maxvals_[j]-minvals_[j]) * (indices[j]+1)/(k[j]+1));
        }
      }
      values.push_back(vals);
      for (size_t j(0); j < minvals_.size(); ++j) {
        if (k[j] == 1)
          continue;
        if (indices[j] < static_cast<size_t>(k[j])-1) {
          ++indices[j];
          break;
        }
        indices[j] = 0;
      }
    } while (indices != init_indices);
    if (log_scale) {
      foreach(vec& v, values) {
        v = exp(v);
      }
    }
    return values;
  }

  vec create_parameter_grid(double minval, double maxval, size_t k,
                            bool log_scale, bool inclusive) {
    assert(k > 1);
    vec values(k, 0.);
    double minval_(minval);
    double maxval_(maxval);
    if (log_scale) {
      assert(minval_ > 0);
      minval_ = std::log(minval_);
      assert(maxval_ > 0);
      maxval_ = std::log(maxval_);
    }
    for (size_t j(0); j < k; ++j) {
      if (inclusive)
        values[j] = minval_ + ((maxval_ - minval_) * j / (k-1));
      else
        values[j] = minval_ + ((maxval_ - minval_) * (j+1) / (k+1));
    }
    if (log_scale) {
      foreach(double& v, values) {
        v = std::exp(v);
      }
    }
    return values;
  }

  std::vector<vec>
  create_parameter_grid_alt(const vec& minvals, const vec& maxvals, size_t k,
                            bool log_scale, bool inclusive) {
    uvec tmpk(maxvals.size(), k);
    return create_parameter_grid_alt(minvals, maxvals, tmpk, log_scale,
                                     inclusive);
  }

  std::vector<vec>
  create_parameter_grid_alt(const vec& minvals, const vec& maxvals,
                            const uvec& k,
                            bool log_scale, bool inclusive) {
    assert(minvals.size() == maxvals.size());
    assert(maxvals.size() == k.size());
    for (size_t i(0); i < k.size(); ++i) {
      assert(k[i] >= 1);
      if (k[i] == 1)
        assert(minvals[i] == maxvals[i]);
    }
    std::vector<vec> values;
    vec vals;
    vec minvals_(minvals);
    vec maxvals_(maxvals);
    if (log_scale) {
      foreach(double& v, minvals_) {
        assert(v > 0);
        v = std::log(v);
      }
      foreach(double& v, maxvals_) {
        assert(v > 0);
        v = std::log(v);
      }
    }
    for (size_t j(0); j < minvals_.size(); ++j) {
      vals.set_size(k[j]);
      if (k[j] == 1) {
        vals[0] = minvals_[j];
      } else {
        for (size_t i(0); i < static_cast<size_t>(k[j]); ++i) {
          if (inclusive)
            vals[i] = minvals_[j] + ((maxvals_[j]-minvals_[j]) * i / (k[j]-1));
          else
            vals[i] = minvals_[j]
              + ((maxvals_[j]-minvals_[j]) * (i+1) / (k[j]+1));
        }
      }
      values.push_back(vals);
    }
    if (log_scale) {
      foreach(vec& v, values) {
        v = exp(v);
      }
    }
    return values;
  }

  std::vector<vec>
  convert_parameter_grid_to_alt(const std::vector<vec>& vals) {
    assert(vals.size() > 0);
    size_t k(vals[0].size()); // dimensionality of parameter vectors
    std::vector<std::set<double> > unique_value_sets(k);
    foreach(const vec& v, vals) {
      assert(v.size() == k);
      for (size_t j(0); j < k; ++j)
        unique_value_sets[j].insert(v[j]);
    }
    std::vector<vec> altvals(k);
    size_t total_vals(1);
    for (size_t j(0); j < k; ++j) {
      altvals[j].set_size(unique_value_sets[j].size());
      sill::copy(forward_range<double>(unique_value_sets[j].begin(),
                                 unique_value_sets[j].end()),
           altvals[j].begin());
      std::sort(altvals[j].begin(), altvals[j].end());
      total_vals *= altvals[j].size();
    }
    if (total_vals != vals.size()) {
      std::cerr << "convert_parameter_grid_to_alt expected " << total_vals
                << " grid points but was only given " << vals.size()
                << std::endl;
      assert(false);
    }
    return altvals;
  }

  std::vector<vec>
  zoom_parameter_grid(const std::vector<vec>& oldgrid, const vec& val,
                      size_t k, bool log_scale) {
    uvec tmpk(val.size(), k);
    return zoom_parameter_grid(oldgrid, val, tmpk, log_scale);
  }

  std::vector<vec>
  zoom_parameter_grid(const std::vector<vec>& oldgrid, const vec& val,
                      const uvec& k, bool log_scale) {
    assert(oldgrid.size() > 0);
    size_t n(val.size());
    // Find bounding box.
    vec minvals(val.size(), -std::numeric_limits<double>::max());
    vec maxvals(val.size(), std::numeric_limits<double>::max());
    foreach(const vec& v, oldgrid) {
      assert(v.size() == n);
      for (size_t i(0); i < n; ++i) {
        if ((v[i] < val[i]) &&
            (fabs(val[i] - minvals[i]) > fabs(val[i] - v[i])))
          minvals[i] = v[i];
        if ((v[i] > val[i]) &&
            (fabs(val[i] - maxvals[i]) > fabs(val[i] - v[i])))
          maxvals[i] = v[i];
      }
    }
    for (size_t i(0); i < val.size(); ++i) {
      if (minvals[i] == -std::numeric_limits<double>::max())
        minvals[i] = val[i];
      if (maxvals[i] == std::numeric_limits<double>::max())
        maxvals[i] = val[i];
      assert(minvals[i] != maxvals[i]);
    }
    std::vector<vec>
      tmpgrid(create_parameter_grid(minvals, maxvals, k, log_scale, false));
    std::vector<vec> newgrid;
    foreach(const vec& v, tmpgrid) {
      if (compare(val,v) != 0) // TO DO: MAKE SURE THIS IS CORRECT
        newgrid.push_back(v);
    }
    return newgrid;
  }

  vec zoom_parameter_grid(const vec& oldgrid, double val, size_t k,
                          bool log_scale) {
    assert(oldgrid.size() > 0);
    // Find bounding box.
    double minval(-std::numeric_limits<double>::max());
    double maxval(std::numeric_limits<double>::max());
    foreach(double v, oldgrid) {
      if ((v < val) && (fabs(val - minval) > fabs(val - v)))
        minval = v;
      if ((v > val) && (fabs(val - maxval) > fabs(val - v)))
        maxval = v;
    }
    if (minval == -std::numeric_limits<double>::max())
      minval = val;
    if (maxval == std::numeric_limits<double>::max())
      maxval = val;
    assert(minval != maxval);
    vec tmpgrid(create_parameter_grid(minval, maxval, k, log_scale, false));
    std::set<double> oldgrid_set(oldgrid.begin(), oldgrid.end());
    std::vector<double> tmpgrid_new;
    foreach(double v, tmpgrid) {
      if (oldgrid_set.count(v) == 0)
        tmpgrid_new.push_back(v);
    }
    vec newgrid(tmpgrid_new.size(), 0);
    for (size_t i(0); i < tmpgrid_new.size(); ++i)
      newgrid[i] = tmpgrid_new[i];
    /*
    std::list<double> newgrid_list(tmpgrid.begin(), tmpgrid.end());
    std::list<double>::iterator it(newgrid_list.begin());
    while (it != newgrid_list.end()) {
      if (oldgrid_set.count(*it) != 0)
        newgrid_list.erase(it);
      else
        ++it;
    }
    vec newgrid(newgrid_list.size(), 0.);
    it = newgrid_list.begin();
    for (size_t i(0); i < newgrid_list.size(); ++i) {
      newgrid[i] = *it;
      ++it;
    }
    */
    return newgrid;
  }

  std::vector<vec>
  zoom_parameter_grid_alt(const std::vector<vec>& oldgrid, const vec& val,
                          size_t k, bool log_scale) {
    uvec tmpk(val.size(), k);
    return zoom_parameter_grid_alt(oldgrid, val, tmpk, log_scale);
  }

  std::vector<vec>
  zoom_parameter_grid_alt(const std::vector<vec>& oldgrid, const vec& val,
                          const uvec& k, bool log_scale) {
    size_t n(val.size());
    // Find bounding box.
    vec minvals(n, -std::numeric_limits<double>::max());
    vec maxvals(n, std::numeric_limits<double>::max());
    assert(oldgrid.size() == n);
    for (size_t i(0); i < oldgrid.size(); ++i) {
      foreach(double d, oldgrid[i]) {
        if ((d < val[i]) && (fabs(val[i] - minvals[i]) > fabs(val[i] - d)))
          minvals[i] = d;
        if ((d > val[i]) && (fabs(val[i] - maxvals[i]) > fabs(val[i] - d)))
          maxvals[i] = d;
      }
    }
    /*    foreach(const vec& v, oldgrid) {
      assert(v.size() == n);
      for (size_t i(0); i < n; ++i) {
        if ((v[i] < val[i]) &&
            (fabs(val[i] - minvals[i]) > fabs(val[i] - v[i])))
          minvals[i] = v[i];
        if ((v[i] > val[i]) &&
            (fabs(val[i] - maxvals[i]) > fabs(val[i] - v[i])))
          maxvals[i] = v[i];
      }
      }*/
    for (size_t i(0); i < n; ++i) {
      if (minvals[i] == -std::numeric_limits<double>::max())
        minvals[i] = val[i];
      if (maxvals[i] == std::numeric_limits<double>::max())
        maxvals[i] = val[i];
      if (k[i] == 1) {
        assert(minvals[i] == maxvals[i]);
      } else {
        assert(minvals[i] != maxvals[i]);
      }
    }
    return create_parameter_grid_alt(minvals, maxvals, k, log_scale, false);
  }

} // namespace sill

#include <sill/macros_undef.hpp>
