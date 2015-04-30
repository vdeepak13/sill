    /**
     * Approximates the conditional entropy H(Y | X) by sampling x ~ X
     * and computing H(Y | X=x).
     * This is useful if the marginalization required to compute H(Y|X) exactly
     * creates giant cliques, as long as the marginalization required to
     * compute H(Y | X=x) does not create giant cliques.
     * This decides it has converged when
     *  (standard error of estimate) / estimate < mult_std_error.
     *
     * @param base             Base of logarithm in entropy.
     * @param mult_std_error   Convergence parameter (> 0).
     * @return <approx conditional entropy, std error of estimate>
     */
    template <typename RandomNumberGenerator>
    std::pair<double, double>
    approx_conditional_entropy(const domain_type& Y, const domain_type& X,
                               double mult_std_error,
                               RandomNumberGenerator& rng,
                               double base = std::exp(1.)) const {
      assert(mult_std_error > 0);
      double val(0.); // accumulate sum
      double val2(0.); // accumulate sum of squares
      size_t APPROX_CHECK_PERIOD = 50;
      for (size_t i(0); i < std::numeric_limits<size_t>::max() - 1; ++i) {
        assignment_type a(sample(rng));
        decomposable Yx_model(*this);
        assignment_type ax;
        foreach(variable_type* v, X)
          ax[v] = a[v];
        Yx_model.condition(ax);
        decomposable Y_given_x_model;
        Yx_model.marginal(Y, Y_given_x_model);
        double result(Y_given_x_model.entropy());
        val += result;
        val2 += result * result;
        if ((i+1) % APPROX_CHECK_PERIOD == 0) {
          double est(val / (i+1.));
          double stderr_(sqrt((val2 / (i+1.)) - est * est)/sqrt(i+1.));
          if (stderr_ / est < mult_std_error)
            return std::make_pair(est, stderr_);
        }
      }
      throw std::runtime_error
        ("decomposable::approx_conditional_entropy() hit sample limit without getting a suitable approximation.");
    }

    /**
     * Computes the mutual information I(A; B),
     * where A,B must be subsets of the arguments of this model.
     * This is computed using I(A; B) = H(A) - H(A | B).
     * This uses approx_conditional_entropy() to compute H(A | B).
     *
     * @param mult_std_error   Convergence parameter (> 0).
     *                         (See approx_conditional_entropy().)
     * @param base   Base of logarithm.
     */
    template <typename RandomNumberGenerator>
    double
    approx_mutual_information(const domain_type& A, const domain_type& B,
                              double mult_std_error, RandomNumberGenerator& rng,
                              double base = std::exp(1.)) const {
      return entropy(A,base)
        - approx_conditional_entropy(A,B,mult_std_error,rng,base).first;
    }

    /**
     * Computes the conditional mutual information I(A; B | C),
     * where A,B,C must be subsets of the arguments of this model.
     * This is computed using I(A; B | C) = H(A | C) - H(A | B,C).
     * This uses approx_conditional_entropy().
     *
     * @param mult_std_error   Convergence parameter (> 0).
     *                         (See approx_conditional_entropy().)
     * @param base   Base of logarithm.
     */
    template <typename RandomNumberGenerator>
    double
    approx_conditional_mutual_information(const domain_type& A,
                                          const domain_type& B,
                                          const domain_type& C,
                                          double mult_std_error,
                                          RandomNumberGenerator& rng,
                                          double base = std::exp(1.)) const {
      return approx_conditional_entropy(A,C,mult_std_error,rng,base).first
        - approx_conditional_entropy(A,set_union(B,C),mult_std_error,rng,base).first;
    }


    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected per-label accuracy of predicting Y given X,
     * where this distribution represents P(Y,X).
     * @param X    Variables (which MUST be a subset of this model's arguments)
     *             to condition on.
     */
    model_per_label_accuracy_functor<decomposable>
    per_label_accuracy(const domain_type& X) const {
      return model_per_label_accuracy_functor<decomposable>(*this, X);
    }

    /**
     * Returns 1 if this predicts all variable values correctly and 0 otherwise.
     */
    size_t accuracy(const assignment_type& a) const {
      assignment_type pred(max_prob_assignment());
      foreach(variable_type* v, args) {
        if (!equal(pred[v],safe_get(a, v))) {
          return 0;
        }
      }
      return 1;
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected accuracy.
     */
    model_accuracy_functor<decomposable>
    accuracy() const {
      return model_accuracy_functor<decomposable>(*this);
    }

    /**
     * Computes the mean squared error (mean over variables).
     * Note: This is equivalent to per_label_accuracy for finite variables.
     *
     * @param a    an assignment to X
     */
    double mean_squared_error(const assignment_type& a) const {
      return (error_measures::squared_error<assignment_type>
              (a, max_prob_assignment(), args) / args.size());
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected mean squared error.
     */
    model_mean_squared_error_functor<decomposable>
    mean_squared_error() const {
      return model_mean_squared_error_functor<decomposable>(*this);
    }

    /**
     * Computes the mean squared error of predicting Y given X,
     * where this model is of P(Y,X).
     */
    double
    mean_squared_error(const assignment_type& a, const domain_type& X) const {
      assignment_type tmpa;
      foreach(variable_type* v, X) {
        tmpa[v] = safe_get(a, v);
      }
      decomposable tmp_model(*this);
      tmp_model.condition(tmpa);
      return (error_measures::squared_error<assignment_type>
              (a, tmp_model.max_prob_assignment(), args) / args.size());
    }

    /**
     * Returns a functor usable with dataset::expected_value() for computing
     * expected mean squared error of predicting Y given X.
     */
    model_mean_squared_error_functor<decomposable>
    mean_squared_error(const domain_type& X) const {
      return model_mean_squared_error_functor<decomposable>(*this, X);
    }


    /**
     * Computes the per-label accuracy (average over X variables).
     * @param a    an assignment to this model's arguments
     */
    double per_label_accuracy(const assignment_type& a) const {
      double acc = 0;
      assignment_type pred(max_prob_assignment());
      foreach(variable_type* v, args) {
        if (equal(pred[v], safe_get(a, v)))
          ++acc;
      }
      return (acc / args.size());
    }

    /**
     * Computes the per-label accuracy of predicting Y given X,
     * where this distribution represents P(Y,X).
     * @param a    an assignment to this model's arguments
     * @param X    Variables (which MUST be a subset of this model's arguments)
     *             to condition on.
     */
    double
    per_label_accuracy(const assignment_type& a, const domain_type& X) const {
      assignment_type tmpa;
      foreach(variable_type* v, X) {
        tmpa[v] = safe_get(a, v);
      }
      decomposable tmp_model(*this);
      tmp_model.condition(tmpa);
      return tmp_model.per_label_accuracy(a);
    }



    /**
     * Computes the conditional log likelihood: log p(y | x),
     * where this distribution represents P(y, x).
     *
     * \param x a subset of this model's arguments to condition on.
     * \todo optimize this
     */
    result_type conditional_log(const assignment_type& a,
                                const domain_type& x) const {
      decomposable tmp(*this);
      tmp.condition(r.assignment(X));
      return tmp_model.log_likelihood(r, base);
    }


   /**
     * This replaces all current factors with the given factors
     * while keeping the current structure of this model.  Each new factor
     * must be covered by a clique in this model.
     *
     * If you are calling this multiple times,
     * it is significantly faster to use the other replace_factors() method.
     *
     * @param factors A readable forward range of objects of type F
     */
    template <typename Range>
    void replace_factors(const Range& factors) {
      std::vector<vertex> factor_vertex_map;
      foreach(const typename Range::value_type& factor, factors) {
        vertex v(find_clique_cover(factor.arguments()));
        assert(v != vertex());
        factor_vertex_map.push_back(v);
      }
      replace_factors(factors, factor_vertex_map);
    }

    /**
     * This replaces all current factors with the given factors
     * while keeping the current structure of this model.  Each new factor
     * must be covered by a clique in this model.
     *
     * @param factors            A readable forward range of objects of type F
     * @param factor_vertex_map  A vector corresponding to the given factor
     *                            range, with each entry being a vertex in this
     *                            model whose clique covers the corresponding
     *                            factor's arguments.
     */
    template <typename Range>
    void replace_factors(const Range& factors,
                         const std::vector<vertex>& factor_vertex_map) {
      concept_assert((ReadableForwardRangeConvertible<Range, F>));

      foreach(const edge_type& e, edges())
        jt_[e] = F(1);
      foreach(size_t v, vertices())
        jt_[v] = F(1);

      // We do not use F for iteration since Range may be over a different
      // factor type that is merely convertible to F.
      size_t j(0);
      foreach(const typename Range::value_type& factor, factors) {
        if (!factor.arguments().empty()) {
          assert(j < factor_vertex_map.size());
          size_t v = factor_vertex_map[j];
          jt_[v] *= factor;
          /*
           // It is sometimes OK for gaussians to be unnormalizable
           // before calibration.
          if (!jt_[v].is_normalizable()) {
            std::cerr << "Cannot normalize this factor:\n"
                      << factor << std::endl;
            throw normalization_error("decomposable::replace_factors ran into factor which could not be normalized.");
          }
          */
        }
        ++j;
      }

      // Depending on the factor type, pre-normalize clique marginals
      // to avoid numerical issues.
      if (impl::decomposable_extra_normalization<F>::value) {
        foreach(size_t v, vertices()) {
          if (jt_[v].is_normalizable())
            jt_[v].normalize();
        }
      }

      // Recalibrate and renormalize the model.
      calibrate();
      normalize();
    }
 

    /**
     * Marginalizes a set of variables out of this decomposable model.
     * For each variable, we find a subtree that includes the variable
     * and contracts all the edges in that subtree.
     *
     * param vars the variables to be marginalized out
     * \todo The implementation somewhat inefficient at the moment.
     */
    void marginalize_out(const domain_type& vars) {
      for (variable_type* var, vars) {
        while (true) {
          // Find a cover for this variable.
          vertex v = find_clique_cover(make_domain(var));
          assert(v != vertex());
          // Look for a neighbor that also has the variable.
          bool done = true;
          foreach(edge e, out_edges(v)) {
            vertex w = e.target();
            if (clique(w).count(var)) {
              // The cliques at v and w both contain the variable.
              // Merge them and then restart this process.
              done = false;
              merge(e);
              break;
            }
          }
          if (done) {
            // None of the cliques neighboring v contain the variable.
            // By the running intersection property, then, no other
            // cliques contain the variable.  So we can safely
            // marginalize the variable out of this clique, and then
            // we are done.
            jt_.set_clique(v, set_difference(clique(v), make_domain(var)));
            jt_[v] = jt_[v].marginal(clique(v));
            remove_if_nonmaximal(v);
            // Move on to the next variable to marginalize out.
            break;
          }
        }
      }

      // Update the arguments.
      args = set_difference(args, vars);
      #ifdef SILL_VERBOSE
        std::cerr << "Result: " << *this << std::endl;
      #endif
    }

    /**
     * Marginalizes a set of variables out of this decomposable model
     * using an approximation.  The variables are marginalized out of
     * each clique independently; since clique merging is avoided,
     * this operation is not exact.
     *
     * @param  vars
     *         the variables to be marginalized out
     *
     * \todo At the moment, the implementation is somewhat inefficient.
     */
    void marginalize_out_approx(const domain_type& vars) {
      #ifdef SILL_VERBOSE
        std::cerr << "Marginalizing out (approx): " << vars << std::endl;
      #endif
      std::vector<vertex> overlapping;
      while (true) {
        // Find a vertex whose clique overlaps vars.
        overlapping.clear();
        find_intersecting_cliques(vars, std::back_inserter(overlapping));
        if (overlapping.empty()) break;
        vertex v = overlapping.front();
        // Remove vars from this clique.
        jt_.set_clique(v, set_difference(clique(v), vars));
        // at this point, the variables have been removed from the separators
        jt_[v] =  jt_[v].marginal(clique(v));

        // Marginalize out the variables from all incident separators.
        foreach(edge e, out_edges(v)) {
          //! \todo In a way, this check should not be necessary
          //!       These optimizations should be performed inside factors
          domain_type sep_args = jt_[e].arguments();
          if (set_intersect(sep_args, vars).size() > 0)
            jt_[e] = sum(jt_[e], vars);
        }
        remove_if_nonmaximal(v);
      }
      // Update the arguments.
      args.erase(vars.begin(), vars.end());
      #ifdef SILL_VERBOSE
        std::cerr << "Result: " << *this << std::endl;
      #endif
    }

    /**
     * Renames (a subset of) the arguments of this factor.
     *
     * @param map
     *        an object such that map[v] maps the variable handle v
     *        a type compatible variable handle; this mapping must be 1:1.
     */
    void subst_args(const std::map<variable_type*, variable_type*>& var_map) {
      // Compute the variables to be replaced.
      domain_type old_vars = keys(var_map);

      // Find all cliques that contain an old variable.
      std::vector<vertex> vertices;
      find_intersecting_cliques(old_vars, std::back_inserter(vertices));

      // Update affected cliques and incident separator marginals
      foreach(vertex v, vertices) {
        jt_.set_clique(v, subst_vars(clique(v), var_map));
        jt_[v].subst_args(var_map);
        foreach(edge e, out_edges(v))
          jt_[e].subst_args(var_map);
      }

    }

