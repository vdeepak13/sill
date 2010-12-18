#include <sill/base/universe.hpp>
#include <sill/learning/dataset/syn_oracle_majority.hpp>

#include <sill/macros_def.hpp>

namespace sill {

    // Private methods
    //==========================================================================

    void syn_oracle_majority::init() {
      assert(params.valid());

      for (size_t j = 0; j < finite_seq.size(); j++)
        assert(finite_seq[j]->size() == 2);

      // Initialize the random number generator
      rng.seed(static_cast<unsigned>(params.random_seed));
      bernoulli_dist = boost::bernoulli_distribution<double>(.5);
      uniform_prob = boost::uniform_real<double>(0,1);

      // Choose the voting features
      voting = randperm(finite_seq.size() - 1, rng);
      /*
      voting.resize(vars.size() - 1);
      for (size_t j = 0; j < vars.size() - 1; ++j)
        voting[j] = j;
      random_shuffle(voting.begin(), voting.end(), rng);
      */
      voting.resize((size_t)(round(params.r_vars * voting.size())));
    }

    // Mutating operations
    //==========================================================================

    bool syn_oracle_majority::next() {
      for (size_t j = 0; j < finite_vars.size()-1; ++j)
        current_rec.fin_ptr->operator[](j) = (bernoulli_dist(rng) == true ?1:0);
      size_t sum = 0;
      foreach(size_t j, voting)
        sum += current_rec.fin_ptr->operator[](j);
      if (sum > voting.size() / 2.)
        current_rec.fin_ptr->back() = 1;
      else
        current_rec.fin_ptr->back() = 0;
      if (params.feature_noise > 0)
        for (size_t j = 0; j < finite_vars.size()-1; ++j)
          if (uniform_prob(rng) < params.feature_noise)
            current_rec.fin_ptr->operator[](j) =
              1 - current_rec.fin_ptr->operator[](j);
      if (params.label_noise > 0)
        if (uniform_prob(rng) < params.label_noise)
          current_rec.fin_ptr->back() = 1 - current_rec.fin_ptr->back();
      return true;
    }

  // Free functions
  //==========================================================================

  syn_oracle_majority
  create_syn_oracle_majority
  (size_t nfeatures, universe& u,
   const syn_oracle_majority::parameters& params) {
    assert(nfeatures > 0);
    finite_var_vector var_order;
    for (size_t j = 0; j < nfeatures+1; j++)
      var_order.push_back(u.new_finite_variable(2));
    return syn_oracle_majority(var_order, params);
  }

} // namespace sill

#include <sill/macros_undef.hpp>
