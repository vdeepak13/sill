#include <sill/learning/discriminative/logistic_regression.hpp>

#include <sill/macros_def.hpp>

namespace sill {

    // Protected methods
    //==========================================================================

    void logistic_regression::init() {
      assert(params.valid());
      if (ds.num_finite() > 1) {
        finite_offset.push_back(0);
        for (size_t j = 0; j < ds.num_finite(); ++j)
          if (j != label_index_) {
            finite_indices.push_back(j);
            finite_offset.push_back(finite_offset.back()+finite_seq[j]->size());
          }
        finite_offset.pop_back();
      }
      if (ds.num_vector() > 0) {
        vector_offset.push_back(0);
        for (size_t j = 0; j < ds.num_vector()-1; ++j)
          vector_offset.push_back(vector_offset.back()+vector_seq[j]->size());
      }
      lambda = params.lambda;
      if (params.regularization == 1 || params.regularization == 2) {
        if (lambda <= 0) {
          if (DEBUG_LOGISTIC_REGRESSION)
            std::cerr << "logistic_regression was told to"
                      << " use regularization but given lambda = " << lambda
                      << std::endl;
          assert(false);
          return;
        }
      }

      w_fin.resize(ds.finite_dim() - label_->size());
      w_vec.resize(ds.vector_dim());
      b = 0;
      grad_fin.resize(w_fin.size());
      grad_vec.resize(w_vec.size());

      if (params.perturb_init > 0) {
        boost::mt11213b rng(static_cast<unsigned>(params.random_seed));
        boost::uniform_real<double> uniform_dist;
        uniform_dist = boost::uniform_real<double>
          (-1 * params.perturb_init, params.perturb_init);
        foreach(double& v, w_fin)
          v = uniform_dist(rng);
        foreach(double& v, w_vec)
          v = uniform_dist(rng);
        b = uniform_dist(rng);
      }

      eta = params.eta;
      train_acc = 0;
      train_log_like = 0;

      switch(params.method) {
      case 0:
      case 1:
        for (size_t i = 0; i < ds.size(); ++i)
          total_train += ds.weight(i);
        break;
      case 2:
        break;
      default:
        assert(false);
      }

      // TODO: Change this once there are non-iterative methods.
      while(iteration_ < params.init_iterations)
        if (!(step()))
          break;
    }

    bool logistic_regression::step_gradient_descent() {
      double prev_train_log_like = 0;
      for (size_t j(0); j < grad_fin.size(); ++j)
        grad_fin[j] = 0;
      for (size_t j(0); j < grad_vec.size(); ++j)
        grad_vec[j] = 0;
      grad_b = 0;
      train_acc = 0;
      train_log_like = 0;
      for (size_t i = 0; i < ds.size(); ++i) {
        const record& rec = ds[i];
        double v(confidence(rec));
        const std::vector<size_t>& findata = rec.finite();
        const vec& vecdata = rec.vector();
        double bin_label = (findata[label_index_] > 0 ? 1 : -1);
        train_acc += ((v > 0) ^ (bin_label == -1) ? ds.weight(i) : 0);
        train_log_like -= ds.weight(i) * std::log(1. + exp(-bin_label * v));
        if (train_log_like == -std::numeric_limits<double>::infinity())
          std::cerr << ""; // TODO: DEBUG THIS:
        /*
          e.g.,
          ./run_learner --learner batch_booster --train_data /Users/jbradley/data/uci/adult/adult-train.sum --test_data /Users/jbradley/data/uci/adult/adult-test.sum --weak_learner log_reg --learner_objective ada
        */
        v = ds.weight(i) * bin_label / (1. + exp(bin_label * v));
        // update gradient
        for (size_t j = 0; j < finite_indices.size(); ++j) {
          size_t val = findata[finite_indices[j]];
          grad_fin[finite_offset[j] + val] += v;
        }
        for (size_t j = 0; j < w_vec.size(); ++j)
          grad_vec[j] += vecdata[j] * v;
        grad_b += v;
      }
      train_acc /= total_train;
      train_log_like /= total_train;
      if (fabs(train_log_like - prev_train_log_like) < params.convergence) {
        if (DEBUG_LOGISTIC_REGRESSION)
          std::cerr << "logistic_regression converged: training log likelihood "
                    << "changed from " << prev_train_log_like << " to "
                    << train_log_like << "; exiting early (iteration "
                    << iteration_ << ")." << std::endl;
        return false;
      }
      prev_train_log_like = train_log_like;
      // update gradients for regularization
      switch(params.regularization) {
      case 1:
        std::cerr << "NOT IMPLEMENTED YET" << std::endl;
        assert(false);
        break;
      case 2:
        for (size_t j(0); j < grad_fin.size(); ++j)
          grad_fin[j] -= lambda * w_fin[j];
        for (size_t j(0); j < grad_vec.size(); ++j)
          grad_vec[j] -= lambda * w_vec[j];
        grad_b -= lambda * b;
        break;
      default:
        break;
      }
      // update weights
      for (size_t j(0); j < grad_fin.size(); ++j)
        w_fin[j] += eta * grad_fin[j];
      for (size_t j(0); j < grad_vec.size(); ++j)
        w_vec[j] += eta * grad_vec[j];
      b += eta * grad_b;
      eta *= params.mu;
      return true;
    } // end of function: bool step_gradient_descent()

    bool logistic_regression::step_stochastic_gradient_descent() {
      if (!(o.next()))
        return false;
      const record& rec = o.current();
      double ex_weight(o.weight());
      total_train += ex_weight;
      double v(confidence(rec));
      const std::vector<size_t>& findata = rec.finite();
      const vec& vecdata = rec.vector();
      double bin_label = (findata[label_index_] > 0 ? 1 : -1);
      train_acc += ((v > 0) ^ (bin_label == -1) ? ex_weight : 0);
      train_log_like -= ex_weight * std::log(1. + exp(-bin_label * v));
      v = ex_weight * bin_label / (1. + exp(bin_label * v));
      // update weights
      for (size_t j(0); j < grad_fin.size(); ++j)
        grad_fin[j] = 0;
      for (size_t j = 0; j < finite_indices.size(); ++j) {
        size_t val(findata[finite_indices[j]]);
        grad_fin[finite_offset[j] + val] += v;
      }
      for (size_t j = 0; j < w_vec.size(); ++j)
        grad_vec[j] += vecdata[j] * v;
      grad_b += v;
      // update weights for regularization
      switch(params.regularization) {
      case 1:
        std::cerr << "NOT IMPLEMENTED YET" << std::endl;
        assert(false);
        break;
      case 2:
        for (size_t j(0); j < grad_fin.size(); ++j)
          grad_fin[j] -= lambda * w_fin[j];
        for (size_t j(0); j < grad_vec.size(); ++j)
          grad_vec[j] -= lambda * w_vec[j];
        grad_b -= lambda * b;
        break;
      default:
        break;
      }
      // update weights
      for (size_t j(0); j < grad_fin.size(); ++j)
        w_fin[j] += eta * grad_fin[j];
      for (size_t j(0); j < grad_vec.size(); ++j)
        w_vec[j] += eta * grad_vec[j];
      b += eta * grad_b;
      eta *= params.mu;
      ++iteration_;
      return true;
    } // end of function: bool step_stochastic_gradient_descent()

    // Getters and helpers
    //==========================================================================

    bool logistic_regression::is_online() const {
      switch(params.method) {
      case 0:
      case 1:
        return false;
      case 2:
        return true;
      default:
        assert(false);
        return false;
      }
    }

    double logistic_regression::train_accuracy() const {
      switch(params.method) {
      case 0:
      case 1:
        return train_acc;
      case 2:
        return (total_train == 0 ? -1 : train_acc / total_train);
      default:
        assert(false);
        return -1;
      }
    }

    // Prediction methods
    //==========================================================================

    double logistic_regression::confidence(const record& example) const {
      double v(b);
      const std::vector<size_t>& findata = example.finite();
      for (size_t j(0); j < finite_indices.size(); ++j) {
        size_t val = findata[finite_indices[j]];
        v += w_fin[finite_offset[j] + val];
      }
      const vec& vecdata = example.vector();
      for (size_t j(0); j < w_vec.size(); ++j)
        v += w_vec[j] * vecdata[j];
      return v;
    }

    double logistic_regression::confidence(const assignment& example) const {
      double v(b);
      const finite_assignment& fa = example.finite();
      for (size_t j(0); j < finite_indices.size(); ++j) {
        size_t val = safe_get(fa, finite_seq[finite_indices[j]]);
        v += w_fin[finite_offset[j] + val];
      }
//      for (size_t j = 0; j < w_fin.size(); ++j)
//        v += w_fin[j] * example[finite_seq[finite_indices[j]]];
      const vector_assignment& va = example.vector();
      for (size_t j(0); j < w_vec.size(); ++j) {
        const vec& vecdata = safe_get(va, vector_seq[j]);
        for (size_t j2(0); j2 < vector_seq[j]->size(); ++j2) {
          size_t ind(vector_offset[j] + j2);
          v += w_vec[ind] * vecdata[j2];
        }
      }
      return v;
    }

    // Methods for iterative learners
    //==========================================================================

    bool logistic_regression::step() {
      switch(params.method) {
      case 0:
        return step_gradient_descent();
      case 1:
        std::cerr << "Newton's method not yet implemented." << std::endl;
        assert(false);
        return false;
      case 2:
        return step_stochastic_gradient_descent();
      default:
        assert(false);
        return false;
      }
    }

    // Save and load methods
    //==========================================================================

    void logistic_regression::save(std::ofstream& out, size_t save_part,
              bool save_name) const {
      base::save(out, save_part, save_name);
      params.save(out);
      out << eta << " " << w_fin << " " << w_vec << " " << b
          << " " << train_acc << " " << train_log_like << " " << iteration_
          << " " << total_train << "\n";
    }

    bool logistic_regression::load(std::ifstream& in, const datasource& ds, size_t load_part) {
      if (!(base::load(in, ds, load_part)))
        return false;
      finite_seq = ds.finite_list();
      vector_seq = ds.vector_list();
      finite_offset.clear();
      finite_indices.clear();
      if (ds.num_finite() > 1) {
        finite_offset.push_back(0);
        for (size_t j = 0; j < ds.num_finite(); ++j)
          if (j != label_index_) {
            finite_indices.push_back(j);
            finite_offset.push_back(finite_offset.back()+finite_seq[j]->size());
          }
        finite_offset.pop_back();
      }
      vector_offset.clear();
      if (ds.num_vector() > 0) {
        vector_offset.push_back(0);
        for (size_t j = 0; j < ds.num_vector() - 1; ++j)
          vector_offset.push_back(vector_offset.back()+vector_seq[j]->size());
      }
      params.load(in);
      std::string line;
      getline(in, line);
      std::istringstream is(line);
      if (!(is >> eta))
        assert(false);
      assert(eta > 0 && eta <= 1);
      read_vec(is, w_fin);
      read_vec(is, w_vec);
      if (!(is >> b))
        assert(false);
      if (!(is >> train_acc))
        assert(false);
      if (!(is >> train_log_like))
        assert(false);
      if (!(is >> iteration_))
        assert(false);
      if (!(is >> total_train))
        assert(false);
      return true;
    }

} // namespace sill

#include <sill/macros_undef.hpp>
