#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <boost/timer.hpp>

#include <sill/base/universe.hpp>
#include <sill/learning/dataset/data_loader.hpp>
#include <sill/learning/dataset/dataset.hpp>
#include <sill/learning/dataset/dataset_view.hpp>
#include <sill/learning/dataset/data_conversions.hpp>
#include <sill/learning/dataset/syn_oracle_knorm.hpp>
#include <sill/learning/dataset/vector_dataset.hpp>

#include <sill/macros_def.hpp>

/**
 * \file dataset_view_timing.cpp Timing tests of datasets and views.
 */
int main(int argc, char* argv[]) {

  using namespace sill;
  using namespace std;

  universe u;
  size_t nruns = 100;
  double tmp = 0;
  boost::timer t;

  // Create a dataset to work with
  size_t nvars = 20;
  syn_oracle_knorm knorm(create_syn_oracle_knorm(2,nvars,u));
  size_t nrecords = 5000;
  vector_dataset<> ds;
  oracle2dataset(knorm, nrecords, ds);
  // Create some views
  dataset_view<> ds_view_range(ds);
  ds_view_range.set_record_range(1000, 2000);
  std::vector<size_t> indices;
  for (size_t i = 1000; i < 2000; i++)
    indices.push_back(i);
  dataset_view<> ds_view_indices(ds);
  ds_view_indices.set_record_indices(indices);
  dataset<>& ds_view_range_ref = ds_view_range;
  dataset<>& ds_view_indices_ref = ds_view_indices;
  // Make sure the compiler doesn't optimize stuff away
  if (argc == 100000)
    ds_view_range_ref = ds;
  if (argc == 100000)
    ds_view_indices_ref = ds;

  // Run timing tests for data access: records(), at(), finite()
  //  for each of: ds, ds_view_range, ds_view_indices, ds_view_*_ref

  cout << "\nCompute average times (over " << nruns << " iterations) of iterating"
       << " over 1000 records and using " << nvars << " vector values in a sum."
       << endl << endl;

  cout << "Test 1: use record_iterator for " << nruns << " iterations" << endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++) {
    vector_dataset<>::record_iterator_type r_it = ds.records().first;
    for (size_t i = 0; i < 1000; i++) {
      const vec& v = (*r_it).vector();
      for (size_t j = 0; j < nvars; j++)
        tmp += v[j];
      ++r_it;
    }
  }
  cout << " vector_dataset: " << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++) {
    dataset_view<>::record_iterator_type r_it = ds_view_range.records().first;
    for (size_t i = 0; i < 1000; i++) {
      const vec& v = (*r_it).vector();
      for (size_t j = 0; j < nvars; j++)
        tmp += v[j];
      ++r_it;
    }
  }
  cout << " dataset_view (range): " << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++) {
    dataset_view<>::record_iterator_type r_it = ds_view_indices.records().first;
    for (size_t i = 0; i < 1000; i++) {
      const vec& v = (*r_it).vector();
      for (size_t j = 0; j < nvars; j++)
        tmp += v[j];
      ++r_it;
    }
  }
  cout << " dataset_view (indices): " << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++) {
    dataset<>::record_iterator_type r_it = ds_view_range_ref.records().first;
    for (size_t i = 0; i < 1000; i++) {
      const vec& v = (*r_it).vector();
      for (size_t j = 0; j < nvars; j++)
        tmp += v[j];
      ++r_it;
    }
  }
  cout << " dataset ref (for dataset_view (range)): "
       << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++) {
    dataset<>::record_iterator_type r_it = ds_view_indices_ref.records().first;
    for (size_t i = 0; i < 1000; i++) {
      const vec& v = (*r_it).vector();
      for (size_t j = 0; j < nvars; j++)
        tmp += v[j];
      ++r_it;
    }
  }
  cout << " dataset ref (for dataset_view (indices)): "
       << t.elapsed() / nruns << std::endl;

  ////////////////////////////////////////

  cout << "Test 2: use at() for " << nruns << " iterations" << endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++) {
    for (size_t i = 0; i < 1000; i++) {
      const record<>& r = ds[i];
      const vec& v = r.vector();
      for (size_t j = 0; j < nvars; j++)
        tmp += v[j];
    }
  }
  cout << " vector_dataset: " << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++) {
    for (size_t i = 0; i < 1000; i++) {
      const record<>& r = ds_view_range[i];
      const vec& v = r.vector();
      for (size_t j = 0; j < nvars; j++)
        tmp += v[j];
    }
  }
  cout << " dataset_view (range): " << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++) {
    for (size_t i = 0; i < 1000; i++) {
      const record<>& r = ds_view_indices[i];
      const vec& v = r.vector();
      for (size_t j = 0; j < nvars; j++)
        tmp += v[j];
    }
  }
  cout << " dataset_view (indices): " << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++) {
    for (size_t i = 0; i < 1000; i++) {
      const record<>& r = ds_view_range_ref[i];
      const vec& v = r.vector();
      for (size_t j = 0; j < nvars; j++)
        tmp += v[j];
    }
  }
  cout << " dataset ref (for dataset_view (range)): "
       << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++) {
    for (size_t i = 0; i < 1000; i++) {
      const record<>& r = ds_view_indices_ref[i];
      const vec& v = r.vector();
      for (size_t j = 0; j < nvars; j++)
        tmp += v[j];
    }
  }
  cout << " dataset ref (for dataset_view (indices)): "
       << t.elapsed() / nruns << std::endl;

  ////////////////////////////////////////

  cout << "Test 3: use vector() for " << nruns << " iterations" << endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++)
    for (size_t i = 0; i < 1000; i++)
      for (size_t j = 0; j < nvars; j++)
        tmp += ds.vector(i,j);
  cout << " vector_dataset: " << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++)
    for (size_t i = 0; i < 1000; i++)
      for (size_t j = 0; j < nvars; j++)
        tmp += ds_view_range.vector(i,j);
  cout << " dataset_view (range): " << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++)
    for (size_t i = 0; i < 1000; i++)
      for (size_t j = 0; j < nvars; j++)
        tmp += ds_view_indices.vector(i,j);
  cout << " dataset_view (indices): " << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++)
    for (size_t i = 0; i < 1000; i++)
      for (size_t j = 0; j < nvars; j++)
        tmp += ds_view_range_ref.vector(i,j);
  cout << " dataset ref (for dataset_view (range)): "
       << t.elapsed() / nruns << std::endl;
  t.restart();
  for (size_t n = 0; n < nruns; n++)
    for (size_t i = 0; i < 1000; i++)
      for (size_t j = 0; j < nvars; j++)
        tmp += ds_view_indices_ref.vector(i,j);
  cout << " dataset ref (for dataset_view (indices)): "
       << t.elapsed() / nruns << std::endl;

  ////////////////////////////////////////

  // Create two views of views which should end up being the same
  indices.clear();
  for (size_t i = 0; i < 500; i++)
    indices.push_back(i);
  dataset_view<> ds_view_view1(ds_view_range);
  ds_view_view1.set_record_indices(indices);
  dataset_view<> ds_view_view2(ds_view_indices);
  ds_view_view2.set_record_range(0, 500);
  // Check that the two views of views are the same
  for (size_t i = 0; i < 500; i++)
    assert(equal(ds_view_view1[i].vector(), ds_view_view2[i].vector()));
  cout << "\nTested creating views of views--it worked." << endl;

  ////////////////////////////////////////

  // Load a dataset with finite variables to test binarizing and merging
  //  variables.
  if (1) { // since I haven't uploaded the UCI datasets to SVN
    symbolic_oracle<>
      o(*(data_loader::load_symbolic_oracle<symbolic_oracle<>::la_type>
          ("/Users/jbradley/data/uci/adult/adult-test.sum", u)));
    size_t nrecords2 = 5000;
    vector_dataset<> ds2;
    oracle2dataset(o, nrecords2, ds2);
    ds2.print_datasource_info();
    // Create binarized view
    finite_variable* original_var = ds2.finite_list().front();
    finite_variable* binary_var = u.new_finite_variable(2);
    dataset_view<> ds_view_binarized(ds2);
    ds_view_binarized.set_binary_indicator(original_var, binary_var, 0);
    cout << "\nCreated a dataset view with binary variable " << binary_var
         << " == 1 iff original variable " << original_var
         << " == 0.  Compare finite data for first few records:\n";
    for (size_t i = 0; i < 6; i++) {
      cout << "Original:  " << ds2[i].finite() << "\n";
      cout << "Binarized: " << ds_view_binarized[i].finite() << "\n";
    }
    // Create merged view
    finite_var_vector merged1_orig_vars, merged2_orig_vars;
    size_t merged1_new_var_size(1);
    size_t merged2_new_var_size(1);
    assert(ds2.num_finite() > 2);
    merged1_orig_vars.push_back(ds2.finite_list()[0]);
    merged1_orig_vars.push_back(ds2.finite_list()[1]);
    merged1_new_var_size *= ds2.finite_list()[0]->size();
    merged1_new_var_size *= ds2.finite_list()[1]->size();
    size_t n_merged2_orig_vars(std::min(ds2.num_finite()/2., 6.));
    merged2_orig_vars.resize(n_merged2_orig_vars);
    for (size_t j = 0; j < n_merged2_orig_vars; ++j) {
      size_t tmpj = n_merged2_orig_vars - j - 1;
      merged2_orig_vars[tmpj] = ds2.finite_list()[j];
      merged2_new_var_size *= ds2.finite_list()[j]->size();
    }
    finite_variable* merged1_new_var =
      u.new_finite_variable(merged1_new_var_size);
    finite_variable* merged2_new_var =
      u.new_finite_variable(merged2_new_var_size);
    dataset_view<> ds_view_merged1(ds2);
    ds_view_merged1.set_merged_variables(merged1_orig_vars, merged1_new_var);
    boost::shared_ptr<dataset_view<> > ds_view_merged1_light
      = ds_view_merged1.create_light_view();
    dataset_view<> ds_view_merged2(ds2);
    ds_view_merged2.set_merged_variables(merged2_orig_vars, merged2_new_var);
    cout << "\nCreated a dataset view with first 2 finite vars merged."
         << "  Compare finite data for first few records:\n";
    cout << "Orig arities: [";
    for (size_t j = 0; j < ds2.num_finite(); ++j)
      cout << ds2.finite_list()[j]->size() << " ";
    cout << "]\n";
    for (size_t i = 0; i < 6; i++) {
      cout << "Original:  " << ds2[i].finite() << "\n";
      cout << "Merged: " << ds_view_merged1[i].finite() << "\n";
      std::vector<size_t> orig_vals(2);
      ds_view_merged1_light->revert_merged_value
        (ds_view_merged1[i].finite().back(), orig_vals);
      cout << "Reverted: " << orig_vals << "\n";
    }
    cout << "\nCreated a dataset view with first " << merged2_orig_vars.size()
         << " finite vars merged. Compare finite data for first few records:\n";
    cout << "Orig arities: [";
    for (size_t j = 0; j < ds2.num_finite(); ++j)
      cout << ds2.finite_list()[j]->size() << " ";
    cout << "]\n";
    cout << "Using load_finite():\n";
    for (size_t i = 0; i < 6; i++) {
      cout << "Original:  " << ds2[i].finite() << "\n";
      cout << "Merged: " << ds_view_merged2[i].finite() << "\n";
      std::vector<size_t> orig_vals(4);
      ds_view_merged2.revert_merged_value
        (ds_view_merged2[i].finite().back(), orig_vals);
      cout << "Reverted: " << orig_vals << "\n";
    }
    cout << "Using load_record():\n";
    vector_dataset<>::record_iterator_type r_it = ds2.begin();
    vector_dataset<>::record_iterator_type r_it2 = ds_view_merged2.begin();
    for (size_t i = 0; i < 6; i++) {
      cout << "Original: " << (*r_it).finite() << "\n"
           << "Merged: " << (*r_it2).finite() << "\n";
      ++r_it; ++r_it2;
    }
    cout << "Using finite(i,j):\n";
    for (size_t i = 0; i < 6; i++) {
      cout << "Original: [";
      for (size_t j = 0; j < ds2.num_finite(); ++j)
        cout << ds2.finite(i,j) << " ";
      cout << "]\n" << "Merged: [";
      for (size_t j = 0; j < ds_view_merged2.num_finite(); ++j)
        cout << ds_view_merged2.finite(i,j) << " ";
      cout << "]\n";
    }
    cout << "Using convert_record():\n";
    record<> tmprec(ds_view_merged2[0]);
    for (size_t i = 0; i < 6; i++) {
      cout << "Original: " << ds2[i].finite() << "\n";
      ds_view_merged2.convert_record(ds2[i], tmprec);
      cout << "Merged: " << tmprec.finite() << "\n";
    }

    // Do a timing test for binarized variables
    size_t nvars2 = ds2.finite_list().size();
    cout << "Now running timing test for views of binarized variables:\n"
         << "Compute average times (over " << nruns << " iterations) of "
         << "iterating over 1000 records and using " << nvars2
         << " finite values in a sum." << endl << endl;

    cout << "Test: use record_iterator for " << nruns << " iterations" << endl;
    t.restart();
    for (size_t n = 0; n < nruns; n++) {
      vector_dataset<>::record_iterator_type r_it = ds2.records().first;
      for (size_t i = 0; i < 1000; i++) {
        const std::vector<size_t>& f = (*r_it).finite();
        for (size_t j = 0; j < nvars2; j++)
          tmp += f[j];
        ++r_it;
      }
    }
    cout << " vector_dataset class: " << t.elapsed() / nruns << std::endl;
    t.restart();
    for (size_t n = 0; n < nruns; n++) {
      dataset_view<>::record_iterator_type r_it = ds_view_binarized.records().first;
      for (size_t i = 0; i < 1000; i++) {
        const std::vector<size_t>& f = (*r_it).finite();
        for (size_t j = 0; j < nvars2; j++)
          tmp += f[j];
        ++r_it;
      }
    }
    cout << " dataset_view (binarized): " << t.elapsed() / nruns << std::endl;

    // Do a timing test for merged variables for record iterator
    cout << "Now running timing test for views of merged variables:\n"
         << "Compute average times (over " << nruns << " iterations) of"
         << " iterating over 1000 records and using the one variable"
         << " (the merged variable for the view) in a sum.\n" << endl;
    cout << "Test: use record_iterator for " << nruns << " iterations" << endl;
    t.restart();
    for (size_t n = 0; n < nruns; n++) {
      vector_dataset<>::record_iterator_type r_it = ds2.records().first;
      for (size_t i = 0; i < 1000; i++) {
        const std::vector<size_t>& f = (*r_it).finite();
        tmp += f.back();
        ++r_it;
      }
    }
    cout << " vector_dataset class: " << t.elapsed() / nruns << std::endl;
    t.restart();
    for (size_t n = 0; n < nruns; n++) {
      dataset_view<>::record_iterator_type r_it = ds_view_merged1.records().first;
      for (size_t i = 0; i < 1000; i++) {
        const std::vector<size_t>& f = (*r_it).finite();
        tmp += f.back();
        ++r_it;
      }
    }
    cout << " dataset_view (merged 2 vars): " << t.elapsed()/nruns << std::endl;
    t.restart();
    for (size_t n = 0; n < nruns; n++) {
      dataset_view<>::record_iterator_type r_it = ds_view_merged2.records().first;
      for (size_t i = 0; i < 1000; i++) {
        const std::vector<size_t>& f = (*r_it).finite();
        tmp += f.back();
        ++r_it;
      }
    }
    cout << " dataset_view (merged " << merged2_orig_vars.size() << " vars): "
         << t.elapsed()/nruns << std::endl;

    // Do a timing test for merged variables for finite()
    cout << "Test: use finite() for " << nruns << " iterations" << endl;
    size_t ds2_last_f_var(ds2.num_finite() - 1);
    t.restart();
    for (size_t n = 0; n < nruns; n++)
      for (size_t i = 0; i < 1000; i++)
        tmp += ds2.finite(i, ds2_last_f_var);
    cout << " vector_dataset class: " << t.elapsed() / nruns << std::endl;
    size_t ds_view_merged1_last_f_var(ds_view_merged1.num_finite() - 1);
    t.restart();
    for (size_t n = 0; n < nruns; n++)
      for (size_t i = 0; i < 1000; i++)
        tmp += ds_view_merged1.finite(i, ds_view_merged1_last_f_var);
    cout << " dataset_view (merged 2 vars): " << t.elapsed()/nruns << std::endl;
    size_t ds_view_merged2_last_f_var(ds_view_merged2.num_finite() - 1);
    t.restart();
    for (size_t n = 0; n < nruns; n++)
      for (size_t i = 0; i < 1000; i++)
        tmp += ds_view_merged2.finite(i, ds_view_merged2_last_f_var);
    cout << " dataset_view (merged " << merged2_orig_vars.size() << " vars): "
         << t.elapsed()/nruns << std::endl;

    // Test save and load
    ds_view_binarized.save("dataset_view_timing.tmp1");
    dataset_view<> ds_view_binarized_reloaded(ds2);
    ds_view_binarized_reloaded.load("dataset_view_timing.tmp1", binary_var,
                                    NULL);
    cout << "First 6 records from ds_view_binarized and re-loaded version:\n";
    for (size_t i = 0; i < 6; i++) {
      cout << "Original: " << ds_view_binarized[i].finite() << "\n"
           << "Reloaded: " << ds_view_binarized_reloaded[i].finite() << "\n";
    }
    ds2.print_datasource_info();
    ds_view_merged2.save("dataset_view_timing.tmp2");
    dataset_view<> ds_view_merged2_reloaded(ds2);
    ds_view_merged2_reloaded.load("dataset_view_timing.tmp2", NULL,
                                  merged2_new_var);
    cout << "First 6 records from ds_view_merged2 and re-loaded version:\n";
    for (size_t i = 0; i < 6; i++) {
      cout << "Original: " << ds_view_merged2[i].finite() << "\n"
           << "Reloaded: " << ds_view_merged2_reloaded[i].finite() << "\n";
    }
  }

  // Test variable views
  dataset_view<> ds_view_vars(ds);
  std::set<size_t> vv_finite_indices;
  vv_finite_indices.insert(0);
  std::set<size_t> vv_vector_indices;
  vv_vector_indices.insert(1);
  vv_vector_indices.insert(2);
  vv_vector_indices.insert(3);
  vv_vector_indices.insert(4);
  vv_vector_indices.insert(5);
  ds_view_vars.set_variable_indices(vv_finite_indices, vv_vector_indices);
  cout << "First 6 records from ds and variable view with only "
       << "finite var 0 and vector vars 1-5:\n"
       << " Finite vals:\n";
  for (size_t i = 0; i < 6; i++) {
    cout << "Original: " << ds[i].finite() << "\n"
         << "Var view: " << ds_view_vars[i].finite() << "\n";
  }
  cout << " Vector vals:\n";
  for (size_t i = 0; i < 6; i++) {
    cout << "Original: " << ds[i].vector() << "\n"
         << "Var view: " << ds_view_vars[i].vector() << "\n";
  }

  if (argc > 3)
    cout << tmp;
  return 0;
}
