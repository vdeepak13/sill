#include <prl/learning/dataset/symbolic_oracle.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  // Protected methods
  //==========================================================================

  void symbolic_oracle::restart_oracle() {
    switch (sym_params.format) {
    case 0:
      f_in.clear();
      f_in.seekg(0,std::ios_base::beg);
      assert(f_in.good());
      // Skip first lines as necessary
      for (size_t i = 0; i < sym_params.skiplines; i++)
	getline(f_in, line);
      break;
    case 1:
      {
	int status = fseek(bin_f_in, 0, SEEK_SET);
	assert(status == 0);
      }
      break;
    default:
      assert(false);
    }
  }

  void symbolic_oracle::init() {
    params.set_parameters(sym_params.nrecords);

    /*
    std::vector<size_t>
      order2fvorder(sym_params.variable_type_ordering.size());
    size_t nf = 0, nv = 0;
    for (size_t j = 0; j < sym_params.variable_type_ordering.size(); ++j) {
      switch(sym_params.variable_type_ordering[j]) {
      case variable::FINITE_VARIABLE:
	order2fvorder[j] = nf;
	++nf;
	break;
      case variable::VECTOR_VARIABLE:
	order2fvorder[j] = nv;
	++nv;
	break;
      default:
	assert(false);
      }
    }

    foreach(size_t j, sym_params.class_variables) {
      if (j >= num_variables()) {
	std::cerr << "symbolic_oracle: The symbolic data format parameters "
		  << "give " << j << " as a class variable index (from 0), "
		  << "but the symbolic_oracle only has "
		  << num_variables() << " variables." << std::endl;
	assert(false);
      }
      switch(sym_params.variable_type_ordering[j]) {
      case variable::FINITE_VARIABLE:
	finite_class_vars.push_back(finite_seq[order2fvorder[j]]);
	break;
      case variable::VECTOR_VARIABLE:
	vector_class_vars.push_back(vector_seq[order2fvorder[j]]);
	break;
      default:
	assert(false);
      }
    }
    */

    switch (sym_params.format) {
    case 0:
      f_in.open(sym_params.data_filename.c_str());
      assert(f_in.good());
      sep = boost::char_separator<char>(sym_params.separator.c_str());
      prefix_length = sym_params.prefix.size();
      // Skip first lines as necessary
      for (size_t i = 0; i < sym_params.skiplines; i++)
	getline(f_in, line);
      break;
    case 1:
      bin_f_in = fopen(sym_params.data_filename.c_str(), "r");
      assert(bin_f_in != NULL);
      finite_buffer = new size_t[finite_seq.size()];
      vector_buffer = new double[dvector];
      break;
    default:
      assert(false);
    }

    if (!sym_params.weighted)
      current_weight = 1;
  } // init()

  //! Read 1 record from a text data file.
  bool symbolic_oracle::next_text() {
    if (f_in.bad() || records_used_ >= params.record_limit)
      return false;
    if (f_in.eof()) {
      if (params.auto_reset)
	restart_oracle();
      else
	return false;
    }
    getline(f_in, line);
    if (line.size() == 0) {
      if (!(params.auto_reset))
	return false;
      restart_oracle();
      getline(f_in, line);
    }
    std::istringstream is;
    tokenizer tokens(line, sep);
    tokenizer::iterator token_iter = tokens.begin();
    for (size_t i = 0; i < sym_params.skipcols; i++) {
      assert(token_iter != tokens.end());
      ++token_iter;
    }

    size_t nf = 0, nv = 0;
    for (size_t j = 0; j < var_type_order.size(); j++) {
      switch(var_type_order[j]) {
      case variable::FINITE_VARIABLE: {
	finite_variable* f = finite_seq[nf];
	if (token_iter == tokens.end()) {
	  if (j > 0)
	    std::cerr << "Record had " << j << " out of "
		      << var_type_order.size() << " variable values.  "
		      << "Does the summary file specify the right number of "
		      << "variables?" << std::endl;
	  else
	    std::cerr << "Is there unnecessary whitespace on this line: \""
		      << line << "\"?" << std::endl;
	  assert(false);
	}
	size_t val;
	is.clear();
	is.str((*token_iter).substr(prefix_length));
	if (!(is >> val)) {
	  std::cerr << "Finite variable (index " << j
		    << ") had value which could not be read as a size_t."
		    << std::endl;
	  assert(false);
	}
	val -= sym_params.index_base;
	if (val >= f->size()) {
	  std::cerr << "Finite variable (index " << j << ") has size "
		    << f->size() << " but value " << val << " in record "
		    << records_used_ << ". "
		    << " Check to make sure the summary file's variable"
		    << " ordering matches the data file." << std::endl;
	  std::cerr << "In case it is helpful, ";
	  if (token_iter == tokens.begin())
	    std::cerr << "this was the first token on the line." << std::endl;
	  else {
	    ++token_iter;
	    if (token_iter != tokens.end())
	      std::cerr << "the next token is \"" << (*token_iter)
			<< "\"." << std::endl;
	    else
	      std::cerr << "this was the last token on the line."
			<< std::endl;
	  }
	  assert(false);
	}
	current_rec.fin_ptr->operator[](finite_numbering_ptr_->operator[](f))
	  = val;
	++token_iter;
	++nf;
	break;
      }
      case variable::VECTOR_VARIABLE: {
	vector_variable* v = vector_seq[nv];
	for (size_t k = vector_numbering_ptr_->operator[](v);
	     k < vector_numbering_ptr_->operator[](v) + v->size();
	     k++) {
	  if (token_iter == tokens.end()) {
	    if (j > 0)
	      std::cerr << "Record had " << j << " out of "
			<< var_type_order.size() << " variable values.  "
			<< "Does the summary file specify the right number "
			<< "of variables?" << std::endl;
	    else
	      std::cerr << "Is there unnecessary whitespace on this line: \""
			<< line << "\"?" << std::endl;
	    assert(false);
	  }
	  is.clear();
	  is.str((*token_iter).substr(prefix_length));
	  if (!(is >> current_rec.vec_ptr->operator[](k)))
	    assert(false);
	  ++token_iter;
	}
	++nv;
	break;
      }
      default:
	assert(false);
      }
    }
    if (sym_params.weighted) {
      is.clear();
      is.str(*token_iter);
      if (!(is >> current_weight))
	assert(false);
    }
    ++records_used_;
    return true;
  } // next_text()

  //! Read 1 record from a binary data file.
  bool symbolic_oracle::next_binary() {
    if (!bin_f_in || records_used_ >= params.record_limit)
      return false;
    if (feof(bin_f_in)) {
      if (params.auto_reset)
	restart_oracle();
      else
	return false;
    }

    assert(finite_buffer);
    size_t rc =
      fread(finite_buffer, sizeof(size_t), finite_seq.size(), bin_f_in);
    if (rc != finite_seq.size()) {
      std::cerr << "Incomplete record (" << rc << " of "
		<< finite_seq.size() << ") finite variable values.  "
		<< "Does the summary file specify the right number of "
		<< "variables?" << std::endl;
      assert(false);
    }
    for (size_t j(0); j < finite_seq.size(); ++j)
      current_rec.fin_ptr->operator[](j) = finite_buffer[j];

    assert(vector_buffer);
    rc = fread(vector_buffer, sizeof(double), dvector, bin_f_in);
    if (rc != dvector) {
      std::cerr << "Incomplete record (" << rc << " of "
		<< dvector << ") vector variable values.  "
		<< "Does the summary file specify the right number of "
		<< "variables?" << std::endl;
      assert(false);
    }
    for (size_t j(0); j < dvector; ++j)
      current_rec.vec_ptr->operator[](j) = vector_buffer[j];

    if (sym_params.weighted) {
      std::cerr << "Binary datasets do not yet support weighted records."
		<< std::endl;
      assert(false);
    }
    ++records_used_;
    return true;
  } // next_binary()

  void symbolic_oracle::free_everything() {
    if (f_in.is_open())
      f_in.close();
    if (bin_f_in) {
      fclose(bin_f_in);
      bin_f_in = NULL;
    }
    if (finite_buffer) {
      delete [] finite_buffer;
      finite_buffer = NULL;
    }
    if (vector_buffer) {
      delete [] vector_buffer;
      vector_buffer = NULL;
    }
  }

  void symbolic_oracle::copy_everything(const symbolic_oracle& o) {
    switch (sym_params.format) {
    case 0:
      f_in.open(sym_params.data_filename.c_str());
      f_in.seekg(o.f_in.tellg());
      break;
    case 1:
      bin_f_in = fopen(sym_params.data_filename.c_str(), "r");
      assert(bin_f_in != NULL);
      if (o.bin_f_in) {
	int status = fseek(bin_f_in, ftell(o.bin_f_in), SEEK_SET);
	assert(status == 0);
      }
      if (o.finite_buffer)
	finite_buffer = new size_t[finite_seq.size()];
      if (o.vector_buffer)
	vector_buffer = new double[dvector];
      break;
    default:
      assert(false);
    }
  }

  // Mutating operations
  //============================================================================

  symbolic_oracle& symbolic_oracle::operator=(const symbolic_oracle& o) {
    datasource::operator=(o);
    params = o.params;
    sym_params = o.sym_params;
    current_rec = o.current_rec;
    current_weight = o.current_weight;
    records_used_ = o.records_used_;
    prefix_length = o.prefix_length;
    line = o.line;
    sep = o.sep;
    free_everything();
    copy_everything(o);
    return *this;
  }

  bool symbolic_oracle::next() {
    switch (sym_params.format) {
    case 0:
      return next_text();
    case 1:
      return next_binary();
    default:
      assert(false);
      return false;
    }
  } // next()

} // namespace prl

#include <prl/macros_undef.hpp>
