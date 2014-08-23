#ifndef SILL_SYMBOLIC_ORACLE_HPP
#define SILL_SYMBOLIC_ORACLE_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <sill/global.hpp>
#include <sill/learning/dataset_old/oracle.hpp>
#include <sill/learning/dataset_old/symbolic.hpp>

#include <boost/tokenizer.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Class for loading data from symbolic data files (text or binary).
   *
   * \author Joseph Bradley
   * \ingroup learning_dataset
   * @see symbolic.hpp
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<double,size_t>)
   */
  template <typename LA = dense_linear_algebra<> >
  class symbolic_oracle : public oracle<LA> {

    // Public type declarations
    //==========================================================================
  public:

    //! Base class
    typedef oracle<LA> base;

    typedef LA la_type;

    typedef record<la_type> record_type;

    struct parameters {

      //! Limit on the number of records which may be drawn from the oracle;
      //! if 0, then set to # examples in dataset file.
      //!  (default = # examples in dataset file)
      size_t record_limit;

      //! If true, then automatically resets to the first record when the
      //! end of the dataset is reached (until record_limit is reached)
      //!  (default = false)
      bool auto_reset;

      parameters()
        : record_limit(0), auto_reset(false) { }

    protected:

      friend class symbolic_oracle;

      void set_parameters(size_t ds_size) {
        if (record_limit == 0) {
          if (ds_size == 0) // then probably unknown size
            record_limit = std::numeric_limits<size_t>::max();
          else
            record_limit = ds_size;
        }
      }

    }; // class parameters

    // Constructors
    //==========================================================================
  public:

    //! Default constructor
    symbolic_oracle()
      : base(finite_var_vector(), vector_var_vector(),
             std::vector<variable::variable_typenames>()),
        bin_f_in(NULL), finite_buffer(NULL), vector_buffer(NULL) {
    }

    /**
     * Constructor for a symbolic file format oracle which uses existing
     * variables; these variables are assumed to be in the same order and
     * of the same arity as in the dataset.
     *
     * NOTE: next() must be called to load the first record.
     *
     * @param filename        data file name
     * @param info            Struct defining the dataset structure.
     * @param params          oracle parameters
     * @see symbolic.hpp
     */
    symbolic_oracle(const std::string& filename,
                    const datasource_info_type& info,
                    parameters params = parameters())
      : base(info), params(params),
        current_rec(finite_numbering_ptr_, vector_numbering_ptr_, dvector),
        records_used_(0),
	bin_f_in(NULL), finite_buffer(NULL), vector_buffer(NULL) {
      this->sym_params.data_filename = filename;
      this->sym_params.dataset_name = filename;
      this->sym_params.datasource_info = info;
      init();
    }

    /**
     * Constructor for a symbolic file format oracle which uses existing
     * variables; these variables are assumed to be in the same order and
     * of the same arity as in the dataset.
     *
     * NOTE: next() must be called to load the first record.
     *
     * @param sym_params      symbolic parameters specifying dataset
     * @param params          oracle parameters
     * @see symbolic.hpp
     */
    symbolic_oracle(const symbolic::parameters& sym_params,
                    parameters params = parameters())
      : base(sym_params.datasource_info), params(params),
        sym_params(sym_params),
        current_rec(finite_numbering_ptr_, vector_numbering_ptr_, dvector),
        records_used_(0),
	bin_f_in(NULL), finite_buffer(NULL), vector_buffer(NULL) {
      init();
    }

    //! Copy constructor
    symbolic_oracle(const symbolic_oracle& o)
      : base(o.finite_list(), o.vector_list(), o.variable_type_order()),
        params(o.params), sym_params(o.sym_params),
	current_rec(o.current_rec), current_weight(o.current_weight),
	records_used_(o.records_used_),
        prefix_length(o.prefix_length), line(o.line), sep(o.sep),
	bin_f_in(NULL), finite_buffer(NULL), vector_buffer(NULL) {
      copy_everything(o);
    }

    //! Destructor.
    ~symbolic_oracle() {
      free_everything();
    }

    // Getters and helpers
    //==========================================================================

    // From datasource
    using base::num_finite;
    using base::num_vector;
    using base::variable_type_order;
    using base::var_order;
    //    using base::var_order_index;
    //    using base::variable_index;
    using base::record_index;
    using base::vector_indices;
    using base::finite_numbering;
    using base::finite_numbering_ptr;
    using base::vector_numbering;
    using base::vector_numbering_ptr;

    //! Returns the current record.
    const record_type& current() const {
      return current_rec;
    }

    //! Returns the weight of the current example.
    double weight() const {
      return current_weight;
    }

    const std::string& dataset_name() const {
      return sym_params.dataset_name;
    }

    //! Returns the example limit of the oracle, or infinity if none.
    size_t limit() const {
      return params.record_limit;
    }

    //! Returns the number of records used so far (including current record)
    size_t records_used() const {
      return records_used_;
    }

    // Mutating operations
    //==========================================================================

    //! Assignment operator.
    symbolic_oracle& operator=(const symbolic_oracle& o);

    //! Increments the oracle to the next record.
    //! This returns true iff the operation was successful; false indicates
    //! a depleted oracle.
    bool next();

    //! Returns the oracle to its original state.
    //! The current record may not be updated; next() must be called to load
    //! the first record.
    void reset() {
      restart_oracle();
      records_used_ = 0;
    }

    // Protected data members
    //==========================================================================
  protected:

    // From datasource
    //    using base::finite_vars;
    using base::finite_seq;
    using base::finite_numbering_ptr_;
    using base::dfinite;
    using base::finite_class_vars;
    //    using base::vector_vars;
    using base::vector_seq;
    using base::vector_numbering_ptr_;
    using base::dvector;
    using base::vector_class_vars;
    using base::var_type_order;
    //    using base::var_order_map;
    //    using base::vector_var_order_map;

    parameters params;

    symbolic::parameters sym_params;

    //! Current record
    record_type current_rec;

    //! Current record weight
    double current_weight;

    //! Count of number of examples drawn
    size_t records_used_;

    // Data members used for parsing text data files
    //----------------------------------------------

    //! Data file for text data.
    //! This is declared mutable b/c it's necessary for the copy constructor
    //! (which must call f_in.tellg()).
    mutable std::ifstream f_in;

    //! Copied from symbolic parameters
    size_t prefix_length;

    //! Temporary used to avoid repeated allocation.
    std::string line;

    //! Tokenizer for reading data file
    typedef boost::tokenizer< boost::char_separator<char> > tokenizer;

    //! Separator used by tokenizer
    boost::char_separator<char> sep;

    // Data members used for parsing binary data files
    //----------------------------------------------

    //! Data file for binary data.
    FILE* bin_f_in;

    //! Buffer for binary data -- finite values
    size_t* finite_buffer;

    //! Buffer for binary data -- vector values
    double* vector_buffer;

    // Protected methods
    //==========================================================================

    void restart_oracle();

    //! Initialize oracle.
    void init();

    //! Read 1 record from a text data file.
    bool next_text();

    //! Read 1 record from a binary data file.
    bool next_binary();

    //! Free pointers and close files.
    void free_everything();

    //! Copy pointers and files.
    void copy_everything(const symbolic_oracle& o);

  }; // class symbolic_oracle

  //============================================================================
  // Implementations of methods in symbolic_oracle
  //============================================================================

  // Mutating operations
  //============================================================================

  template <typename LA>
  symbolic_oracle<LA>&
  symbolic_oracle<LA>::operator=(const symbolic_oracle& o) {
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

  template <typename LA>
  bool symbolic_oracle<LA>::next() {
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

  // Protected methods
  //==========================================================================

  template <typename LA>
  void symbolic_oracle<LA>::restart_oracle() {
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

  template <typename LA>
  void symbolic_oracle<LA>::init() {
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
  template <typename LA>
  bool symbolic_oracle<LA>::next_text() {
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
  template <typename LA>
  bool symbolic_oracle<LA>::next_binary() {
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

  template <typename LA>
  void symbolic_oracle<LA>::free_everything() {
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

  template <typename LA>
  void symbolic_oracle<LA>::copy_everything(const symbolic_oracle& o) {
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

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_SYMBOLIC_ORACLE_HPP
