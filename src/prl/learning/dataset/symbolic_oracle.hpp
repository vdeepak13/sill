
#ifndef PRL_SYMBOLIC_ORACLE_HPP
#define PRL_SYMBOLIC_ORACLE_HPP

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <prl/global.hpp>
#include <prl/learning/dataset/oracle.hpp>
#include <prl/learning/dataset/symbolic.hpp>

#include <boost/tokenizer.hpp>

#include <prl/macros_def.hpp>

namespace prl {

  /**
   * Class for loading data from symbolic data files (text or binary).
   *
   * \author Joseph Bradley
   * \ingroup learning_dataset
   * @see symbolic.hpp
   */
  class symbolic_oracle : public oracle {

    // Public type declarations
    //==========================================================================
  public:

    //! Base class
    typedef oracle base;

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

    // Protected data members
    //==========================================================================
  protected:

    parameters params;

    symbolic::parameters sym_params;

    //! Current record
    record current_rec;

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

    //! Returns the current record.
    const record& current() const {
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

  }; // class symbolic_oracle

} // namespace prl

#include <prl/macros_undef.hpp>

#endif // #ifndef PRL_SYMBOLIC_ORACLE_HPP
