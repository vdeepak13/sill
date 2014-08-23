namespace sill {

  struct symbolic_format {
    // options (separator, prefix, skiplines, weighted position)
    // mapping from finite variables to enum values
    // parse summary file
    var_vector vars; // the variables in the natural order -- do we have it here?
    // parse finite variable
    
    load_summary(const std::string& filename, universe& u, var_vector& vars);
    save_summary(const std::string& filename);
    load_config();
    save_config();
  };

  // row parser
  template <typename T>
  finite_row_parser {
    finite_var_vector vars; // ordering of variables
    format format;
    double operator()(const std::string& row, std::vector<T>& values);
  };
  
  template <typename T>
  vector_row_parser {
    vector_var_vector v;
    format format;
    T operator()(const std::string& row, std::vector<T>& values);
  };

  template <typename Tf, typename Tv>
  hybrid_row_parser {
    var_vector v;
    format format;
    Tv operator()(const std::string& row, ...);
  };


  // finite_dataset_io
  // vector_dataset_io
  // hybrid_dataset_io

  void load(std::iarchive& in, const symbolic_format& params, finite_dataset<T>& data) {
    
  }

  void load(std::iarchive& in, const symbolic_format& params, vector_dataset<T>& data) {
    // super simple - just parse each row into std::vector<T>
  }
  
  load(const std::string& filename, const symbolic_format& params, finite_dataset<T>& data);
  load(const std::string& filename, const symbolic_format& params, vector_dataset<T>& data);
  load(const std::string& filename, const symbolic_format& params, hybrid_dataset<T>& data);
  
  save(const std::string& filename, const symbolic_format& params, ...);

} // namespace sill



  
