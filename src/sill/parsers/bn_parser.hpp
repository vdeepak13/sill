#ifndef SILL_BN_PARSER_HPP
#define SILL_BN_PARSER_HPP

#include <string>
#include <vector>
#include <iostream>
#include <istream>
#include <fstream>
#include <sstream>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/bayesian_network.hpp>
#include <sill/factor/table_factor.hpp>
//#include <sill/factor/log_table_factor.hpp>
#include <sill/macros_def.hpp>

namespace sill {

  typedef bayesian_network<table_factor>::factor_type factor_type;
  typedef bayesian_network<table_factor>::variable_type variable_type;

  /**
   * This function fills in a Bayesian network using the file
   * provided in filename.  The function returns true if parsing
   * succeeded.
   *
   * The file format is:
   * variables:
   * VARNAME1 // VAR1ARITY
   * ...
   * factors:
   * PARENTNAME1 / PARENTNAME2 / CHILDNAME // VALUES
   *
   * NOTE: Values are NOT stored in log form, unlike in the other model formats.
   */
  bool parse_bn(universe& u, bayesian_network<table_factor>& bn, 
                const std::string& filename) {
    // NOTE: the code below uses table_factor::table(). this code will only
    // work with table_factor and not log_table_factor
    typedef factor_type::table_type table_type;
    
    // Open an input file stream
    std::ifstream fin(filename.c_str());
    assert(fin.good());
    std::string line;
    size_t line_number = 1;
    
    // Read the first line which should be "variables:"
    assert(getline(fin,line,line_number));
    assert(line == "variables:");
    
    // Read all the variables and create a map from the variable name
    // (string) to the variable* prl variable pointer.
    std::map<std::string, variable_type*> variable_map;
    while(fin.good() && getline(fin, line, line_number) && line != "factors:") {
      size_t begin_arity = line.find_last_of('/')+1;
      size_t end_var_name = begin_arity - 2;
      std::string var_name = trim(line.substr(0,end_var_name));
      int arity;
      std::istringstream iss(line.substr(begin_arity));
      if (!(iss >> arity))
        assert(false);
      variable_type* v = u.new_finite_variable(var_name,(size_t)(arity));
      assert(v != NULL);
      variable_map[var_name] = v;
    }
    
    // Starting to read factors
    assert(line == "factors:");
    
    // track the max value and min value lines
    double max_value = -std::numeric_limits<double>::max();
    double min_value = std::numeric_limits<double>::max();
    size_t max_value_line = line_number;
    size_t min_value_line = line_number;
    
    while(fin.good() && getline(fin, line, line_number)) {
      // Process the arguments
      size_t end_of_variables = line.find_last_of('/')-1;
      std::vector<variable_type*> args;
      for(size_t i = 0; i < end_of_variables; 
          i = line.find_first_of('/', i) + 1) {
        std::string variable_name = 
          trim(line.substr(i, line.find_first_of('/',i) - i));
        variable_type* var = variable_map[variable_name];
        assert(var != NULL);
        args.push_back(var);
      }  
      variable_type* child_arg = args.back();
      // Initialize a factor and get the table
      factor_type factor(args, 0.0);
      table_type& tbl = factor.table();
      
      // This is really scarry MAKES CRAZY ASSUMPTION THAT TABLE
      // ORDERING AND TABLE FACTOR ITERATORS MATCH which might
      // actually currently be true
      table_type::iterator iter = tbl.begin();
      std::istringstream tbl_values;
      tbl_values.str(line.substr(line.find_last_of('/')+1)); 
      double value = 0.0;
      size_t valcnt = 0;
      for( ;
           tbl_values.good() && tbl_values >> value && iter != tbl.end();
           ++iter){
        // Get the weight from the value (NOT IN LOG FORM)
        double weight = value;
        // Assert that the weight is a probability.
        assert(weight >= 0 && weight <= 1);
        // For debugging purposes I track the lines with the highest and lowest
        // weight
        if(weight > max_value) {
          max_value = weight;
          max_value_line = line_number;
        }
        if(weight < min_value) {
          min_value = weight;
          min_value_line = line_number;
        }
        // Save the weight into the table for this factor
        *iter = weight;
        ++valcnt;
      }
      // Must be at end of table
      if (iter != tbl.end()) {
        std::cerr << "We should be at the end of the table but are not; "
                  << "we have loaded " << valcnt << " values, while the table "
                  << "has size " << tbl.size() << std::endl;
        std::cerr << "This error occurred while parsing line:\n"
                  << line << std::endl;
        assert(false);
      }
      // Must be at the end of the line
      assert(tbl_values.good() == false);
      // If this factor has the highest or smallest parameter so far
      // then print it out for debugging purposes
      if(max_value_line == line_number || min_value_line == line_number) {
        std::cout << "Extreme Factor: " << factor << std::endl;
        std::cout << "Max Value: " << max_value << " on line " << max_value_line << std::endl;
        std::cout << "Min Value: " << min_value << " on line " << min_value_line << std::endl;
        
      } // End of for(tbl_values ...)
      // Finished processing factor so add it to the model
      bn.add_factor(child_arg, factor);
    } // End of processing all factors should be end of file
    assert(fin.good() == false);
    fin.close();
    std::cout << "Max Value: " << max_value << " on line " << max_value_line << std::endl;
    std::cout << "Min Value: " << min_value << " on line " << min_value_line << std::endl;
    return true; // Parsing successful
  } // end of parse_bn method
  
  /**
   * This function fills in a Bayesian network using the file
   * provided in filename.  The function returns true if parsing
   * succeeded.
   * Parameter var_order gives variables to be used, in order.
   *
   * The file format is:
   * variables:
   * VARNAME1 // VAR1ARITY
   * ...
   * factors:
   * PARENTNAME1 / PARENTNAME2 / CHILDNAME // VALUES
   *
   * NOTE: Values are NOT stored in log form, unlike in the other model formats.
   */
  bool parse_bn(finite_var_vector& var_order, bayesian_network<table_factor>& bn, 
                const std::string& filename) {
    // Define types
    typedef factor_type::table_type table_type;
    
    // Open an input file stream
    std::ifstream fin(filename.c_str());
    assert(fin.good());
    std::string line;
    size_t line_number = 1;
    
    // Read the first line which should be "variables:"
    assert(getline(fin,line,line_number));
    assert(line == "variables:");
    
    // Read all the variables and create a map from the variable name
    // (string) to the variable* prl variable pointer.
    std::map<std::string, variable_type*> variable_map;
    size_t nvars = 0;
    while(fin.good() && getline(fin, line, line_number) && line != "factors:") {
      size_t begin_arity = line.find_last_of('/')+1;
      size_t end_var_name = begin_arity - 2;
      std::string var_name = trim(line.substr(0,end_var_name));
      int arity;
      std::istringstream iss(line.substr(begin_arity));
      if (!(iss >> arity))
        assert(false);
      ++nvars;
      assert(var_order.size() >= nvars);
      variable_type* v = var_order[nvars-1];
      assert(v != NULL);
      assert(v->size() == (size_t)(arity));
      variable_map[var_name] = v;
    }
    
    // Starting to read factors
    assert(line == "factors:");
    
    // track the max value and min value lines
    double max_value = -std::numeric_limits<double>::max();
    double min_value = std::numeric_limits<double>::max();
    size_t max_value_line = line_number;
    size_t min_value_line = line_number;
    
    while(fin.good() && getline(fin, line, line_number)) {
      // Process the arguments
      size_t end_of_variables = line.find_last_of('/')-1;
      std::vector<variable_type*> args;
      for(size_t i = 0; i < end_of_variables; 
          i = line.find_first_of('/', i) + 1) {
        std::string variable_name = 
          trim(line.substr(i, line.find_first_of('/',i) - i));
        variable_type* var = variable_map[variable_name];
        assert(var != NULL);
        args.push_back(var);
      }  
      variable_type* child_arg = args.back();
      // Initialize a factor and get the table
      factor_type factor(args, 0.0);
      table_type& tbl = factor.table();
      
      // This is really scarry MAKES CRAZY ASSUMPTION THAT TABLE
      // ORDERING AND TABLE FACTOR ITERATORS MATCH which might
      // actually currently be true
      table_type::iterator iter = tbl.begin();
      std::istringstream tbl_values;
      tbl_values.str(line.substr(line.find_last_of('/')+1)); 
      double value = 0.0;
      size_t valcnt = 0;
      for( ;
           tbl_values.good() && tbl_values >> value && iter != tbl.end();
           ++iter){
        // Get the weight from the value (NOT IN LOG FORM)
        double weight = value;
        // Assert that the weight is a probability.
        assert(weight >= 0 && weight <= 1);
        // For debugging purposes I track the lines with the highest and lowest
        // weight
        if(weight > max_value) {
          max_value = weight;
          max_value_line = line_number;
        }
        if(weight < min_value) {
          min_value = weight;
          min_value_line = line_number;
        }
        // Save the weight into the table for this factor
        *iter = weight;
        ++valcnt;
      }
      // Must be at end of table
      if (iter != tbl.end()) {
        std::cerr << "We should be at the end of the table but are not; "
                  << "we have loaded " << valcnt << " values, while the table "
                  << "has size " << tbl.size() << std::endl;
        std::cerr << "This error occurred while parsing line:\n"
                  << line << std::endl;
        assert(false);
      }
      // Must be at the end of the line
      assert(tbl_values.good() == false);
      // If this factor has the highest or smallest parameter so far
      // then print it out for debugging purposes
      if(max_value_line == line_number || min_value_line == line_number) {
        std::cout << "Extreme Factor: " << factor << std::endl;
        std::cout << "Max Value: " << max_value << " on line " << max_value_line << std::endl;
        std::cout << "Min Value: " << min_value << " on line " << min_value_line << std::endl;
        
      } // End of for(tbl_values ...)
      // Finished processing factor so add it to the model
      bn.add_factor(child_arg, factor);
    } // End of processing all factors should be end of file
    assert(fin.good() == false);
    fin.close();
    std::cout << "Max Value: " << max_value << " on line " << max_value_line << std::endl;
    std::cout << "Min Value: " << min_value << " on line " << min_value_line << std::endl;
    return true; // Parsing successful
  } // end of parse_bn method

} // namespace sill

#include <sill/macros_undef.hpp>

#endif
