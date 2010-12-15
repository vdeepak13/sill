#ifndef PRL_DECOMPOSABLE_PARSER_HPP
#define PRL_DECOMPOSABLE_PARSER_HPP

#include <string>
#include <vector>
#include <iostream>
#include <istream>
#include <fstream>
#include <sstream>

#include <prl/base/universe.hpp>
#include <prl/factor/table_factor.hpp>
#include <prl/model/decomposable.hpp>

#include <prl/macros_def.hpp>
namespace prl {

  typedef decomposable<table_factor>::factor_type factor_type;
  typedef decomposable<table_factor>::variable_type variable_type;

  /**
   * Removes trailing and leading white space from a string
   */
  std::string trim(const std::string& str) {
    std::string::size_type pos1 = str.find_first_not_of(" \t");
    std::string::size_type pos2 = str.find_last_not_of(" \t");
    return str.substr(pos1 == std::string::npos ? 0 : pos1, 
                      pos2 == std::string::npos ? str.size()-1 : pos2-pos1+1);
  }
  

  /**
   * same as the stl string get line but this also increments a line
   * counter which is useful for debugging purposes
   */
  std::istream& getline(std::ifstream& fin, std::string& line,
                        size_t& line_number) {
    ++line_number;
    return getline(fin, line); 
  }

  /**
   * This function fills in a decomposable model using the file
   * provided in filename.  The function returns true if parsing
   * succeeded.
   */
  bool parse_decomposable(universe& universe,
                          decomposable<table_factor>& fg, 
                          const std::string& filename) {
    // NOTE: the code below uses table_factor::table(). this code will only
    // work with table_factor and not log_table_factor
    
    // Define types
    typedef factor_type::table_type table_type;
    
    // Open an input file stream
    std::ifstream fin(filename.c_str());
    assert(fin.good());
    std::string line;
    size_t line_number = 1;
    
    // Read the first line which should be "variable:"
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
      variable_type* v = universe.new_finite_variable(var_name,(size_t)(arity));
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
      for( ;
           tbl_values.good() && tbl_values >> value && iter != tbl.end();
           ++iter){
        // Get the weight from the value (recall that the features are stored
        // in log form
        double weight = exp(value);
        // Assert that the weight is positive
        assert(weight > 0);
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
      }
      // Must be at end of table
      assert(iter == tbl.end());
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
      fg *= factor;
    } // End of processing all factors should be end of file
    assert(fin.good() == false);
    fin.close();
    std::cout << "Max Value: " << max_value << " on line " << max_value_line << std::endl;
    std::cout << "Min Value: " << min_value << " on line " << min_value_line << std::endl;
    return true; // Parsing successful
  } // end of parse_decomposable method
  
} // End of namespace
#include <prl/macros_undef.hpp>


#endif // PRL_DECOMPOSABLE_PARSER_HPP
