#ifndef SILL_ALCHEMY_HPP
#define SILL_ALCHEMY_HPP

#include <string>
#include <vector>
#include <iostream>
#include <istream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <dirent.h>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/canonical_table.hpp>
#include <sill/model/lifted_factor_graph_model.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/parsers/string_functions.hpp>


#include <sill/macros_def.hpp>
namespace sill {

  /**
   * Print a single factor as a single line of text in the alchemy
   * factor format.
   */
  void print_alchemy(const canonical_table & f,
                     std::ostream& out){
    bool first_line = true;
    foreach(const finite_variable* v, f.arg_vector()){
      if(!first_line) out << " ";
      out << v.name() << " /";
      first_line = false;
    }
    out << "/ ";
    foreach(logarithmic<double> entry, f.values()) {
      out << log(entry) << " ";
    }
    out << std::endl;
  } // print alchemy format for table factors


  void print_alchemy(const table_factor & f,
                     std::ostream& out){
    bool first_line = true;
    foreach(const finite_variable* v, f.arg_vector()){
      if(!first_line) out << " ";
      out << v.name() << " /";
      first_line = false;
    }
    out << "/ ";
    foreach(double entry, f.values()) {
      out << log(entry) << " ";
    }
    out << std::endl;
  } // print alchemy format for table factors


  /**
   * Print an entire factor graph in alchemy format.
   */
  template<typename Factor>
  void print_alchemy(const factor_graph_model<Factor>& fg,
                     std::ostream& out) {
    // Print the variables
    out << "variables:" << std::endl;
    foreach(const finite_variable* v, fg.arguments()) {
      out << v.name() << '\t' << v.size() << std::endl;
    } // end of foreach variable
    // Print the factors in alchemy format
    out << "factors:" << std::endl;
    foreach(const Factor& factor, fg.factors()) {
      print_alchemy(factor, out);
    }
  } // end of print_alchemy_format for factor graph


  /**
   * Read a single factor from the line of text filling in the factor
   * argument.  This function makes insanely strong assumptions so
   * read the code before using.  In particular this function assumes
   * that the entries in the file are in the same order that they
   * should be in the linear table representing the factor.
   */
  void parse_alchemy_factor(canonical_table& factor,
                      std::vector<size_t> &weight,
                     const std::string& line){
  // NOTE: the code below uses canonical_table::table(). this code will only
    // work with canonical_table and not table_factor
    
    typedef canonical_table factor_type;
    typedef factor_type::table_type table_type;

    table_type& tbl = factor.table();
    // This is really scarry MAKES CRAZY ASSUMPTION THAT TABLE
    // ORDERING AND TABLE FACTOR ITERATORS MATCH which might
    // actually currently be true
    table_type::iterator iter = tbl.begin();
    std::istringstream tbl_values;
    size_t weightpos = line.find("///");
    if (weightpos == std::string::npos) {
      tbl_values.str(line.substr(line.find("//") + 2));
    }
    else {
      size_t startpos = line.find("//") + 2;
      tbl_values.str(line.substr(startpos, weightpos - startpos));
    }
    // Loop through the line processing each entry
    for(double value = 0.0;
         tbl_values.good() && tbl_values >> value && iter != tbl.end();
         ++iter){
      // Get the weight from the value (recall that the features are
      // stored in log form
      double val = exp(value);
      // Assert that the weight is positive
      assert(val > 0);
      // Save the weight into the table for this factor
      *iter = val;
    }
    // Must be at end of table
    assert(iter == tbl.end());
    // Must be at the end of the line
    assert(tbl_values.good() == false);
    
    weight.clear();
    if (weightpos != std::string::npos) {
      std::istringstream weightvalues;
      weightvalues.str(line.substr(weightpos+3));
      while(weightvalues.good()) {
        size_t weightval;
        weightvalues >> weightval;
        weight.push_back(weightval);
      }
    }
  } // end of read alchemy format



  /**
   * This function fills in a factor graph with with alchemy file
   * provided in filename.  The function returns true if parsing
   * succeeded.
   */
  template <typename F>
  bool parse_alchemy(universe& universe,
                    factor_graph_model<F>& fg,
                    const std::string& filename) {
    // Define types
    typedef typename factor_graph_model<F>::variable_type
      variable_type;

    // Open an input file stream
    std::ifstream fin(filename.c_str());
    assert(fin.good());
    std::string line;
    size_t line_number = 1;

    // Read the first line which should be "variable:"
    assert(getline(fin,line,line_number));
    line = trim(line);
    assert(line == "variables:");
    bool warnprinted = false;
    // Read all the variables and create a map from the variable name
    // (string) to the variable* prl variable pointer.
    std::map<std::string, variable_type> variable_map;
    while(fin.good() &&
          getline(fin, line, line_number) &&
          trim(line) != "factors:") {
      // Separate into name and size
      line = trim(line);
      assert(line.length() > 0);
      size_t namelen = line.find_last_of('\t');
      size_t varsize = 2;
      // if their is a '\t' character then the variable size follows it
      if(namelen != std::string::npos) {
        std::stringstream istrm(trim(line.substr(namelen)));
        istrm >> varsize;
      }
      // Get the variable name
      std::string var_name = trim(line.substr(0, namelen));
      // Create a new finite variable in the universe
      variable_type v = universe.new_finite_variable(var_name, varsize);
      assert(v != NULL);
      // Store the variable in the local variable map
      variable_map[var_name] = v;
    }

    // Starting to read factors
    assert(trim(line) == "factors:");

    // track the max value and min value lines.  We use these to
    // assess dynamic range when parsing. Probably not necessary here.
    double max_value = -std::numeric_limits<double>::max();
    double min_value = std::numeric_limits<double>::max();
    size_t max_value_line = line_number;
    size_t min_value_line = line_number;

    while(fin.good() && getline(fin, line, line_number)) {
      /// if the line is empty skip it
      if(trim(line).length() == 0) continue;


      // the factor being read may contain the same variable multiple
      // times to account for that, we first read a temporary factors,
      // making every variable unique artificially, and then convert
      // it to the factor we actually need

      // Process the arguments
      size_t end_of_variables = line.find("//")-1;
      std::vector<variable_type>
        tmp_args,  // temporary arguments not as a set
        args_as_read_from_file; // exact arguments read from the file



      // the map "true variable" --> its copies in the temporary factor
      std::map<variable_type, std::vector<variable_type> >
        var_occurence_map;

      // Read in all the variables in the factor and store them
      for(size_t i = 0; i < end_of_variables;
          i = line.find_first_of('/', i) + 1) {
        // Read the next variable as a string
        std::string variable_name =
          trim(line.substr(i, line.find_first_of('/',i) - i));
        // Look up the variable in the variable map
        variable_type var = variable_map[variable_name];
        // The variable must have already been defined in the preamble
        // containing all the variables
        assert(var != NULL);
        // Save the arguments read from the file
        args_as_read_from_file.push_back(var);
        // If we have already read this variable before then we create
        // a new unique variable for this copy
        if( !var_occurence_map[var].empty() ) {
          // create a unique temporary variable for each duplicate
          // occurrence
          variable_type unique_copy =
            universe.new_finite_variable(var->size());
          // map the original variable to this copy (there may be
          // others)
          var_occurence_map[var].push_back(unique_copy);
          // Construct a temporary set of argument
          tmp_args.push_back(unique_copy);
        } else {
          // Store the variable
          var_occurence_map[var].push_back(var);
          // otherwise keep the original
          tmp_args.push_back(var);
        }
      } // end of first pass through variable
      
      // Fill in a factor
      F factor;
      std::vector<size_t> weights;
      if(args_as_read_from_file.size() == var_occurence_map.size()){
        // all the variables read from file are unique, no need to
        // remap.  just read the weights directly from the file this
        // is just an efficiency optimization
        factor = F(args_as_read_from_file, 0.0);
        parse_alchemy_factor(factor, weights,line);
      } else { //non-unique vars, read to temp first and then remap
        // First read table from file
        F temp_factor(tmp_args, 0.0);
        parse_alchemy_factor(temp_factor, weights,line);
        // remapping taking duplicate values into account
        std::set<variable_type> args = keys(var_occurence_map);
        factor = F(args, 0.0);

        // Loop through all possible assignments.  For each assignment
        // if the assignment is consistent with all duplicated
        // variables then write the assignment plus value to the true
        // factor
        foreach(const finite_assignment& tmp_assg,
                temp_factor.assignments()){
          // see if the duplicate copy of each duplicated variable
          // has the same value in this assignment
          bool duplicates_consistent = true;
          // For all of the true variable arguments check against
          // duplicates
          foreach(variable_type unique_var, args){
            // If the unique variable only occurs once then its
            // trivially consistent
            if(var_occurence_map[unique_var].size() == 1) {
              continue;
            } else {
              // check each consequetive duplicate of a
              // unique_variable and see if has the same assignment as
              // the last and then by transitivity conclude that the
              // assignments is consistent
              for(size_t i=1;
                  i < var_occurence_map[unique_var].size() &&
                    duplicates_consistent;
                  i++ ) {
                if( safe_get(tmp_assg, var_occurence_map[unique_var][i-1]) !=
                    safe_get(tmp_assg, var_occurence_map[unique_var][i]) ) {
                  duplicates_consistent = false;
                } // end of if statement
              }
            }
          } // end of fore each
          // if this is duplicate consistent then we can record the
          // result in the new factor as the assignment.
          if(duplicates_consistent){
            factor.set_v(tmp_assg, temp_factor.v(tmp_assg));
          }
        } // end of loop over all tmp_assgs
      } // end of outer if statement for whether factor is unique

      // collect some additional statistics
      if(max_value < factor.maximum()){
        max_value = factor.maximum();
        max_value_line = line_number;
      }
      if(min_value > factor.minimum()){
        min_value = factor.minimum();
        min_value_line = line_number;
      }

      // Finished processing factor so add it to the factor graph
      fg.add_factor(factor);
      // if there are weights, make sure they are all one
      if(weights.size() > 0 && warnprinted == false) {
        assert(weights.size() == factor.arg_vector().size());
        for (size_t i = 0; i < weights.size(); ++i) {
          if (weights[i] != 1 && warnprinted == false) {
            std::cerr << "Warning. Loading Lifted model into regular factor_graph\n";
            warnprinted = true;
          }
        }
      }
      

    } // End of outer while loop over factors should be end of file

    assert(fin.good() == false);
    fin.close();

    // Extra output
//     std::cout << "Max Value: " << max_value
//               << " on line " << max_value_line << std::endl;
//     std::cout << "Min Value: " << min_value
//               << " on line " << min_value_line << std::endl;
//     std::cout << "Total variables: " << fg.arguments().size()
//               << std::endl;
    (void)max_value_line;
    (void)min_value_line;

    return true; // Parsing successful
  } // end of parse_alchemy method











  /**
   * This function fills in a lifted factor graph with with alchemy file
   * provided in filename.  The function returns true if parsing
   * succeeded.
   * TODO: Refactor the 2 versions of parse_alchemy
   */
  template <typename F>
  bool parse_alchemy(universe& universe,
                    lifted_factor_graph_model<F>& fg,
                    const std::string& filename) {
    // Define types
    typedef typename factor_graph_model<F>::variable_type
      variable_type;

    // Open an input file stream
    std::ifstream fin(filename.c_str());
    assert(fin.good());
    std::string line;
    size_t line_number = 1;

    // Read the first line which should be "variable:"
    assert(getline(fin,line,line_number));
    line = trim(line);
    assert(line == "variables:");

    // Read all the variables and create a map from the variable name
    // (string) to the variable* prl variable pointer.
    std::map<std::string, variable_type> variable_map;
    while(fin.good() &&
          getline(fin, line, line_number) &&
          trim(line) != "factors:") {
      // Separate into name and size
      line = trim(line);
      assert(line.length() > 0);
      size_t namelen = line.find_last_of('\t');
      size_t varsize = 2;
      // if their is a '\t' character then the variable size follows it
      if(namelen != std::string::npos) {
        std::stringstream istrm(trim(line.substr(namelen)));
        istrm >> varsize;
      }
      // Get the variable name
      std::string var_name = trim(line.substr(0, namelen));
      // Create a new finite variable in the universe
      variable_type v = universe.new_finite_variable(var_name, varsize);
      assert(v != NULL);
      // Store the variable in the local variable map
      variable_map[var_name] = v;
    }

    // Starting to read factors
    assert(trim(line) == "factors:");

    // track the max value and min value lines.  We use these to
    // assess dynamic range when parsing. Probably not necessary here.
    double max_value = -std::numeric_limits<double>::max();
    double min_value = std::numeric_limits<double>::max();
    size_t max_value_line = line_number;
    size_t min_value_line = line_number;

    while(fin.good() && getline(fin, line, line_number)) {
      /// if the line is empty skip it
      if(trim(line).length() == 0) continue;


      // the factor being read may contain the same variable multiple
      // times to account for that, we first read a temporary factors,
      // making every variable unique artificially, and then convert
      // it to the factor we actually need

      // Process the arguments
      size_t end_of_variables = line.find("//")-1;
      std::vector<variable_type>
        tmp_args,  // temporary arguments not as a set
        args_as_read_from_file; // exact arguments read from the file



      // the map "true variable" --> its copies in the temporary factor
      std::map<variable_type, std::vector<variable_type> >
        var_occurence_map;

      // Read in all the variables in the factor and store them
      for(size_t i = 0; i < end_of_variables;
          i = line.find_first_of('/', i) + 1) {
        // Read the next variable as a string
        std::string variable_name =
          trim(line.substr(i, line.find_first_of('/',i) - i));
        // Look up the variable in the variable map
        variable_type var = variable_map[variable_name];
        // The variable must have already been defined in the preamble
        // containing all the variables
        assert(var != NULL);
        // Save the arguments read from the file
        args_as_read_from_file.push_back(var);
        // If we have already read this variable before then we create
        // a new unique variable for this copy
        if( !var_occurence_map[var].empty() ) {
          // create a unique temporary variable for each duplicate
          // occurrence
          variable_type unique_copy =
            universe.new_finite_variable(var->size());
          // map the original variable to this copy (there may be
          // others)
          var_occurence_map[var].push_back(unique_copy);
          // Construct a temporary set of argument
          tmp_args.push_back(unique_copy);
        } else {
          // Store the variable
          var_occurence_map[var].push_back(var);
          // otherwise keep the original
          tmp_args.push_back(var);
        }
      } // end of first pass through variable
      
      // Fill in a factor
      F factor;
      std::vector<size_t> weights;
      if(args_as_read_from_file.size() == var_occurence_map.size()){
        // all the variables read from file are unique, no need to
        // remap.  just read the weights directly from the file this
        // is just an efficiency optimization
        factor = F(args_as_read_from_file, 0.0);
        parse_alchemy_factor(factor, weights,line);
      } else { //non-unique vars, read to temp first and then remap
        // First read table from file
        F temp_factor(tmp_args, 0.0);
        parse_alchemy_factor(temp_factor, weights,line);
        // remapping taking duplicate values into account
        std::set<variable_type> args = keys(var_occurence_map);
        factor = F(args, 0.0);

        // Loop through all possible assignments.  For each assignment
        // if the assignment is consistent with all duplicated
        // variables then write the assignment plus value to the true
        // factor
        foreach(const finite_assignment& tmp_assg,
                temp_factor.assignments()){
          // see if the duplicate copy of each duplicated variable
          // has the same value in this assignment
          bool duplicates_consistent = true;
          // For all of the true variable arguments check against
          // duplicates
          foreach(variable_type unique_var, args){
            // If the unique variable only occurs once then its
            // trivially consistent
            if(var_occurence_map[unique_var].size() == 1) {
              continue;
            } else {
              // check each consequetive duplicate of a
              // unique_variable and see if has the same assignment as
              // the last and then by transitivity conclude that the
              // assignments is consistent
              for(size_t i=1;
                  i < var_occurence_map[unique_var].size() &&
                    duplicates_consistent;
                  i++ ) {
                if( safe_get(tmp_assg, var_occurence_map[unique_var][i-1]) !=
                    safe_get(tmp_assg, var_occurence_map[unique_var][i]) ) {
                  duplicates_consistent = false;
                } // end of if statement
              }
            }
          } // end of fore each
          // if this is duplicate consistent then we can record the
          // result in the new factor as the assignment.
          if(duplicates_consistent){
            factor.set_v(tmp_assg, temp_factor.v(tmp_assg));
          }
        } // end of loop over all tmp_assgs
      } // end of outer if statement for whether factor is unique

      // collect some additional statistics
      if(max_value < factor.maximum()){
        max_value = factor.maximum();
        max_value_line = line_number;
      }
      if(min_value > factor.minimum()){
        min_value = factor.minimum();
        min_value_line = line_number;
      }

      // Finished processing factor so add it to the factor graph
      size_t factorid = fg.add_factor(factor);
      if(weights.size() > 0) {
        assert(weights.size() == factor.arg_vector().size());
        for (size_t i = 0; i < weights.size(); ++i) {
          finite_variable* f = factor.arg_vector()[i];
          typename factor_graph_model<F>::vertex_type varvertex(f);
          size_t varid = fg.vertex2id(varvertex);
          
          fg.set_weight(factorid, varid, weights[i]);
        }
      }
      

    } // End of outer while loop over factors should be end of file

    assert(fin.good() == false);
    fin.close();

    // Extra output
    // std::cout << "Max Value: " << max_value
    //           << " on line " << max_value_line << std::endl;
    // std::cout << "Min Value: " << min_value
    //           << " on line " << min_value_line << std::endl;
    // std::cout << "Total variables: " << fg.arguments().size()
    //           << std::endl;
    (void)max_value_line;
    (void)min_value_line;

    return true; // Parsing successful
  } // end of parse_alchemy method



  inline bool alchemy_parse_truth(universe &u,
                            finite_assignment &truth,
                           std::string filename,bool predicatevalue,
                           bool silent=false) {
    std::cout << "Parsing " << filename << " as " << predicatevalue << "\n";
    std::ifstream fin;
    fin.open(filename.c_str());
    if (fin.fail()) {
      std::cerr<< "Unable to open " << filename << std::endl;
      return false;
    }
    std::string line;
    while(fin.good() && std::getline(fin, line)) {
      std::string varname = trim(line);
      // look for the variable name in the universe
      variable* v = u.var_from_name(varname);
      if (v == NULL) {
        if (!silent) std::cout << "var " << varname << " not in network" << std::endl;
      }
      else {
        finite_variable *fv = dynamic_cast<finite_variable*>(v);
        if (predicatevalue) truth[fv] = 1;
 			  else truth[fv] = 0;
      }
    }
    fin.close();
    return true;
  }


  /*
    A directory of files XXxx-true, or xxxx-false
    where the Xxx-true files contain true variables and xxx-false files
    contain false variables.
    (Matthai DBN format)
  */
  bool alchemy_parse_truthdir(universe &u,
                              finite_assignment &truth,
                              std::string dirname,
                              bool silent=false) {
    //enumerate the directory
    if (dirname[dirname.length()-1]=='/') {
      dirname = dirname.substr(0,dirname.length()-1);
    }

    DIR* dir = opendir(dirname.c_str());
    if (dir==NULL) {
      std::cerr<< "Unable to open " << dirname << std::endl;
      return false;
    }
    struct dirent *ent;
    while ((ent = readdir(dir))) {
      std::string fname = ent->d_name;
      bool predicatevalue = (tolower(fname).find("true") != std::string::npos);
      fname = dirname+"/"+fname;
      alchemy_parse_truth(u, truth,fname,predicatevalue,silent);
    }
    closedir(dir);
    return true;
  }

  template <typename F>
  bool alchemy_parse_belief_file(universe &u,
                                std::map<finite_variable*, F> &truth,
                                std::string filename,
                                bool silent = false) {
    std::ifstream fin;
    fin.open(filename.c_str());
    if (fin.fail()) {
      std::cerr<< "Unable to open " << filename << std::endl;
      return false;
    }
    std::string line;
    while(fin.good() && std::getline(fin, line)) {
      line = trim(line);
      if (line.length() == 0) continue;
      size_t commapos = line.find_last_of(",");
      assert(commapos != line.npos);
      
      std::string beliefline = trim(line.substr(commapos+1));


      std::string varname = trim(line.substr(0,commapos));
      // now strip the quotes if they exist
      if (varname[0] == '\"') {
        varname = trim(varname.substr(1,varname.length()-2));
      }

      variable* v = u.var_from_name(varname);
      if (v == NULL) {
        if (!silent) std::cout << "var " << varname << " not in network" << std::endl;
      }
      else {
        finite_variable *fv = dynamic_cast<finite_variable*>(v);
        
        
        
        // if beliefline only has one entry, let fv(1) equal that value
        // and fv(0) equal 1 - that value. (original alchemy format)
        // Otherwise, loop through the size of the variable and assign
        // every element (extension of alchemy format)
        truth[fv]=F(make_domain(fv),0.0).normalize();
        if (beliefline.find(" ")  == beliefline.npos) {
          double blf = atof(beliefline.c_str());
          truth[fv].set_v(1, blf);
          truth[fv].set_v(0, 1.0 - blf);
        }
        else {
          std::stringstream strm(beliefline);
          for (size_t i = 0;i <fv.size(); ++i) {
            double value = 0.0;
            strm >> value;
            truth[fv].set_v(i, value);
          }
          truth[fv].normalize();
        } 
      }
    }
    fin.close();
    return true;
  }


  /**
    Saves the engine beliefs in an alchemy like belief file
  */
  template <typename F>
  void write_alchemy_beliefs(universe &u,
                             std::map<finite_variable*, F> &beliefs,
                             std::string& output_filename) {
    // Create an ouptut filestream
    std::ofstream fout(output_filename.c_str());
    assert(fout.good());
    foreach(finite_variable* v, keys(beliefs)) {
      fout<<"\"" << v.name() << "\", ";
      for (size_t i = 0;i < beliefs[v].size(); ++i) {
        fout << (double)(beliefs[v](i)) << " ";
      }
      fout << std::endl;
    }
    
    fout.flush();
    fout.close();
  }
} // End of namespace
#include <sill/macros_undef.hpp>


#endif // SILL_ALCHEMY_HPP
