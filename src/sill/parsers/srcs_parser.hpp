/**
 * \file srcs_parser.hpp Parsers for common-sense Horn clauses from Matthai Philipose
 *
 * \author Anton
 */


#ifndef SILL_SRCS_PARSER_HPP
#define SILL_SRCS_PARSER_HPP

#include <string>
#include <vector>
#include <iostream>
#include <istream>
#include <fstream>
#include <map>
#include <cstdlib>
#include <sstream>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/factor/log_table_factor.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/parsers/string_functions.hpp>
#include <sill/parsers/alchemy.hpp>

#include <sill/macros_def.hpp>

namespace sill{

  typedef std::pair< std::vector<std::string>, double> weighted_horn_clause_type;

  inline void print_weighted_horn_clause(const weighted_horn_clause_type& clause,
                                         std::ostream& out){
    out << "(";
    for(size_t i = 0; i + 1 < clause.first.size(); i++)
      out << clause.first[i] << ", ";

    out << "--> " << *clause.first.rbegin() << ")  ";
    out << clause.second << std::endl;
  }

  /**
   * Replaces the symbols that Alchemy parser uses as separators (spaces and
   * forward slashes) with underscores
   *
   * inlined to avoid double definitions
   * \todo move to a cpp file to ensure separate compilation
   */
  inline void sanitize_srcs_variable_name(std::string& var_name){
    for(size_t i =0; i<var_name.length(); i++)
      if(var_name[i] == ' ' || var_name[i] == '/' || var_name[i] == '\t')
        var_name[i] =  '_';
  }

  /**
   * Parses a single weighted Horn clause. Stores the resulting wvariable
   * names, ids, and clause weights in the corresponding parameters.
   *
   * inlined to avoid double definitions
   * \todo move to a cpp file to ensure separate compilation
   */
  inline void parse_single_srcs_clause( const std::string& clause_line,
                                        std::vector<std::string>& var_names,
                                        std::vector<std::size_t>& var_ids,
                                        double& clause_weight){

    const std::string ARROW_STRING = "-->";

    assert(var_names.empty());
    assert(var_ids.empty());

    assert(clause_line[0] == '(');

    std::string::size_type last_brace_pos = clause_line.find_last_of(')');
    assert(last_brace_pos != std::string::npos); //there has to be a closing brace

    assert(clause_line[last_brace_pos + 1] == ';');
    assert(clause_line[last_brace_pos + 2] == '*');

    std::string clause_weight_line = clause_line.substr(last_brace_pos + 3, clause_line.length() - last_brace_pos - 3);
    clause_weight = atof(clause_weight_line.c_str());


    std::string var_list_line = clause_line.substr(1, last_brace_pos - 1);
    var_list_line += ";"; //appending semicolon to make handling the last var simpler
    while(var_list_line.length() > 0){
      std::string::size_type atom_name_end_pos = var_list_line.find_first_of(';');
      assert(atom_name_end_pos != std::string::npos);

      std::string atom_full = var_list_line.substr(0, atom_name_end_pos);

      //find the --> arrow that separates variable name from integer id
      std::string::size_type arrow_loc = atom_full.find(ARROW_STRING);
      assert(arrow_loc != std::string::npos);

      std::string atom_name = trim(atom_full.substr(0, arrow_loc));
      std::string atom_id_str = trim(atom_full.substr(arrow_loc + ARROW_STRING.length(),
                                     atom_name_end_pos + 1 - arrow_loc - ARROW_STRING.length()));

      //replace spaces with underscores in atom name
      sanitize_srcs_variable_name(atom_name);

      var_names.push_back(atom_name);
      var_ids.push_back(atoi(atom_id_str.c_str()));

      if(atom_name_end_pos < var_list_line.length() - 1)
        var_list_line = var_list_line.substr(atom_name_end_pos + 1, var_list_line.length() - atom_name_end_pos - 1);
      else
        var_list_line = "";
    } // end while(var_list_line.length() > 0)

  }

  /**
   * Returns a mapping from atom name to the graph node ids in the srcs file.
   * Can be used e.g. to figure out whether there are atoms mapped to
   * multiple nodes (which should not happen ideally, but does in the data).
   *
   * inlined to avoid double definitions
   * \todo move to a cpp file to ensure separate compilation
   */
  inline std::map<std::string, std::set<size_t> >
    srcs_get_var_name_to_id_map(const std::string& filename){

    const std::string ARROW_STRING = "-->";

    std::map<std::string, std::set<size_t> > result;

    // Open an input file stream
    std::ifstream fin(filename.c_str());
    assert(fin.good());
    std::string line;
    size_t line_number = 1;

    // Read the first line which should be "variable:"
    while(fin.good()){
      getline(fin, line, line_number);

      //cut off the braces and clause weight
      line = trim(line);
      if(line.length() == 0)
        continue;

      std::vector<std::string> var_names;
      std::vector<std::size_t> var_ids;
      double clause_weight;

      parse_single_srcs_clause( line, var_names, var_ids, clause_weight);
      for(size_t i=0; i<var_names.size(); i++)
        result[var_names[i]].insert(var_ids[i]);

    } //end while(fin.good())

    fin.close();
    return result;
  } // end of srcs_get_var_name_to_id_map()



  inline std::vector<weighted_horn_clause_type>
  srcs_get_all_unique_clauses(const std::string& filename){
    typedef std::set<std::string> unordered_clause_lhs_type;
    typedef std::pair<unordered_clause_lhs_type, std::string> unordered_clause_type;

    std::map<unordered_clause_type, double> weighted_unordered_clauses;

    // Open an input file stream
    std::ifstream fin(filename.c_str());
    assert(fin.good());
    std::string line;
    size_t line_number = 0;

    while(fin.good()){
      getline(fin, line, line_number);

      //cut off the braces and clause weight
      line = trim(line);
      if(line.length() == 0)
        continue;

      weighted_horn_clause_type weighted_clause;
      std::vector<size_t> var_ids;
      parse_single_srcs_clause( line,
                                weighted_clause.first,
                                var_ids,
                                weighted_clause.second);

      //check if the clause is unique
      unordered_clause_type unordered_clause;
      unordered_clause.first.insert(weighted_clause.first.begin(),
                                    weighted_clause.first.end());
      unordered_clause.second = *weighted_clause.first.rbegin();
      unordered_clause.first.erase(unordered_clause.second);

      if(weighted_unordered_clauses.find(unordered_clause) == weighted_unordered_clauses.end())
        weighted_unordered_clauses[unordered_clause] = weighted_clause.second;
      else
        weighted_unordered_clauses[unordered_clause] += weighted_clause.second;
    } //end while(fin.good())
    fin.close();

    std::vector<weighted_horn_clause_type> result;
    for(std::map<unordered_clause_type, double>::iterator i = weighted_unordered_clauses.begin();
                                                          i != weighted_unordered_clauses.end();
                                                          i++){
      weighted_horn_clause_type wc;
      wc.second = i->second;
      wc.first.insert(wc.first.end(), i->first.first.begin(), i->first.first.end());
      wc.first.push_back(i->first.second);
      result.push_back(wc);
    }

    return result;
  } //end of srcs_get_all_unique_clauses()


  inline std::map<std::string, std::vector<std::pair<double, double> > >
  srcs_get_ground_truth_intervals(const std::string& filename, double& max_time){
    assert(max_time >= 0);

    std::map<std::string, std::vector<std::pair<double, double> > > result;

    // Open an input file stream
    std::ifstream fin(filename.c_str());
    assert(fin.good());
    std::string line;
    size_t line_number = 0;

    while(fin.good()){
      getline(fin, line, line_number);

      //cut off the braces and clause weight
      line = trim(line);
      if(line.length() == 0)
        continue;

      std::string::size_type space_pos = line.find_first_of(')');
      std::string var_name = trim(line.substr(0, space_pos + 1));

      //remove spaces and forward slashes from the variable name
      sanitize_srcs_variable_name(var_name);

      tokenizer t(line.substr(space_pos + 1, line.length() - space_pos - 1));
      std::string start_time_str = t.next_token();
      std::string end_time_str =  t.next_token();

      std::pair<double, double> interval;
      interval.first = atof(start_time_str.c_str());
      interval.second = atof(end_time_str.c_str());

      assert(interval.second > interval.first);
      assert(interval.first >= 0);

      max_time = std::max(max_time, interval.second);

      result[var_name].push_back(interval);

    } //end while(fin.good())
    return result;
  }


  inline std::map<std::string, std::vector<double> >
  srcs_get_object_usage_times(const std::string& filename, double& max_time){

    const std::string PREFIX("useinferred(Object:"), SUFFIX(")");
    const std::string START_TAG("start_tag"), STOP_TAG("stop_tag");
    const char TIME_NAME_SEPARATOR = ':';

    std::map<std::string, std::vector<double> > result;
    max_time = 0;

    // Open an input file stream
    std::ifstream fin(filename.c_str());
    assert(fin.good());
    std::string line;
    size_t line_number = 0;

    while(fin.good()){
      getline(fin, line, line_number);

      //cut off the braces and clause weight
      line = trim(line);
      if(line.length() == 0)
        continue;

      std::string::size_type space_pos = line.find_first_of(' ');
      double tag_time = atof(line.substr(0, space_pos).c_str());
      assert(tag_time >= 0);
      max_time = std::max(tag_time, max_time);

      assert(line[space_pos + 1] == TIME_NAME_SEPARATOR);
      std::string tag_name = trim(line.substr(space_pos + 2, line.length() - space_pos - 2));

      sanitize_srcs_variable_name(tag_name);

      if(tag_name.compare(START_TAG) == 0)
        continue;
      if(tag_name.compare(STOP_TAG) == 0)
        continue;

      result[PREFIX + tag_name + SUFFIX].push_back(tag_time);

    } //end while(fin.good())

    return result;
  } //end of srcs_get_object_usage_times()

  inline std::set<size_t>
  srcs_get_true_timeslices(const std::vector<std::pair<double, double> >& intervals,
                           double slice_duration,
                           size_t total_slices){
    std::set<size_t> result;
    for(size_t i=0; i < total_slices ; i++){
      double start = i * slice_duration;
      double stop = (i + 1 ) * slice_duration;

      typedef std::pair<double, double> double_pair_type;
      foreach(const double_pair_type& interval, intervals){
        assert(interval.second >= interval.first);
        if(interval.first >= start && interval.first < stop)
          result.insert(i);
        if (interval.second >= start && interval.second < stop)
          result.insert(i);
        if(interval.first < start && interval.second >= start)
          result.insert(i);
      }
    }
    return result;
  }

  inline std::set<size_t>
  srcs_get_true_timeslices(const std::vector<double>& occurences,
                           double slice_duration,
                           size_t total_slices){
    std::set<size_t> result;
    for(size_t i=0; i < total_slices ; i++){
      double start = i * slice_duration;
      double stop = (i + 1 ) * slice_duration;

      foreach(size_t o, occurences){
        if(o >= start && o < stop){
          result.insert(i);
          break;
        }
      }
    }
    return result;
  }

  inline void srcs_parse_trace(
    const std::string& trace_filename,
    const std::string& ground_truth_filename,
    double timeslice_duration,
    size_t& timeslices_number,
    std::map<std::string, std::set<size_t> >& tag_used_slices,
    std::map<std::string, std::set<size_t> >& ground_truth_true_slices){

    typedef std::pair<double, double> double_pair_type;
    typedef std::map<std::string, std::vector<double_pair_type> > string_to_intervals_map_type;
    typedef std::map<std::string, std::vector<double> > string_to_doubles_map_type;

    //read the tag usage times and ground truth intervals
    double max_time = 0;
    string_to_doubles_map_type tag_times = srcs_get_object_usage_times(trace_filename, max_time);
    string_to_intervals_map_type gt_intervals = srcs_get_ground_truth_intervals(ground_truth_filename, max_time);

    //figure out the number of timeslices
    timeslices_number = ceil(max_time / timeslice_duration);

    assert(tag_used_slices.size() == 0);
    assert(ground_truth_true_slices.size() == 0);

    //fill in the ground truth slices
    for(string_to_intervals_map_type::iterator i = gt_intervals.begin();
                                                   i != gt_intervals.end();
                                                   i++){
      std::set<size_t> slices = srcs_get_true_timeslices(i->second, timeslice_duration, timeslices_number);
      ground_truth_true_slices[i->first] = slices;
    }

    //fill in the true tag slices
    for(string_to_doubles_map_type::iterator  i = tag_times.begin();
                                              i != tag_times.end();
                                              i++){
      std::set<size_t> slices = srcs_get_true_timeslices(i->second, timeslice_duration, timeslices_number);
      tag_used_slices[i->first] = slices;
    }
  }

  inline void srcs_print_positive_ground_truth(
      const std::string& trace_filename,
      const std::string& ground_truth_filename,
      double timeslice_duration,
      std::ostream& out){

    size_t timeslices_number;
    std::map<std::string, std::set<size_t> > tag_used_slices, ground_truth_true_slices;

    srcs_parse_trace(trace_filename, ground_truth_filename, timeslice_duration,
                     timeslices_number, tag_used_slices, ground_truth_true_slices);

    for(std::map<std::string, std::set<size_t> >::iterator i = ground_truth_true_slices.begin();
                                                           i != ground_truth_true_slices.end();
                                                           i++){
      foreach(size_t slice, i->second)
        out << i->first << "_" << slice << std::endl;
    }
  }

  inline void srcs_print_positive_evidence(
      const std::string& trace_filename,
      const std::string& ground_truth_filename,
      double timeslice_duration,
      std::ostream& out){

    size_t timeslices_number;
    std::map<std::string, std::set<size_t> > tag_used_slices, ground_truth_true_slices;

    srcs_parse_trace(trace_filename, ground_truth_filename, timeslice_duration,
                     timeslices_number, tag_used_slices, ground_truth_true_slices);

    for(std::map<std::string, std::set<size_t> >::iterator i = tag_used_slices.begin();
                                                           i != tag_used_slices.end();
                                                           i++){
      foreach(size_t slice, i->second)
        out << i->first << "_" << slice << std::endl;
    }

  }

  inline void srcs_print_partial_negative_evidence(
      const std::string& trace_filename,
      const std::string& ground_truth_filename,
      double timeslice_duration,
      std::ostream& out){

    size_t timeslices_number;
    std::map<std::string, std::set<size_t> > tag_used_slices, ground_truth_true_slices;

    srcs_parse_trace(trace_filename, ground_truth_filename, timeslice_duration,
                     timeslices_number, tag_used_slices, ground_truth_true_slices);

    for(std::map<std::string, std::set<size_t> >::iterator i = tag_used_slices.begin();
                                                           i != tag_used_slices.end();
                                                           i++){
      for(size_t slice=0; slice<timeslices_number; slice++)
        if(i->second.find(slice) == i->second.end())
          out << i->first << "_" << slice << std::endl;
    }
  }

  inline void srcs_print_trace_mrf(
      const std::string& clauses_filename,
      const std::string& trace_filename,
      const std::string& ground_truth_filename,
      double timeslice_duration,
      std::ostream& out){

    using namespace std;

    size_t timeslices_number;
    std::map<std::string, std::set<size_t> > tag_used_slices, ground_truth_true_slices;
    srcs_parse_trace(trace_filename, ground_truth_filename, timeslice_duration,
                     timeslices_number, tag_used_slices, ground_truth_true_slices);

    std::vector<weighted_horn_clause_type> all_clauses = srcs_get_all_unique_clauses(clauses_filename);

    std::set<string> all_var_names;
    foreach(const weighted_horn_clause_type& clause, all_clauses)
      foreach(const string& var_name, clause.first)
        all_var_names.insert(var_name);

    //output variables for every timeslice
    out << "variables:" << endl;
    foreach(const string& var_name, all_var_names)
      for(size_t i=0; i<timeslices_number; i++)
        out << var_name << "_" << i << endl;

    out << "factors:" << endl;

    //output vactors for every timeslice
    foreach(const weighted_horn_clause_type& clause, all_clauses){
      for(size_t i=0; i<timeslices_number; i++){
        assert(clause.second > 0);

        universe u;
        std::vector<finite_variable *> arguments;
        BOOST_FOREACH(string var_name, clause.first){
          std::ostringstream full_var_name_stream;
          full_var_name_stream << var_name << "_" << i;
          finite_variable *v = u.new_finite_variable(full_var_name_stream.str(), 2);
          arguments.push_back(v);
        }

        log_table_factor factor(arguments, logarithmic<double>(0.0, log_tag()));
        finite_assignment assignment;

        // nontrivially satisfying assignment - everything is true
        foreach(finite_variable *v, arguments)
          assignment[v] = 1;
        factor(assignment) = logarithmic<double>(clause.second, sill::log_tag());

        //not satisfying assignment - LHS is all true, RHS is false
        assignment[*arguments.rbegin()] = 0;
        factor(assignment) = logarithmic<double>(-clause.second, log_tag());

        //print the alchemy format
        print_factor_alchemy_format(factor, out);
      }
    }

    //output factors connecting the timeslices
    foreach(const string& var_name, all_var_names){
      for(size_t i=0; i+1<timeslices_number; i++){
        universe u;
        std::ostringstream from_var_name_stream, to_var_name_stream;
        from_var_name_stream << var_name << "_" << i;
        to_var_name_stream << var_name << "_" << (i+1);
        finite_variable *from = u.new_finite_variable(from_var_name_stream.str(), 2);
        finite_variable *to = u.new_finite_variable(to_var_name_stream.str(), 2);
        std::vector<finite_variable *> arguments;
        arguments.push_back(from);
        arguments.push_back(to);
        log_table_factor factor(arguments, logarithmic<double>(0.0, log_tag()));
        finite_assignment assignment;
        assignment[from] = 1;
        assignment[to] = 1;
        factor(assignment) = 0.95;
        assignment[to] = 0;
        factor(assignment) = 0.05;
        assignment[from] = 0;
        assignment[to] = 1;
        factor(assignment) = 0.095;
        assignment[to] = 0;
        factor(assignment) = 0.905;
        print_factor_alchemy_format(factor, out);
      }
    }

  }

} // End of namespace sill
#include <sill/macros_undef.hpp>


#endif /* SRCS_PARSER_HPP_ */
