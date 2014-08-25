/*
 * bif_parser.hpp
 *
 *  Created on: May 24, 2009
 *      Author: antonc
 */

#ifndef BIF_PARSER_HPP
#define BIF_PARSER_HPP

#include <sstream>
#include <fstream>
#include <algorithm>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/factor_graph_model.hpp>
#include <sill/parsers/string_functions.hpp>
#include <sill/parsers/tokenizer.hpp>

#include <boost/random/uniform_real.hpp>
#include <boost/random/mersenne_twister.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * This function fills in a factor graph with with *.bif file
   * provided in filename.  The function returns true if parsing
   * succeeded.
   */
  template <typename F>
  bool parse_bif(universe& universe,
                 factor_graph_model<F>& fg,
                 std::vector<finite_variable*>& reverse_elimination_order,
                 std::set<finite_variable*>& leaves,
                 const std::string& filename){

    assert(fg.arguments().size() == 0);
    assert(fg.size() == 0);
    assert(reverse_elimination_order.size() == 0);
    assert(leaves.size() == 0);

    const std::string NETWORK("network"), VARIABLE("variable");
    const std::string PROBABILITY("probability"), TYPE("type");
    const std::string DISCRETE("discrete"), TABLE("table");
    const std::string OPEN_BRACE("{"), CLOSE_BRACE("}");
    const std::string PROPERTY("property"), COMMENT_START("//");

    std::stringstream file_single_line_stream;

    std::ifstream fin(filename.c_str());
    assert(fin.good());
    std::string line;
    size_t line_number = 1;
    while(fin.good()){
      getline(fin,line,line_number);

      //filter out comments and properties
      tokenizer t(line);

      if(!t.has_token())
        continue;

      std::string token = t.next_token();
      if(PROPERTY.compare(token) == 0)
        continue;

      size_t comment_start_index = line.find(COMMENT_START);
      if(comment_start_index == std::string::npos)
        file_single_line_stream << line << " " ;
      else
        file_single_line_stream << line.substr(0, comment_start_index) << " " ;
    }
    fin.close();

    //get all the file contents into a single string
    //so that we can use a tokenizer on it
    std::string file_single_line = file_single_line_stream.str();

    std::map<std::string, sill::finite_variable*> name_to_var_map;
    std::map<finite_variable*, std::map<std::string, finite_variable::value_type> > var_to_val_to_id_map;
    std::map<finite_variable*, std::set<finite_variable*> > var_to_parents;
    std::map<finite_variable*, std::set<finite_variable*> > var_to_children;

    tokenizer t(file_single_line, "\" ,;|=()[]\t\n");
    while(t.has_token()){
      std::string token = t.next_token();

      if(NETWORK.compare(token) == 0){
        token = t.next_token(); //network name
        token = t.next_token();
        assert(OPEN_BRACE.compare(token) == 0);
        token = t.next_token();
        assert(CLOSE_BRACE.compare(token) == 0);
      }
      else if(VARIABLE.compare(token) == 0){
        std::string var_name = t.next_token();
        assert(name_to_var_map.find(var_name) == name_to_var_map.end());

        token = t.next_token();
        assert(OPEN_BRACE.compare(token) == 0);
          token = t.next_token();
          assert(TYPE.compare(token) == 0);
          token = t.next_token();
          assert(DISCRETE.compare(token) == 0);
//          token = t.next_token();
//          assert(OPEN_SQUARE.compare(token) == 0);

            std::string var_cardinality_str = t.next_token();
            size_t cardinality = atoi(var_cardinality_str.c_str());
            finite_variable* v = universe.new_finite_variable(var_name, cardinality);
            name_to_var_map[var_name] = v;

//          token = t.next_token();
//          assert(CLOSE_SQUARE.compare(token) == 0);
          token = t.next_token();
          assert(OPEN_BRACE.compare(token) == 0);

          //for every value map it to an integer
          for(size_t i=0 ; i<cardinality; i++){
            token = t.next_token();
            assert(var_to_val_to_id_map[v].find(token) == var_to_val_to_id_map[v].end());
            var_to_val_to_id_map[v][token] = i;
          }

          token = t.next_token();
          assert(CLOSE_BRACE.compare(token) == 0);
        token = t.next_token();
        assert(CLOSE_BRACE.compare(token) == 0);
      }
      else if(PROBABILITY.compare(token) == 0){
        finite_var_vector v_vector;

        //reading off the domain
        token = t.next_token();
        do{
          finite_variable* v = name_to_var_map[token];
          v_vector.push_back(v);
          token = t.next_token();
        } while(OPEN_BRACE.compare(token) != 0);

        finite_var_vector reverse_v_vector(v_vector.rbegin(), v_vector.rend());

        F reverse_factor(reverse_v_vector, 0.0);

        token = t.next_token();
        if(TABLE.compare(token) == 0){
          typename F::table_type::iterator table_iter = reverse_factor.table().begin();
          for(size_t i=0; i<reverse_factor.size(); i++){
            assert(table_iter != reverse_factor.table().end());

            token = t.next_token();
            double value = atof(token.c_str());
            double REGULARIZATION = 1.0E-10;
            if (typeid(F) == typeid(table_factor)) {
              *table_iter = std::max(value, REGULARIZATION);
            }
            else if (typeid(F) == typeid(log_table_factor)) {
              *table_iter = std::log(std::max(value, REGULARIZATION));
            }
            table_iter++;
          }

          assert(table_iter == reverse_factor.table().end());

        }
        else
          do{
            finite_assignment assignment;

            if(v_vector.size() == 1)
              assert(TABLE.compare(token) == 0);
            else{
              finite_variable* v = v_vector[1];
              assignment[v] = var_to_val_to_id_map[v][token];
              for(size_t i=2; i<v_vector.size(); i++){
                token = t.next_token();
                finite_variable* v = v_vector[i];
                assignment[v] = var_to_val_to_id_map[v][token];
              }
            }

            for(size_t i=0; i<v_vector[0]->size(); i++){
              assignment[v_vector[0]] = i;
              token = t.next_token();
              double value = atof(token.c_str());
              double REGULARIZATION = 1.0E-10;

              reverse_factor(assignment) = std::max(value, REGULARIZATION);
            }
            token = t.next_token();
          }while(CLOSE_BRACE.compare(token) != 0);

        F factor(v_vector, 0.0);
        BOOST_FOREACH(const sill::finite_assignment& a, reverse_factor.assignments())
          factor(a) = reverse_factor(a);

        fg.add_factor(factor);

        for(size_t i=1; i<v_vector.size(); i++){
          var_to_children[v_vector[i]].insert(v_vector[0]);
          var_to_parents[v_vector[0]].insert(v_vector[i]);
        }
      }

    }

    //find the reverse elimination order - top-down
    std::set<finite_variable*> already_visited, fringe;

    BOOST_FOREACH(finite_variable *v, fg.arguments())
      if(var_to_parents[v].size() == 0)
        fringe.insert(v);
      else if(var_to_children[v].size() == 0)
        leaves.insert(v);

    while(fringe.size() > 0){
      std::set<finite_variable*> newly_visited;
      BOOST_FOREACH(finite_variable *v, fringe){
        bool all_parents_visited = true;

        BOOST_FOREACH(finite_variable *w, var_to_parents[v])
          if(already_visited.find(w) == already_visited.end())
            all_parents_visited = false;

        if(all_parents_visited)
          newly_visited.insert(v);
      }

      assert(newly_visited.size() > 0);
      already_visited.insert(newly_visited.begin(), newly_visited.end());

      BOOST_FOREACH(finite_variable *v, newly_visited){
        reverse_elimination_order.push_back(v);
        fringe.erase(v);
      }

      BOOST_FOREACH(finite_variable *v, newly_visited)
        fringe.insert(var_to_children[v].begin(), var_to_children[v].end());

    }

    return true;
  }

  template <typename F>
  bool parse_bif(universe& universe,
                 factor_graph_model<F>& fg,
                 const std::string& filename){
    std::vector<finite_variable*> reverse_elimination_order;
    std::set<finite_variable*> leaves;

    return parse_bif(universe, fg, reverse_elimination_order, leaves, filename);
  }

  //! draw a random sample from a factor
  template <typename F>
  size_t sample_from_factor(F& factor, boost::mt11213b& rng) {
     boost::uniform_real<double> uniform_real_;
     assert(factor.arguments().size() == 1);
     double sum = 0;
     double r = uniform_real_(rng) * factor.norm_constant();
     size_t index = 0;
     foreach(double d, factor.values()) {
       sum += d;
       if(r <= sum) return index;
       else index++;
     }
     assert(false);
     return -1;
   } // end of sample from a factor


  template <typename F>
  void sample_from_bayes_net(const factor_graph_model<F>& fg,
                             const std::vector<finite_variable*>& reverse_elimination_order,
                             boost::mt11213b& rng,
                             finite_assignment& result){

    assert(result.size() == 0);
    for(std::vector<finite_variable*>::const_iterator i = reverse_elimination_order.begin();
                                                      i != reverse_elimination_order.end();
                                                      i++){

      std::set<const F*> CPTs;
      BOOST_FOREACH(const F& f, fg.factors())
        if(*(f.arg_vector().begin()) == *i)
          CPTs.insert(&f);

      assert(CPTs.size() == 1);

      const F* CPT = *CPTs.begin();
      F conditional_probability = CPT->restrict(result);
      assert(conditional_probability.arguments().size() == 1);
      assert(conditional_probability.arguments().contains(*i));

      result[*i] = sample_from_factor(conditional_probability, rng);

    }

  }

} // namespace sill

#endif
