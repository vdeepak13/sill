/**
 * \file string_functions.hpp String functions used by parsers.
 *
 * \author Anton
 */

#ifndef SILL_PARSERS_STRING_FUNCTIONS_HPP
#define SILL_PARSERS_STRING_FUNCTIONS_HPP

#include <cstdlib>
#include <string>
#include <cassert>
#include <istream>

namespace sill {

  class tokenizer{
  private:
    const std::string s_, delims_;
    std::string::size_type position_;

  public:
    tokenizer(const std::string& s, const std::string& delims = " \t") :
      s_(s), delims_(delims){
      position_ = s_.find_first_not_of(delims);
    };

    bool has_token(){
      assert((position_ < s_.length()) || (position_ == std::string::npos));
      return position_ < s_.length();
    }

    std::string next_token(){
      assert(has_token());
      //assert that the position_ points to a non-delimiter
      assert(delims_.find_first_of(s_[position_]) == std::string::npos);

      std::string::size_type token_end_pos = s_.find_first_of(delims_, position_);
      std::string result;
      if(token_end_pos == std::string::npos){
        //no delimiters in the remainder, return the whole suffix
        result = s_.substr(position_, s_.length() - position_);
        position_ = std::string::npos;
      }
      else{
        result = s_.substr(position_, token_end_pos - position_);
        position_ = s_.find_first_not_of(delims_, token_end_pos);
      }

      return result;
    }
  };

  /**
   * Removes trailing and leading white space from a string
   */
  inline std::string trim(const std::string& str) {
    std::string::size_type pos1 = str.find_first_not_of(" \t\r");
    std::string::size_type pos2 = str.find_last_not_of(" \t\r");
    return str.substr(pos1 == std::string::npos ? 0 : pos1,
                      pos2 == std::string::npos ? str.size()-1 : pos2-pos1+1);
  }


  /**
   * Removes trailing and leading white space from a string
   */
  inline std::string lcase(const std::string& str) {
    std::string ret = str;
    for (size_t i = 0; i < ret.length(); ++i) {
      ret[i] = std::tolower(ret[i]);
    }
    return ret;
  }

  /**
   * same as the stl string get line but this also increments a line
   * counter which is useful for debugging purposes
   */
  inline std::istream& getline(std::ifstream& fin, std::string& line,
                        size_t& line_number) {
    ++line_number;
//    if(line_number % 10000 == 0)
//      std::cout << "line " << line_number << std::endl;
    return std::getline(fin, line);
  }
} // End of namespace

#endif /* STRING_FUNCTIONS_HPP_ */
