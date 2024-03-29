#ifndef SILL_SIMPLE_CONFIG_HPP
#define SILL_SIMPLE_CONFIG_HPP

#include <sill/global.hpp>
#include <sill/parsers/string_functions.hpp>

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * A class for loading and saving configurations in a simple INI format.
   * \see http://en.wikipedia.org/wiki/INI_file
   */
  class simple_config {
  public:
    //! The entries for a single section, listed in the order specified in the file
    typedef std::vector< std::pair<std::string, std::string> > config_entries;
    
    //! A single entry in the configuration
    typedef std::pair<std::string, std::string> config_entry;

    //! Default constructor
    simple_config() { }
    
    //! Returns the entries for a given section
    config_entries& operator[](const std::string& section) {
      return sections[section];
    }

    //! Adds an entry to a section
    void add(const std::string& section, const std::string& key, const std::string& value) {
      sections[section].push_back(std::make_pair(key, value));
    }

    //! Adds an entry to a section by casting the value to a std::string
    template <typename T>
    void add(const std::string& section, const std::string& key, const T& value) {
      sections[section].push_back(std::make_pair(key, to_string(value)));
    }

    //! Loads the configuration from a file
    void load(const std::string& filename) {
      std::ifstream in(filename);
      if (!in) {
        throw std::runtime_error("Could not open the file " + filename);
      }

      size_t line_number = 0;
      std::string line;
      std::string section;
      while (getline(in, line)) {
        ++line_number;

        // trim and get rid of comments
        line = trim(line.substr(0, line.find('#')));
        if (line.empty()) {
          continue;
        }

        // parse the section or the value
        if (line.front() == '[' && line.back() == ']') {
          section = line.substr(1, line.size() - 2);
        } else {
          size_t pos = line.find('=');
          if (pos == std::string::npos) {
            throw std::runtime_error("Line " + to_string(line_number) + ": missing '='");
          } else if (pos == 0) {
            throw std::runtime_error("Line " + to_string(line_number) + ": missing key");
          } else {
            add(section, trim(line, 0, pos-1), trim(line, pos+1));
          }
        }
      }
    }

    //! Saves the configuration to a file
    void save(const std::string& filename) const {
      std::ofstream out(filename);
      if (!out) {
        throw std::runtime_error("Could not open the file " + filename);
      }

      typedef std::pair<const std::string, config_entries> value_type;
      foreach(const value_type& section, sections) {
        out << '[' << section.first << ']' << std::endl;
        foreach(const config_entry& entry, section.second) {
          out << entry.first << '=' << entry.second << std::endl;
        }
        out << std::endl;
      }
    }

  private:
    std::map<std::string, config_entries> sections;

  }; // class simple_config
    
} // namespace sill

#include <sill/macros_undef.hpp>

#endif
