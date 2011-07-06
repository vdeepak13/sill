
#ifndef SILL_LEARNING_DISCRIMINATIVE_LEARNER_HPP
#define SILL_LEARNING_DISCRIMINATIVE_LEARNER_HPP

#include <fstream>
#include <sstream>
#include <string>

#include <sill/learning/dataset/dataset_statistics.hpp>
#include <sill/learning/dataset/oracle.hpp>
#include <sill/stl_io.hpp>

#include <sill/macros_def.hpp>

namespace sill {

  /**
   * Learner interface.
   *
   * Learners may be divided into 3 parts.  These parts are not defined by
   * interfaces, but save() and load() functions do distinguish between them.
   *  - prediction function: can make predictions given a new example
   *  - engine: holds a current prediction function and the info need to
   *            train the learner further
   *  - shell: does not hold a prediction function; holds parameters required
   *           to create a new engine or train a prediction function
   *
   * Implementation conventions:
   *  - Member data in non-final classes
   *     - Interfaces and non-final classes are allowed to have member data.
   *       They should be completely responsible for it: they should save,
   *       load, initialize, clean up, etc.
   *     - If an interface or non-final class has member data which must be
   *       handled by the final classes, it should clearly specify what needs
   *       to be done.  If an interface inherits such data members but does not
   *       handle them itself, then it should likewise clearly specify those.
   *  - Data, save(), load():
   *     - Each class is only responsible for saving and loading the data
   *       members which are declared in it.
   *     - This base class saves and checks the name of the learner.
   *     - Classes are not required to use the save_part parameter,
   *       but they may do so in order to save space.  Note that they must
   *       save in a format such that the save_part parameter is not needed by
   *       the load function.
   *     - Parameter save_name is used by metalearners to save space.
   *  - Parameters:
   *     - Parameter classes may be defined in interfaces, but the set functions
   *       must be defined in the base classes.
   *     - Parameters are used to set metalearners.
   *  - Learner names:
   *     - All learners are required to have distinct names; the name() function
   *       should return LEARNER_NAME or LEARNER_NAME<TEMPLATE_PARAMETERS>.
   *       Names and template parameters cannot contain these characters: "<>;,"
   *
   * \author Joseph Bradley
   * \ingroup learning_discriminative
   * @todo Figure out how to save the state of random number generators!
   * @todo Make a generalization of test_accuracy() which computes
   *       std dev, as well as true/false positive/negative rates.
   *
   * @tparam LA  Linear algebra type specifier
   *             (default = dense_linear_algebra<>)
   */
  template <typename LA = dense_linear_algebra<> >
  class learner {

    // Public types
    //==========================================================================
  public:

    typedef LA la_type;

    typedef record<la_type>                record_type;
    typedef typename la_type::value_type   value_type;
    typedef typename la_type::vector_type  vector_type;
    typedef typename la_type::matrix_type  matrix_type;
    typedef arma::Col<value_type>          dense_vector_type;
    typedef arma::Mat<value_type>          dense_matrix_type;

    // Constructors and destructors
    //==========================================================================
  public:

    virtual ~learner() { }

    // Getters and helpers
    //==========================================================================

    //! Return a name for the algorithm without template parameters.
    virtual std::string name() const = 0;

    //! Return a name for the algorithm with comma-separated template parameters
    //! (e.g., objective).
    virtual std::string fullname() const = 0;

    //! Returns true iff learner is naturally an online learner.
    virtual bool is_online() const = 0;

    //! Returns the total training time (so far) for the learner.
    //! This is not guaranteed to be implemented.
    virtual value_type training_time() const {
      return -1;
    }

    //! Print classifier
    //! This is not guaranteed to be implemented
    virtual void print(std::ostream& out) const {
      out << "(learner)\n";
    }

    // Learning and mutating operations
    //==========================================================================

    //! Resets the random seed in the learner's random number generator
    //! and parameters.
    virtual void random_seed(value_type value) { }

    // Methods for iterative learners
    // (None of these are implemented by non-iterative learners.)
    //==========================================================================

    //! Returns the current iteration number (from 0)
    //!  (i.e., the number of learning iterations completed).
    //! ITERATIVE ONLY: This must be implemented by iterative learners.
    virtual size_t iteration() const {
      assert(false);
      return 0;
    }

    //! Returns the total time elapsed after each iteration.
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    virtual std::vector<value_type> elapsed_times() const {
      return std::vector<value_type>();
    }

    //! Does the next step of training (updating the current classifier).
    //! @return true iff the learner may be trained further
    //! ITERATIVE ONLY: This must be implemented by iterative learners.
    virtual bool step() {
      assert(false);
      return false;
    }

    //! Resets the data source to be used in future rounds of training.
    //! @param  n   max number of examples which may be drawn from the oracle
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    virtual void
    reset_datasource(oracle<dense_linear_algebra<> >& o, size_t n) {
      assert(false);
    }

    //! Resets the data source to be used in future rounds of training.
    //! ITERATIVE ONLY: This may be implemented by iterative learners.
    virtual void
    reset_datasource(dataset_statistics<dense_linear_algebra<> >& stats) {
      assert(false);
    }

    // Save and load methods
    //==========================================================================

    //! Output the learner to a human-readable file which can be reloaded.
    //! @param save_part  0: function, 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    virtual void save(std::ofstream& out, size_t save_part = 0,
                      bool save_name = true) const {
      if (save_name)
        out << fullname() << "\n";
      out << save_part << "\n";
    }

    //! Output the learner to a human-readable file which can be reloaded.
    //! @param save_part  0: save function, 1: engine, 2: shell
    //! @param save_name  If true, this saves the name of the learner.
    void save(const std::string& filename, size_t save_part,
              bool save_name) const {
      std::ofstream out(filename.c_str(), std::ios::out);
      save(out, save_part, save_name);
      out.flush();
      out.close();
    }

    //! Output the learner to a human-readable file which can be reloaded.
    //! This saves the function only, and it saves the name.
    void save(const std::string& filename) const {
      std::ofstream out(filename.c_str(), std::ios::out);
      save(out, 0, true);
      out.flush();
      out.close();
    }

    /**
     * Input the learner from a human-readable file.
     * @param in          input filestream for file holding the saved learner
     * @param ds          datasource used to get variables
     * @param load_part   0: load function (default), 1: engine, 2: shell
     * This assumes that the learner name has already been checked.
     * @return true if successful
     */
    virtual bool
    load(std::ifstream& in, const datasource& ds, size_t load_part) = 0;

    /**
     * Input the learner from a human-readable file.
     * @param in          input filestream for file holding the saved learner
     * @param ds          datasource used to get variables
     * @param check_name  true iff this should expect the learner name
     *                    at the beginning of the input
     * @return true if successful
     */
    bool load(std::ifstream& in, const datasource& ds, bool check_name = true) {
      std::string line;
      if (check_name) {
        getline(in, line);
        if (line.compare(fullname()) != 0) {
          std::cerr << "Tried to load learner of type"
                    << "\"" << line << "\" into one of type"
                    << "\"" << fullname() << "\"" << std::endl;
          assert(false);
          return false;
        }
      }
      getline(in, line);
      std::istringstream is(line);
      size_t load_part;
      if (!(is >> load_part))
        assert(false);
      return load(in, ds, load_part);
    }

    /**
     * Input the learner from a human-readable file.
     * @param filename    file holding the saved learner
     * @param ds          datasource used to get variables
     * @param check_name  true iff this should expect the learner name
     *                    at the beginning of the input
     * @return true if successful
     */
    bool load(const std::string& filename, const datasource& ds,
              bool check_name) {
      std::ifstream in(filename.c_str(), std::ios::in);
      bool val = load(in, ds, check_name);
      in.close();
      return val;
    }

    /**
     * Input the learner from a human-readable file.
     * @param filename    file holding the saved learner
     * @param ds          datasource used to get variables
     * This assumes check_name = true.
     * @return true if successful
     */
    bool load(const std::string& filename, const datasource& ds) {
      std::ifstream in(filename.c_str(), std::ios::in);
      bool val = load(in, ds, true);
      in.close();
      return val;
    }

  }; // class learner

  template <typename LA>
  std::ostream& operator<<(std::ostream& out, const learner<LA>& l) {
    l.print(out);
    return out;
  }

} // namespace sill

#include <sill/macros_undef.hpp>

#endif // #ifndef SILL_LEARNING_DISCRIMINATIVE_LEARNER_HPP
