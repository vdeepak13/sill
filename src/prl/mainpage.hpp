/* mainpage.hpp
 * This file holds documentation which appears on the index.html page of the
 * Doxygen-generated HTML.
 * The mainpage command can appear anywhere, but it's standard to use a
 * separate header file.
 */

/**
 * \mainpage Probabilistic Reasoning Library (PRL)
 *
 * \section mainpage_intro Introduction
 * This is a friendly package for Probabilistic Graphical Models (PGMs).
 * To get started, see the README file in the package for installation
 * instructions, and check out the
 * \ref mainpage_important_classes "Important Classes" below.
 *
 * \section mainpage_installation Installation
 * See the README file for installation instructions for various platforms.
 *
 * \section mainpage_important_classes Important Classes
 * This section outlines some of the major PRL classes and modules to know and
 * some of the main libraries we use from Boost.  The PRL classes generally have
 * synopses or examples in their documentation.  There are synopses and
 * tutorials for the Boost libraries below (linked from this list).
 *
 *  - Intro to Using PRL
 *    - Basic Data Structures:
 *      - \ref prl::variable "Variable"
 *        - \ref prl::finite_variable "Finite Variable"
 *        - \ref prl::vector_variable "Vector (Real) Variable"
 *      - \ref prl::assignment "Assignment"
 *        - \ref prl::finite_assignment "Finite Assignment"
 *        - \ref prl::vector_assignment "Vector (Real) Assignment"
 *      - \ref prl::factor "Factor"
 *      - \ref prl::crf_factor "CRF (Conditional) Factor"
 *    - \ref graph "Graphs"
 *      - \ref prl::directed_graph "Directed Graph"
 *      - \ref prl::undirected_graph "Undirected Graph"
 *    - PGM Structures:
 *      - \ref prl::bayesian_graph "Bayesian Graph"
 *      - \ref prl::junction_tree "Junction Tree"
 *      - \ref prl::markov_graph "Markov Graph"
 *      - \ref prl::crf_graph "Conditional Random Field (CRF) Graph"
 *    - Probabilistic Graphical Models:
 *      - \ref prl::bayesian_network "Bayesian Network" (based on Bayesian Graph)
 *      - \ref prl::decomposable "Decomposable" (based on Junction Tree)
 *        - Multiply in factors; automatically renormalizes.
 *        - Also can e.g. compute marginals, condition on evidence,
 *            compute entropies
 *      - \ref prl::markov_network "Markov Network" (based on Markov Graph)
 *      - \ref prl::crf_model "Conditional Random Field (CRF)" (based on CRF Graph)
 *    - \ref prl::dataset "Dataset"
 *    - Learning
 *      - \ref learning_param "Parameter Learning"
 *      - \ref learning_structure "Structure Learning" (\ref prl::chow_liu "Chow Liu", \ref prl::pwl_crf_learner "Tree CRF Learner")
 *      - \ref learning_discriminative "Discriminative Learning"
 *    - Other Data Structures
 *      - \ref prl::set "Set" (based on STL set)
 *      - \ref prl::map "Map" (based on STL map)
 *      - \ref prl::table_base "Tables" (\ref prl::dense_table "Dense" and
 *        \ref prl::sparse_table_t "Sparse")
 *      - \ref prl::copy_ptr "Copy-on-Write Pointer"
 *  - Intro to Programming in PRL (with Boost)
 *    - \ref range "Ranges" (based on Boost ranges)
 *    - Boost Concept Checking
 *      - Interfaces are to classes as concepts are to templates.
 *      - http://www.boost.org/libs/concept_check/concept_check.htm
 *      - \ref group_concepts "List of PRL Concepts"
 *      - \ref subpage_concepts "List of Outside Concepts"
 *    - Boost Serialization
 *      - For saving and loading classes to and from files.
 *      - http://www.boost.org/libs/serialization/doc/index.html
 *    - others (less important):
 *      - Boost Random: http://www.boost.org/libs/random/
 *      - Boost Program Options: http://www.boost.org/doc/html/program_options.html
 *      - Boost Tuple: http://www.boost.org/libs/tuple/doc/tuple_users_guide.html
 *      - Boost Smart Pointers: http://www.boost.org/libs/smart_ptr/smart_ptr.htm
 *
 * \todo In the Class Hierarchy, is there a way to make templates only appear
 *    once?  Right now, there is one item for each instantiation of the template
 *    (with different template parameters).
 *
 * \todo Why do some items in the Class Hierarchy appear in black (w/o links)?
 *    I think they are all STL/outside classes which are used as parents of
 *    PRL classes.  I wonder if there's a way to make the outside classes not
 *    show up.
 *
 * \todo See what the 'test' tag does.
 *
 * \todo Simple shortcuts to external concepts.
 *
 * \todo Make docs more readable (a la Java).
 *
 * \todo Add info about how to construct ranges: pass an STL vector or list,
 *       or a pair of pointers (?), or a pair of iterators.
 *
 * \todo Group constructors, etc. using \@name NAME \\ \@{ ... \\ \@}
 */

/**
 * \defgroup base Base
 *   \defgroup 
 *   \defgroup base_concepts Concepts
 *     \ingroup base
 *     \author Anton Chechetka, 
 *   \defgroup base_types Classes
 *     Elementary classes, such as variables, domains, and assignments.
 *     \ingroup base
 *     \author Stanislav Funiak, Anton Chechetka, Mark Paskin
 *
 * \defgroup math Math
 *   \defgroup math_number Number representation
 *     \ingroup math
 *     \author Mark Paskin
 *   \defgroup math_constants Constants
 *     \ingroup math
 *     \author Stanislav Funiak
 *   \defgroup math_functions Functions
 *     \ingroup math
 *     \author Stanislav Funiak
 *   \defgroup math_linalg Linear Algebra
 *     \ingroup math
 *     \author Stanislav Funiak
 *   \defgroup math_random Random Number and Distribution Generators
 *     \ingroup math
 *     \author Joseph Bradley
 *
 * \defgroup datastructure Data Structures
 *   Basic datastructures, such as multidimensional tables, 
 *   queues, BSP trees, etc.
 *   \author Mark Paskin, Stanislav Funiak
 *
 * \defgroup iterator Iterators
 *   Iterators used in various parts of the implementation in PRL;
 *   these are rarely used outside of the core classes.
 *   \author Mark Paskin, Stanislav Funiak, Joseph Bradley
 *
 * \defgroup range Range
 *   \defgroup range_concepts Range concepts
 *     Shortcuts for common Range concepts.
 *     \ingroup range
 *     \author Stanislav Funiak
 *
 *   \defgroup range_algorithm Range algorithms
 *     Range-based versions of STL algorithms.
 *     See http://www.cplusplus.com/reference/algorithm/
 *     \ingroup range
 *     \author Stanislav Funiak
 *
 *   \defgroup range_numeric Range numerics
 *    Range-based versions of STL numerics.
 *     \ingroup range
 *     \author Stanislav Funiak
 *
 *   \defgroup range_adapters Range adapters
 *    Adapters for obtaining a view of a range.
 *     \ingroup range
 *     \author Stanislav Funiak
 *
 * \defgroup graph Graph
 *   \defgroup graph_types Classes
 *    Classes for representing graphs and graph operations.
 *     \ingroup graph
 *     \author Joey Gonzales, Stanislav Funiak, Mark Paskin
 *   \defgroup graph_algorithms Algorithms
 *    Algorithms for traversing and triangulating graphs, and extracting
 *    graph properties.
 *     \ingroup graph
 *     \author Mark Paskin, Stanislav Funiak
 *
 * \defgroup factor Factors
 *   \defgroup factor_concepts Concepts
 *     Concepts that characterize different types of factors.
 *     \ingroup factor
 *     \author Stanislav Funiak
 *   \defgroup factor_types Classes
 *     Classes for representing different types of distributions or conditional
 *     distributions in a factorized probabilistic model.
 *     \ingroup factor
 *     \author Stanislav Funiak, Mark Paskin
 *   \defgroup factor_operations Standard operations
 *     Operations supported for all factor types. These operations are 
 *     automatically included in the factor/factor.hpp file and are 
 *     implemented by invoking the factor's combine() and collapse() functions.
 *     \ingroup factor
 *     \author Stanislav Funiak
 *   \defgroup factor_exceptions Exceptions
 *     Exceptions thrown by factor operations
 *     \ingroup factor
 *     \author Stanislav Funiak
 *   \defgroup factor_random Random factors
 *     Functions for creating random discrete factors.
 *     \ingroup factor
 *     \author Stanislav Funiak, Joseph Bradley
 *   \defgroup factor_approx Gaussian approximation
 *     Classes for approximating a non-linear distribution with a Gaussian.
 *     \ingroup factor
 *     \author Stanislav Funiak
 *
 * \defgroup model Models
 *   Probabilistic graphical models.
 *   \author Stanislav Funiak, Joseph Bradley, Mark Paskin
 *
 * \defgroup inference Inference
 *   Algorithms for static and dynamic inference in probabilistic graphical
 *   models.
 *   \author Stanislav Funiak, Joseph Bradley
 *
 * \defgroup learning Learning
 *   \defgroup learning_discriminative Discriminative learning
 *      Classes for discriminative learning algorithms for classification and
 *      regression: decision stumps, logistic regression, boosting, etc.
 *      \ingroup learning
 *      \author Joseph Bradley
 *   \defgroup learning_param Parameter learning
 *      Classes for learning parameters of graphical models.
 *      \ingroup learning
 *      \author Stanislav Funiak, Joseph Bradley
 *   \defgroup learning_structure Structure learning
 *      Classes for learning the structure of graphical models.
 *      \ingroup learning
 *      \author Joseph Bradley
 *   \defgroup learning_dataset Datasets
 *      Datasets, online data sources, synthetic data generators.
 *      \ingroup learning
 *      \author Joseph Bradley, Stanislav Funiak
 *
 * \defgroup geometry Computational geometry
 *    \author Stanislav Funiak
 *
 * \defgroup serialization Serialization
 *   Tools for storing factors to an XML file.
 *
 * \defgroup optimization Optimization
 *   General convex optimization.
 *   \author Joseph Bradley
 *   \defgroup optimization_concepts Optimization concepts
 *     \ingroup optimization
 *     \author Joseph Bradley
 *   \defgroup optimization_algorithms Optimization algorithms
 *     \ingroup optimization
 *     \author Joseph Bradley
 *   \defgroup optimization_classes Optimization classes
 *     \ingroup optimization
 *     \author Joseph Bradley
 */

/* DON'T DELETE THIS YET; WE SHOULD PUT IT SOMEWHERE SINCE IT HAS INFO ON
 * USING DEPOT FOR FACILITIZED MACHINES.
 * \section mainpage_installation Installing PRL
 *
 * -# Prerequisites:
 *    - cmake: http://www.cmake.org
 *    - lapack: http://www.netlib.org/lapack/
 *      - lapack3 (for Debian GNU/Linux)
 *      - Intel Math Kernel Library 10.0:
 *         http://www.intel.com/cd/software/products/asmo-na/eng/307757.htm
 *        - for all Intel processors
 *        - includes blas
 *      - Accelerate:
 *        - for Mac Os X
 *        -includes blas
 *    - blas: http://www.netlib.org/blas/
 *      - blas3 (for Debian GNU/Linux)
 *    - NOTE: On Linux, you can install all of the above by running:
 *       'yum install cmake blas blas-devel lapack lapack-devel'
 *    - Boost: http://www.boost.org
 *      - Download Boost and unzip it.
 *      - Run:
 *        - ./configure
 *        - make
 *        - make install
 *      - Note: On CMU SCS facilitized machines, more needs to be done:
 *        - Install Boost in /usr/local:
 *          - ./configure --prefix=/usr/local
 *          - make
 *          - make install
 *        - Copy:
 *          - /usr/local/lib/libboost*
 *            to /usr0/prl_tools/boost_installation/usr_local_lib/
 *          - /usr/local/include/boost-1_35
 *            to /usr0/prl_tools/boost_installation/usr_local_include/
 *        - Add lines to files:
 *          - file: /usr/local/depot/depot.pref.local
 *            - 'path boost /usr0/prl_tools/boost_installation'
 *          - file: /usr0/prl_tools/boost_installation/depot.conf
 *            - 'usr_local_lib lib'
 *            - 'usr_local_include include'
 *        - Run: /usr/local/bin/dosupdepot
 *        - Note: As soon as you run dosupdepot, it updates the /usr/local
 *          directory, so you can go ahead and check to make sure that your
 *          changes were successful.
 *    - doxygen (for documentation only): http://www.stack.nl/~dimitri/doxygen/
 * -# Use SVN to check out projects/prl.
 * -# To create debug and release directories for compiling code,
 *    run the 'configure' script in the prl root directory.
 *    - This will create a separate folder called 'build' in which it will
 *      place the executables. (ditto for 'release')
 *    - Note there is a bug with the link libraries: (Is this comment outdated?)
 *      In Mac Os X, you need to link against
 *      '${PRL_SOURCE_DIR}/../libs/boost-trunk/build/lib/libboost_serialization.a'
 *      and in Linux, you need to link against
 *      '${PRL_SOURCE_DIR}/../libs/boost-trunk/build/lib/libboost_serialization-gcc41-d.a'
 */
