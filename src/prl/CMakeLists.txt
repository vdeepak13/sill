add_subdirectory(base)
add_subdirectory(factor)
add_subdirectory(learning)
add_subdirectory(math)
add_subdirectory(model)
add_subdirectory(mpi)
add_subdirectory(optimization)
add_subdirectory(random)
add_subdirectory(serialization)

add_prefix(PRL_BASE_SOURCES "base/")
add_prefix(PRL_FACTOR_SOURCES "factor/")
add_prefix(PRL_LEARNING_SOURCES "learning/")
add_prefix(PRL_MATH_SOURCES "math/")
add_prefix(PRL_MODEL_SOURCES "model/")
add_prefix(PRL_MPI_SOURCES "mpi/")
add_prefix(PRL_OPTIMIZATION_SOURCES "optimization/")
add_prefix(PRL_RANDOM_SOURCES "random/")
add_prefix(PRL_SERIALIZATION_SOURCES "serialization/")
add_library(prl
  ${PRL_BASE_SOURCES}
  ${PRL_FACTOR_SOURCES}
  ${PRL_LEARNING_SOURCES}
  ${PRL_MATH_SOURCES}
  ${PRL_MPI_SOURCES}
  ${PRL_OPTIMIZATION_SOURCES}
  ${PRL_RANDOM_SOURCES}
  ${PRL_SERIALIZATION_SOURCES})

	    
