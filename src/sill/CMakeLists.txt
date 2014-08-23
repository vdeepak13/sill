add_subdirectory(base)
add_subdirectory(factor)
add_subdirectory(inference)
add_subdirectory(learning)
add_subdirectory(math)
add_subdirectory(model)
add_subdirectory(optimization)
add_subdirectory(serialization)

add_prefix(PRL_BASE_SOURCES "base/")
add_prefix(PRL_FACTOR_SOURCES "factor/")
add_prefix(PRL_INFERENCE_SOURCES "inference/")
add_prefix(PRL_LEARNING_SOURCES "learning/")
add_prefix(PRL_MATH_SOURCES "math/")
add_prefix(PRL_MODEL_SOURCES "model/")
add_prefix(PRL_OPTIMIZATION_SOURCES "optimization/")
add_prefix(PRL_SERIALIZATION_SOURCES "serialization/")
add_library(sill
  ${PRL_BASE_SOURCES}
  ${PRL_FACTOR_SOURCES}
  ${PRL_INFERENCE_SOURCES}
  ${PRL_LEARNING_SOURCES}
  ${PRL_MATH_SOURCES}
  ${PRL_MODEL_SOURCES}
  ${PRL_OPTIMIZATION_SOURCES}
  ${PRL_SERIALIZATION_SOURCES})

	    
