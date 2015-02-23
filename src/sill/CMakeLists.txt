add_subdirectory(base)
add_subdirectory(factor)
add_subdirectory(inference)
#add_subdirectory(learning)
add_subdirectory(math)
add_subdirectory(model)
#add_subdirectory(optimization)
add_subdirectory(serialization)

add_prefix(SILL_BASE_SOURCES "base/")
add_prefix(SILL_FACTOR_SOURCES "factor/")
add_prefix(SILL_INFERENCE_SOURCES "inference/")
add_prefix(SILL_LEARNING_SOURCES "learning/")
add_prefix(SILL_MATH_SOURCES "math/")
add_prefix(SILL_MODEL_SOURCES "model/")
#add_prefix(SILL_OPTIMIZATION_SOURCES "optimization/")
add_prefix(SILL_SERIALIZATION_SOURCES "serialization/")
add_library(sill
  ${SILL_BASE_SOURCES}
  ${SILL_FACTOR_SOURCES}
  ${SILL_INFERENCE_SOURCES}
#  ${SILL_LEARNING_SOURCES}
  ${SILL_MATH_SOURCES}
  ${SILL_MODEL_SOURCES}
#  ${SILL_OPTIMIZATION_SOURCES}
  ${SILL_SERIALIZATION_SOURCES})

	    
