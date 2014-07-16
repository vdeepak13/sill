subdirs(
  parallel
  mpi
)

add_executable(belief_propagation belief_propagation.cpp)
add_executable(bethe_bp bethe_bp.cpp)
#add_executable(denoise denoise.cpp)
add_executable(gbp gbp.cpp)
add_executable(gibbs_sampler gibbs_sampler.cpp)
#add_executable(image_segmentation image_segmentation.cpp)
add_executable(junction_tree_inference junction_tree_inference.cpp)
add_executable(kikuchi kikuchi.cpp)
add_executable(line_graph_generator line_graph_generator.cpp)
add_executable(markov_logic markov_logic.cpp)
#add_executable(mln_cuts mln_cuts.cpp)
#add_executable(protein_cuts protein_cuts.cpp)
#target_link_libraries(protein_cuts metis)
add_executable(protein_inference protein_inference.cpp)
add_executable(taueps_quant taueps_quant.cpp)
add_executable(variable_elimination variable_elimination.cpp)

find_package(TCMALLOC)
if(TCMALLOC_FOUND) 
  target_link_libraries(markov_logic tcmalloc)
  target_link_libraries(protein_inference tcmalloc)
endif()

add_test(belief_propagation belief_propagation)
add_test(gbp gbp)
add_test(junction_tree_inference junction_tree_inference)
add_test(kikuchi kikuchi)
add_test(variable_elimination variable_elimination)
