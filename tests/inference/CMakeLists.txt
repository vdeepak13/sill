
subdirs(
  parallel
  mpi
)

#add_executable(approximate_inference_engine approximate_inference_engine.cpp)
add_executable(belief_propagation belief_propagation.cpp)
add_executable(gibbs_sampler gibbs_sampler.cpp)
add_executable(bethe_bp bethe_bp.cpp)
add_executable(junction_tree_inference junction_tree_inference.cpp)
#add_executable(srs_denoise denoise.cpp)

add_executable(protein_inference protein_inference.cpp)
target_link_libraries(protein_inference tcmalloc)

#add_executable(protein_cuts protein_cuts.cpp)
#target_link_libraries(protein_cuts metis)

#add_executable(mln_cuts mln_cuts.cpp)
#target_link_libraries(mln_cuts metis )

add_executable(line_graph_generator line_graph_generator.cpp)

#target_link_libraries(srs_denoise jpeg tcmalloc)


#add_executable(image_segmentation image_segmentation.cpp)

add_executable(markov_logic markov_logic.cpp)
target_link_libraries(markov_logic tcmalloc)

add_executable(taueps_quant taueps_quant.cpp)

# add_executable(markov_logic markov_logic.cpp)
# set_target_properties(markov_logic PROPERTIES COMPILE_FLAGS "-DBLF_SPLASH")
# target_link_libraries(markov_logic tcmalloc)


# add_executable(markov_logic_rbp markov_logic.cpp)
# target_link_libraries(markov_logic_rbp tcmalloc)
# set_target_properties(markov_logic_rbp PROPERTIES COMPILE_FLAGS "-DRESIDUAL_SPLASH")

# add_executable(markov_logic_robin markov_logic.cpp)
# target_link_libraries(markov_logic_robin tcmalloc)
# set_target_properties(markov_logic_robin PROPERTIES COMPILE_FLAGS "-DROUND_ROBIN")

# add_executable(markov_logic_gibbs markov_logic.cpp)
# target_link_libraries(markov_logic_gibbs tcmalloc)
# set_target_properties(markov_logic_gibbs PROPERTIES COMPILE_FLAGS "-DGIBBS")

add_executable(kikuchi kikuchi)
add_executable(gbp gbp)

