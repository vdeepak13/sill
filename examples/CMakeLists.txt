project(examples)
link_libraries(prl ${Boost_LIBRARIES})
subdirs(filter graph grid_motion ipf loopy_bp)
