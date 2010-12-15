project(examples)
link_libraries(${PRL_LIBRARIES} ${Boost_LIBRARIES})
subdirs(filter graph grid_motion ipf loopy_bp)
