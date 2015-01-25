project(examples)
link_libraries(sill ${Boost_LIBRARIES})
subdirs(
  filter
  graph
  grid_motion
  ipf
  logistic_regression
  loopy_bp
  markov_chain
  naive_bayes
)
