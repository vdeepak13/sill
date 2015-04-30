project(examples)
link_libraries(sill ${Boost_LIBRARIES})
subdirs(
  filter
  grid_motion
  ipf
  logistic_regression
  loopy_bp
  markov_chain
  naive_bayes
)
