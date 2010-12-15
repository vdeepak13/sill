This example illustrates how to construct and manipulate discrete
distributions. The example simulates a robot that moves in a discrete
2D grid. At every step, the distribution spreads uniformly around the
location at the previous time step. The model defines two sets of 
variables, pos_t = {x_t, y_t} and pos_tp = {x_t', y_t'}. After 
constructing the distribution over the location at the first step,
the program repeatedly evolves the state, by performing variable 
elimination in the underlying HMM.
