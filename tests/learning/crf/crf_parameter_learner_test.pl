# Script for running crf_parameter_learner_test.cpp a bunch of times to test
# parameter learning and different types of regularization with Gaussians.

use strict;
use warnings;

my $regtype = shift @ARGV;

my @ntrains = qw/10 20 50 100/;
my @modelsizes = qw/2 4 10/;
#my @regtypes = qw/0 2 3 4 5 6/;

my $seed = 35733998;

foreach my $ntrain (@ntrains) {
  foreach my $modelsize (@modelsizes) {
#    foreach my $regtype (@regtypes) {
      my $testname = "cpl_test_" . $regtype . "_" . $ntrain . "_" . $modelsize . ".txt";
      system("time ./crf_parameter_learner_test --ntrain $ntrain --ntest 1000 --model_size $modelsize --model_type tree --factor_type gaussian --cpl_iterations 200 --cpl_opt_method 1 --line_search_type 1 --line_search_stopping 1 --convergence_zero .000001 --regularization_type $regtype --do_cv --random_seed $seed > $testname");
      sleep 2;
#    }
  }
}
