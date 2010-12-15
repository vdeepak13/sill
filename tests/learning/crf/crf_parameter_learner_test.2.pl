# Script for parsing the results of crf_parameter_learner_test.pl
#
# Prints out a table with each line listing one completed test:
#  modelsize  regtype  ntrain  lambda1  lambda2  trainll  testll
#      true_trainll  true_testll

use strict;
use warnings;

my $filename_re = qr/cpl_test_([0-9])_([0-9]+)_([0-9]+)/;
my $lambdas_re = qr/Chose lambdas = \[([0-9\-e\.]+) ([0-9\-e\.]+)\]/;
my $trainll_re = qr/CRF's avg training data log likelihood after parameter learning: ([0-9\-e\.]+)/;
my $testll_re = qr/CRF's avg test data log likelihood after parameter learning: ([0-9\-e\.]+)/;
my $true_trainll_re = qr/True model's avg training data log likelihood: ([0-9\-e\.]+)/;
my $true_testll_re = qr/True model's avg test data log likelihood: ([0-9\-e\.]+)/;

print "modelsize\tregtype\tntrain\tlambda1\tlambda2\ttrainll\ttestll\ttrue_trainll\ttrue_testll\n" .
    "-------------------------------------------------------------------\n";

foreach my $file (@ARGV) {
    chomp $file;
    if (!($file =~ $filename_re)) {
	die();
    }
    my $regtype = $1;
    my $ntrain = $2;
    my $modelsize = $3;
    my $lambda1;
    my $lambda2;
    my $trainll;
    my $testll;
    my $true_trainll;
    my $true_testll;
    open INPUT, $file;
    while (<INPUT>) {
	my $line = $_;
	chomp $line;
	if ($line =~ $lambdas_re) {
	    $lambda1 = $1;
	    $lambda2 = $2;
	} elsif ($line =~ $trainll_re) {
	    $trainll = $1;
	} elsif ($line =~ $testll_re) {
	    $testll = $1;
	} elsif ($line =~ $true_trainll_re) {
	    $true_trainll = $1;
	} elsif ($line =~ $true_testll_re) {
	    $true_testll = $1;
	}
    }
    close INPUT;
    if (defined($lambda1) || defined($lambda2) ||
	defined($trainll) || defined($testll) ||
	defined($true_trainll) || defined($true_testll)) {
	if (defined($lambda1) && defined($lambda2) &&
	    defined($trainll) && defined($testll) &&
	    defined($true_trainll) && defined($true_testll)) {
	    print $modelsize . "\t" . $regtype . "\t" . $ntrain . "\t" .
		$lambda1 . "\t" . $lambda2 . "\t" . $trainll . "\t" .
		$testll . "\t" . $true_trainll . "\t" . $true_testll . "\n";
	} else {
	    print "modelsize = $modelsize\n"
		. "regtype = $regtype\n"
		. "ntrain = $ntrain\n"
		. "lambda1 = $lambda1\n"
		. "lambda2 = $lambda2\n"
		. "trainll = $trainll\n"
		. "testll = $testll\n"
		. "true_trainll = $true_trainll\n"
		. "true_testll = $true_testll\n";
	    die("some is, some ain't\n")
	}
    }
}
