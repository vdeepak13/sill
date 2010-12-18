# Script for converting a saved AdaBoost classifier with Haar-like features
#  from PRL to OpenCV format.
# usage: perl prl2opencv.pl < [prl file] > [opencv file]

use warnings;
use strict;

sub read_vec {
  my $l = shift;
  my @line = split ' ', $l;
  my $re1 = qr/^\[(.*)/;
  my $re2 = qr/^(.*)\]$/;
  my @vec;
  if (!($line[0] =~ $re1)) { die "ERROR"; }
  push @vec, $1;
  if ($vec[0] =~ $re2) {
    $vec[0] = $1;
  } else {
    for (my $i = 1; $i < @line - 1; $i++) {
      if ($line[$i] =~ $re2) {
        push @vec, $1;
        return \@vec;
      }
      push @vec, $line[$i];
    }
    if (!($line[$#line] =~ $re2)) { die "ERROR"; }
    push @vec, $1;
  }
  return \@vec;
}

my $in = <>;
chomp $in;
my $classifier_type = $in;
if ($in ne "batch_booster" && $in ne "filtering_booster") { die "ERROR"; }
$in = <>; # booster type
$in = <>; # parameters
$in = <>; # haar
$in = <>; # haar
$in = <>; # haar
if ($in eq "batch_booster") {
  $in = <>; # resampler
}
$in = <>;
chomp $in;
my @line = split ' ', $in;
shift @line; # class_variable_index
shift @line; # train_acc
my $iter = shift @line;
$in = join(' ', @line);
my $re1 = qr/^(\[.*\])(.*)$/;
if (!($in =~ $re1)) { die "ERROR"; }
my $tmp = read_vec($1);
my @alphas = @{$tmp};
$in = $2;
my @classifiers;
for (my $i = 0; $i < $iter; $i++) {
  $in = <>;
  $in = <>;
  $in = <>;
  chomp $in;
  @line = split ' ', $in;
  shift @line; shift @line; shift @line; shift @line; shift @line; # params
  shift @line; # class_variable_index
  shift @line; # train_acc
  shift @line; # window_h
  shift @line; # window_w
  $in = join(' ', @line);
  if (!($in =~ $re1)) { die "ERROR"; }
  $tmp = read_vec($1);
  my @multipliers = @{$tmp};
  $in = $2;
  @line = split ' ', $in;
  my $predictA = shift @line;
  my $predictB = shift @line;
  shift @line; # train_objective
  if (scalar(@line) != scalar(@multipliers)) { die "ERROR"; }
  my $re3 = qr/^<(\d+),(\d+)>$/;
  my @coordinates;
  for (my $j = 0; $j < @multipliers; $j++) {
    if ($line[$j] =~ $re3) { push @coordinates, [$1, $2]; }
  }
  push @classifiers, [\@multipliers, \@coordinates, $predictA, $predictB];
}
if (scalar(@alphas) != scalar(@classifiers)) {
  print "ERROR\n";
  exit;
}

my ($n2,$n3,$n4) = (0,0,0);
print "<?xml version=\"1.0\"?>\n"
  . "<opencv_storage>\n"
  . "<prl_booster type_id=\"opencv-haar-classifier\">\n"
  . "  <size>\n"
  . "    24 24</size>\n"
  . "  <stages>\n"
  . "    <_>\n"
  . "      <!-- stage 0 -->\n"
  . "      <trees>\n";
for (my $i = 0; $i < @classifiers; $i++) {
  my @multipliers = @{$classifiers[$i][0]};
  my @coordinates = @{$classifiers[$i][1]};
  my $predictA = $classifiers[$i][2];
  my $predictB = $classifiers[$i][3];
  if (@multipliers == 6) { # 2 rectangle
    $n2++;
    @multipliers = (-1., 2.);
    @coordinates = ([$coordinates[0][0], $coordinates[0][1],
                     $coordinates[5][0]-$coordinates[0][0],
                     $coordinates[5][1]-$coordinates[0][1]],
                    [$coordinates[1][0], $coordinates[1][1],
                     $coordinates[5][0]-$coordinates[1][0],
                     $coordinates[5][1]-$coordinates[1][1]]);
  } elsif (@multipliers == 8) { # 3 rectangle
    $n3++;
    @multipliers = (1., -3.);
    @coordinates = ([$coordinates[0][0], $coordinates[0][1],
                     $coordinates[7][0]-$coordinates[0][0],
                     $coordinates[7][1]-$coordinates[0][1]],
                    [$coordinates[1][0], $coordinates[1][1],
                     $coordinates[6][0]-$coordinates[1][0],
                     $coordinates[6][1]-$coordinates[1][1]]);
  } elsif (@multipliers == 9) { # 4 rectangle
    $n4++;
    @multipliers = (1., -2., -2.);
    @coordinates = ([$coordinates[0][0], $coordinates[0][1],
                     $coordinates[8][0]-$coordinates[0][0],
                     $coordinates[8][1]-$coordinates[0][1]],
                    [$coordinates[0][0], $coordinates[0][1],
                     $coordinates[4][0]-$coordinates[0][0],
                     $coordinates[4][1]-$coordinates[0][1]],
                    [$coordinates[4][0], $coordinates[4][1],
                     $coordinates[8][0]-$coordinates[4][0],
                     $coordinates[8][1]-$coordinates[4][1]]);
  } else { die "ERROR: " . scalar(@multipliers) . " multipliers\n"; }
  
  print "        <_>\n"
    . "          <!-- tree " . $i . " -->\n"
    . "          <_>\n"
    . "            <!-- root node -->\n"
    . "            <feature>\n"
    . "              <rects>\n";
  for (my $j = 0; $j < @multipliers; $j++) {
    if (!($multipliers[$j] =~ m/\./)) { $multipliers[$j] .= "."; }
    print "                <_>\n"
      . "                  ";
    print join(' ', @{$coordinates[$j]}) . " " . $multipliers[$j];
    print "</_>\n";
  }
  if (!($predictA =~ m/\./)) { $predictA .= "."; }
  if (!($predictB =~ m/\./)) { $predictB .= "."; }
  print "                               </rects>\n"
    . "              <tilted>0</tilted></feature>\n"
    . "            <threshold>0.</threshold>\n"
    . "            <left_val>" . ($predictB * $alphas[$i]) . "</left_val>\n"
    . "            <right_val>" . ($predictA * $alphas[$i]) . "</right_val></_></_>\n";
}
print "        </trees>\n"
  . "      <stage_threshold>0.</stage_threshold>\n"
  . "      <parent>-1</parent>\n"
  . "      <next>-1</next></_></stages></prl_booster>\n"
  . "</opencv_storage>\n";

print STDERR "n2=$n2 n3=$n3 n4=$n4\n";
