export PATH=$PATH:/mnt/stor2/jegonzal/prl/branches/paraml/release/tests/inference


nparts=120
alg="kmetis"

for f in `find * -type d`
do
    cd $f
    echo "starting $BIN in $f" 
    mln_cuts *.out $nparts --weighted=true --alg=$alg | grep "Running Time: "
    cd ..
done

