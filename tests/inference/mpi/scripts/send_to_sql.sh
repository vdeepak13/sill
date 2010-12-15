
mysql --host=bigbro2.ml.cmu.edu \
    --user=paraml \
    --password=paraml \
    --database=paraml \
    --execute="LOAD DATA LOCAL INFILE \"cpuactivity.txt\" INTO TABLE cpuusage"

mysql --host=bigbro2.ml.cmu.edu \
    --user=paraml \
    --password=paraml \
    --database=paraml \
    --execute="LOAD DATA LOCAL INFILE \"netactivity.txt\" INTO TABLE networkusage"

mysql --host=bigbro2.ml.cmu.edu \
    --user=paraml \
    --password=paraml \
    --database=paraml \
    --execute="LOAD DATA LOCAL INFILE \"runtime.txt\" INTO TABLE summary"