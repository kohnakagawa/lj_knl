#!/bin/sh

execs=("./knl_1x8_aos_v1.out" "./knl_1x8_aos_v2.out" "./knl_1x8_aos_v3.out" "./knl_ref_aos.out" "./knl_1x8_soa_v1.out" "./knl_ref_soa.out")
files=("1x8_aos_v1.txt" "1x8_aos_v2.txt" "1x8_aos_v3.txt" "ref_aos.txt" "1x8_soa_v1.txt" "ref_soa.txt")
Ni="100003"
seed=$RANDOM

for i in `seq 0 5`
do
    rm -f ${files[$i]}
    touch ${files[$i]}
    for Nj in `seq 4 1 203`
    do
        ${execs[$i]} $Ni $Nj $seed >> ${files[$i]}
    done
done

echo "# num_iloop num_jloop aos_v1 aos_v2 aos_v3 ref_aos" > result_aos_$seed.txt
echo "# num_iloop num_jloop soa_v1 ref_soa" > result_soa_$seed.txt
paste 1x8_aos_v1.txt 1x8_aos_v2.txt 1x8_aos_v3.txt ref_aos.txt | awk '{print $1, $2, $3, $7, $11, $15, $4}' >> result_aos_$seed.txt
paste 1x8_soa_v1.txt ref_soa.txt | awk '{print $1, $2, $3, $7, $4}' >> result_soa_$seed.txt
rm ${files[@]}
