AOS = aos_pair.out aos_next.out aos_sorted.out aos_intrin_v1.out aos_intrin_v2.out\
	aos_intrin_v3.out aos_intrin_v4.out aos_intrin_v1_reactless.out aos_intrin_v2_reactless.out \
	aos_sorted_reactless.out aos_bench.out
SOA = soa_pair.out soa_sorted.out soa_next.out soa_intrin_v1.out soa_intrin_v2.out\
	soa_sorted_2d.out

TARGET= $(AOS) $(SOA)
AOS_ASM = force_aos.s
SOA_ASM = force_soa.s

all: $(TARGET) $(AOS_ASM) $(SOA_ASM)
aos: $(AOS) $(AOS_ASM)
soa: $(SOA) $(SOA_ASM)

.SUFFIXES:
.SUFFIXES: .cpp .s
.cpp.s:
	icpc -O3 -xMIC-AVX512 -std=c++11 -S $< -o $@

aos_pair.out: force_aos.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DPAIR $< -o $@

aos_sorted.out: force_aos.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DSORTED $< -o $@

aos_sorted_reactless.out: force_aos.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DSORTED -DREACTLESS $< -o $@

aos_next.out: force_aos.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DNEXT $< -o $@

aos_intrin_v1.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 -DINTRIN_v1 $< -o $@

aos_intrin_v2.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 -DINTRIN_v2 $< -o $@

aos_intrin_v3.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 -DINTRIN_v3 $< -o $@

aos_intrin_v4.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 -DINTRIN_v4 $< -o $@

aos_intrin_v1_reactless.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 -DINTRIN_v1 -DREACTLESS $< -o $@

aos_intrin_v2_reactless.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 -DINTRIN_v2 -DREACTLESS $< -o $@

aos_bench.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 $< -o $@

soa_pair.out: force_soa.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DPAIR $< -o $@

soa_sorted.out: force_soa.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DSORTED $< -o $@

soa_next.out: force_soa.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DNEXT $< -o $@

soa_intrin_v1.out: force_soa.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DINTRIN_v1 $< -o $@

soa_intrin_v2.out: force_soa.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DINTRIN_v2 $< -o $@

soa_sorted_2d.out: force_soa.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DSORTED_2D $< -o $@

clean:
	rm -f $(TARGET) $(ASM)
