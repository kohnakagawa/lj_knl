AOS = aos_pair.out aos_next.out aos_sorted.out aos_intrin_v1.out aos_intrin_v2.out aos_intrin_v3.out aos_intrin_v1_reactless.out aos_intrin_v2_reactless.out aos_intrin_v3_reactless.out
SOA = soa_pair.out soa_next.out soa_intrin_v1.out
LOOP_DEP = knl_1x8_aos_v1.out knl_1x8_aos_v2.out knl_1x8_aos_v3.out knl_1x8_soa_v1.out knl_ref_aos.out knl_ref_soa.out

TARGET= $(AOS) $(SOA) $(LOOP_DEP)

ASM = force_aos.s force_soa.s force_aos_loop_dep.s force_soa_loop_dep.s

all: $(TARGET) $(ASM)

aos: $(AOS)
soa: $(SOA)
loop: $(LOOP_DEP)

.SUFFIXES:
.SUFFIXES: .cpp .s
.cpp.s:
	icpc -O3 -xMIC-AVX512 -std=c++11 -S $< -o $@

aos_pair.out: force_aos.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DPAIR $< -o $@

aos_sorted.out: force_aos.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DSORTED $< -o $@

aos_next.out: force_aos.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DNEXT $< -o $@

aos_intrin_v1.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 -DINTRIN_v1 $< -o $@

aos_intrin_v2.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 -DINTRIN_v2 $< -o $@

aos_intrin_v3.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 -DINTRIN_v3 $< -o $@

aos_intrin_v1_reactless.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 -DINTRIN_v1 -DREACTLESS $< -o $@

aos_intrin_v2_reactless.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 -DINTRIN_v2 -DREACTLESS $< -o $@

aos_intrin_v3_reactless.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 -DINTRIN_v3 -DREACTLESS $< -o $@

soa_pair.out: force_soa.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DPAIR $< -o $@

soa_next.out: force_soa.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DNEXT $< -o $@

soa_intrin_v1.out: force_soa.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 -DINTRIN_v1 -DREACTLESS $< -o $@

knl_ref_aos.out: force_aos_loop_dep.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DREFERENCE $< -o $@

knl_1x8_aos_v1.out: force_aos_loop_dep.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DUSE1x8_v1 $< -o $@

knl_1x8_aos_v2.out: force_aos_loop_dep.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DUSE1x8_v2 $< -o $@

knl_1x8_aos_v3.out: force_aos_loop_dep.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DUSE1x8_v3 $< -o $@

knl_ref_soa.out: force_soa_loop_dep.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DREFERENCE $< -o $@

knl_1x8_soa_v1.out: force_soa_loop_dep.cpp
	icpc -O3 -xMIC-AVX512 -std=c++11 -DUSE1x8_v1 $< -o $@

clean:
	rm -f $(TARGET) $(ASM)

test_aos: $(AOS)
	./aos_pair.out > aos_pair.dat
	./aos_next.out > aos_next.dat
	./aos_sorted.out > aos_sorted.dat
	./aos_intrin_v1.out > aos_intrin_v1.dat
	./aos_intrin_v2.out > aos_intrin_v2.dat
	./aos_intrin_v3.out > aos_intrin_v3.dat
	./aos_intrin_v1_reactless.out > aos_intrin_v1_reactless.dat
	./aos_intrin_v2_reactless.out > aos_intrin_v2_reactless.dat
	./aos_intrin_v3_reactless.out > aos_intrin_v3_reactless.dat
	diff aos_pair.dat aos_next.dat
	diff aos_next.dat aos_sorted.dat
	diff aos_sorted.dat aos_intrin_v1.dat
	diff aos_intrin_v1.dat aos_intrin_v2.dat
	diff aos_intrin_v2.dat aos_intrin_v3.dat
	diff aos_intrin_v3.dat aos_intrin_v1_reactless.dat
	diff aos_intrin_v1_reactless.dat aos_intrin_v2_reactless.dat
	diff aos_intrin_v2_reactless.dat aos_intrin_v3_reactless.dat

test_soa: $(SOA)
	./soa_pair.out > soa_pair.dat
	./soa_next.out > soa_next.dat
	./soa_intrin_v1.out > soa_intrin_v1.dat
	diff soa_pair.dat soa_next.dat
	diff soa_next.dat soa_intrin_v1.dat

test_loop: $(LOOP_DEP)
	./run_ofp.sh
