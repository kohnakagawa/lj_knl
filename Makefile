TARGET= aos.out aos_pair.out aos_next.out aos_intrin_v1.out aos_intrin_v2.out soa.out soa_pair.out soa_next.out soa_intrin_v1.out knl_1x8_aos_v1.out knl_1x8_aos_v2.out knl_1x8_soa_v1.out knl_ref_aos.out knl_ref_soa.out
ASM = force_aos.s force_soa.s force_aos_loop_dep.s force_soa_loop_dep.s

all: $(TARGET) $(ASM)

.SUFFIXES:
.SUFFIXES: .cpp .s
.cpp.s:
	icpc -O3 -axMIC-AVX512 -std=c++11 -S $< -o $@

aos.out: force_aos.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 $< -o $@

aos_pair.out: force_aos.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 -DPAIR $< -o $@

aos_next.out: force_aos.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 -DNEXT $< -o $@

aos_intrin_v1.out: force_aos.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 -DINTRIN_v1 $< -o $@

aos_intrin_v2.out: force_aos.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 -DINTRIN_v2 $< -o $@

soa.out: force_soa.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 $< -o $@

soa_pair.out: force_soa.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 -DPAIR $< -o $@

soa_next.out: force_soa.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 -DNEXT $< -o $@

soa_intrin_v1.out: force_soa.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 -DINTRIN_v1 $< -o $@

knl_ref_aos.out: force_aos_loop_dep.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 -DREFERENCE $< -o $@

knl_1x8_aos_v1.out: force_aos_loop_dep.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 -DUSE1x8_v1 $< -o $@

knl_1x8_aos_v2.out: force_aos_loop_dep.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 -DUSE1x8_v2 $< -o $@

knl_ref_soa.out: force_soa_loop_dep.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 -DREFERENCE $< -o $@

knl_1x8_soa_v1.out: force_soa_loop_dep.cpp
	icpc -O3 -axMIC-AVX512 -std=c++11 -DUSE1x8_v1 $< -o $@

clean:
	rm -f $(TARGET) $(ASM)

test_aos: aos_pair.out aos_next.out aos_intrin_v1.out aos_intrin_v2.out
	./aos_pair.out > aos_pair.dat
	./aos_next.out > aos_next.dat
	./aos_intrin_v1.out > aos_intrin_v1.dat
	./aos_intrin_v2.out > aos_intrin_v2.dat
	diff aos_pair.dat aos_next.dat
	diff aos_next.dat aos_intrin_v1.dat
	diff aos_intrin_v1.dat aos_intrin_v2.dat

test_soa: soa_pair.out soa_next.out soa_intrin_v1.out
	./soa_pair.out > soa_pair.dat
	./soa_next.out > soa_next.dat
	./soa_intrin_v1.out > soa_intrin_v1.dat
	diff soa_pair.dat soa_next.dat
	diff soa_next.dat soa_intrin_v1.dat

test_loop: knl_ref_aos.out knl_1x8_aos_v1.out knl_1x8_aos_v2.out knl_ref_soa.out knl_1x8_soa_v1.out
	./run_ofp.sh
