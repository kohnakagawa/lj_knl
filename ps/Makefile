AOS = aos_pair.out aos_next.out aos_sorted.out aos_intrin_v1.out aos_intrin_v2.out aos_intrin_v3.out aos_intrin_v1_reactless.out aos_intrin_v2_reactless.out aos_bench.out

TARGET= $(AOS)
ASM = force_aos.s

all: $(TARGET) $(ASM)

aos: $(AOS)

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

aos_bench.out: force_aos.cpp
	icpc -O3 -qopt-prefetch=4 -xMIC-AVX512 -std=c++11 $< -o $@

clean:
	rm -f $(TARGET) $(ASM)
