TARGET = aos_skl.out aos_knl.out

OPT_FLAGS = -qopt-prefetch=4 -O3
CXX_FLAGS = -std=c++11 $(OPT_FLAGS)

all: $(TARGET)

aos_skl.out: main.cpp
	icpc -xCORE-AVX512 $(CXX_FLAGS) $< -o $@

aos_knl.out: main.cpp
	icpc -xMIC-AVX512 $(CXX_FLAGS) $< -o $@

clean:
	rm -f $(TARGET)
