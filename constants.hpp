#pragma once

constexpr double SL = 3.3;
constexpr double CL = 3.0;
constexpr double CL2 = CL * CL;
constexpr double dt = 0.001;
constexpr double L = 50.0;
constexpr int32_t N = 400000;

#ifdef LOW_DENSITY
constexpr double density = 0.5;
#else
constexpr double density = 1.0;
#endif

constexpr uint64_t knl_features = (_FEATURE_AVX512F | _FEATURE_AVX512ER |
                                   _FEATURE_AVX512PF | _FEATURE_AVX512CD);
constexpr uint64_t skl_features = (_FEATURE_AVX512F | _FEATURE_AVX512CD |
                                   _FEATURE_AVX512VL | _FEATURE_AVX512DQ |
                                   _FEATURE_AVX512BW);
