#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <cmath>
#include <cassert>
#include <memory>
#include <x86intrin.h>

#include "vec.hpp"
#include "constants.hpp"
#include "conf_maker.hpp"
#include "make_nlist.hpp"
#include "simd_utils.hpp"
#include "force_aos.hpp"

void print_result(Vec4* p, const int32_t pn) {
  for (int32_t i = 0; i < 5; i++) {
    printf("%.10f %.10f %.10f\n", p[i].x, p[i].y, p[i].z);
  }
  for (int32_t i = pn - 5; i < pn; i++) {
    printf("%.10f %.10f %.10f\n", p[i].x, p[i].y, p[i].z);
  }
}

#ifdef LOW_DENSITY
const char* const refdata_name = "./ref_data/density0_5.bin";
#else
const char* const refdata_name = "./ref_data/density1.bin";
#endif
void dump_refdata(const Vec4* p,
                  const int32_t pn) {
  std::ofstream fout(refdata_name, std::ios::binary);
  fout.write((char*)&pn, sizeof(int32_t));
  fout.write((char*)&p[0].x, pn * sizeof(Vec4));
  fout.close();
}

void load_ref_results(Vec4* p_ref) {
  std::ifstream ifs(refdata_name, std::ios::binary);
  if (!ifs) {
    std::cerr << "Fail to open reference momentum data" << std::endl;
    std::cerr << __FILE__ << ":" << __LINE__ << std::endl;
    std::exit(1);
  }

  int32_t pn;
  ifs.read((char*)&pn, sizeof(int32_t));
  ifs.read((char*)p_ref, pn * sizeof(Vec4));
}

void compare_result(const Vec4* p_ref,
                    const Vec4* p,
                    const int pn) {
  constexpr double eps = 1.0e-9;
  for (int i = 0; i < pn; i++) {
    const auto ex = std::abs((p_ref[i].x - p[i].x) / std::min(p_ref[i].x, p[i].x));
    const auto ey = std::abs((p_ref[i].y - p[i].y) / std::min(p_ref[i].y, p[i].y));
    const auto ez = std::abs((p_ref[i].z - p[i].z) / std::min(p_ref[i].z, p[i].z));
    if (!(ex < eps && ey < eps && ez < eps)) {
      std::cerr << "error = " << ex << " " << ey << " " << ez << std::endl;
      std::cerr << "p[" << i << "] = "
                << p[i].x << ", " << p[i].y << ", " << p[i].z << std::endl;
      std::cerr << "p_ref[" << i << "] = "
                << p_ref[i].x << ", " << p_ref[i].y << ", " << p_ref[i].z << std::endl;
      return;
    }
  }
}

template <class Func, class... Args>
void measure(const char* name, const int pn,
             Func func, Args... args) {
  using namespace std::chrono;
  const int loop = 100;
  const auto beg = system_clock::now();
  for (int i = 0; i < loop; i++) {
    func(args...);
  }
  const auto end = system_clock::now();
  const long dur = duration_cast<milliseconds>(end - beg).count();
  fprintf(stderr, "N=%d, %s %ld [ms]\n", pn, name, dur);
}

void clear_p(Vec4* p, const int32_t pn) {
  std::for_each(p, p + pn, [](Vec4& e){e.x = e.y = e.z = e.w = 0.0;});
}

#define STRINGIZE(s) #s
#define MEASURE(func, ...)                            \
  do {                                                \
    measure(STRINGIZE(func), pn, func, __VA_ARGS__);  \
    compare_result(p_ref.get(), p, pn);               \
    clear_p(p, pn);                                   \
  } while (0)

#define MEASURE_Z(func, ...)                          \
  do {                                                \
    copy_to_z(z, q, pn);                              \
    measure(STRINGIZE(func), pn, func, __VA_ARGS__);  \
    copy_from_z(p, z, pn);                            \
    compare_result(p_ref.get(), p, pn);               \
    clear_p(p, pn);                                   \
  } while (0)

int main() {
  Vec4 *q = nullptr, *p = nullptr;
  int32_t pn = 0;
  std::tie(q, p, pn) = make_particles(density);
  Vec8* z = make_z();

  auto p_ref = std::unique_ptr<Vec4[]>(new Vec4 [pn]);
  load_ref_results(p_ref.get());

  NeighborList nlist;
  if (!nlist.loadpair(pn)) {
    std::cout << "Now make neighbor list." << std::endl;
    nlist.make_sorted_list(pn, SL, q);
  }

  const int32_t np                  = nlist.number_of_pairs();
  const int32_t* number_of_partners = nlist.number_of_partners();
  const int32_t* pointer            = nlist.pointer();
  const int32_t* sorted_list        = nlist.sorted_list();
  const int32_t* i_particles        = nlist.i_particles();
  const int32_t* j_particles        = nlist.j_particles();

  { // bench
    // scalar
    MEASURE(scalar::force_pair, q, p, i_particles, j_particles, np);
    MEASURE(scalar::force_sorted, q, p, number_of_partners, pointer, sorted_list, pn);
    MEASURE(scalar::force_next, q, p, number_of_partners, pointer, sorted_list, pn);

    if (_may_i_use_cpu_feature(skl_features)) {
      MEASURE(skl::avx512::force_intrin, q, p, number_of_partners, pointer, sorted_list, pn);
      MEASURE(skl::avx512::force_intrin_swp, q, p, number_of_partners, pointer, sorted_list, pn);
      MEASURE_Z(skl::avx512::force_intrin_z, z, number_of_partners, pointer, sorted_list, pn);
      MEASURE_Z(skl::avx512::force_intrin_z_swp, z, number_of_partners, pointer, sorted_list, pn);
      MEASURE(skl::avx2::force_intrin, q, p, number_of_partners, pointer, sorted_list, pn);
      MEASURE(skl::avx2::force_intrin_swp, q, p, number_of_partners, pointer, sorted_list, pn);
      MEASURE_Z(skl::avx2::force_intrin_z, z, number_of_partners, pointer, sorted_list, pn);
      MEASURE_Z(skl::avx2::force_intrin_z_swp, z, number_of_partners, pointer, sorted_list, pn);
    }

    if (_may_i_use_cpu_feature(knl_features)) {
#ifdef REACTLESS
      MEASURE(knl::avx512::force_intrin_reactless, q, p, number_of_partners, pointer, sorted_list, pn);
#else
      MEASURE(knl::avx512::force_intrin, q, p, number_of_partners, pointer, sorted_list, pn);
      MEASURE(knl::avx512::force_intrin_swp, q, p, number_of_partners, pointer, sorted_list, pn);
      MEASURE(knl::avx512::force_intrin_swp_pf, q, p, number_of_partners, pointer, sorted_list, pn);
      MEASURE_Z(knl::avx512::force_intrin_z_swp, z, number_of_partners, pointer, sorted_list, pn);
#endif
    }
  }

  nlist.savepair(pn);
  delete_particles(q, p, z);
}
