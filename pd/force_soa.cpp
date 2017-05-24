#include <iostream>
#include <algorithm>
#include <fstream>
#include <cassert>
#include <cstdlib>
#include <random>
#include <cmath>
#include <chrono>
#include <x86intrin.h>
#include <sys/stat.h>
//----------------------------------------------------------------------
 const double density = 1.0;
// const double density = 0.5;
const int N = 400000;
const int MAX_PAIRS = 30 * N;
const double L = 50.0;
const double dt = 0.001;

const char* pairlist_cache_file_name = "pair.dat";

double* __restrict qx = nullptr;
double* __restrict qy = nullptr;
double* __restrict qz = nullptr;

double* __restrict px = nullptr;
double* __restrict py = nullptr;
double* __restrict pz = nullptr;

const int D = 3;
enum {X, Y, Z};
double q[D][N];
double p[D][N];

int particle_number = 0;
int number_of_pairs = 0;
int* __restrict number_of_partners = nullptr;
int i_particles[MAX_PAIRS];
int j_particles[MAX_PAIRS];
int32_t* __restrict pointer = nullptr;
int32_t pointer2[N];
int* __restrict sorted_list = nullptr;

const double CUTOFF_LENGTH = 3.0;
const double SEARCH_LENGTH = 3.3;
const double CL2 = CUTOFF_LENGTH * CUTOFF_LENGTH;
//----------------------------------------------------------------------
void
print512(__m512d r) {
  union {
    __m512d r;
    double elem[8];
  } tmp;
  tmp.r = r;
  printf("%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n",
         tmp.elem[0], tmp.elem[1], tmp.elem[2], tmp.elem[3],
         tmp.elem[4], tmp.elem[5], tmp.elem[6], tmp.elem[7]);
}
//----------------------------------------------------------------------
void
add_particle(double x, double y, double z) {
  static std::mt19937 mt(2);
  std::uniform_real_distribution<double> ud(0.0, 0.1);
  qx[particle_number] = x + ud(mt);
  qy[particle_number] = y + ud(mt);
  qz[particle_number] = z + ud(mt);
  particle_number++;
}
//----------------------------------------------------------------------
void
register_pair(int index1, int index2) {
  int i, j;
  if (index1 < index2) {
    i = index1;
    j = index2;
  } else {
    i = index2;
    j = index1;
  }
  i_particles[number_of_pairs] = i;
  j_particles[number_of_pairs] = j;
  number_of_partners[i]++;
  number_of_pairs++;
}
//----------------------------------------------------------------------
void
sortpair(void){
  const int pn = particle_number;
  int pos = 0;
  pointer[0] = 0;
  for (int i = 0; i < pn - 1; i++) {
    pos += number_of_partners[i];
    pointer[i + 1] = pos;
  }
  for (int i = 0; i < pn; i++) {
    pointer2[i] = 0;
  }
  const int s = number_of_pairs;
  for (int k = 0; k < s; k++) {
    int i = i_particles[k];
    int j = j_particles[k];
    int index = pointer[i] + pointer2[i];
    sorted_list[index] = j;
    pointer2[i] ++;
  }
}
//----------------------------------------------------------------------
void
makepair(void) {
  const double SL2 = SEARCH_LENGTH * SEARCH_LENGTH;
  const int pn = particle_number;
  for (int i = 0; i < pn; i++) {
    number_of_partners[i] = 0;
  }
  for (int i = 0; i < particle_number - 1; i++) {
    for (int j = i + 1; j < particle_number; j++) {
      const double dx = qx[i] - qx[j];
      const double dy = qy[i] - qy[j];
      const double dz = qz[i] - qz[j];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < SL2) {
        register_pair(i, j);
      }
    }
  }
}
//----------------------------------------------------------------------
void
allocate(void) {
  posix_memalign((void**)(&qx), 64, sizeof(double) * N);
  posix_memalign((void**)(&qy), 64, sizeof(double) * N);
  posix_memalign((void**)(&qz), 64, sizeof(double) * N);

  posix_memalign((void**)(&px), 64, sizeof(double) * N);
  posix_memalign((void**)(&py), 64, sizeof(double) * N);
  posix_memalign((void**)(&pz), 64, sizeof(double) * N);

  posix_memalign((void**)(&number_of_partners), 64, sizeof(int) * N);
  posix_memalign((void**)(&pointer), 64, sizeof(int32_t) * N);
  sorted_list = new int [MAX_PAIRS];

  std::fill(number_of_partners,
            number_of_partners + N,
            0);
  std::fill(pointer,
            pointer + N,
            0);
  std::fill(sorted_list,
            sorted_list + MAX_PAIRS,
            0);
}
//----------------------------------------------------------------------
void
deallocate(void) {
  free(qx);
  free(qy);
  free(qz);

  free(px);
  free(py);
  free(pz);

  free(number_of_partners);
  free(pointer);
  delete [] sorted_list;
}
//----------------------------------------------------------------------
 void
 random_shfl() {
   std::mt19937 mt(10);
   const auto pn = particle_number;
   for (int i = 0; i < pn; i++) {
     const auto kp = pointer[i];
     const auto np = number_of_partners[i];
     std::shuffle(&sorted_list[kp], &sorted_list[kp + np], mt);
   }
 }
//----------------------------------------------------------------------
void
init(void) {
  const double s = 1.0 / std::pow(density * 0.25, 1.0 / 3.0);
  const double hs = s * 0.5;
  int sx = static_cast<int>(L / s);
  int sy = static_cast<int>(L / s);
  int sz = static_cast<int>(L / s);
  for (int iz = 0; iz < sz; iz++) {
    for (int iy = 0; iy < sy; iy++) {
      for (int ix = 0; ix < sx; ix++) {
        double x = ix*s;
        double y = iy*s;
        double z = iz*s;
        add_particle(x     ,y   ,z);
        add_particle(x     ,y+hs,z+hs);
        add_particle(x+hs  ,y   ,z+hs);
        add_particle(x+hs  ,y+hs,z);
      }
    }
  }
  for (int i = 0; i < particle_number; i++) {
    px[i] = 0.0;
    py[i] = 0.0;
    pz[i] = 0.0;
  }
}
//----------------------------------------------------------------------
void
force_pair(void){
  const auto nps = number_of_pairs;
  for(int k=0;k<number_of_pairs;k++){
    const int i = i_particles[k];
    const int j = j_particles[k];
    double dx = qx[j] - qx[i];
    double dy = qy[j] - qy[i];
    double dz = qz[j] - qz[i];
    double r2 = (dx * dx + dy * dy + dz * dz);
    if (r2 > CL2) continue;
    double r6 = r2 * r2 * r2;
    double df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
    px[i] += df * dx;
    py[i] += df * dy;
    pz[i] += df * dz;
    px[j] -= df * dx;
    py[j] -= df * dy;
    pz[j] -= df * dz;
  }
}
//----------------------------------------------------------------------
void
force_sorted(void) {
  const auto pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const auto qx_key = qx[i];
    const auto qy_key = qy[i];
    const auto qz_key = qz[i];
    const auto np = number_of_partners[i];
    double pfx = 0, pfy = 0, pfz = 0;
    const auto kp = pointer[i];
    for (int k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = qx[j] - qx_key;
      const auto dy = qy[j] - qy_key;
      const auto dz = qz[j] - qz_key;
      const auto r2 = (dx*dx + dy*dy + dz*dz);
      if (r2 > CL2) continue;
      const auto r6 = r2*r2*r2;
      const auto df = ((24.0 * r6 - 48.0)/(r6 * r6 * r2)) * dt;
      pfx += df*dx;
      pfy += df*dy;
      pfz += df*dz;
      px[j] -= df*dx;
      py[j] -= df*dy;
      pz[j] -= df*dz;
    } // end of k loop
    px[i] += pfx;
    py[i] += pfy;
    pz[i] += pfz;
  } // end of i loop
}
//----------------------------------------------------------------------
void
force_sorted_2d(void) {
  const auto pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const auto qx_key = q[X][i];
    const auto qy_key = q[Y][i];
    const auto qz_key = q[Z][i];
    const auto np = number_of_partners[i];
    double pfx = 0, pfy = 0, pfz = 0;
    const auto kp = pointer[i];
    for (int k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[X][j] - qx_key;
      const auto dy = q[Y][j] - qy_key;
      const auto dz = q[Z][j] - qz_key;
      const auto r2 = (dx*dx + dy*dy + dz*dz);
      if (r2 > CL2) continue;
      const auto r6 = r2*r2*r2;
      const auto df = ((24.0 * r6 - 48.0)/(r6 * r6 * r2)) * dt;
      pfx += df*dx;
      pfy += df*dy;
      pfz += df*dz;
      p[X][j] -= df*dx;
      p[Y][j] -= df*dy;
      p[Z][j] -= df*dz;
    } // end of k loop
    p[X][i] += pfx;
    p[Y][i] += pfy;
    p[Z][i] += pfz;
  } // end of i loop
}

//----------------------------------------------------------------------
void
force_next(void) {
  const int pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const double qx_key = qx[i];
    const double qy_key = qy[i];
    const double qz_key = qz[i];
    double pfx = 0;
    double pfy = 0;
    double pfz = 0;
    const int kp = pointer[i];
    int ja = sorted_list[kp];
    double dxa = qx[ja] - qx_key;
    double dya = qy[ja] - qy_key;
    double dza = qz[ja] - qz_key;
    double df = 0.0;
    double dxb = 0.0, dyb = 0.0, dzb = 0.0;
    int jb = 0;
    const int np = number_of_partners[i];
    for (int k = kp; k < np + kp; k++) {
      const double dx = dxa;
      const double dy = dya;
      const double dz = dza;
      double r2 = (dx * dx + dy * dy + dz * dz);
      const int j = ja;
      ja = sorted_list[k + 1];
      dxa = qx[ja] - qx_key;
      dya = qy[ja] - qy_key;
      dza = qz[ja] - qz_key;
      if (r2 > CL2)continue;
      pfx += df * dxb;
      pfy += df * dyb;
      pfz += df * dzb;

      px[jb] -= df * dxb;
      py[jb] -= df * dyb;
      pz[jb] -= df * dzb;
      const double r6 = r2 * r2 * r2;
      df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      jb = j;
      dxb = dx;
      dyb = dy;
      dzb = dz;
    }
    px[jb] -= df * dxb;
    py[jb] -= df * dyb;
    pz[jb] -= df * dzb;
    px[i] += pfx + df * dxb;
    py[i] += pfy + df * dyb;
    pz[i] += pfz + df * dzb;
  }
}
//----------------------------------------------------------------------
static inline void transpose_4x4x2(__m512d& va,
                                   __m512d& vb,
                                   __m512d& vc,
                                   __m512d& vd) {
  const auto t_a = _mm512_unpacklo_pd(va, vb);
  const auto t_b = _mm512_unpackhi_pd(va, vb);
  const auto t_c = _mm512_unpacklo_pd(vc, vd);
  const auto t_d = _mm512_unpackhi_pd(vc, vd);

  va = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_c);
  vb = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_d);
  vc = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_c);
  vd = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_d);
}
//----------------------------------------------------------------------
static inline void transpose_4x4x2(const __m512d& va,
                                   const __m512d& vb,
                                   const __m512d& vc,
                                   const __m512d& vd,
                                   __m512d& vx,
                                   __m512d& vy,
                                   __m512d& vz) {
  const auto t_a = _mm512_unpacklo_pd(va, vb);
  const auto t_b = _mm512_unpackhi_pd(va, vb);
  const auto t_c = _mm512_unpacklo_pd(vc, vd);
  const auto t_d = _mm512_unpackhi_pd(vc, vd);

  vx = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_c);
  vy = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_d);
  vz = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_c);
}
//----------------------------------------------------------------------
void
force_intrin_v1(void) {
  const auto vc24  = _mm512_set1_pd(24.0 * dt);
  const auto vc48  = _mm512_set1_pd(48.0 * dt);
  const auto vcl2  = _mm512_set1_pd(CL2);
  const auto vzero = _mm512_setzero_pd();
  const auto pn = particle_number;

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_pd(qx[i]);
    const auto vqyi = _mm512_set1_pd(qy[i]);
    const auto vqzi = _mm512_set1_pd(qz[i]);

    auto vpxi = _mm512_setzero_pd();
    auto vpyi = _mm512_setzero_pd();
    auto vpzi = _mm512_setzero_pd();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    for (int k = 0; k < (np / 8) * 8; k += 8) {
      const auto vindex = _mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k]));

      const auto vqxj = _mm512_i32gather_pd(vindex, qx, 8);
      const auto vqyj = _mm512_i32gather_pd(vindex, qy, 8);
      const auto vqzj = _mm512_i32gather_pd(vindex, qz, 8);

      const auto vdx = _mm512_sub_pd(vqxj, vqxi);
      const auto vdy = _mm512_sub_pd(vqyj, vqyi);
      const auto vdz = _mm512_sub_pd(vqzj, vqzi);

      auto vpxj = _mm512_i32gather_pd(vindex, px, 8);
      auto vpyj = _mm512_i32gather_pd(vindex, py, 8);
      auto vpzj = _mm512_i32gather_pd(vindex, pz, 8);

      const auto vr2 = _mm512_fmadd_pd(vdz,
                                       vdz,
                                       _mm512_fmadd_pd(vdy,
                                                       vdy,
                                                       _mm512_mul_pd(vdx, vdx)));
      const auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);

      auto vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                               _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));
      vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);

      vpxi = _mm512_fmadd_pd(vdf, vdx, vpxi);
      vpyi = _mm512_fmadd_pd(vdf, vdy, vpyi);
      vpzi = _mm512_fmadd_pd(vdf, vdz, vpzi);

      vpxj = _mm512_fnmadd_pd(vdf, vdx, vpxj);
      vpyj = _mm512_fnmadd_pd(vdf, vdy, vpyj);
      vpzj = _mm512_fnmadd_pd(vdf, vdz, vpzj);

      _mm512_i32scatter_pd(px, vindex, vpxj, 8);
      _mm512_i32scatter_pd(py, vindex, vpyj, 8);
      _mm512_i32scatter_pd(pz, vindex, vpzj, 8);
    } // end of k loop
    px[i] += _mm512_reduce_add_pd(vpxi);
    py[i] += _mm512_reduce_add_pd(vpyi);
    pz[i] += _mm512_reduce_add_pd(vpzi);

    // remaining loop
    double pfx = 0.0, pfy = 0.0, pfz = 0.0;
    auto qx_key = qx[i], qy_key = qy[i], qz_key = qz[i];
    for (int k = (np / 8) * 8; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = qx[j] - qx_key;
      const auto dy = qy[j] - qy_key;
      const auto dz = qz[j] - qz_key;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
      pfx   += df * dx;
      pfy   += df * dy;
      pfz   += df * dz;
      px[j] -= df * dx;
      py[j] -= df * dy;
      pz[j] -= df * dz;
    } // end of k loop
    px[i] += pfx;
    py[i] += pfy;
    pz[i] += pfz;
  } // end of i loop
}
//----------------------------------------------------------------------
void
force_intrin_v2(void) {
  const auto vc24  = _mm512_set1_pd(24.0 * dt);
  const auto vc48  = _mm512_set1_pd(48.0 * dt);
  const auto vcl2  = _mm512_set1_pd(CL2);
  const auto vzero = _mm512_setzero_pd();
  const auto pn = particle_number;
  const auto vpitch = _mm512_set1_epi64(8);

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_pd(qx[i]);
    const auto vqyi = _mm512_set1_pd(qy[i]);
    const auto vqzi = _mm512_set1_pd(qz[i]);

    auto vpxi = _mm512_setzero_pd();
    auto vpyi = _mm512_setzero_pd();
    auto vpzi = _mm512_setzero_pd();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    const auto vnp = _mm512_set1_epi64(np);
    auto vk_idx = _mm512_set_epi64(7LL, 6LL, 5LL, 4LL,
                                   3LL, 2LL, 1LL, 0LL);
    const auto num_loop = ((np - 1) / 8 + 1) * 8;

    for (int k = 0; k < num_loop; k += 8) {
      const auto vindex = _mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k]));

      const auto mask = _mm512_cmp_epi64_mask(vk_idx,
                                              vnp,
                                              _MM_CMPINT_LT);

      const auto vqxj = _mm512_i32gather_pd(vindex, &qx[0], 8);
      const auto vqyj = _mm512_i32gather_pd(vindex, &qy[0], 8);
      const auto vqzj = _mm512_i32gather_pd(vindex, &qz[0], 8);

      const auto vdx = _mm512_sub_pd(vqxj, vqxi);
      const auto vdy = _mm512_sub_pd(vqyj, vqyi);
      const auto vdz = _mm512_sub_pd(vqzj, vqzi);

      auto vpxj = _mm512_i32gather_pd(vindex, &px[0], 8);
      auto vpyj = _mm512_i32gather_pd(vindex, &py[0], 8);
      auto vpzj = _mm512_i32gather_pd(vindex, &pz[0], 8);

      const auto vr2 = _mm512_fmadd_pd(vdz,
                                       vdz,
                                       _mm512_fmadd_pd(vdy,
                                                       vdy,
                                                       _mm512_mul_pd(vdx, vdx)));

      const auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);

      auto vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                               _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));

      vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);

      vdf = _mm512_mask_blend_pd(mask, vzero, vdf);

      vpxi = _mm512_fmadd_pd(vdf, vdx, vpxi);
      vpyi = _mm512_fmadd_pd(vdf, vdy, vpyi);
      vpzi = _mm512_fmadd_pd(vdf, vdz, vpzi);

      vpxj = _mm512_fnmadd_pd(vdf, vdx, vpxj);
      vpyj = _mm512_fnmadd_pd(vdf, vdy, vpyj);
      vpzj = _mm512_fnmadd_pd(vdf, vdz, vpzj);

      _mm512_mask_i32scatter_pd(&px[0], mask, vindex, vpxj, 8);
      _mm512_mask_i32scatter_pd(&py[0], mask, vindex, vpyj, 8);
      _mm512_mask_i32scatter_pd(&pz[0], mask, vindex, vpzj, 8);

      vk_idx = _mm512_add_epi64(vk_idx, vpitch);
    } // end of k loop
    px[i] += _mm512_reduce_add_pd(vpxi);
    py[i] += _mm512_reduce_add_pd(vpyi);
    pz[i] += _mm512_reduce_add_pd(vpzi);
  } // end of i loop
}
//----------------------------------------------------------------------
void
force_intrin_v3(void) {
  const auto vc24  = _mm512_set1_pd(24.0 * dt);
  const auto vc48  = _mm512_set1_pd(48.0 * dt);
  const auto vcl2  = _mm512_set1_pd(CL2);
  const auto vzero = _mm512_setzero_pd();
  const auto pn = particle_number;
  const auto vpitch = _mm512_set1_epi64(8);

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_pd(qx[i]);
    const auto vqyi = _mm512_set1_pd(qy[i]);
    const auto vqzi = _mm512_set1_pd(qz[i]);

    auto vpxi = _mm512_setzero_pd();
    auto vpyi = _mm512_setzero_pd();
    auto vpzi = _mm512_setzero_pd();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    const auto vnp = _mm512_set1_epi64(np);
    auto vk_idx = _mm512_set_epi64(7LL, 6LL, 5LL, 4LL,
                                   3LL, 2LL, 1LL, 0LL);
    const auto num_loop = ((np - 1) / 8 + 1) * 8;

    // initial force calculation
    // load position
    auto vindex_a = _mm256_lddqu_si256((const __m256i*)(&sorted_list[kp]));
    auto mask_a = _mm512_cmp_epi64_mask(vk_idx,
                                        vnp,
                                        _MM_CMPINT_LT);
    auto vqxj = _mm512_i32gather_pd(vindex_a, &qx[0], 8);
    auto vqyj = _mm512_i32gather_pd(vindex_a, &qy[0], 8);
    auto vqzj = _mm512_i32gather_pd(vindex_a, &qz[0], 8);

    // calc distance
    auto vdx_a = _mm512_sub_pd(vqxj, vqxi);
    auto vdy_a = _mm512_sub_pd(vqyj, vqyi);
    auto vdz_a = _mm512_sub_pd(vqzj, vqzi);
    auto vr2 = _mm512_fmadd_pd(vdz_a,
                               vdz_a,
                               _mm512_fmadd_pd(vdy_a,
                                               vdy_a,
                                               _mm512_mul_pd(vdx_a, vdx_a)));

    // calc force norm
    auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);

    auto vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                             _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));
    vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                               vzero, vdf);
    vdf = _mm512_mask_blend_pd(mask_a, vzero, vdf);

    for (int k = 8; k < num_loop; k += 8) {
      // load position
      auto vindex_b = _mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k]));
      vk_idx = _mm512_add_epi64(vk_idx, vpitch);
      auto mask_b = _mm512_cmp_epi64_mask(vk_idx,
                                          vnp,
                                          _MM_CMPINT_LT);
      vqxj = _mm512_i32gather_pd(vindex_b, &qx[0], 8);
      vqyj = _mm512_i32gather_pd(vindex_b, &qy[0], 8);
      vqzj = _mm512_i32gather_pd(vindex_b, &qz[0], 8);

      // calc distance
      auto vdx_b = _mm512_sub_pd(vqxj, vqxi);
      auto vdy_b = _mm512_sub_pd(vqyj, vqyi);
      auto vdz_b = _mm512_sub_pd(vqzj, vqzi);
      vr2 = _mm512_fmadd_pd(vdz_b,
                            vdz_b,
                            _mm512_fmadd_pd(vdy_b,
                                            vdy_b,
                                            _mm512_mul_pd(vdx_b,
                                                          vdx_b)));

      // write back j particle momentum
      vpxi = _mm512_fmadd_pd(vdf, vdx_a, vpxi);
      vpyi = _mm512_fmadd_pd(vdf, vdy_a, vpyi);
      vpzi = _mm512_fmadd_pd(vdf, vdz_a, vpzi);

      auto vpxj = _mm512_i32gather_pd(vindex_a, &px[0], 8);
      auto vpyj = _mm512_i32gather_pd(vindex_a, &py[0], 8);
      auto vpzj = _mm512_i32gather_pd(vindex_a, &pz[0], 8);

      vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
      vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
      vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

      _mm512_mask_i32scatter_pd(&px[0], mask_a, vindex_a, vpxj, 8);
      _mm512_mask_i32scatter_pd(&py[0], mask_a, vindex_a, vpyj, 8);
      _mm512_mask_i32scatter_pd(&pz[0], mask_a, vindex_a, vpzj, 8);

      // calc force norm
      vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);
      vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                          _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));
      vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);
      vdf = _mm512_mask_blend_pd(mask_b, vzero, vdf);

      // send to next
      vindex_a = vindex_b;
      mask_a   = mask_b;
      vdx_a    = vdx_b;
      vdy_a    = vdy_b;
      vdz_a    = vdz_b;
    } // end of k loop

    // final write back momentum
    // write back j particle momentum
    vpxi = _mm512_fmadd_pd(vdf, vdx_a, vpxi);
    vpyi = _mm512_fmadd_pd(vdf, vdy_a, vpyi);
    vpzi = _mm512_fmadd_pd(vdf, vdz_a, vpzi);

    auto vpxj = _mm512_i32gather_pd(vindex_a, &px[0], 8);
    auto vpyj = _mm512_i32gather_pd(vindex_a, &py[0], 8);
    auto vpzj = _mm512_i32gather_pd(vindex_a, &pz[0], 8);

    vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
    vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
    vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

    _mm512_mask_i32scatter_pd(&px[0], mask_a, vindex_a, vpxj, 8);
    _mm512_mask_i32scatter_pd(&py[0], mask_a, vindex_a, vpyj, 8);
    _mm512_mask_i32scatter_pd(&pz[0], mask_a, vindex_a, vpzj, 8);

    // write back i particle momentum
    px[i] += _mm512_reduce_add_pd(vpxi);
    py[i] += _mm512_reduce_add_pd(vpyi);
    pz[i] += _mm512_reduce_add_pd(vpzi);
  } // end of i loop
}
//----------------------------------------------------------------------
void
measure(void(*pfunc)(), const char *name) {
  const auto beg = std::chrono::system_clock::now();
  const int LOOP = 100;
  for (int i = 0; i < LOOP; i++) {
    pfunc();
  }
  const auto end = std::chrono::system_clock::now();
  const long dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
  fprintf(stderr, "N=%d, %s %ld [ms]\n", particle_number, name, dur);
}
//----------------------------------------------------------------------
void
loadpair(void){
  std::ifstream ifs(pairlist_cache_file_name,std::ios::binary);
  ifs.read((char*)&number_of_pairs,sizeof(int));
  ifs.read((char*)number_of_partners,sizeof(int)*N);
  ifs.read((char*)i_particles,sizeof(int)*MAX_PAIRS);
  ifs.read((char*)j_particles,sizeof(int)*MAX_PAIRS);
}
//----------------------------------------------------------------------
void
savepair(void){
  makepair();
  random_shfl();
  std::ofstream ofs(pairlist_cache_file_name,std::ios::binary);
  ofs.write((char*)&number_of_pairs,sizeof(int));
  ofs.write((char*)number_of_partners,sizeof(int)*N);
  ofs.write((char*)i_particles,sizeof(int)*MAX_PAIRS);
  ofs.write((char*)j_particles,sizeof(int)*MAX_PAIRS);
}
//----------------------------------------------------------------------
void
print_result(void){
  for (int i = 0; i < 5; i++) {
    printf("%.10f %.10f %.10f\n", px[i], py[i], pz[i]);
  }
  for (int i = particle_number-5; i < particle_number; i++) {
    printf("%.10f %.10f %.10f\n", px[i], py[i], pz[i]);
  }
}
//----------------------------------------------------------------------
int
main(void) {
  allocate();
  init();
  for(int i=0;i<particle_number;i++){
    q[X][i] = qx[i];
    q[Y][i] = qy[i];
    q[Z][i] = qz[i];
    p[X][i] = px[i];
    p[Y][i] = py[i];
    p[Z][i] = pz[i];
  }
  struct stat st;
  int ret = stat(pairlist_cache_file_name, &st);
  if (ret == 0) {
    std::cerr << "A pair-file is found. I use it." << std::endl;
    loadpair();
  } else {
    std::cerr << "Make pairlist." << std::endl;
    savepair();
  }
  std::cerr << "Number of pairs: " << number_of_pairs << std::endl;
  sortpair();
#ifdef PAIR
  measure(&force_pair, "pair");
  print_result();
#elif SORTED
  measure(&force_sorted, "sorted");
  print_result();
#elif SORTED_2D
  measure(&force_sorted_2d, "sorted_2d");
  for(int i=0;i<particle_number;i++){
    qx[i] = q[X][i];
    qy[i] = q[Y][i];
    qz[i] = q[Z][i];
    px[i] = p[X][i];
    py[i] = p[Y][i];
    pz[i] = p[Z][i];
  }
  print_result();
#elif NEXT
  measure(&force_next, "next");
  print_result();
#elif INTRIN_v1
  measure(&force_intrin_v1, "with scatter & gather");
  print_result();
#elif INTRIN_v2
  measure(&force_intrin_v2, "with scatter & gather, remaining loop opt");
  print_result();
#elif INTRIN_v3
  measure(&force_intrin_v3, "with scatter & gather, remaining loop opt, swp");
  print_result();
#else
  measure(&force_pair, "pair");
  measure(&force_sorted, "sorted");
  measure(&force_sorted_2d, "sorted_2d");
  measure(&force_next, "next");
  measure(&force_intrin_v1, "with scatter & gather");
  measure(&force_intrin_v2, "with scatter & gather, remaining loop opt");
  measure(&force_intrin_v3, "with scatter & gather, remaining loop opt, swp");
#endif
  deallocate();
}
//----------------------------------------------------------------------
