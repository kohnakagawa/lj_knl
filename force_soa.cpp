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
const int N = 400000;
#ifdef REACTLESS
const int MAX_PAIRS = 60 * N;
#else
const int MAX_PAIRS = 30 * N;
#endif
const double L = 50.0;
const double dt = 0.001;

#ifdef REACTLESS
const char* pairlist_cache_file_name = "pair_all.dat";
#else
const char* pairlist_cache_file_name = "pair.dat";
#endif

double* qx = nullptr;
double* qy = nullptr;
double* qz = nullptr;

double* px = nullptr;
double* py = nullptr;
double* pz = nullptr;

int particle_number = 0;
int number_of_pairs = 0;
int* number_of_partners = nullptr;
int i_particles[MAX_PAIRS];
int j_particles[MAX_PAIRS];
int32_t* pointer = nullptr;
int32_t pointer2[N];
int* sorted_list = nullptr;

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
#ifdef REACTLESS
  i = index1;
  j = index2;
#else
  if (index1 < index2) {
    i = index1;
    j = index2;
  } else {
    i = index2;
    j = index1;
  }
#endif
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
#ifdef REACTLESS
  for (int i = 0; i < particle_number; i++) {
    for (int j = 0; j < particle_number; j++) {
      if (i == j) continue;
#else
  for (int i = 0; i < particle_number - 1; i++) {
    for (int j = i + 1; j < particle_number; j++) {
#endif
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
first_touch(void) {
  const int est_N = 119164;
#ifdef REACTLESS
  const int est_pair_num = 2 * 7839886;
#else
  const int est_pair_num = 7839886;
#endif

#pragma omp parallel for
  for (int i = 0; i < est_N; i++) {
    qx[i] = qy[i] = qz[i] = 0.0;
    px[i] = py[i] = pz[i] = 0.0;
    number_of_partners[i] = pointer[i] = 0;
  }

#pragma omp parallel for
  for (int i = 0; i < est_pair_num; i++) {
    sorted_list[i] = 0;
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

#ifdef _OPENMP
  first_touch();
#endif

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
#pragma omp parallel
  {
    const auto nps = number_of_pairs;
#pragma omp for nowait
#pragma novector
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
#pragma omp atomic
      px[i] += df * dx;
#pragma omp atomic
      py[i] += df * dy;
#pragma omp atomic
      pz[i] += df * dz;
#pragma omp atomic
      px[j] -= df * dx;
#pragma omp atomic
      py[j] -= df * dy;
#pragma omp atomic
      pz[j] -= df * dz;
    }
  }
}
//----------------------------------------------------------------------
void
force_next(void) {
#pragma omp parallel
  {
    const int pn = particle_number;
#pragma omp for nowait
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
#pragma omp atomic
        px[jb] -= df * dxb;
#pragma omp atomic
        py[jb] -= df * dyb;
#pragma omp atomic
        pz[jb] -= df * dzb;
        const double r6 = r2 * r2 * r2;
        df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
        jb = j;
        dxb = dx;
        dyb = dy;
        dzb = dz;
      }
#pragma omp atomic
      px[jb] -= df * dxb;
#pragma omp atomic
      py[jb] -= df * dyb;
#pragma omp atomic
      pz[jb] -= df * dzb;
#pragma omp atomic
      px[i] += pfx + df * dxb;
#pragma omp atomic
      py[i] += pfy + df * dyb;
#pragma omp atomic
      pz[i] += pfz + df * dzb;
    }
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
#pragma omp parallel
  {
    const auto vc24  = _mm512_set1_pd(24.0 * dt);
    const auto vc48  = _mm512_set1_pd(48.0 * dt);
    const auto vcl2  = _mm512_set1_pd(CL2);
    const auto vzero = _mm512_setzero_pd();
    const auto pn = particle_number;

#pragma omp for nowait
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

        const auto vqxj = _mm512_i32gather_pd(vindex, &qx[0], 8);
        const auto vqyj = _mm512_i32gather_pd(vindex, &qy[0], 8);
        const auto vqzj = _mm512_i32gather_pd(vindex, &qz[0], 8);

        const auto vdx = _mm512_sub_pd(vqxj, vqxi);
        const auto vdy = _mm512_sub_pd(vqyj, vqyi);
        const auto vdz = _mm512_sub_pd(vqzj, vqzi);

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
      } // end of k loop
#if 0
      // horizontal sum
      auto vpwi = _mm512_setzero_pd();
      transpose_4x4x2(vpxi, vpyi, vpzi, vpwi);
      // vpxi = {vpia, vpie}
      // vpyi = {vpib, vpif}
      // vpzi = {vpic, vpig}
      // vpwi = {vpid, vpih}

      auto vpi_hilo = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(vpxi,
                                                                vpyi),
                                                  vpzi),
                                    vpwi);
      auto vpi_lohi = _mm512_castsi512_pd(
                                          _mm512_alignr_epi64(_mm512_castpd_si512(vpi_hilo),
                                                              _mm512_castpd_si512(vpi_hilo),
                                                              4));
      auto vpi = _mm512_castpd512_pd256(_mm512_add_pd(vpi_hilo, vpi_lohi));
      double* pi = reinterpret_cast<double*>(&vpi);

      auto pfx = px[i] + pi[0], pfy = py[i] + pi[1], pfz = pz[i] + pi[2];
#else
      auto pfx = px[i] + _mm512_reduce_add_pd(vpxi);
      auto pfy = py[i] + _mm512_reduce_add_pd(vpyi);
      auto pfz = pz[i] + _mm512_reduce_add_pd(vpzi);
#endif

#pragma novector
      for (int k = (np / 8) * 8; k < np; k++) {
        const auto j = sorted_list[kp + k];
        const auto dx = qx[j] - qx[i];
        const auto dy = qy[j] - qy[i];
        const auto dz = qz[j] - qz[i];
        const auto r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > CL2) continue;
        const auto r6 = r2 * r2 * r2;
        const auto df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
        pfx += df * dx;
        pfy += df * dy;
        pfz += df * dz;
      } // end of k loop
      px[i] = pfx;
      py[i] = pfy;
      pz[i] = pfz;
    } // end of i loop
  } // end of pragma omp parallel
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
  fprintf(stderr, "N=%d, %s %ld [microsec]\n", particle_number, name, dur);
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
#elif NEXT
  measure(&force_next, "next");
  print_result();
#elif defined INTRIN_v1 && defined REACTLESS
  measure(&force_intrin_v1, "intrin_v1");
  print_result();
#endif
  deallocate();
}
//----------------------------------------------------------------------
