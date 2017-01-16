#include <iostream>
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
const int MAX_PAIRS = 30 * N;
double L = 50.0;
const double dt = 0.001;

double* __restrict qx = nullptr;
double* __restrict qy = nullptr;
double* __restrict qz = nullptr;

double* __restrict px = nullptr;
double* __restrict py = nullptr;
double* __restrict pz = nullptr;

int particle_number = 0;
int number_of_pairs = 0;
int number_of_partners[N];
int i_particles[MAX_PAIRS];
int j_particles[MAX_PAIRS];
int32_t pointer[N], pointer2[N];
int sorted_list[MAX_PAIRS];

const double CUTOFF_LENGTH = 3.0;
const double SEARCH_LENGTH = 3.3;
const double CL2 = CUTOFF_LENGTH * CUTOFF_LENGTH;
//----------------------------------------------------------------------
typedef double v4df __attribute__((vector_size(32)));
typedef double v8df __attribute__((vector_size(64)));
typedef int32_t v8si __attribute__((vector_size(32)));
//----------------------------------------------------------------------
void
print256(v4df r) {
  double *a = (double*)(&r);
  printf("%.10f %.10f %.10f %.10f\n", a[0], a[1], a[2], a[3]);
}
//----------------------------------------------------------------------
void
print512(v8df r) {
  union {
    v8df r;
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
}
//----------------------------------------------------------------------
void
init(void) {
  const double s = 1.0 / pow(density * 0.25, 1.0 / 3.0);
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
static inline void transpose_4x4x2(v8df& va,
                                   v8df& vb,
                                   v8df& vc,
                                   v8df& vd) {
  v8df t_a = _mm512_unpacklo_pd(va, vb);
  v8df t_b = _mm512_unpackhi_pd(va, vb);
  v8df t_c = _mm512_unpacklo_pd(vc, vd);
  v8df t_d = _mm512_unpackhi_pd(vc, vd);

  va = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_c);
  vb = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_d);
  vc = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_c);
  vd = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_d);
}
//----------------------------------------------------------------------
static inline void transpose_4x4x2(const v8df& va,
                                   const v8df& vb,
                                   const v8df& vc,
                                   const v8df& vd,
                                   v8df& vx,
                                   v8df& vy,
                                   v8df& vz) {
  v8df t_a = _mm512_unpacklo_pd(va, vb);
  v8df t_b = _mm512_unpackhi_pd(va, vb);
  v8df t_c = _mm512_unpacklo_pd(vc, vd);
  v8df t_d = _mm512_unpackhi_pd(vc, vd);

  vx = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_c);
  vy = _mm512_permutex2var_pd(t_b, _mm512_set_epi64(0xd, 0xc, 0x5, 0x4, 0x9, 0x8, 0x1, 0x0), t_d);
  vz = _mm512_permutex2var_pd(t_a, _mm512_set_epi64(0xf, 0xe, 0x7, 0x6, 0xb, 0xa, 0x3, 0x2), t_c);
}
//----------------------------------------------------------------------
void
force_intrin_v1(void) {
  const v8df vc24  = _mm512_set1_pd(24.0 * dt);
  const v8df vc48  = _mm512_set1_pd(48.0 * dt);
  const v8df vcl2  = _mm512_set1_pd(CL2);
  const v8df vzero = _mm512_setzero_pd();
  for (int i = 0; i < N; i++) {
    v8df vqxi = _mm512_set1_pd(qx[i]);
    v8df vqyi = _mm512_set1_pd(qy[i]);
    v8df vqzi = _mm512_set1_pd(qz[i]);

    v8df vpxi = _mm512_setzero_pd();
    v8df vpyi = _mm512_setzero_pd();
    v8df vpzi = _mm512_setzero_pd();

    const int np = number_of_partners[i];
    const int kp = pointer[i];
    for (int k = 0; k < (np / 8) * 8; k += 8) {
      v8si vindex = _mm256_load_si256(reinterpret_cast<__m256i*>(&sorted_list[kp + k]));

      v8df vqxj = _mm512_i32gather_pd(vindex, &qx[0], 8);
      v8df vqyj = _mm512_i32gather_pd(vindex, &qy[0], 8);
      v8df vqzj = _mm512_i32gather_pd(vindex, &qz[0], 8);

      v8df vpxj = _mm512_i32gather_pd(vindex, &px[0], 8);
      v8df vpyj = _mm512_i32gather_pd(vindex, &py[0], 8);
      v8df vpzj = _mm512_i32gather_pd(vindex, &pz[0], 8);

      v8df vdx = vqxj - vqxi;
      v8df vdy = vqyj - vqyi;
      v8df vdz = vqzj - vqzi;

      v8df vr2 = vdx * vdx + vdy * vdy + vdz * vdz;
      v8df vr6 = vr2 * vr2 * vr2;
      v8df vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);

      vpxi += vdf * vdx;
      vpyi += vdf * vdy;
      vpzi += vdf * vdz;

      vpxj -= vdf * vdx;
      vpyj -= vdf * vdy;
      vpzj -= vdf * vdz;

      _mm512_i32scatter_pd(&px[0], vindex, vpxj, 8);
      _mm512_i32scatter_pd(&py[0], vindex, vpyj, 8);
      _mm512_i32scatter_pd(&pz[0], vindex, vpzj, 8);
    }
    // horizontal sum
    v8df vpwi = _mm512_setzero_pd();
    transpose_4x4x2(vpxi, vpyi, vpzi, vpwi);
    // vpxi = {vpia, vpie}
    // vpyi = {vpib, vpif}
    // vpzi = {vpic, vpig}
    // vpwi = {vpid, vpih}

    v8df vpi_hilo = vpxi + vpyi + vpzi + vpwi;
    v8df vpi_lohi = _mm512_permutexvar_pd(_mm512_set_epi64(0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4),
                                          vpi_hilo);
    v4df vpi = static_cast<v4df>(_mm512_castpd512_pd256(vpi_hilo + vpi_lohi));
    vpi += static_cast<v4df>(_mm256_set_pd(0.0, pz[i], py[i], px[i]));
    double* pi = reinterpret_cast<double*>(&vpi);

    px[i] = pi[0];
    py[i] = pi[1];
    pz[i] = pi[2];

    for (int k = (np / 8) * 8; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = qx[j] - qx[i];
      const auto dy = qy[j] - qy[i];
      const auto dz = qz[j] - qz[i];
      const auto r2 = (dx * dx + dy * dy + dz * dz);
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
      px[i] += df * dx;
      py[i] += df * dy;
      pz[i] += df * dz;
      px[j] -= df * dx;
      py[j] -= df * dy;
      pz[j] -= df * dz;
    }
  }
}
//----------------------------------------------------------------------
void
measure(void(*pfunc)(), const char *name) {
  const auto beg = std::chrono::system_clock::now();
  const int LOOP = 20;
  for (int i = 0; i < LOOP; i++) {
    pfunc();
  }
  const auto end = std::chrono::system_clock::now();
  const long dur = std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count();
  fprintf(stderr, "N=%d, %s %ld [microsec]\n", particle_number, name, dur);
}
//----------------------------------------------------------------------
void
loadpair(void){
  std::ifstream ifs("pair.dat",std::ios::binary);
  ifs.read((char*)&number_of_pairs,sizeof(int));
  ifs.read((char*)number_of_partners,sizeof(int)*N);
  ifs.read((char*)i_particles,sizeof(int)*MAX_PAIRS);
  ifs.read((char*)j_particles,sizeof(int)*MAX_PAIRS);
}
//----------------------------------------------------------------------
void
savepair(void){
  makepair();
  std::ofstream ofs("pair.dat",std::ios::binary);
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
  int ret = stat("pair.dat", &st);
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
#elif INTRIN_v1
  measure(&force_intrin_v1, "intrin_v1");
  print_result();
#else
  measure(&force_pair, "pair");
  measure(&force_next, "sorted_next");
  measure(&force_intrin_v1, "intrin_v1");
#endif
  deallocate();
}
//----------------------------------------------------------------------
