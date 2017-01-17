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

struct double4 {double x, y, z, w;};
double4* __restrict q = nullptr;
double4* __restrict p = nullptr;

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
  q[particle_number].x = x + ud(mt);
  q[particle_number].y = y + ud(mt);
  q[particle_number].z = z + ud(mt);
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
      const double dx = q[i].x - q[j].x;
      const double dy = q[i].y - q[j].y;
      const double dz = q[i].z - q[j].z;
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
  posix_memalign((void**)(&q), 64, sizeof(double4) * N);
  posix_memalign((void**)(&p), 64, sizeof(double4) * N);
}
//----------------------------------------------------------------------
void
deallocate(void) {
  free(q);
  free(p);
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
    p[i].x = 0.0;
    p[i].y = 0.0;
    p[i].z = 0.0;
  }
}
//----------------------------------------------------------------------
void
force_pair(void){
  for(int k=0;k<number_of_pairs;k++){
    const int i = i_particles[k];
    const int j = j_particles[k];
    double dx = q[j].x - q[i].x;
    double dy = q[j].y - q[i].y;
    double dz = q[j].z - q[i].z;
    double r2 = (dx * dx + dy * dy + dz * dz);
    if (r2 > CL2) continue;
    double r6 = r2 * r2 * r2;
    double df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
    p[i].x += df * dx;
    p[i].y += df * dy;
    p[i].z += df * dz;
    p[j].x -= df * dx;
    p[j].y -= df * dy;
    p[j].z -= df * dz;
  }
}
//----------------------------------------------------------------------
void
force_next(void) {
  const int pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const double qx_key = q[i].x;
    const double qy_key = q[i].y;
    const double qz_key = q[i].z;
    double pfx = 0;
    double pfy = 0;
    double pfz = 0;
    const int kp = pointer[i];
    int ja = sorted_list[kp];
    double dxa = q[ja].x - qx_key;
    double dya = q[ja].y - qy_key;
    double dza = q[ja].z - qz_key;
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
      dxa = q[ja].x - qx_key;
      dya = q[ja].y - qy_key;
      dza = q[ja].z - qz_key;
      if (r2 > CL2)continue;
      pfx += df * dxb;
      pfy += df * dyb;
      pfz += df * dzb;
      p[jb].x -= df * dxb;
      p[jb].y -= df * dyb;
      p[jb].z -= df * dzb;
      const double r6 = r2 * r2 * r2;
      df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      jb = j;
      dxb = dx;
      dyb = dy;
      dzb = dz;
    }
    p[jb].x -= df * dxb;
    p[jb].y -= df * dyb;
    p[jb].z -= df * dzb;
    p[i].x += pfx + df * dxb;
    p[i].y += pfy + df * dyb;
    p[i].z += pfz + df * dzb;
  }
}
//----------------------------------------------------------------------
static inline v8df
_mm512_load2_m256d(const double* hiaddr,
                   const double* loaddr) {
  v8df ret;
  ret = _mm512_insertf64x4(ret, _mm256_load_pd(loaddr), 0x0);
  ret = _mm512_insertf64x4(ret, _mm256_load_pd(hiaddr), 0x1);
  return ret;
}
//----------------------------------------------------------------------
static inline void
_mm512_store2_m256d(double* hiaddr,
                    double* loaddr,
                    const v8df& dat) {
  _mm256_store_pd(loaddr, _mm512_castpd512_pd256(dat));
  _mm256_store_pd(hiaddr, _mm512_extractf64x4_pd(dat, 0x1));
}
//----------------------------------------------------------------------
static inline void
transpose_4x4x2(v8df& va,
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
static inline void
transpose_4x4x2(const v8df& va,
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
    v8df vqxi = _mm512_set1_pd(q[i].x);
    v8df vqyi = _mm512_set1_pd(q[i].y);
    v8df vqzi = _mm512_set1_pd(q[i].z);

    v8df vpxi = _mm512_setzero_pd();
    v8df vpyi = _mm512_setzero_pd();
    v8df vpzi = _mm512_setzero_pd();

    const int np = number_of_partners[i];
    const int kp = pointer[i];
    for (int k = 0; k < (np / 8) * 8; k += 8) {
      v8si vindex = _mm256_lddqu_si256(reinterpret_cast<__m256i*>(&sorted_list[kp + k]));
      vindex = _mm256_slli_epi32(vindex, 2);

      v8df vqxj = _mm512_i32gather_pd(vindex, &q[0].x, 8);
      v8df vqyj = _mm512_i32gather_pd(vindex, &q[0].y, 8);
      v8df vqzj = _mm512_i32gather_pd(vindex, &q[0].z, 8);

      v8df vpxj = _mm512_i32gather_pd(vindex, &p[0].x, 8);
      v8df vpyj = _mm512_i32gather_pd(vindex, &p[0].y, 8);
      v8df vpzj = _mm512_i32gather_pd(vindex, &p[0].z, 8);
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

      _mm512_i32scatter_pd(&p[0].x, vindex, vpxj, 8);
      _mm512_i32scatter_pd(&p[0].y, vindex, vpyj, 8);
      _mm512_i32scatter_pd(&p[0].z, vindex, vpzj, 8);
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
    v8df vpi = vpi_hilo + vpi_lohi;

    // store
    vpi += static_cast<v8df>(_mm512_castpd256_pd512(_mm256_load_pd(&p[i].x)));
    _mm256_store_pd(&p[i].x, _mm512_castpd512_pd256(vpi));

    for (int k = (np / 8) * 8; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - q[i].x;
      const auto dy = q[j].y - q[i].y;
      const auto dz = q[j].z - q[i].z;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = (24.0 * r6 - 48.0) / (r6 * r6 * r2) * dt;
      p[i].x += df * dx;
      p[i].y += df * dy;
      p[i].z += df * dz;
      p[j].x -= df * dx;
      p[j].y -= df * dy;
      p[j].z -= df * dz;
    }
  }
}
//----------------------------------------------------------------------
void
force_intrin_v2(void) {
  const v8df vc24  = _mm512_set1_pd(24.0 * dt);
  const v8df vc48  = _mm512_set1_pd(48.0 * dt);
  const v8df vcl2  = _mm512_set1_pd(CL2);
  const v8df vzero = _mm512_setzero_pd();
  for (int i = 0; i < N; i++) {
    v8df vqi = _mm512_castpd256_pd512(_mm256_load_pd(&q[i].x));
    vqi = _mm512_insertf64x4(vqi, _mm512_castpd512_pd256(vqi), 0x1);
    v8df vpi = _mm512_setzero_pd();

    const int np = number_of_partners[i];
    const int kp = pointer[i];
    for (int k = 0; k < (np / 8) * 8; k += 8) {
      const auto j_a = sorted_list[kp + k    ];
      const auto j_b = sorted_list[kp + k + 1];
      const auto j_c = sorted_list[kp + k + 2];
      const auto j_d = sorted_list[kp + k + 3];

      const auto j_e = sorted_list[kp + k + 4];
      const auto j_f = sorted_list[kp + k + 5];
      const auto j_g = sorted_list[kp + k + 6];
      const auto j_h = sorted_list[kp + k + 7];

      v8df vpj_ea = _mm512_load2_m256d(&p[j_e].x, &p[j_a].x);
      v8df vpj_fb = _mm512_load2_m256d(&p[j_f].x, &p[j_b].x);
      v8df vpj_gc = _mm512_load2_m256d(&p[j_g].x, &p[j_c].x);
      v8df vpj_hd = _mm512_load2_m256d(&p[j_h].x, &p[j_d].x);

      v8df vqj_ea = _mm512_load2_m256d(&q[j_e].x, &q[j_a].x);
      v8df vqj_fb = _mm512_load2_m256d(&q[j_f].x, &q[j_b].x);
      v8df vqj_gc = _mm512_load2_m256d(&q[j_g].x, &q[j_c].x);
      v8df vqj_hd = _mm512_load2_m256d(&q[j_h].x, &q[j_d].x);

      v8df vdq_ea = vqj_ea - vqi;
      v8df vdq_fb = vqj_fb - vqi;
      v8df vdq_gc = vqj_gc - vqi;
      v8df vdq_hd = vqj_hd - vqi;

      v8df vdx, vdy, vdz;
      transpose_4x4x2(vdq_ea, vdq_fb, vdq_gc, vdq_hd,
                      vdx, vdy, vdz);

      v8df vr2 = vdx * vdx + vdy * vdy + vdz * vdz;
      v8df vr6 = vr2 * vr2 * vr2;
      v8df vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

      vdf = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);

      v8df vdf_ea = _mm512_permutex_pd(vdf, 0x00);
      v8df vdf_fb = _mm512_permutex_pd(vdf, 0x55);
      v8df vdf_gc = _mm512_permutex_pd(vdf, 0xaa);
      v8df vdf_hd = _mm512_permutex_pd(vdf, 0xff);

      vpi    += vdf_ea * vdq_ea;
      vpj_ea -= vdf_ea * vdq_ea;

      vpi    += vdf_fb * vdq_fb;
      vpj_fb -= vdf_fb * vdq_fb;

      vpi    += vdf_gc * vdq_gc;
      vpj_gc -= vdf_gc * vdq_gc;

      vpi    += vdf_hd * vdq_hd;
      vpj_hd -= vdf_hd * vdq_hd;

      _mm512_store2_m256d(&p[j_e].x, &p[j_a].x, vpj_ea);
      _mm512_store2_m256d(&p[j_f].x, &p[j_b].x, vpj_fb);
      _mm512_store2_m256d(&p[j_g].x, &p[j_c].x, vpj_gc);
      _mm512_store2_m256d(&p[j_h].x, &p[j_d].x, vpj_hd);
    }
    vpi = _mm512_add_pd(vpi,
                        _mm512_permutexvar_pd(_mm512_set_epi64(0x3, 0x2, 0x1, 0x0, 0x7, 0x6, 0x5, 0x4),
                                              vpi));
    vpi = _mm512_add_pd(vpi, _mm512_castpd256_pd512(_mm256_load_pd(&p[i].x)));
    _mm256_store_pd(&p[i].x, _mm512_castpd512_pd256(vpi));

    for (int k = (np / 8) * 8; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - q[i].x;
      const auto dy = q[j].y - q[i].y;
      const auto dz = q[j].z - q[i].z;
      const auto r2 = (dx * dx + dy * dy + dz * dz);
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
      p[i].x += df * dx;
      p[i].y += df * dy;
      p[i].z += df * dz;
      p[j].x -= df * dx;
      p[j].y -= df * dy;
      p[j].z -= df * dz;
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
    printf("%.10f %.10f %.10f\n", p[i].x, p[i].y, p[i].z);
  }
  for (int i = particle_number-5; i < particle_number; i++) {
    printf("%.10f %.10f %.10f\n", p[i].x, p[i].y, p[i].z);
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
#elif INTRIN_v2
  measure(&force_intrin_v2, "intrin_v2");
  print_result();
#else
  measure(&force_pair, "pair");
  measure(&force_next, "sorted_next");
  measure(&force_intrin_v1, "intrin_v1");
  measure(&force_intrin_v2, "intrin_v2");
#endif
  deallocate();
}
//----------------------------------------------------------------------
