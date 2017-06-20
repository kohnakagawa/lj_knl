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
const float density = 1.0f;
// const float density = 0.5f;
const int N = 400000;
const int MAX_PAIRS = 30 * N;
const float L = 50.0f;
const float dt = 0.001f;

const char* pairlist_cache_file_name = "pair.dat";

float* __restrict qx = nullptr;
float* __restrict qy = nullptr;
float* __restrict qz = nullptr;

float* __restrict px = nullptr;
float* __restrict py = nullptr;
float* __restrict pz = nullptr;

const int D = 3;
enum {X, Y, Z};
float q[D][N];
float p[D][N];

int particle_number = 0;
int number_of_pairs = 0;
int* __restrict number_of_partners = nullptr;
int i_particles[MAX_PAIRS];
int j_particles[MAX_PAIRS];
int32_t* __restrict pointer = nullptr;
int32_t pointer2[N];
int* __restrict sorted_list = nullptr;

const float CUTOFF_LENGTH = 3.0f;
const float SEARCH_LENGTH = 3.3f;
const float CL2 = CUTOFF_LENGTH * CUTOFF_LENGTH;
//----------------------------------------------------------------------
void
print512(__m512 r) {
  union {
    __m512 r;
    float elem[16];
  } tmp;
  tmp.r = r;
  for (int i = 0; i < 16; i++) {
    printf("%.10f ", tmp.elem[i]);
  }
  printf("\n");
}
//----------------------------------------------------------------------
void
add_particle(float x, float y, float z) {
  static std::mt19937 mt(2);
  std::uniform_real_distribution<float> ud(0.0, 0.1);
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
  const float SL2 = SEARCH_LENGTH * SEARCH_LENGTH;
  const int pn = particle_number;
  for (int i = 0; i < pn; i++) {
    number_of_partners[i] = 0;
  }
  for (int i = 0; i < particle_number - 1; i++) {
    for (int j = i + 1; j < particle_number; j++) {
      const float dx = qx[i] - qx[j];
      const float dy = qy[i] - qy[j];
      const float dz = qz[i] - qz[j];
      const float r2 = dx * dx + dy * dy + dz * dz;
      if (r2 < SL2) {
        register_pair(i, j);
      }
    }
  }
}
//----------------------------------------------------------------------
void
allocate(void) {
  posix_memalign((void**)(&qx), 64, sizeof(float) * N);
  posix_memalign((void**)(&qy), 64, sizeof(float) * N);
  posix_memalign((void**)(&qz), 64, sizeof(float) * N);

  posix_memalign((void**)(&px), 64, sizeof(float) * N);
  posix_memalign((void**)(&py), 64, sizeof(float) * N);
  posix_memalign((void**)(&pz), 64, sizeof(float) * N);

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
  const float s = 1.0f / std::pow(density * 0.25f, 1.0f / 3.0f);
  const float hs = s * 0.5f;
  int sx = static_cast<int>(L / s);
  int sy = static_cast<int>(L / s);
  int sz = static_cast<int>(L / s);
  for (int iz = 0; iz < sz; iz++) {
    for (int iy = 0; iy < sy; iy++) {
      for (int ix = 0; ix < sx; ix++) {
        float x = ix*s;
        float y = iy*s;
        float z = iz*s;
        add_particle(x     ,y   ,z);
        add_particle(x     ,y+hs,z+hs);
        add_particle(x+hs  ,y   ,z+hs);
        add_particle(x+hs  ,y+hs,z);
      }
    }
  }
  for (int i = 0; i < particle_number; i++) {
    px[i] = 0.0f;
    py[i] = 0.0f;
    pz[i] = 0.0f;
  }
}
//----------------------------------------------------------------------
void
force_pair(void){
  const auto nps = number_of_pairs;
  for(int k=0;k<number_of_pairs;k++){
    const int i = i_particles[k];
    const int j = j_particles[k];
    float dx = qx[j] - qx[i];
    float dy = qy[j] - qy[i];
    float dz = qz[j] - qz[i];
    float r2 = (dx * dx + dy * dy + dz * dz);
    if (r2 > CL2) continue;
    float r6 = r2 * r2 * r2;
    float df = ((24.0f * r6 - 48.0f) / (r6 * r6 * r2)) * dt;
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
    float pfx = 0, pfy = 0, pfz = 0;
    const auto kp = pointer[i];
#pragma simd
    for (int k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = qx[j] - qx_key;
      const auto dy = qy[j] - qy_key;
      const auto dz = qz[j] - qz_key;
      const auto r2 = (dx*dx + dy*dy + dz*dz);
      if (r2 > CL2) continue;
      const auto r6 = r2*r2*r2;
      const auto df = ((24.0f * r6 - 48.0f)/(r6 * r6 * r2)) * dt;
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
    float pfx = 0, pfy = 0, pfz = 0;
    const auto kp = pointer[i];
#pragma simd
    for (int k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[X][j] - qx_key;
      const auto dy = q[Y][j] - qy_key;
      const auto dz = q[Z][j] - qz_key;
      const auto r2 = (dx*dx + dy*dy + dz*dz);
      if (r2 > CL2) continue;
      const auto r6 = r2*r2*r2;
      const auto df = ((24.0f * r6 - 48.0f)/(r6 * r6 * r2)) * dt;
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
    const float qx_key = qx[i];
    const float qy_key = qy[i];
    const float qz_key = qz[i];
    float pfx = 0;
    float pfy = 0;
    float pfz = 0;
    const int kp = pointer[i];
    int ja = sorted_list[kp];
    float dxa = qx[ja] - qx_key;
    float dya = qy[ja] - qy_key;
    float dza = qz[ja] - qz_key;
    float df = 0.0;
    float dxb = 0.0, dyb = 0.0, dzb = 0.0;
    int jb = 0;
    const int np = number_of_partners[i];
    for (int k = kp; k < np + kp; k++) {
      const float dx = dxa;
      const float dy = dya;
      const float dz = dza;
      float r2 = (dx * dx + dy * dy + dz * dz);
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
      const float r6 = r2 * r2 * r2;
      df = ((24.0f * r6 - 48.0f) / (r6 * r6 * r2)) * dt;
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
void
force_intrin_v1(void) {
  const auto vc24  = _mm512_set1_ps(24.0f * dt);
  const auto vc48  = _mm512_set1_ps(48.0f * dt);
  const auto vcl2  = _mm512_set1_ps(CL2);
  const auto vzero = _mm512_setzero_ps();
  const auto pn = particle_number;

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_ps(qx[i]);
    const auto vqyi = _mm512_set1_ps(qy[i]);
    const auto vqzi = _mm512_set1_ps(qz[i]);

    auto vpxi = _mm512_setzero_ps();
    auto vpyi = _mm512_setzero_ps();
    auto vpzi = _mm512_setzero_ps();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    for (int k = 0; k < (np / 16) * 16; k += 16) {
      const auto vindex = _mm512_loadu_si512(&sorted_list[kp + k]);

      const auto vqxj = _mm512_i32gather_ps(vindex, qx, 4);
      const auto vqyj = _mm512_i32gather_ps(vindex, qy, 4);
      const auto vqzj = _mm512_i32gather_ps(vindex, qz, 4);

      const auto vdx = _mm512_sub_ps(vqxj, vqxi);
      const auto vdy = _mm512_sub_ps(vqyj, vqyi);
      const auto vdz = _mm512_sub_ps(vqzj, vqzi);

      auto vpxj = _mm512_i32gather_ps(vindex, px, 4);
      auto vpyj = _mm512_i32gather_ps(vindex, py, 4);
      auto vpzj = _mm512_i32gather_ps(vindex, pz, 4);

      const auto vr2 = _mm512_fmadd_ps(vdz,
                                       vdz,
                                       _mm512_fmadd_ps(vdy,
                                                       vdy,
                                                       _mm512_mul_ps(vdx, vdx)));
      const auto vr6 = _mm512_mul_ps(_mm512_mul_ps(vr2, vr2), vr2);

      auto vdf = _mm512_div_ps(_mm512_fmsub_ps(vc24, vr6, vc48),
                               _mm512_mul_ps(_mm512_mul_ps(vr6, vr6), vr2));
      vdf = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);

      vpxi = _mm512_fmadd_ps(vdf, vdx, vpxi);
      vpyi = _mm512_fmadd_ps(vdf, vdy, vpyi);
      vpzi = _mm512_fmadd_ps(vdf, vdz, vpzi);

      vpxj = _mm512_fnmadd_ps(vdf, vdx, vpxj);
      vpyj = _mm512_fnmadd_ps(vdf, vdy, vpyj);
      vpzj = _mm512_fnmadd_ps(vdf, vdz, vpzj);

      _mm512_i32scatter_ps(px, vindex, vpxj, 4);
      _mm512_i32scatter_ps(py, vindex, vpyj, 4);
      _mm512_i32scatter_ps(pz, vindex, vpzj, 4);
    } // end of k loop
    px[i] += _mm512_reduce_add_ps(vpxi);
    py[i] += _mm512_reduce_add_ps(vpyi);
    pz[i] += _mm512_reduce_add_ps(vpzi);

    // remaining loop
    float pfx = 0.0, pfy = 0.0, pfz = 0.0;
    auto qx_key = qx[i], qy_key = qy[i], qz_key = qz[i];
    for (int k = (np / 16) * 16; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = qx[j] - qx_key;
      const auto dy = qy[j] - qy_key;
      const auto dz = qz[j] - qz_key;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = (24.0f * r6 - 48.0f) / (r6 * r6 * r2) * dt;
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
  const auto vc24   = _mm512_set1_ps(24.0f * dt);
  const auto vc48   = _mm512_set1_ps(48.0f * dt);
  const auto vcl2   = _mm512_set1_ps(CL2);
  const auto vzero  = _mm512_setzero_ps();
  const auto pn     = particle_number;
  const auto vpitch = _mm512_set1_epi32(16);

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_ps(qx[i]);
    const auto vqyi = _mm512_set1_ps(qy[i]);
    const auto vqzi = _mm512_set1_ps(qz[i]);

    auto vpxi = _mm512_setzero_ps();
    auto vpyi = _mm512_setzero_ps();
    auto vpzi = _mm512_setzero_ps();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    const int* ptr_list = &sorted_list[kp];

    const auto vnp = _mm512_set1_epi32(np);
    auto vk_idx = _mm512_set_epi32(15, 14, 13, 12,
                                   11, 10, 9 , 8,
                                   7 , 6 , 5 , 4,
                                   3 , 2 , 1 , 0);
    for (int k = 0; k < np; k += 16) {
      const auto vindex = _mm512_loadu_si512(ptr_list);
      ptr_list += 16;

      const auto lt_np = _mm512_cmp_epi32_mask(vk_idx,
                                               vnp,
                                               _MM_CMPINT_LT);

      const auto vqxj = _mm512_mask_i32gather_ps(vzero, lt_np, vindex, &qx[0], 4);
      const auto vqyj = _mm512_mask_i32gather_ps(vzero, lt_np, vindex, &qy[0], 4);
      const auto vqzj = _mm512_mask_i32gather_ps(vzero, lt_np, vindex, &qz[0], 4);

      const auto vdx = _mm512_sub_ps(vqxj, vqxi);
      const auto vdy = _mm512_sub_ps(vqyj, vqyi);
      const auto vdz = _mm512_sub_ps(vqzj, vqzi);

      const auto vr2 = _mm512_fmadd_ps(vdz,
                                       vdz,
                                       _mm512_fmadd_ps(vdy,
                                                       vdy,
                                                       _mm512_mul_ps(vdx, vdx)));

      const auto vr6 = _mm512_mul_ps(_mm512_mul_ps(vr2, vr2), vr2);

      const auto vdf_nume = _mm512_fmsub_ps(vc24, vr6, vc48);
      const auto vdf_deno = _mm512_mul_ps(_mm512_mul_ps(vr6, vr6), vr2);
      const auto vdf_deno_inv = _mm512_rcp28_ps(vdf_deno);
      auto vdf = _mm512_mul_ps(vdf_nume, vdf_deno_inv);

      const auto le_cl2 = _mm512_cmp_ps_mask(vr2, vcl2, _CMP_LE_OS);
      const auto mask = _mm512_kand(lt_np, le_cl2);

      vdf = _mm512_mask_blend_ps(mask, vzero, vdf);

      vpxi = _mm512_fmadd_ps(vdf, vdx, vpxi);
      vpyi = _mm512_fmadd_ps(vdf, vdy, vpyi);
      vpzi = _mm512_fmadd_ps(vdf, vdz, vpzi);

      auto vpxj = _mm512_mask_i32gather_ps(vzero, mask, vindex, &px[0], 4);
      auto vpyj = _mm512_mask_i32gather_ps(vzero, mask, vindex, &py[0], 4);
      auto vpzj = _mm512_mask_i32gather_ps(vzero, mask, vindex, &pz[0], 4);

      vpxj = _mm512_fnmadd_ps(vdf, vdx, vpxj);
      vpyj = _mm512_fnmadd_ps(vdf, vdy, vpyj);
      vpzj = _mm512_fnmadd_ps(vdf, vdz, vpzj);

      _mm512_mask_i32scatter_ps(&px[0], mask, vindex, vpxj, 4);
      _mm512_mask_i32scatter_ps(&py[0], mask, vindex, vpyj, 4);
      _mm512_mask_i32scatter_ps(&pz[0], mask, vindex, vpzj, 4);

      vk_idx = _mm512_add_epi32(vk_idx, vpitch);
    } // end of k loop

    px[i] += _mm512_reduce_add_ps(vpxi);
    py[i] += _mm512_reduce_add_ps(vpyi);
    pz[i] += _mm512_reduce_add_ps(vpzi);
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
    printf("%.5f %.5f %.5f\n", px[i], py[i], pz[i]);
  }
  for (int i = particle_number-5; i < particle_number; i++) {
    printf("%.5f %.5f %.5f\n", px[i], py[i], pz[i]);
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
#else
  measure(&force_pair, "pair");
  measure(&force_sorted, "sorted");
  measure(&force_sorted_2d, "sorted_2d");
  measure(&force_next, "next");
  measure(&force_intrin_v1, "with scatter & gather");
  measure(&force_intrin_v2, "with scatter & gather, remaining loop opt");
#endif
  deallocate();
}
//----------------------------------------------------------------------
