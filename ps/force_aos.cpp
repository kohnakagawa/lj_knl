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
// const float density = 1.0f;
const float density = 0.5f;
const int N = 400000;
#ifdef REACTLESS
const int MAX_PAIRS = 60 * N;
#else
const int MAX_PAIRS = 30 * N;
#endif
const float L = 50.0f;
const float dt = 0.001f;

#ifdef REACTLESS
const char* pairlist_cache_file_name = "pair_all.dat";
#else
const char* pairlist_cache_file_name = "pair.dat";
#endif

struct float4 {float x, y, z, w;};
typedef float4 Vec;

Vec* q = nullptr;
Vec* p = nullptr;

int particle_number = 0;
int number_of_pairs = 0;
int* number_of_partners = nullptr;
int i_particles[MAX_PAIRS];
int j_particles[MAX_PAIRS];
int32_t* pointer = nullptr;
int32_t pointer2[N];
int* sorted_list = nullptr;

const float CUTOFF_LENGTH = 3.0f;
const float SEARCH_LENGTH = 3.3f;
const float CL2 = CUTOFF_LENGTH * CUTOFF_LENGTH;
//----------------------------------------------------------------------
void
print512d(__m512d r) {
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
print512(__m512 r) {
  union {
    __m512 r;
    float elem[16];
  } tmp;
  tmp.r = r;
  printf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n",
         tmp.elem[0],  tmp.elem[1],  tmp.elem[2],  tmp.elem[3],
         tmp.elem[4],  tmp.elem[5],  tmp.elem[6],  tmp.elem[7],
         tmp.elem[8],  tmp.elem[9],  tmp.elem[10], tmp.elem[11],
         tmp.elem[12], tmp.elem[13], tmp.elem[14], tmp.elem[15]);
}
//----------------------------------------------------------------------
void
add_particle(float x, float y, float z) {
  static std::mt19937 mt(2);
  std::uniform_real_distribution<float> ud(0.0f, 0.1f);
  q[particle_number].x = x + ud(mt);
  q[particle_number].y = y + ud(mt);
  q[particle_number].z = z + ud(mt);
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
  const float SL2 = SEARCH_LENGTH * SEARCH_LENGTH;
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
      const float dx = q[i].x - q[j].x;
      const float dy = q[i].y - q[j].y;
      const float dz = q[i].z - q[j].z;
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
  posix_memalign((void**)(&q), 64, sizeof(Vec) * N);
  posix_memalign((void**)(&p), 64, sizeof(Vec) * N);

  posix_memalign((void**)(&number_of_partners), 64, sizeof(int) * N);
  posix_memalign((void**)(&pointer), 64, sizeof(int32_t) * N);
  posix_memalign((void**)(&sorted_list), 64, sizeof(int32_t) * MAX_PAIRS);

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
  free(q);
  free(p);
  free(number_of_partners);
  free(pointer);
  free(sorted_list);
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
    p[i].x = 0.0f;
    p[i].y = 0.0f;
    p[i].z = 0.0f;
  }
}
//----------------------------------------------------------------------
void
force_pair(void){
  const auto nps = number_of_pairs;
  for (int k = 0; k < nps; k++) {
    const int i = i_particles[k];
    const int j = j_particles[k];
    auto dx = q[j].x - q[i].x;
    auto dy = q[j].y - q[i].y;
    auto dz = q[j].z - q[i].z;
    auto r2 = (dx * dx + dy * dy + dz * dz);
    if (r2 > CL2) continue;
    auto r6 = r2 * r2 * r2;
    auto df = ((24.0f * r6 - 48.0f) / (r6 * r6 * r2)) * dt;
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
force_sorted(void) {
  const auto pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const auto qx_key = q[i].x;
    const auto qy_key = q[i].y;
    const auto qz_key = q[i].z;
    const auto np = number_of_partners[i];
    float pfx = 0.0f, pfy = 0.0f, pfz = 0.0f;
    const auto kp = pointer[i];
    for (int k = 0; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qx_key;
      const auto dy = q[j].y - qy_key;
      const auto dz = q[j].z - qz_key;
      const auto r2 = (dx*dx + dy*dy + dz*dz);
      if (r2 > CL2) continue;
      const auto r6 = r2*r2*r2;
      const auto df = ((24.0f * r6 - 48.0f)/(r6 * r6 * r2)) * dt;
      pfx += df*dx;
      pfy += df*dy;
      pfz += df*dz;
      p[j].x -= df*dx;
      p[j].y -= df*dy;
      p[j].z -= df*dz;
    } // end of k loop
    p[i].x += pfx;
    p[i].y += pfy;
    p[i].z += pfz;
  } // end of i loop
}
//----------------------------------------------------------------------
void
force_next(void) {
  const int pn = particle_number;
  for (int i = 0; i < pn; i++) {
    const float qx_key = q[i].x;
    const float qy_key = q[i].y;
    const float qz_key = q[i].z;
    float pfx = 0.0f, pfy = 0.0f, pfz = 0.0f;
    const int kp = pointer[i];
    int ja = sorted_list[kp];
    float dxa = q[ja].x - qx_key;
    float dya = q[ja].y - qy_key;
    float dza = q[ja].z - qz_key;
    float df = 0.0;
    float dxb = 0.0f, dyb = 0.0f, dzb = 0.0f;
    int jb = 0;

    const int np = number_of_partners[i];
    for (int k = kp; k < np + kp; k++) {

      const float dx = dxa;
      const float dy = dya;
      const float dz = dza;
      float r2 = (dx * dx + dy * dy + dz * dz);
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
      const float r6 = r2 * r2 * r2;
      df = ((24.0f * r6 - 48.0f) / (r6 * r6 * r2)) * dt;
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
// intrin (with scatter & gather)
void
force_intrin_v1(void) {
  const auto vc24  = _mm512_set1_ps(24.0f * dt);
  const auto vc48  = _mm512_set1_ps(48.0f * dt);
  const auto vcl2  = _mm512_set1_ps(CL2);
  const auto vzero = _mm512_setzero_ps();
  const auto pn = particle_number;

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_ps(q[i].x);
    const auto vqyi = _mm512_set1_ps(q[i].y);
    const auto vqzi = _mm512_set1_ps(q[i].z);

    auto vpxi = _mm512_setzero_ps();
    auto vpyi = _mm512_setzero_ps();
    auto vpzi = _mm512_setzero_ps();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    for (int k = 0; k < (np / 16) * 16; k += 16) {
      const auto vindex = _mm512_slli_epi32(_mm512_loadu_si512((const __m512i*)(&sorted_list[kp + k])),
                                            2);

      const auto vqxj = _mm512_i32gather_ps(vindex, &q[0].x, 4);
      const auto vqyj = _mm512_i32gather_ps(vindex, &q[0].y, 4);
      const auto vqzj = _mm512_i32gather_ps(vindex, &q[0].z, 4);

      const auto vdx = _mm512_sub_ps(vqxj, vqxi);
      const auto vdy = _mm512_sub_ps(vqyj, vqyi);
      const auto vdz = _mm512_sub_ps(vqzj, vqzi);

      auto vpxj = _mm512_i32gather_ps(vindex, &p[0].x, 4);
      auto vpyj = _mm512_i32gather_ps(vindex, &p[0].y, 4);
      auto vpzj = _mm512_i32gather_ps(vindex, &p[0].z, 4);

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

      _mm512_i32scatter_ps(&p[0].x, vindex, vpxj, 4);
      _mm512_i32scatter_ps(&p[0].y, vindex, vpyj, 4);
      _mm512_i32scatter_ps(&p[0].z, vindex, vpzj, 4);
    } // end of k loop

    // remaining loop
    float pfx = _mm512_reduce_add_ps(vpxi);
    float pfy = _mm512_reduce_add_ps(vpyi);
    float pfz = _mm512_reduce_add_ps(vpzi);;
    auto qx_key = q[i].x, qy_key = q[i].y, qz_key = q[i].z;
    for (int k = (np / 16) * 16; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qx_key;
      const auto dy = q[j].y - qy_key;
      const auto dz = q[j].z - qz_key;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = ((24.0f * r6 - 48.0f) / (r6 * r6 * r2)) * dt;
      pfx    += df * dx;
      pfy    += df * dy;
      pfz    += df * dz;
      p[j].x -= df * dx;
      p[j].y -= df * dy;
      p[j].z -= df * dz;
    } // end of k loop
    p[i].x += pfx;
    p[i].y += pfy;
    p[i].z += pfz;
  } // end of i loop
}
//----------------------------------------------------------------------
// intrin (with scatter & gather) + remove remaining loop
void
force_intrin_v2(void) {
  const auto vc24  = _mm512_set1_ps(24.0f * dt);
  const auto vc48  = _mm512_set1_ps(48.0f * dt);
  const auto vcl2  = _mm512_set1_ps(CL2);
  const auto vzero = _mm512_setzero_ps();
  const auto pn = particle_number;
  const auto vpitch = _mm512_set1_epi32(16);

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_ps(q[i].x);
    const auto vqyi = _mm512_set1_ps(q[i].y);
    const auto vqzi = _mm512_set1_ps(q[i].z);

    auto vpxi = _mm512_setzero_ps();
    auto vpyi = _mm512_setzero_ps();
    auto vpzi = _mm512_setzero_ps();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    const auto vnp = _mm512_set1_epi32(np);
    auto vk_idx = _mm512_set_epi32(15, 14, 13, 12,
                                   11, 10, 9, 8,
                                   7, 6, 5, 4,
                                   3, 2, 1, 0);
    const auto num_loop = ((np - 1) / 16 + 1) * 16;

    for (int k = 0; k < num_loop; k += 16) {
      const auto vindex = _mm512_slli_epi32(_mm512_loadu_si512((const __m512i*)(&sorted_list[kp + k])),
                                            2);

      const auto mask = _mm512_cmp_epi32_mask(vk_idx,
                                              vnp,
                                              _MM_CMPINT_LT);

      const auto vqxj = _mm512_i32gather_ps(vindex, &q[0].x, 4);
      const auto vqyj = _mm512_i32gather_ps(vindex, &q[0].y, 4);
      const auto vqzj = _mm512_i32gather_ps(vindex, &q[0].z, 4);

      const auto vdx = _mm512_sub_ps(vqxj, vqxi);
      const auto vdy = _mm512_sub_ps(vqyj, vqyi);
      const auto vdz = _mm512_sub_ps(vqzj, vqzi);

      auto vpxj = _mm512_i32gather_ps(vindex, &p[0].x, 4);
      auto vpyj = _mm512_i32gather_ps(vindex, &p[0].y, 4);
      auto vpzj = _mm512_i32gather_ps(vindex, &p[0].z, 4);

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

      vdf = _mm512_mask_blend_ps(mask, vzero, vdf);

      vpxi = _mm512_fmadd_ps(vdf, vdx, vpxi);
      vpyi = _mm512_fmadd_ps(vdf, vdy, vpyi);
      vpzi = _mm512_fmadd_ps(vdf, vdz, vpzi);

      vpxj = _mm512_fnmadd_ps(vdf, vdx, vpxj);
      vpyj = _mm512_fnmadd_ps(vdf, vdy, vpyj);
      vpzj = _mm512_fnmadd_ps(vdf, vdz, vpzj);

      _mm512_mask_i32scatter_ps(&p[0].x, mask, vindex, vpxj, 4);
      _mm512_mask_i32scatter_ps(&p[0].y, mask, vindex, vpyj, 4);
      _mm512_mask_i32scatter_ps(&p[0].z, mask, vindex, vpzj, 4);

      vk_idx = _mm512_add_epi32(vk_idx, vpitch);
    } // end of k loop

    p[i].x += _mm512_reduce_add_ps(vpxi);
    p[i].y += _mm512_reduce_add_ps(vpyi);
    p[i].z += _mm512_reduce_add_ps(vpzi);
  } // end of i loop
}
//----------------------------------------------------------------------
// intrin (with scatter & gather) + swp + remove remaining loop
void
force_intrin_v3(void) {
  const auto vc24  = _mm512_set1_ps(24.0f * dt);
  const auto vc48  = _mm512_set1_ps(48.0f * dt);
  const auto vcl2  = _mm512_set1_ps(CL2);
  const auto vzero = _mm512_setzero_ps();
  const auto pn = particle_number;
  const auto vpitch = _mm512_set1_epi32(16);

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_ps(q[i].x);
    const auto vqyi = _mm512_set1_ps(q[i].y);
    const auto vqzi = _mm512_set1_ps(q[i].z);

    auto vpxi = _mm512_setzero_ps();
    auto vpyi = _mm512_setzero_ps();
    auto vpzi = _mm512_setzero_ps();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    const auto vnp = _mm512_set1_epi32(np);
    auto vk_idx = _mm512_set_epi32(15, 14, 13, 12,
                                   11, 10, 9, 8,
                                   7, 6, 5, 4,
                                   3, 2, 1, 0);
    const auto num_loop = ((np - 1) / 16 + 1) * 16;

    // initial force calculation
    // load position
    auto vindex_a = _mm512_slli_epi32(_mm512_loadu_si512((const __m512i*)(&sorted_list[kp])),
                                      2);
    auto mask_a = _mm512_cmp_epi32_mask(vk_idx,
                                        vnp,
                                        _MM_CMPINT_LT);
    auto vqxj = _mm512_i32gather_ps(vindex_a, &q[0].x, 4);
    auto vqyj = _mm512_i32gather_ps(vindex_a, &q[0].y, 4);
    auto vqzj = _mm512_i32gather_ps(vindex_a, &q[0].z, 4);

    // calc distance
    auto vdx_a = _mm512_sub_ps(vqxj, vqxi);
    auto vdy_a = _mm512_sub_ps(vqyj, vqyi);
    auto vdz_a = _mm512_sub_ps(vqzj, vqzi);
    auto vr2   = _mm512_fmadd_ps(vdz_a,
                                 vdz_a,
                                 _mm512_fmadd_ps(vdy_a,
                                                 vdy_a,
                                                 _mm512_mul_ps(vdx_a, vdx_a)));

    // calc force norm
    auto vr6 = _mm512_mul_ps(_mm512_mul_ps(vr2, vr2), vr2);

    auto vdf = _mm512_div_ps(_mm512_fmsub_ps(vc24, vr6, vc48),
                             _mm512_mul_ps(_mm512_mul_ps(vr6, vr6), vr2));
    vdf = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(vr2, vcl2, _CMP_LE_OS),
                               vzero, vdf);
    vdf = _mm512_mask_blend_ps(mask_a, vzero, vdf);

    for (int k = 16; k < num_loop; k += 16) {
      // load position
      auto vindex_b = _mm512_slli_epi32(_mm512_loadu_si512((const __m512i*)(&sorted_list[kp + k])),
                                        2);
      vk_idx = _mm512_add_epi32(vk_idx, vpitch);
      auto mask_b = _mm512_cmp_epi32_mask(vk_idx,
                                          vnp,
                                          _MM_CMPINT_LT);
      vqxj = _mm512_i32gather_ps(vindex_b, &q[0].x, 4);
      vqyj = _mm512_i32gather_ps(vindex_b, &q[0].y, 4);
      vqzj = _mm512_i32gather_ps(vindex_b, &q[0].z, 4);

      // calc distance
      auto vdx_b = _mm512_sub_ps(vqxj, vqxi);
      auto vdy_b = _mm512_sub_ps(vqyj, vqyi);
      auto vdz_b = _mm512_sub_ps(vqzj, vqzi);
      vr2 = _mm512_fmadd_ps(vdz_b,
                            vdz_b,
                            _mm512_fmadd_ps(vdy_b,
                                            vdy_b,
                                            _mm512_mul_ps(vdx_b,
                                                          vdx_b)));

      // write back j particle momentum
      vpxi = _mm512_fmadd_ps(vdf, vdx_a, vpxi);
      vpyi = _mm512_fmadd_ps(vdf, vdy_a, vpyi);
      vpzi = _mm512_fmadd_ps(vdf, vdz_a, vpzi);

      auto vpxj = _mm512_i32gather_ps(vindex_a, &p[0].x, 4);
      auto vpyj = _mm512_i32gather_ps(vindex_a, &p[0].y, 4);
      auto vpzj = _mm512_i32gather_ps(vindex_a, &p[0].z, 4);

      vpxj = _mm512_fnmadd_ps(vdf, vdx_a, vpxj);
      vpyj = _mm512_fnmadd_ps(vdf, vdy_a, vpyj);
      vpzj = _mm512_fnmadd_ps(vdf, vdz_a, vpzj);

      _mm512_mask_i32scatter_ps(&p[0].x, mask_a, vindex_a, vpxj, 4);
      _mm512_mask_i32scatter_ps(&p[0].y, mask_a, vindex_a, vpyj, 4);
      _mm512_mask_i32scatter_ps(&p[0].z, mask_a, vindex_a, vpzj, 4);

      // calc force norm
      vr6 = _mm512_mul_ps(_mm512_mul_ps(vr2, vr2), vr2);
      vdf = _mm512_div_ps(_mm512_fmsub_ps(vc24, vr6, vc48),
                          _mm512_mul_ps(_mm512_mul_ps(vr6, vr6), vr2));
      vdf = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(vr2, vcl2, _CMP_LE_OS),
                                 vzero, vdf);
      vdf = _mm512_mask_blend_ps(mask_b, vzero, vdf);

      // send to next
      vindex_a = vindex_b;
      mask_a   = mask_b;
      vdx_a    = vdx_b;
      vdy_a    = vdy_b;
      vdz_a    = vdz_b;
    } // end of k loop

    // final write back momentum
    // write back j particle momentum
    vpxi = _mm512_fmadd_ps(vdf, vdx_a, vpxi);
    vpyi = _mm512_fmadd_ps(vdf, vdy_a, vpyi);
    vpzi = _mm512_fmadd_ps(vdf, vdz_a, vpzi);

    auto vpxj = _mm512_i32gather_ps(vindex_a, &p[0].x, 4);
    auto vpyj = _mm512_i32gather_ps(vindex_a, &p[0].y, 4);
    auto vpzj = _mm512_i32gather_ps(vindex_a, &p[0].z, 4);

    vpxj = _mm512_fnmadd_ps(vdf, vdx_a, vpxj);
    vpyj = _mm512_fnmadd_ps(vdf, vdy_a, vpyj);
    vpzj = _mm512_fnmadd_ps(vdf, vdz_a, vpzj);

    _mm512_mask_i32scatter_ps(&p[0].x, mask_a, vindex_a, vpxj, 4);
    _mm512_mask_i32scatter_ps(&p[0].y, mask_a, vindex_a, vpyj, 4);
    _mm512_mask_i32scatter_ps(&p[0].z, mask_a, vindex_a, vpzj, 4);

    // write back i particle momentum
    p[i].x += _mm512_reduce_add_ps(vpxi);
    p[i].y += _mm512_reduce_add_ps(vpyi);
    p[i].z += _mm512_reduce_add_ps(vpzi);
  } // end of i loop
}
//----------------------------------------------------------------------
void
force_intrin_v1_reactless(void) {
  const auto vc24  = _mm512_set1_ps(24.0f * dt);
  const auto vc48  = _mm512_set1_ps(48.0f * dt);
  const auto vcl2  = _mm512_set1_ps(CL2);
  const auto vzero = _mm512_setzero_ps();
  const auto pn = particle_number;

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_ps(q[i].x);
    const auto vqyi = _mm512_set1_ps(q[i].y);
    const auto vqzi = _mm512_set1_ps(q[i].z);

    auto vpxi = _mm512_setzero_ps();
    auto vpyi = _mm512_setzero_ps();
    auto vpzi = _mm512_setzero_ps();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    for (int k = 0; k < (np / 16) * 16; k += 16) {
      const auto vindex = _mm512_slli_epi32(_mm512_loadu_si512((const __m512i*)(&sorted_list[kp + k])),
                                            2);

      const auto vqxj = _mm512_i32gather_ps(vindex, &q[0].x, 4);
      const auto vqyj = _mm512_i32gather_ps(vindex, &q[0].y, 4);
      const auto vqzj = _mm512_i32gather_ps(vindex, &q[0].z, 4);

      const auto vdx = _mm512_sub_ps(vqxj, vqxi);
      const auto vdy = _mm512_sub_ps(vqyj, vqyi);
      const auto vdz = _mm512_sub_ps(vqzj, vqzi);

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
    } // end of k loop

    auto pfx = p[i].x + _mm512_reduce_add_ps(vpxi);
    auto pfy = p[i].y + _mm512_reduce_add_ps(vpyi);
    auto pfz = p[i].z + _mm512_reduce_add_ps(vpzi);
    auto qx_key = q[i].x, qy_key = q[i].y, qz_key = q[i].z;
#pragma novector
    for (int k = (np / 16) * 16; k < np; k++) {
      const auto j = sorted_list[kp + k];
      const auto dx = q[j].x - qx_key;
      const auto dy = q[j].y - qy_key;
      const auto dz = q[j].z - qz_key;
      const auto r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > CL2) continue;
      const auto r6 = r2 * r2 * r2;
      const auto df = (24.0f * r6 - 48.0f) / (r6 * r6 * r2) * dt;
      pfx += df * dx;
      pfy += df * dy;
      pfz += df * dz;
    } // end of k loop
    p[i].x = pfx;
    p[i].y = pfy;
    p[i].z = pfz;
  } // end of i loop
}
//----------------------------------------------------------------------
void
force_intrin_v2_reactless(void) {
  const auto vc24  = _mm512_set1_ps(24.0f * dt);
  const auto vc48  = _mm512_set1_ps(48.0f * dt);
  const auto vcl2  = _mm512_set1_ps(CL2);
  const auto vzero = _mm512_setzero_ps();
  const auto pn = particle_number;
  const auto vpitch = _mm512_set1_epi32(16);

  for (int i = 0; i < pn; i++) {
    const auto vqxi = _mm512_set1_ps(q[i].x);
    const auto vqyi = _mm512_set1_ps(q[i].y);
    const auto vqzi = _mm512_set1_ps(q[i].z);

    auto vpxi = _mm512_setzero_ps();
    auto vpyi = _mm512_setzero_ps();
    auto vpzi = _mm512_setzero_ps();

    const auto np = number_of_partners[i];
    const auto kp = pointer[i];
    const auto vnp = _mm512_set1_epi32(np);
    auto vk_idx = _mm512_set_epi32(15, 14, 13, 12,
                                   11, 10, 9, 8,
                                   7, 6, 5, 4,
                                   3, 2, 1, 0);
    const auto num_loop = ((np - 1) / 16 + 1) * 16;

    for (int k = 0; k < num_loop; k += 16) {
      const auto vindex = _mm512_slli_epi32(_mm512_loadu_si512((const __m512i*)(&sorted_list[kp + k])),
                                            2);

      const auto mask = _mm512_cmp_epi32_mask(vk_idx,
                                              vnp,
                                              _MM_CMPINT_LT);

      const auto vqxj = _mm512_i32gather_ps(vindex, &q[0].x, 4);
      const auto vqyj = _mm512_i32gather_ps(vindex, &q[0].y, 4);
      const auto vqzj = _mm512_i32gather_ps(vindex, &q[0].z, 4);

      const auto vdx = _mm512_sub_ps(vqxj, vqxi);
      const auto vdy = _mm512_sub_ps(vqyj, vqyi);
      const auto vdz = _mm512_sub_ps(vqzj, vqzi);

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

      vdf = _mm512_mask_blend_ps(mask, vzero, vdf);

      vpxi = _mm512_fmadd_ps(vdf, vdx, vpxi);
      vpyi = _mm512_fmadd_ps(vdf, vdy, vpyi);
      vpzi = _mm512_fmadd_ps(vdf, vdz, vpzi);

      vk_idx = _mm512_add_epi32(vk_idx, vpitch);
    } // end of k loop

    p[i].x += _mm512_reduce_add_ps(vpxi);
    p[i].y += _mm512_reduce_add_ps(vpyi);
    p[i].z += _mm512_reduce_add_ps(vpzi);
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
check(void) {
  for (int i = 0; i < number_of_pairs; i++) {
    assert(sorted_list[i] >= 0 && sorted_list[i] < particle_number);
  }
  for (int i = 0; i < particle_number; i++) {
    assert(number_of_partners[i] >= 0 && number_of_partners[i] < particle_number);
    assert(pointer[i] >= 0 && pointer[i] <= number_of_pairs);
  }
}
//----------------------------------------------------------------------
void
print_result(void){
  for (int i = 0; i < 5; i++) {
    printf("%f %f %f\n", p[i].x, p[i].y, p[i].z);
  }
  for (int i = particle_number-5; i < particle_number; i++) {
    printf("%f %f %f\n", p[i].x, p[i].y, p[i].z);
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
  check();
#ifdef PAIR
  measure(&force_pair, "pair");
  print_result();
#elif NEXT
  measure(&force_next, "next");
  print_result();
#elif SORTED
  measure(&force_sorted, "sorted");
  print_result();
#elif defined INTRIN_v1 && defined REACTLESS
  measure(&force_intrin_v1_reactless, "with gather, reactless");
  print_result();
#elif defined INTRIN_v2 && defined REACTLESS
  measure(&force_intrin_v2_reactless, "without gather, reactless");
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
  measure(&force_next, "next");
  measure(&force_intrin_v1, "with scatter & gather");
  measure(&force_intrin_v2, "with scatter & gather, remaining loop opt");
  measure(&force_intrin_v3, "with scatter & gather, remaining loop opt, swp");
#endif
  deallocate();
}
//----------------------------------------------------------------------
