#pragma once

namespace skl {
  namespace avx512 {
    __attribute__((noinline))
    void force_intrin(const Vec4* __restrict q,
                      Vec4* __restrict p,
                      const int32_t* __restrict number_of_partners,
                      const int32_t* __restrict pointer,
                      const int32_t* __restrict sorted_list,
                      const int pn) {
      const auto vc24   = _mm512_set1_pd(24.0 * dt);
      const auto vc48   = _mm512_set1_pd(48.0 * dt);
      const auto vcl2   = _mm512_set1_pd(CL2);
      const auto vzero  = _mm512_setzero_pd();
      const auto vpitch = _mm512_set1_epi64(8);

      for (int i = 0; i < pn; i++) {
        const auto vqxi = _mm512_set1_pd(q[i].x);
        const auto vqyi = _mm512_set1_pd(q[i].y);
        const auto vqzi = _mm512_set1_pd(q[i].z);

        auto vpxi = _mm512_setzero_pd();
        auto vpyi = _mm512_setzero_pd();
        auto vpzi = _mm512_setzero_pd();

        const auto np = number_of_partners[i];
        const auto kp = pointer[i];
        const auto vnp = _mm512_set1_epi64(np);
        auto vk_idx = _mm512_set_epi64(7LL, 6LL, 5LL, 4LL,
                                       3LL, 2LL, 1LL, 0LL);

        for (int k = 0; k < np; k += 8) {
          const auto vindex = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k])),
                                                2);
          auto mask = _mm512_cmp_epi64_mask(vk_idx,
                                            vnp,
                                            _MM_CMPINT_LT);

          const auto vqxj = _mm512_i32gather_pd(vindex, &q[0].x, 8);
          const auto vqyj = _mm512_i32gather_pd(vindex, &q[0].y, 8);
          const auto vqzj = _mm512_i32gather_pd(vindex, &q[0].z, 8);

          const auto vdx = vqxj - vqxi;
          const auto vdy = vqyj - vqyi;
          const auto vdz = vqzj - vqzi;

          auto vpxj = _mm512_i32gather_pd(vindex, &p[0].x, 8);
          auto vpyj = _mm512_i32gather_pd(vindex, &p[0].y, 8);
          auto vpzj = _mm512_i32gather_pd(vindex, &p[0].z, 8);

          const auto vr2 = vdx*vdx + vdy*vdy + vdz*vdz;
          const auto vr6 = vr2 * vr2 * vr2;

          auto vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);
          const auto le_cl2 =  _mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS);
          mask = _mm512_kand(mask, le_cl2);
          vdf = _mm512_mask_blend_pd(mask, vzero, vdf);

          vpxi += vdf * vdx;
          vpyi += vdf * vdy;
          vpzi += vdf * vdz;

          vpxj -= vdf * vdx;
          vpyj -= vdf * vdy;
          vpzj -= vdf * vdz;

          _mm512_mask_i32scatter_pd(&p[0].x, mask, vindex, vpxj, 8);
          _mm512_mask_i32scatter_pd(&p[0].y, mask, vindex, vpyj, 8);
          _mm512_mask_i32scatter_pd(&p[0].z, mask, vindex, vpzj, 8);

          vk_idx += vpitch;
        } // end of k loop

        p[i].x += _mm512_reduce_add_pd(vpxi);
        p[i].y += _mm512_reduce_add_pd(vpyi);
        p[i].z += _mm512_reduce_add_pd(vpzi);
      } // end of i loop
    }

    __attribute__((noinline))
    void force_intrin_swp(const Vec4* __restrict q,
                          Vec4* __restrict p,
                          const int32_t* __restrict number_of_partners,
                          const int32_t* __restrict pointer,
                          const int32_t* __restrict sorted_list,
                          const int pn) {
      const auto vc24  = _mm512_set1_pd(24.0 * dt);
      const auto vc48  = _mm512_set1_pd(48.0 * dt);
      const auto vcl2  = _mm512_set1_pd(CL2);
      const auto vzero = _mm512_setzero_pd();
      const auto vpitch = _mm512_set1_epi64(8);

      for (int i = 0; i < pn; i++) {
        const auto vqxi = _mm512_set1_pd(q[i].x);
        const auto vqyi = _mm512_set1_pd(q[i].y);
        const auto vqzi = _mm512_set1_pd(q[i].z);

        auto vpxi = _mm512_setzero_pd();
        auto vpyi = _mm512_setzero_pd();
        auto vpzi = _mm512_setzero_pd();

        const auto np = number_of_partners[i];
        const auto kp = pointer[i];
        const auto vnp = _mm512_set1_epi64(np);
        auto vk_idx = _mm512_set_epi64(7LL, 6LL, 5LL, 4LL,
                                       3LL, 2LL, 1LL, 0LL);

        auto vindex_a = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp])),
                                          2);
        auto mask_a = _mm512_cmp_epi64_mask(vk_idx, vnp, _MM_CMPINT_LT);
        auto vqxj = _mm512_i32gather_pd(vindex_a, &q[0].x, 8);
        auto vqyj = _mm512_i32gather_pd(vindex_a, &q[0].y, 8);
        auto vqzj = _mm512_i32gather_pd(vindex_a, &q[0].z, 8);

        auto vdx_a = vqxj - vqxi;
        auto vdy_a = vqyj - vqyi;
        auto vdz_a = vqzj - vqzi;
        auto vr2   = vdx_a*vdx_a + vdy_a*vdy_a + vdz_a*vdz_a;
        auto vr6   = vr2 * vr2 * vr2;

        auto vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);
        auto le_cl2 = _mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS);
        mask_a = _mm512_kand(mask_a, le_cl2);
        vdf = _mm512_mask_blend_pd(mask_a, vzero, vdf);

        for (int k = 8; k < np; k += 8) {
          auto vindex_b = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k])),
                                            2);
          vk_idx += vpitch;
          auto mask_b = _mm512_cmp_epi64_mask(vk_idx,
                                              vnp,
                                              _MM_CMPINT_LT);
          vqxj = _mm512_i32gather_pd(vindex_b, &q[0].x, 8);
          vqyj = _mm512_i32gather_pd(vindex_b, &q[0].y, 8);
          vqzj = _mm512_i32gather_pd(vindex_b, &q[0].z, 8);

          auto vdx_b = vqxj - vqxi;
          auto vdy_b = vqyj - vqyi;
          auto vdz_b = vqzj - vqzi;
          vr2 = vdx_b*vdx_b + vdy_b*vdy_b + vdz_b*vdz_b;

          vpxi += vdf * vdx_a;
          vpyi += vdf * vdy_a;
          vpzi += vdf * vdz_a;

          auto vpxj = _mm512_i32gather_pd(vindex_a, &p[0].x, 8);
          auto vpyj = _mm512_i32gather_pd(vindex_a, &p[0].y, 8);
          auto vpzj = _mm512_i32gather_pd(vindex_a, &p[0].z, 8);

          vpxj -= vdf * vdx_a;
          vpyj -= vdf * vdy_a;
          vpzj -= vdf * vdz_a;

          _mm512_mask_i32scatter_pd(&p[0].x, mask_a, vindex_a, vpxj, 8);
          _mm512_mask_i32scatter_pd(&p[0].y, mask_a, vindex_a, vpyj, 8);
          _mm512_mask_i32scatter_pd(&p[0].z, mask_a, vindex_a, vpzj, 8);

          vr6 = vr2 * vr2 * vr2;
          vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

          le_cl2 = _mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS);
          mask_b = _mm512_kand(mask_b, le_cl2);
          vdf = _mm512_mask_blend_pd(mask_b, vzero, vdf);

          vindex_a = vindex_b;
          mask_a   = mask_b;
          vdx_a    = vdx_b;
          vdy_a    = vdy_b;
          vdz_a    = vdz_b;
        } // end of k loop

        vpxi += vdf * vdx_a;
        vpyi += vdf * vdy_a;
        vpzi += vdf * vdz_a;

        auto vpxj = _mm512_i32gather_pd(vindex_a, &p[0].x, 8);
        auto vpyj = _mm512_i32gather_pd(vindex_a, &p[0].y, 8);
        auto vpzj = _mm512_i32gather_pd(vindex_a, &p[0].z, 8);

        vpxj -= vdf * vdx_a;
        vpyj -= vdf * vdy_a;
        vpzj -= vdf * vdz_a;

        _mm512_mask_i32scatter_pd(&p[0].x, mask_a, vindex_a, vpxj, 8);
        _mm512_mask_i32scatter_pd(&p[0].y, mask_a, vindex_a, vpyj, 8);
        _mm512_mask_i32scatter_pd(&p[0].z, mask_a, vindex_a, vpzj, 8);

        p[i].x += _mm512_reduce_add_pd(vpxi);
        p[i].y += _mm512_reduce_add_pd(vpyi);
        p[i].z += _mm512_reduce_add_pd(vpzi);
      } // end of i loop
    }

    __attribute__((noinline))
    void force_intrin_z(Vec8* __restrict z,
                        const int32_t* __restrict number_of_partners,
                        const int32_t* __restrict pointer,
                        const int32_t* __restrict sorted_list,
                        const int pn) {
      const auto vc24   = _mm512_set1_pd(24.0 * dt);
      const auto vc48   = _mm512_set1_pd(48.0 * dt);
      const auto vcl2   = _mm512_set1_pd(CL2);
      const auto vzero  = _mm512_setzero_pd();
      const auto vpitch = _mm512_set1_epi64(8);

      for (int i = 0; i < pn; i++) {
        const auto vqxi = _mm512_set1_pd(z[i].x);
        const auto vqyi = _mm512_set1_pd(z[i].y);
        const auto vqzi = _mm512_set1_pd(z[i].z);

        auto vpxi = _mm512_setzero_pd();
        auto vpyi = _mm512_setzero_pd();
        auto vpzi = _mm512_setzero_pd();

        const auto np = number_of_partners[i];
        const auto kp = pointer[i];
        const auto vnp = _mm512_set1_epi64(np);
        auto vk_idx = _mm512_set_epi64(7LL, 6LL, 5LL, 4LL,
                                       3LL, 2LL, 1LL, 0LL);

        for (int k = 0; k < np; k += 8) {
          const auto vindex = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k])),
                                                3);
          auto mask = _mm512_cmp_epi64_mask(vk_idx,
                                            vnp,
                                            _MM_CMPINT_LT);

          const auto vqxj = _mm512_i32gather_pd(vindex, &z[0].x, 8);
          const auto vqyj = _mm512_i32gather_pd(vindex, &z[0].y, 8);
          const auto vqzj = _mm512_i32gather_pd(vindex, &z[0].z, 8);

          const auto vdx = vqxj - vqxi;
          const auto vdy = vqyj - vqyi;
          const auto vdz = vqzj - vqzi;

          auto vpxj = _mm512_i32gather_pd(vindex, &z[0].px, 8);
          auto vpyj = _mm512_i32gather_pd(vindex, &z[0].py, 8);
          auto vpzj = _mm512_i32gather_pd(vindex, &z[0].pz, 8);

          const auto vr2 = vdx*vdx + vdy*vdy + vdz*vdz;
          const auto vr6 = vr2 * vr2 * vr2;

          auto vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);
          const auto le_cl2 =  _mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS);
          mask = _mm512_kand(mask, le_cl2);
          vdf = _mm512_mask_blend_pd(mask, vzero, vdf);

          vpxi += vdf * vdx;
          vpyi += vdf * vdy;
          vpzi += vdf * vdz;

          vpxj -= vdf * vdx;
          vpyj -= vdf * vdy;
          vpzj -= vdf * vdz;

          _mm512_mask_i32scatter_pd(&z[0].px, mask, vindex, vpxj, 8);
          _mm512_mask_i32scatter_pd(&z[0].py, mask, vindex, vpyj, 8);
          _mm512_mask_i32scatter_pd(&z[0].pz, mask, vindex, vpzj, 8);

          vk_idx += vpitch;
        } // end of k loop

        z[i].px += _mm512_reduce_add_pd(vpxi);
        z[i].py += _mm512_reduce_add_pd(vpyi);
        z[i].pz += _mm512_reduce_add_pd(vpzi);
      } // end of i loop
    }

    __attribute__((noinline))
    void force_intrin_z_swp(Vec8* __restrict z,
                            const int32_t* __restrict number_of_partners,
                            const int32_t* __restrict pointer,
                            const int32_t* __restrict sorted_list,
                            const int pn) {
      const auto vc24   = _mm512_set1_pd(24.0 * dt);
      const auto vc48   = _mm512_set1_pd(48.0 * dt);
      const auto vcl2   = _mm512_set1_pd(CL2);
      const auto vzero  = _mm512_setzero_pd();
      const auto vpitch = _mm512_set1_epi64(8);

      for (int i = 0; i < pn; i++) {
        const auto vqxi = _mm512_set1_pd(z[i].x);
        const auto vqyi = _mm512_set1_pd(z[i].y);
        const auto vqzi = _mm512_set1_pd(z[i].z);

        auto vpxi = _mm512_setzero_pd();
        auto vpyi = _mm512_setzero_pd();
        auto vpzi = _mm512_setzero_pd();

        const auto np = number_of_partners[i];
        const auto kp = pointer[i];
        const int* ptr_list = sorted_list + kp;

        const auto vnp = _mm512_set1_epi64(np);
        auto vk_idx = _mm512_set_epi64(7LL, 6LL, 5LL, 4LL,
                                       3LL, 2LL, 1LL, 0LL);

        auto vindex_a = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)ptr_list), 3);
        auto mask_a = _mm512_cmp_epi64_mask(vk_idx, vnp, _MM_CMPINT_LT);
        auto vqxj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].x, 8);
        auto vqyj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].y, 8);
        auto vqzj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].z, 8);

        auto vdx_a = vqxj - vqxi;
        auto vdy_a = vqyj - vqyi;
        auto vdz_a = vqzj - vqzi;
        auto vr2   = vdx_a*vdx_a + vdy_a*vdy_a + vdz_a*vdz_a;
        auto vr6   = vr2 * vr2 * vr2;
        auto vdf   = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

        auto le_cl2 = _mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS);
        mask_a = _mm512_kand(mask_a, le_cl2);
        vdf = _mm512_mask_blend_pd(mask_a, vzero, vdf);

        for (int k = 8; k < np; k += 8) {
          ptr_list += 8;
          auto vindex_b = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)ptr_list), 3);
          vk_idx += vpitch;
          auto mask_b = _mm512_cmp_epi64_mask(vk_idx, vnp, _MM_CMPINT_LT);
          vqxj = _mm512_mask_i32gather_pd(vzero, mask_b, vindex_b, &z[0].x, 8);
          vqyj = _mm512_mask_i32gather_pd(vzero, mask_b, vindex_b, &z[0].y, 8);
          vqzj = _mm512_mask_i32gather_pd(vzero, mask_b, vindex_b, &z[0].z, 8);

          auto vdx_b = vqxj - vqxi;
          auto vdy_b = vqyj - vqyi;
          auto vdz_b = vqzj - vqzi;
          vr2 = vdx_b*vdx_b + vdy_b*vdy_b + vdz_b*vdz_b;

          vpxi += vdf * vdx_a;
          vpyi += vdf * vdy_a;
          vpzi += vdf * vdz_a;

          auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].px, 8);
          auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].py, 8);
          auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].pz, 8);

          vpxj -= vdf * vdx_a;
          vpyj -= vdf * vdy_a;
          vpzj -= vdf * vdz_a;

          _mm512_mask_i32scatter_pd(&z[0].px, mask_a, vindex_a, vpxj, 8);
          _mm512_mask_i32scatter_pd(&z[0].py, mask_a, vindex_a, vpyj, 8);
          _mm512_mask_i32scatter_pd(&z[0].pz, mask_a, vindex_a, vpzj, 8);

          vr6 = vr2 * vr2 * vr2;
          vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);

          le_cl2 = _mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS);
          mask_b = _mm512_kand(mask_b, le_cl2);
          vdf = _mm512_mask_blend_pd(mask_b, vzero, vdf);

          vindex_a = vindex_b;
          mask_a   = mask_b;
          vdx_a    = vdx_b;
          vdy_a    = vdy_b;
          vdz_a    = vdz_b;
        }

        vpxi += vdf * vdx_a;
        vpyi += vdf * vdy_a;
        vpzi += vdf * vdz_a;

        auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].px, 8);
        auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].py, 8);
        auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].pz, 8);

        vpxj -= vdf * vdx_a;
        vpyj -= vdf * vdy_a;
        vpzj -= vdf * vdz_a;

        _mm512_mask_i32scatter_pd(&z[0].px, mask_a, vindex_a, vpxj, 8);
        _mm512_mask_i32scatter_pd(&z[0].py, mask_a, vindex_a, vpyj, 8);
        _mm512_mask_i32scatter_pd(&z[0].pz, mask_a, vindex_a, vpzj, 8);

        z[i].px += _mm512_reduce_add_pd(vpxi);
        z[i].py += _mm512_reduce_add_pd(vpyi);
        z[i].pz += _mm512_reduce_add_pd(vpzi);
      }
    }
  } // end of namespace avx512

  namespace avx2 {
    // modified version of
    // https://github.com/kaityo256/lj_simdstep/blob/master/step6/force.cpp
    // void force_sorted_intrin(void);
    __attribute__((noinline))
    void force_intrin(const Vec4* __restrict q,
                      Vec4* __restrict p,
                      const int32_t* __restrict number_of_partners,
                      const int32_t* __restrict pointer,
                      const int32_t* __restrict sorted_list,
                      const int pn) {
      const auto vzero = _mm256_setzero_pd();
      const auto vcl2 = _mm256_set1_pd(CL2);
      const auto vc24 = _mm256_set1_pd(24.0 * dt);
      const auto vc48 = _mm256_set1_pd(48.0 * dt);
      for (int i = 0; i < pn; i++) {
        const auto vqi = _mm256_load_pd((double*)&q[i].x);
        auto vpi = _mm256_setzero_pd();
        const auto np = number_of_partners[i];
        const auto kp = pointer[i];
        for (int k = 0; k < (np / 4) * 4; k += 4) {
          const auto j_a = sorted_list[kp + k];
          auto vqj_a = _mm256_load_pd((double*)&q[j_a].x);
          auto vdq_a = vqj_a - vqi;

          const auto j_b = sorted_list[kp + k + 1];
          auto vqj_b = _mm256_load_pd((double*)&q[j_b].x);
          auto vdq_b = vqj_b - vqi;

          const auto j_c = sorted_list[kp + k + 2];
          auto vqj_c = _mm256_load_pd((double*)&q[j_c].x);
          auto vdq_c = vqj_c - vqi;

          const auto j_d = sorted_list[kp + k + 3];
          auto vqj_d = _mm256_load_pd((double*)&q[j_d].x);
          auto vdq_d = vqj_d - vqi;

          __m256d vx, vy, vz;
          transpose_4x4(vdq_a, vdq_b, vdq_c, vdq_d, vx, vy, vz);
          const auto vr2 = vx*vx + vy*vy + vz*vz;
          const auto vr6 = vr2 * vr2 * vr2;
          auto vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);
          const auto mask = vcl2 - vr2;
          vdf = _mm256_blendv_pd(vdf, vzero, mask);

          const auto vdf_a = _mm256_permute4x64_pd(vdf, 0);
          const auto vdf_b = _mm256_permute4x64_pd(vdf, 85);
          const auto vdf_c = _mm256_permute4x64_pd(vdf, 170);
          const auto vdf_d = _mm256_permute4x64_pd(vdf, 255);

          auto vpj_a = _mm256_load_pd((double*)&p[j_a].x);
          vpi   += vdq_a * vdf_a;
          vpj_a -= vdq_a * vdf_a;
          _mm256_store_pd((double*)&p[j_a].x, vpj_a);

          auto vpj_b = _mm256_load_pd((double*)&p[j_b].x);
          vpi   += vdq_b * vdf_b;
          vpj_b -= vdq_b * vdf_b;
          _mm256_store_pd((double*)&p[j_b].x, vpj_b);

          auto vpj_c = _mm256_load_pd((double*)&p[j_c].x);
          vpi   += vdq_c * vdf_c;
          vpj_c -= vdq_c * vdf_c;
          _mm256_store_pd((double*)&p[j_c].x, vpj_c);

          auto vpj_d = _mm256_load_pd((double*)&p[j_d].x);
          vpi   += vdq_d * vdf_d;
          vpj_d -= vdq_d * vdf_d;
          _mm256_store_pd((double*)&p[j_d].x, vpj_d);
        }
        vpi += _mm256_load_pd((double*)&p[i].x);
        _mm256_store_pd((double*)&p[i].x, vpi);
        for (int k = (np / 4) * 4; k < np; k++) {
          const int j = sorted_list[kp + k];
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
    }

    // modified version of
    // https://github.com/kaityo256/lj_simdstep/blob/master/step6/force.cpp
    // void force_sorted_swp_intrin_mat_transp(void);
    __attribute__((noinline))
    void force_intrin_swp(const Vec4* __restrict q,
                          Vec4* __restrict p,
                          const int32_t* __restrict number_of_partners,
                          const int32_t* __restrict pointer,
                          const int32_t* __restrict sorted_list,
                          const int pn) {
      const auto vzero = _mm256_setzero_pd();
      const auto vcl2  = _mm256_set1_pd(CL2);
      const auto vc24  = _mm256_set1_pd(24.0 * dt);
      const auto vc48  = _mm256_set1_pd(48.0 * dt);
      for (int i = 0; i < pn; i++) {
        const auto vqi = _mm256_load_pd((double*)(q + i));
        auto vpf = _mm256_setzero_pd();
        const auto kp = pointer[i];
        auto ja_1 = sorted_list[kp];
        auto ja_2 = sorted_list[kp + 1];
        auto ja_3 = sorted_list[kp + 2];
        auto ja_4 = sorted_list[kp + 3];
        auto vqj_1 = _mm256_load_pd((double*)(q + ja_1));
        auto vdqa_1 = vqj_1 - vqi;
        auto vqj_2 = _mm256_load_pd((double*)(q + ja_2));
        auto vdqa_2 = vqj_2 - vqi;
        auto vqj_3 = _mm256_load_pd((double*)(q + ja_3));
        auto vdqa_3 = vqj_3 - vqi;
        auto vqj_4 = _mm256_load_pd((double*)(q + ja_4));
        auto vdqa_4 = vqj_4 - vqi;

        auto vdf = _mm256_setzero_pd();

        auto vdqb_1 = _mm256_setzero_pd();
        auto vdqb_2 = _mm256_setzero_pd();
        auto vdqb_3 = _mm256_setzero_pd();
        auto vdqb_4 = _mm256_setzero_pd();

        int jb_1 = 0, jb_2 = 0, jb_3 = 0, jb_4 = 0;
        const auto np = number_of_partners[i];
        for (int k = 0; k < (np / 4) * 4; k += 4) {
          const auto j_1 = ja_1;
          const auto j_2 = ja_2;
          const auto j_3 = ja_3;
          const auto j_4 = ja_4;
          auto vdq_1 = vdqa_1;
          auto vdq_2 = vdqa_2;
          auto vdq_3 = vdqa_3;
          auto vdq_4 = vdqa_4;

          ja_1 = sorted_list[kp + k + 4];
          ja_2 = sorted_list[kp + k + 5];
          ja_3 = sorted_list[kp + k + 6];
          ja_4 = sorted_list[kp + k + 7];

          auto tmp0 = _mm256_unpacklo_pd(vdq_1, vdq_2);
          auto tmp1 = _mm256_unpackhi_pd(vdq_1, vdq_2);
          auto tmp2 = _mm256_unpacklo_pd(vdq_3, vdq_4);
          auto tmp3 = _mm256_unpackhi_pd(vdq_3, vdq_4);

          auto vdx = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
          auto vdy = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
          auto vdz = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);

          auto vdf_1 = _mm256_permute4x64_pd(vdf, 0);
          auto vdf_2 = _mm256_permute4x64_pd(vdf, 85);
          auto vdf_3 = _mm256_permute4x64_pd(vdf, 170);
          auto vdf_4 = _mm256_permute4x64_pd(vdf, 255);

          vqj_1 = _mm256_load_pd((double*)(q + ja_1));
          vdqa_1 = vqj_1 - vqi;
          vpf += vdf_1 * vdqb_1;

          auto vpjb_1 = _mm256_load_pd((double*)(p + jb_1));
          vpjb_1 -= vdf_1 * vdqb_1;
          _mm256_store_pd((double*)(p + jb_1), vpjb_1);

          vqj_2 = _mm256_load_pd((double*)(q + ja_2));
          vdqa_2 = vqj_2 - vqi;
          vpf += vdf_2 * vdqb_2;

          auto vpjb_2 = _mm256_load_pd((double*)(p + jb_2));
          vpjb_2 -= vdf_2 * vdqb_2;
          _mm256_store_pd((double*)(p + jb_2), vpjb_2);

          vqj_3 = _mm256_load_pd((double*)(q + ja_3));
          vdqa_3 = vqj_3 - vqi;
          vpf += vdf_3 * vdqb_3;

          auto vpjb_3 = _mm256_load_pd((double*)(p + jb_3));
          vpjb_3 -= vdf_3 * vdqb_3;
          _mm256_store_pd((double*)(p + jb_3), vpjb_3);

          vqj_4 = _mm256_load_pd((double*)(q + ja_4));
          vdqa_4 = vqj_4 - vqi;
          vpf += vdf_4 * vdqb_4;

          auto vpjb_4 = _mm256_load_pd((double*)(p + jb_4));
          vpjb_4 -= vdf_4 * vdqb_4;
          _mm256_store_pd((double*)(p + jb_4), vpjb_4);

          auto vr2 = vdx * vdx + vdy * vdy + vdz * vdz;
          auto vr6 = vr2 * vr2 * vr2;
          vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);
          auto mask = vcl2 - vr2;
          vdf = _mm256_blendv_pd(vdf, vzero, mask);

          jb_1 = j_1;
          jb_2 = j_2;
          jb_3 = j_3;
          jb_4 = j_4;
          vdqb_1 = vdq_1;
          vdqb_2 = vdq_2;
          vdqb_3 = vdq_3;
          vdqb_4 = vdq_4;
        }
        auto vdf_1 = _mm256_permute4x64_pd(vdf, 0);
        auto vdf_2 = _mm256_permute4x64_pd(vdf, 85);
        auto vdf_3 = _mm256_permute4x64_pd(vdf, 170);
        auto vdf_4 = _mm256_permute4x64_pd(vdf, 255);

        auto vpjb_1 = _mm256_load_pd((double*)(p + jb_1));
        vpjb_1 -= vdf_1 * vdqb_1;
        _mm256_store_pd((double*)(p + jb_1), vpjb_1);

        auto vpjb_2 = _mm256_load_pd((double*)(p + jb_2));
        vpjb_2 -= vdf_2 * vdqb_2;
        _mm256_store_pd((double*)(p + jb_2), vpjb_2);

        auto vpjb_3 = _mm256_load_pd((double*)(p + jb_3));
        vpjb_3 -= vdf_3 * vdqb_3;
        _mm256_store_pd((double*)(p + jb_3), vpjb_3);

        auto vpjb_4 = _mm256_load_pd((double*)(p + jb_4));
        vpjb_4 -= vdf_4 * vdqb_4;
        _mm256_store_pd((double*)(p + jb_4), vpjb_4);

        auto vpi = _mm256_load_pd((double*)(p + i));
        vpf += vdf_1 * vdqb_1;
        vpf += vdf_2 * vdqb_2;
        vpf += vdf_3 * vdqb_3;
        vpf += vdf_4 * vdqb_4;
        vpi += vpf;
        _mm256_store_pd((double*)(p + i), vpi);
        const auto qix = q[i].x;
        const auto qiy = q[i].y;
        const auto qiz = q[i].z;
        double pfx = 0.0;
        double pfy = 0.0;
        double pfz = 0.0;
        for (int k = (np / 4) * 4; k < np; k++) {
          const auto j = sorted_list[k + kp];
          const auto dx = q[j].x - qix;
          const auto dy = q[j].y - qiy;
          const auto dz = q[j].z - qiz;
          const auto r2 = (dx * dx + dy * dy + dz * dz);
          const auto r6 = r2 * r2 * r2;
          auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
          if (r2 > CL2) df = 0.0;
          pfx += df * dx;
          pfy += df * dy;
          pfz += df * dz;
          p[j].x -= df * dx;
          p[j].y -= df * dy;
          p[j].z -= df * dz;
        }
        p[i].x += pfx;
        p[i].y += pfy;
        p[i].z += pfz;
      }
    }

    // modified version of
    // https://github.com/kaityo256/lj_simdstep/blob/master/step7/force.cpp
    // void force_sorted_z_intrin(void);
    __attribute__((noinline))
    void force_intrin_z(Vec8* __restrict z,
                        const int32_t* __restrict number_of_partners,
                        const int32_t* __restrict pointer,
                        const int32_t* __restrict sorted_list,
                        const int pn) {
      const auto vzero = _mm256_setzero_pd();
      const auto vcl2 = _mm256_set1_pd(CL2);
      const auto vc24 = _mm256_set1_pd(24.0 * dt);
      const auto vc48 = _mm256_set1_pd(48.0 * dt);
      for (int i = 0; i < pn; i++) {
        const auto vqi = _mm256_load_pd((double*)&z[i].x);
        auto vpi = _mm256_setzero_pd();
        const auto np = number_of_partners[i];
        const auto kp = pointer[i];
        for (int k = 0; k < (np / 4) * 4; k += 4) {
          const auto j_a = sorted_list[kp + k];
          auto vqj_a = _mm256_load_pd((double*)&z[j_a].x);
          auto vdq_a = vqj_a - vqi;

          const auto j_b = sorted_list[kp + k + 1];
          auto vqj_b = _mm256_load_pd((double*)&z[j_b].x);
          auto vdq_b = vqj_b - vqi;

          const auto j_c = sorted_list[kp + k + 2];
          auto vqj_c = _mm256_load_pd((double*)&z[j_c].x);
          auto vdq_c = vqj_c - vqi;

          const auto j_d = sorted_list[kp + k + 3];
          auto vqj_d = _mm256_load_pd((double*)&z[j_d].x);
          auto vdq_d = vqj_d - vqi;

          __m256d vx, vy, vz;
          transpose_4x4(vdq_a, vdq_b, vdq_c, vdq_d, vx, vy, vz);
          const auto vr2 = vx*vx + vy*vy + vz*vz;
          const auto vr6 = vr2 * vr2 * vr2;
          auto vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);
          const auto mask = vcl2 - vr2;
          vdf = _mm256_blendv_pd(vdf, vzero, mask);

          const auto vdf_a = _mm256_permute4x64_pd(vdf, 0);
          const auto vdf_b = _mm256_permute4x64_pd(vdf, 85);
          const auto vdf_c = _mm256_permute4x64_pd(vdf, 170);
          const auto vdf_d = _mm256_permute4x64_pd(vdf, 255);

          auto vpj_a = _mm256_load_pd((double*)&z[j_a].px);
          vpi   += vdq_a * vdf_a;
          vpj_a -= vdq_a * vdf_a;
          _mm256_store_pd((double*)&z[j_a].px, vpj_a);

          auto vpj_b = _mm256_load_pd((double*)&z[j_b].px);
          vpi   += vdq_b * vdf_b;
          vpj_b -= vdq_b * vdf_b;
          _mm256_store_pd((double*)&z[j_b].px, vpj_b);

          auto vpj_c = _mm256_load_pd((double*)&z[j_c].px);
          vpi   += vdq_c * vdf_c;
          vpj_c -= vdq_c * vdf_c;
          _mm256_store_pd((double*)&z[j_c].px, vpj_c);

          auto vpj_d = _mm256_load_pd((double*)&z[j_d].px);
          vpi   += vdq_d * vdf_d;
          vpj_d -= vdq_d * vdf_d;
          _mm256_store_pd((double*)&z[j_d].px, vpj_d);
        }
        vpi += _mm256_load_pd((double*)&z[i].px);
        _mm256_store_pd((double*)&z[i].px, vpi);
        for (int k = (np / 4) * 4; k < np; k++) {
          const int j = sorted_list[kp + k];
          double dx = z[j].x - z[i].x;
          double dy = z[j].y - z[i].y;
          double dz = z[j].z - z[i].z;
          double r2 = (dx * dx + dy * dy + dz * dz);
          if (r2 > CL2) continue;
          double r6 = r2 * r2 * r2;
          double df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
          z[i].px += df * dx;
          z[i].py += df * dy;
          z[i].pz += df * dz;
          z[j].px -= df * dx;
          z[j].py -= df * dy;
          z[j].pz -= df * dz;
        }
      }
    }

    // modified version of
    // https://github.com/kaityo256/lj_simdstep/blob/master/step7/force.cpp
    // void force_sorted_z_intrin_swp(void);
    __attribute__((noinline))
    void force_intrin_z_swp(Vec8* __restrict z,
                            const int32_t* __restrict number_of_partners,
                            const int32_t* __restrict pointer,
                            const int32_t* __restrict sorted_list,
                            const int pn) {
      const auto vzero = _mm256_setzero_pd();
      const auto vcl2  = _mm256_set1_pd(CL2);
      const auto vc24  = _mm256_set1_pd(24.0 * dt);
      const auto vc48  = _mm256_set1_pd(48.0 * dt);
      for (int i = 0; i < pn; i++) {
        const auto vqi = _mm256_load_pd((double*)&z[i].x);
        auto vpf = _mm256_setzero_pd();
        const auto kp = pointer[i];
        auto ja_1 = sorted_list[kp];
        auto ja_2 = sorted_list[kp + 1];
        auto ja_3 = sorted_list[kp + 2];
        auto ja_4 = sorted_list[kp + 3];
        auto vqj_1 = _mm256_load_pd((double*)&z[ja_1].x);
        auto vdqa_1 = vqj_1 - vqi;
        auto vqj_2 = _mm256_load_pd((double*)&z[ja_2].x);
        auto vdqa_2 = vqj_2 - vqi;
        auto vqj_3 = _mm256_load_pd((double*)&z[ja_3].x);
        auto vdqa_3 = vqj_3 - vqi;
        auto vqj_4 = _mm256_load_pd((double*)&z[ja_4].x);
        auto vdqa_4 = vqj_4 - vqi;

        auto vdf = _mm256_setzero_pd();

        auto vdqb_1 = _mm256_setzero_pd();
        auto vdqb_2 = _mm256_setzero_pd();
        auto vdqb_3 = _mm256_setzero_pd();
        auto vdqb_4 = _mm256_setzero_pd();

        int jb_1 = 0, jb_2 = 0, jb_3 = 0, jb_4 = 0;
        const auto np = number_of_partners[i];
        for (int k = 0; k < (np / 4) * 4; k += 4) {
          const auto j_1 = ja_1;
          const auto j_2 = ja_2;
          const auto j_3 = ja_3;
          const auto j_4 = ja_4;
          auto vdq_1 = vdqa_1;
          auto vdq_2 = vdqa_2;
          auto vdq_3 = vdqa_3;
          auto vdq_4 = vdqa_4;

          ja_1 = sorted_list[kp + k + 4];
          ja_2 = sorted_list[kp + k + 5];
          ja_3 = sorted_list[kp + k + 6];
          ja_4 = sorted_list[kp + k + 7];

          auto tmp0 = _mm256_unpacklo_pd(vdq_1, vdq_2);
          auto tmp1 = _mm256_unpackhi_pd(vdq_1, vdq_2);
          auto tmp2 = _mm256_unpacklo_pd(vdq_3, vdq_4);
          auto tmp3 = _mm256_unpackhi_pd(vdq_3, vdq_4);

          auto vdx = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);
          auto vdy = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);
          auto vdz = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);

          auto vdf_1 = _mm256_permute4x64_pd(vdf, 0);
          auto vdf_2 = _mm256_permute4x64_pd(vdf, 85);
          auto vdf_3 = _mm256_permute4x64_pd(vdf, 170);
          auto vdf_4 = _mm256_permute4x64_pd(vdf, 255);

          vqj_1 = _mm256_load_pd((double*)&z[ja_1].x);
          vdqa_1 = vqj_1 - vqi;
          vpf += vdf_1 * vdqb_1;
          auto vpjb_1 = _mm256_load_pd((double*)&z[jb_1].px);
          vpjb_1 -= vdf_1 * vdqb_1;
          _mm256_store_pd((double*)&z[jb_1].px, vpjb_1);

          vqj_2 = _mm256_load_pd((double*)&z[ja_2].x);
          vdqa_2 = vqj_2 - vqi;
          vpf += vdf_2 * vdqb_2;
          auto vpjb_2 = _mm256_load_pd((double*)&z[jb_2].px);
          vpjb_2 -= vdf_2 * vdqb_2;
          _mm256_store_pd((double*)&z[jb_2].px, vpjb_2);

          vqj_3 = _mm256_load_pd((double*)&z[ja_3].x);
          vdqa_3 = vqj_3 - vqi;
          vpf += vdf_3 * vdqb_3;
          auto vpjb_3 = _mm256_load_pd((double*)&z[jb_3].px);
          vpjb_3 -= vdf_3 * vdqb_3;
          _mm256_store_pd((double*)&z[jb_3].px, vpjb_3);

          vqj_4 = _mm256_load_pd((double*)&z[ja_4].x);
          vdqa_4 = vqj_4 - vqi;
          vpf += vdf_4 * vdqb_4;
          auto vpjb_4 = _mm256_load_pd((double*)&z[jb_4].px);
          vpjb_4 -= vdf_4 * vdqb_4;
          _mm256_store_pd((double*)&z[jb_4].px, vpjb_4);

          auto vr2 = vdx*vdx + vdy*vdy + vdz*vdz;
          auto vr6 = vr2 * vr2 * vr2;
          vdf = (vc24 * vr6 - vc48) / (vr6 * vr6 * vr2);
          auto mask = vcl2 - vr2;
          vdf = _mm256_blendv_pd(vdf, vzero, mask);

          jb_1 = j_1;
          jb_2 = j_2;
          jb_3 = j_3;
          jb_4 = j_4;
          vdqb_1 = vdq_1;
          vdqb_2 = vdq_2;
          vdqb_3 = vdq_3;
          vdqb_4 = vdq_4;
        }
        auto vdf_1 = _mm256_permute4x64_pd(vdf, 0);
        auto vdf_2 = _mm256_permute4x64_pd(vdf, 85);
        auto vdf_3 = _mm256_permute4x64_pd(vdf, 170);
        auto vdf_4 = _mm256_permute4x64_pd(vdf, 255);

        auto vpjb_1 = _mm256_load_pd((double*)&z[jb_1].px);
        vpjb_1 -= vdf_1 * vdqb_1;
        _mm256_store_pd((double*)&z[jb_1].px, vpjb_1);

        auto vpjb_2 = _mm256_load_pd((double*)&z[jb_2].px);
        vpjb_2 -= vdf_2 * vdqb_2;
        _mm256_store_pd((double*)&z[jb_2].px, vpjb_2);

        auto vpjb_3 = _mm256_load_pd((double*)&z[jb_3].px);
        vpjb_3 -= vdf_3 * vdqb_3;
        _mm256_store_pd((double*)&z[jb_3].px, vpjb_3);

        auto vpjb_4 = _mm256_load_pd((double*)&z[jb_4].px);
        vpjb_4 -= vdf_4 * vdqb_4;
        _mm256_store_pd((double*)&z[jb_4].px, vpjb_4);

        auto vpi = _mm256_load_pd((double*)&z[i].px);
        vpf += vdf_1 * vdqb_1;
        vpf += vdf_2 * vdqb_2;
        vpf += vdf_3 * vdqb_3;
        vpf += vdf_4 * vdqb_4;
        vpi += vpf;
        _mm256_store_pd((double*)&z[i].px, vpi);
        const auto qix = z[i].x;
        const auto qiy = z[i].y;
        const auto qiz = z[i].z;
        double pfx = 0.0;
        double pfy = 0.0;
        double pfz = 0.0;
        for (int k = (np / 4) * 4; k < np; k++) {
          const auto j = sorted_list[k + kp];
          const auto dx = z[j].x - qix;
          const auto dy = z[j].y - qiy;
          const auto dz = z[j].z - qiz;
          const auto r2 = (dx * dx + dy * dy + dz * dz);
          const auto r6 = r2 * r2 * r2;
          auto df = ((24.0 * r6 - 48.0) / (r6 * r6 * r2)) * dt;
          if (r2 > CL2) df = 0.0;
          pfx += df * dx;
          pfy += df * dy;
          pfz += df * dz;
          z[j].px -= df * dx;
          z[j].py -= df * dy;
          z[j].pz -= df * dz;
        }
        z[i].px += pfx;
        z[i].py += pfy;
        z[i].pz += pfz;
      }
    }
  } // end of namespace avx2
} // end of namespace skl
