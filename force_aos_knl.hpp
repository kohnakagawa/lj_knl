#pragma once

namespace knl {
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

          const auto vdx = _mm512_sub_pd(vqxj, vqxi);
          const auto vdy = _mm512_sub_pd(vqyj, vqyi);
          const auto vdz = _mm512_sub_pd(vqzj, vqzi);

          auto vpxj = _mm512_i32gather_pd(vindex, &p[0].x, 8);
          auto vpyj = _mm512_i32gather_pd(vindex, &p[0].y, 8);
          auto vpzj = _mm512_i32gather_pd(vindex, &p[0].z, 8);

          const auto vr2 = _mm512_fmadd_pd(vdz,
                                           vdz,
                                           _mm512_fmadd_pd(vdy,
                                                           vdy,
                                                           _mm512_mul_pd(vdx, vdx)));

          const auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);

          auto vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                                   _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));

          const auto le_cl2 =  _mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS);
          mask = _mm512_kand(mask, le_cl2);
          vdf = _mm512_mask_blend_pd(mask, vzero, vdf);

          vdf = _mm512_mask_blend_pd(mask, vzero, vdf);

          vpxi = _mm512_fmadd_pd(vdf, vdx, vpxi);
          vpyi = _mm512_fmadd_pd(vdf, vdy, vpyi);
          vpzi = _mm512_fmadd_pd(vdf, vdz, vpzi);

          vpxj = _mm512_fnmadd_pd(vdf, vdx, vpxj);
          vpyj = _mm512_fnmadd_pd(vdf, vdy, vpyj);
          vpzj = _mm512_fnmadd_pd(vdf, vdz, vpzj);

          _mm512_mask_i32scatter_pd(&p[0].x, mask, vindex, vpxj, 8);
          _mm512_mask_i32scatter_pd(&p[0].y, mask, vindex, vpyj, 8);
          _mm512_mask_i32scatter_pd(&p[0].z, mask, vindex, vpzj, 8);

          vk_idx = _mm512_add_epi64(vk_idx, vpitch);
        } // end of k loop

        p[i].x += _mm512_reduce_add_pd(vpxi);
        p[i].y += _mm512_reduce_add_pd(vpyi);
        p[i].z += _mm512_reduce_add_pd(vpzi);
      } // end of i loop
    }

    __attribute__((noinline))
    void force_intrin_reactless(const Vec4* __restrict q,
                                Vec4* __restrict p,
                                const int32_t* __restrict number_of_partners,
                                const int32_t* __restrict pointer,
                                const int32_t* __restrict sorted_list,
                                const int pn) {
      const auto vc24  = _mm512_set1_pd(24.0 * dt);
      const auto vc48  = _mm512_set1_pd(48.0 * dt);
      const auto vcl2  = _mm512_set1_pd(CL2);
      const auto v2    = _mm512_set1_pd(2.0);
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
        const int* ptr_list = &sorted_list[kp];

        const auto vnp = _mm512_set1_epi64(np);
        auto vk_idx = _mm512_set_epi64(7LL, 6LL, 5LL, 4LL,
                                       3LL, 2LL, 1LL, 0LL);

        for (int k = 0; k < np; k += 8) {
          const auto vindex = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)ptr_list), 2);
          ptr_list += 8;

          const auto lt_np = _mm512_cmp_epi64_mask(vk_idx,
                                                   vnp,
                                                   _MM_CMPINT_LT);

          const auto vqxj = _mm512_mask_i32gather_pd(vzero, lt_np, vindex, &q[0].x, 8);
          const auto vqyj = _mm512_mask_i32gather_pd(vzero, lt_np, vindex, &q[0].y, 8);
          const auto vqzj = _mm512_mask_i32gather_pd(vzero, lt_np, vindex, &q[0].z, 8);

          const auto vdx = _mm512_sub_pd(vqxj, vqxi);
          const auto vdy = _mm512_sub_pd(vqyj, vqyi);
          const auto vdz = _mm512_sub_pd(vqzj, vqzi);

          const auto vr2 = _mm512_fmadd_pd(vdz,
                                           vdz,
                                           _mm512_fmadd_pd(vdy,
                                                           vdy,
                                                           _mm512_mul_pd(vdx, vdx)));
          const auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);
          const auto vdf_nume      = _mm512_fmsub_pd(vc24, vr6, vc48);
          const auto vdf_deno      = _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2);
          const auto vdf_deno_inv  = _mm512_rcp28_pd(vdf_deno);
          auto vdf_deno_inv2 = _mm512_fnmadd_pd(vdf_deno, vdf_deno_inv, v2);
          vdf_deno_inv2      = _mm512_mul_pd(vdf_deno_inv2, vdf_deno_inv);
          auto vdf           = _mm512_mul_pd(vdf_nume, vdf_deno_inv2);

          const auto le_cl2 = _mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS);
          const auto mask = _mm512_kand(lt_np, le_cl2);
          vdf = _mm512_mask_blend_pd(mask, vzero, vdf);

          vpxi = _mm512_fmadd_pd(vdf, vdx, vpxi);
          vpyi = _mm512_fmadd_pd(vdf, vdy, vpyi);
          vpzi = _mm512_fmadd_pd(vdf, vdz, vpzi);

          vk_idx = _mm512_add_epi64(vk_idx, vpitch);
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
        auto mask_a = _mm512_cmp_epi64_mask(vk_idx,
                                            vnp,
                                            _MM_CMPINT_LT);
        auto vqxj = _mm512_i32gather_pd(vindex_a, &q[0].x, 8);
        auto vqyj = _mm512_i32gather_pd(vindex_a, &q[0].y, 8);
        auto vqzj = _mm512_i32gather_pd(vindex_a, &q[0].z, 8);

        auto vdx_a = _mm512_sub_pd(vqxj, vqxi);
        auto vdy_a = _mm512_sub_pd(vqyj, vqyi);
        auto vdz_a = _mm512_sub_pd(vqzj, vqzi);
        auto vr2 = _mm512_fmadd_pd(vdz_a,
                                   vdz_a,
                                   _mm512_fmadd_pd(vdy_a,
                                                   vdy_a,
                                                   _mm512_mul_pd(vdx_a, vdx_a)));

        auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);

        auto vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                                 _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));
        auto le_cl2 = _mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS);
        mask_a = _mm512_kand(mask_a, le_cl2);
        vdf = _mm512_mask_blend_pd(mask_a, vzero, vdf);

        for (int k = 8; k < np; k += 8) {
          auto vindex_b = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)(&sorted_list[kp + k])),
                                            2);
          vk_idx = _mm512_add_epi64(vk_idx, vpitch);
          auto mask_b = _mm512_cmp_epi64_mask(vk_idx,
                                              vnp,
                                              _MM_CMPINT_LT);
          vqxj = _mm512_i32gather_pd(vindex_b, &q[0].x, 8);
          vqyj = _mm512_i32gather_pd(vindex_b, &q[0].y, 8);
          vqzj = _mm512_i32gather_pd(vindex_b, &q[0].z, 8);

          auto vdx_b = _mm512_sub_pd(vqxj, vqxi);
          auto vdy_b = _mm512_sub_pd(vqyj, vqyi);
          auto vdz_b = _mm512_sub_pd(vqzj, vqzi);
          vr2 = _mm512_fmadd_pd(vdz_b,
                                vdz_b,
                                _mm512_fmadd_pd(vdy_b,
                                                vdy_b,
                                                _mm512_mul_pd(vdx_b,
                                                              vdx_b)));

          vpxi = _mm512_fmadd_pd(vdf, vdx_a, vpxi);
          vpyi = _mm512_fmadd_pd(vdf, vdy_a, vpyi);
          vpzi = _mm512_fmadd_pd(vdf, vdz_a, vpzi);

          auto vpxj = _mm512_i32gather_pd(vindex_a, &p[0].x, 8);
          auto vpyj = _mm512_i32gather_pd(vindex_a, &p[0].y, 8);
          auto vpzj = _mm512_i32gather_pd(vindex_a, &p[0].z, 8);

          vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
          vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
          vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

          _mm512_mask_i32scatter_pd(&p[0].x, mask_a, vindex_a, vpxj, 8);
          _mm512_mask_i32scatter_pd(&p[0].y, mask_a, vindex_a, vpyj, 8);
          _mm512_mask_i32scatter_pd(&p[0].z, mask_a, vindex_a, vpzj, 8);

          vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);
          vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                              _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));

          le_cl2 = _mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS);
          mask_b = _mm512_kand(mask_b, le_cl2);
          vdf = _mm512_mask_blend_pd(mask_b, vzero, vdf);

          vindex_a = vindex_b;
          mask_a   = mask_b;
          vdx_a    = vdx_b;
          vdy_a    = vdy_b;
          vdz_a    = vdz_b;
        } // end of k loop

        vpxi = _mm512_fmadd_pd(vdf, vdx_a, vpxi);
        vpyi = _mm512_fmadd_pd(vdf, vdy_a, vpyi);
        vpzi = _mm512_fmadd_pd(vdf, vdz_a, vpzi);

        auto vpxj = _mm512_i32gather_pd(vindex_a, &p[0].x, 8);
        auto vpyj = _mm512_i32gather_pd(vindex_a, &p[0].y, 8);
        auto vpzj = _mm512_i32gather_pd(vindex_a, &p[0].z, 8);

        vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
        vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
        vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

        _mm512_mask_i32scatter_pd(&p[0].x, mask_a, vindex_a, vpxj, 8);
        _mm512_mask_i32scatter_pd(&p[0].y, mask_a, vindex_a, vpyj, 8);
        _mm512_mask_i32scatter_pd(&p[0].z, mask_a, vindex_a, vpzj, 8);

        p[i].x += _mm512_reduce_add_pd(vpxi);
        p[i].y += _mm512_reduce_add_pd(vpyi);
        p[i].z += _mm512_reduce_add_pd(vpzi);
      } // end of i loop
    }

    __attribute__((noinline))
    void force_intrin_swp_pf(const Vec4* __restrict q,
                             Vec4* __restrict p,
                             const int32_t* __restrict number_of_partners,
                             const int32_t* __restrict pointer,
                             const int32_t* __restrict sorted_list,
                             const int pn) {
      const auto vc24   = _mm512_set1_pd(24.0 * dt);
      const auto vc48   = _mm512_set1_pd(48.0 * dt);
      const auto vcl2   = _mm512_set1_pd(CL2);
      const auto vzero  = _mm512_setzero_pd();
      const auto vpitch = _mm512_set1_epi32(8);

      for (int i = 0; i < pn; i++) {
        const auto vqxi = _mm512_set1_pd(q[i].x);
        const auto vqyi = _mm512_set1_pd(q[i].y);
        const auto vqzi = _mm512_set1_pd(q[i].z);

        auto vpxi = _mm512_setzero_pd();
        auto vpyi = _mm512_setzero_pd();
        auto vpzi = _mm512_setzero_pd();

        const auto np = number_of_partners[i];
        const auto kp = pointer[i];
        const auto vnp = _mm512_set1_epi32(np);
        auto vk_idx = _mm512_set_epi32(15, 14, 13, 12,
                                       11, 10, 9,  8,
                                       7,  6,  5,  4,
                                       3,  2,  1,  0);

        // initial force calculation
        // load position
        auto vindex_a = _mm512_slli_epi32(_mm512_loadu_si512(&sorted_list[kp]),
                                          2);
        auto mask_a = _mm512_cmp_epi32_mask(vk_idx,
                                            vnp,
                                            _MM_CMPINT_LT);
        auto vqxj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                             _mm512_castsi512_si256(vindex_a), &q[0].x, 8);
        auto vqyj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                             _mm512_castsi512_si256(vindex_a), &q[0].y, 8);
        auto vqzj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                             _mm512_castsi512_si256(vindex_a), &q[0].z, 8);

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

        for (int k = 8; k < np; k += 8) {
          // load position
          auto vindex_b = _mm512_slli_epi32(_mm512_loadu_si512(&sorted_list[kp + k]),
                                            2);
          vk_idx = _mm512_add_epi32(vk_idx, vpitch);
          auto mask_b = _mm512_cmp_epi32_mask(vk_idx,
                                              vnp,
                                              _MM_CMPINT_LT);
          vqxj = _mm512_mask_i32gather_pd(vzero, mask_b,
                                          _mm512_castsi512_si256(vindex_b), &q[0].x, 8);
          vqyj = _mm512_mask_i32gather_pd(vzero, mask_b,
                                          _mm512_castsi512_si256(vindex_b), &q[0].y, 8);
          vqzj = _mm512_mask_i32gather_pd(vzero, mask_b,
                                          _mm512_castsi512_si256(vindex_b), &q[0].z, 8);

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

          auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                               _mm512_castsi512_si256(vindex_a), &p[0].x, 8);
          auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                               _mm512_castsi512_si256(vindex_a), &p[0].y, 8);
          auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                               _mm512_castsi512_si256(vindex_a), &p[0].z, 8);

          vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
          vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
          vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

          _mm512_mask_i32scatter_pd(&p[0].x, mask_a,
                                    _mm512_castsi512_si256(vindex_a), vpxj, 8);
          _mm512_mask_i32scatter_pd(&p[0].y, mask_a,
                                    _mm512_castsi512_si256(vindex_a), vpyj, 8);
          _mm512_mask_i32scatter_pd(&p[0].z, mask_a,
                                    _mm512_castsi512_si256(vindex_a), vpzj, 8);

          // calc force norm
          vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);
          vdf = _mm512_div_pd(_mm512_fmsub_pd(vc24, vr6, vc48),
                              _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2));

          // prefetch next line
          _mm512_mask_prefetch_i32gather_pd(_mm512_castsi512_si256(vindex_b),
                                            mask_b, &p[0].x, 8, _MM_HINT_T0);
          __mmask8 mask_b_hi = mask_b >> 8;
          _mm512_mask_prefetch_i32gather_pd(_mm256_castpd_si256(_mm512_extractf64x4_pd(_mm512_castsi512_pd(vindex_b), 1)),
                                            mask_b_hi, &q[0].x, 8, _MM_HINT_T0);

          // cutoff
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

        auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                             _mm512_castsi512_si256(vindex_a), &p[0].x, 8);
        auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                             _mm512_castsi512_si256(vindex_a), &p[0].y, 8);
        auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a,
                                             _mm512_castsi512_si256(vindex_a), &p[0].z, 8);

        vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
        vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
        vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

        _mm512_mask_i32scatter_pd(&p[0].x, mask_a,
                                  _mm512_castsi512_si256(vindex_a), vpxj, 8);
        _mm512_mask_i32scatter_pd(&p[0].y, mask_a,
                                  _mm512_castsi512_si256(vindex_a), vpyj, 8);
        _mm512_mask_i32scatter_pd(&p[0].z, mask_a,
                                  _mm512_castsi512_si256(vindex_a), vpzj, 8);

        // write back i particle momentum
        p[i].x += _mm512_reduce_add_pd(vpxi);
        p[i].y += _mm512_reduce_add_pd(vpyi);
        p[i].z += _mm512_reduce_add_pd(vpzi);
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
      const auto v2     = _mm512_set1_pd(2.0);
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

        auto vdx_a = _mm512_sub_pd(vqxj, vqxi);
        auto vdy_a = _mm512_sub_pd(vqyj, vqyi);
        auto vdz_a = _mm512_sub_pd(vqzj, vqzi);
        auto vr2 = _mm512_fmadd_pd(vdz_a,
                                   vdz_a,
                                   _mm512_fmadd_pd(vdy_a,
                                                   vdy_a,
                                                   _mm512_mul_pd(vdx_a, vdx_a)));

        auto vr6 = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);
        auto vdf_nume      = _mm512_fmsub_pd(vc24, vr6, vc48);
        auto vdf_deno      = _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2);
        auto vdf_deno_inv  = _mm512_rcp28_pd(vdf_deno);
        auto vdf_deno_inv2 = _mm512_fnmadd_pd(vdf_deno, vdf_deno_inv, v2);
        vdf_deno_inv2      = _mm512_mul_pd(vdf_deno_inv2, vdf_deno_inv);
        auto vdf           = _mm512_mul_pd(vdf_nume, vdf_deno_inv2);

        auto le_cl2 = _mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS);
        mask_a = _mm512_kand(mask_a, le_cl2);
        vdf = _mm512_mask_blend_pd(mask_a, vzero, vdf);

        for (int k = 8; k < np; k += 8) {
          ptr_list += 8;
          auto vindex_b = _mm256_slli_epi32(_mm256_lddqu_si256((const __m256i*)ptr_list), 3);
          vk_idx = _mm512_add_epi64(vk_idx, vpitch);
          auto mask_b = _mm512_cmp_epi64_mask(vk_idx,
                                              vnp,
                                              _MM_CMPINT_LT);
          vqxj = _mm512_mask_i32gather_pd(vzero, mask_b, vindex_b, &z[0].x, 8);
          vqyj = _mm512_mask_i32gather_pd(vzero, mask_b, vindex_b, &z[0].y, 8);
          vqzj = _mm512_mask_i32gather_pd(vzero, mask_b, vindex_b, &z[0].z, 8);

          auto vdx_b = _mm512_sub_pd(vqxj, vqxi);
          auto vdy_b = _mm512_sub_pd(vqyj, vqyi);
          auto vdz_b = _mm512_sub_pd(vqzj, vqzi);
          vr2 = _mm512_fmadd_pd(vdz_b,
                                vdz_b,
                                _mm512_fmadd_pd(vdy_b,
                                                vdy_b,
                                                _mm512_mul_pd(vdx_b,
                                                              vdx_b)));

          vpxi = _mm512_fmadd_pd(vdf, vdx_a, vpxi);
          vpyi = _mm512_fmadd_pd(vdf, vdy_a, vpyi);
          vpzi = _mm512_fmadd_pd(vdf, vdz_a, vpzi);

          auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].px, 8);
          auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].py, 8);
          auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].pz, 8);

          vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
          vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
          vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

          _mm512_mask_i32scatter_pd(&z[0].px, mask_a, vindex_a, vpxj, 8);
          _mm512_mask_i32scatter_pd(&z[0].py, mask_a, vindex_a, vpyj, 8);
          _mm512_mask_i32scatter_pd(&z[0].pz, mask_a, vindex_a, vpzj, 8);

          vr6           = _mm512_mul_pd(_mm512_mul_pd(vr2, vr2), vr2);
          vdf_nume      = _mm512_fmsub_pd(vc24, vr6, vc48);
          vdf_deno      = _mm512_mul_pd(_mm512_mul_pd(vr6, vr6), vr2);
          vdf_deno_inv  = _mm512_rcp28_pd(vdf_deno);
          vdf_deno_inv2 = _mm512_fnmadd_pd(vdf_deno, vdf_deno_inv, v2);
          vdf_deno_inv2 = _mm512_mul_pd(vdf_deno_inv2, vdf_deno_inv);
          vdf           = _mm512_mul_pd(vdf_nume, vdf_deno_inv2);

          le_cl2 = _mm512_cmp_pd_mask(vr2, vcl2, _CMP_LE_OS);
          mask_b = _mm512_kand(mask_b, le_cl2);
          vdf = _mm512_mask_blend_pd(mask_b, vzero, vdf);

          vindex_a = vindex_b;
          mask_a   = mask_b;
          vdx_a    = vdx_b;
          vdy_a    = vdy_b;
          vdz_a    = vdz_b;
        }

        vpxi = _mm512_fmadd_pd(vdf, vdx_a, vpxi);
        vpyi = _mm512_fmadd_pd(vdf, vdy_a, vpyi);
        vpzi = _mm512_fmadd_pd(vdf, vdz_a, vpzi);

        auto vpxj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].px, 8);
        auto vpyj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].py, 8);
        auto vpzj = _mm512_mask_i32gather_pd(vzero, mask_a, vindex_a, &z[0].pz, 8);

        vpxj = _mm512_fnmadd_pd(vdf, vdx_a, vpxj);
        vpyj = _mm512_fnmadd_pd(vdf, vdy_a, vpyj);
        vpzj = _mm512_fnmadd_pd(vdf, vdz_a, vpzj);

        _mm512_mask_i32scatter_pd(&z[0].px, mask_a, vindex_a, vpxj, 8);
        _mm512_mask_i32scatter_pd(&z[0].py, mask_a, vindex_a, vpyj, 8);
        _mm512_mask_i32scatter_pd(&z[0].pz, mask_a, vindex_a, vpzj, 8);

        z[i].px += _mm512_reduce_add_pd(vpxi);
        z[i].py += _mm512_reduce_add_pd(vpyi);
        z[i].pz += _mm512_reduce_add_pd(vpzi);
      }
    }
  } // end of namespace avx512
} // end of namespace knl
