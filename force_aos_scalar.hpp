#pragma once

#ifdef ENBLE_AUTOVECT
#define PRAGMA_SIMD _Pragma("omp parallel for simd")
#else
#define PRAGMA_SIMD
#endif

namespace scalar {
  __attribute__((noinline))
  void force_pair(const Vec4* __restrict q,
                  Vec4* __restrict p,
                  const int32_t* __restrict i_particles,
                  const int32_t* __restrict j_particles,
                  const int32_t nps){
    PRAGMA_SIMD
    for (int32_t k = 0; k < nps; k++) {
      const auto i = i_particles[k];
      const auto j = j_particles[k];
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

  __attribute__((noinline))
  void force_sorted(const Vec4* __restrict q,
                    Vec4* __restrict p,
                    const int32_t* __restrict number_of_partners,
                    const int32_t* __restrict pointer,
                    const int32_t* __restrict sorted_list,
                    const int pn) {
    for (int32_t i = 0; i < pn; i++) {
      const auto qx_key = q[i].x;
      const auto qy_key = q[i].y;
      const auto qz_key = q[i].z;
      const auto np = number_of_partners[i];
      double pfx = 0, pfy = 0, pfz = 0;
      const auto kp = pointer[i];

      PRAGMA_SIMD
      for (int k = 0; k < np; k++) {
        const auto j = sorted_list[kp + k];
        const auto dx = q[j].x - qx_key;
        const auto dy = q[j].y - qy_key;
        const auto dz = q[j].z - qz_key;
        const auto r2 = (dx*dx + dy*dy + dz*dz);
        if (r2 > CL2) continue;
        const auto r6 = r2*r2*r2;
        const auto df = ((24.0 * r6 - 48.0)/(r6 * r6 * r2)) * dt;
        pfx += df*dx;
        pfy += df*dy;
        pfz += df*dz;
        p[j].x -= df*dx;
        p[j].y -= df*dy;
        p[j].z -= df*dz;
      }
      p[i].x += pfx;
      p[i].y += pfy;
      p[i].z += pfz;
    }
  }

  __attribute__((noinline))
  void force_next(const Vec4* __restrict q,
                  Vec4* __restrict p,
                  const int32_t* __restrict number_of_partners,
                  const int32_t* __restrict pointer,
                  const int32_t* __restrict sorted_list,
                  const int pn) {
    for (int i = 0; i < pn; i++) {
      auto qx_key = q[i].x;
      auto qy_key = q[i].y;
      auto qz_key = q[i].z;
      double pfx = 0, pfy = 0, pfz = 0;
      const auto kp = pointer[i];
      auto ja = sorted_list[kp];
      auto dxa = q[ja].x - qx_key;
      auto dya = q[ja].y - qy_key;
      auto dza = q[ja].z - qz_key;
      double df = 0.0;
      double dxb = 0.0, dyb = 0.0, dzb = 0.0;
      int jb = 0;

      const auto np = number_of_partners[i];
      for (int k = kp; k < np + kp; k++) {

        const auto dx = dxa;
        const auto dy = dya;
        const auto dz = dza;
        auto r2 = (dx * dx + dy * dy + dz * dz);
        const auto j = ja;
        ja = sorted_list[k + 1];
        dxa = q[ja].x - qx_key;
        dya = q[ja].y - qy_key;
        dza = q[ja].z - qz_key;
        if (r2 > CL2) continue;
        pfx += df * dxb;
        pfy += df * dyb;
        pfz += df * dzb;
        p[jb].x -= df * dxb;
        p[jb].y -= df * dyb;
        p[jb].z -= df * dzb;
        const auto r6 = r2 * r2 * r2;
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

  __attribute__((noinline))
  void force_sorted_z(Vec8* __restrict z,
                      const int32_t* __restrict number_of_partners,
                      const int32_t* __restrict pointer,
                      const int32_t* __restrict sorted_list,
                      const int pn) {
    for (int32_t i = 0; i < pn; i++) {
      const auto qx_key = z[i].x;
      const auto qy_key = z[i].y;
      const auto qz_key = z[i].z;
      const auto np = number_of_partners[i];
      double pfx = 0, pfy = 0, pfz = 0;
      const auto kp = pointer[i];

      PRAGMA_SIMD
      for (int k = 0; k < np; k++) {
        const auto j = sorted_list[kp + k];
        const auto dx = z[j].x - qx_key;
        const auto dy = z[j].y - qy_key;
        const auto dz = z[j].z - qz_key;
        const auto r2 = (dx*dx + dy*dy + dz*dz);
        if (r2 > CL2) continue;
        const auto r6 = r2*r2*r2;
        const auto df = ((24.0 * r6 - 48.0)/(r6 * r6 * r2)) * dt;
        pfx += df*dx;
        pfy += df*dy;
        pfz += df*dz;
        z[j].px -= df*dx;
        z[j].py -= df*dy;
        z[j].pz -= df*dz;
      }
      z[i].px += pfx;
      z[i].py += pfy;
      z[i].pz += pfz;
    }
  }
}
