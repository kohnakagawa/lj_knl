#pragma once

void add_particle(double x, double y, double z,
                  Vec4* q, int32_t& particle_number) {
  static std::mt19937 mt(2);
  std::uniform_real_distribution<double> ud(0.0, 0.1);
  q[particle_number].x = x + ud(mt);
  q[particle_number].y = y + ud(mt);
  q[particle_number].z = z + ud(mt);
  particle_number++;
}

std::tuple<Vec4*, Vec4*, int32_t> make_particles(const double density) {
  Vec4 *q = nullptr, *p = nullptr;
  int32_t pn = 0;

  posix_memalign((void**)(&q), 64, sizeof(Vec4) * N);
  posix_memalign((void**)(&p), 64, sizeof(Vec4) * N);

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
        add_particle(x     ,y   ,z, q, pn);
        add_particle(x     ,y+hs,z+hs, q, pn);
        add_particle(x+hs  ,y   ,z+hs, q, pn);
        add_particle(x+hs  ,y+hs,z, q, pn);
      }
    }
  }
  for (int i = 0; i < pn; i++) {
    p[i].x = 0.0;
    p[i].y = 0.0;
    p[i].z = 0.0;
  }

  return std::make_tuple(q, p, pn);
}

Vec8* make_z() {
  Vec8* z = nullptr;
  posix_memalign((void**)(&z), 64, sizeof(Vec8) * N);
  return z;
}

void copy_to_z(Vec8* z, const Vec4* q, const int32_t pn) {
  for (int32_t i = 0; i < pn; i++) {
    z[i].x  = q[i].x; z[i].y  = q[i].y; z[i].z  = q[i].z;
    z[i].px = 0.0; z[i].py = 0.0; z[i].pz = 0.0;
  }
}

void copy_from_z(Vec4* p, const Vec8* z, const int32_t pn) {
  for (int32_t i = 0; i < pn; i++) {
    p[i].x = z[i].px; p[i].y = z[i].py; p[i].z = z[i].pz;
  }
}

void delete_particles(Vec4* q, Vec4* p, Vec8* z) {
  free(p);
  free(q);
  free(z);
}
