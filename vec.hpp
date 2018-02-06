#pragma once

struct double4 {
  double x, y, z, w;
};

struct double8 {
  double x, y, z, w;
  double px, py, pz, pw;
};

using Vec4 = double4;
using Vec8 = double8;
