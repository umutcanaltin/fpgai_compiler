#pragma once
#include <cstdint>

// Minimal ap_fixed stub for host compilation only.
// Uses float internally; enough for functional simulation.
template<int W, int I>
struct ap_fixed {
  float v;
  ap_fixed() : v(0.0f) {}
  ap_fixed(float x) : v(x) {}
  ap_fixed(double x) : v((float)x) {}
  ap_fixed(int x) : v((float)x) {}

  operator float() const { return v; }
  ap_fixed& operator=(float x) { v = x; return *this; }
};

