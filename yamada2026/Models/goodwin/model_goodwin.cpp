#include "../../Lib/model_interface.h"
#include <cmath>
#include <vector>

extern "C" {

// Model: Goodwin Oscillator (Cyclic Inhibition)
// Corresponds to var3_Temp.m
// Fixed number of variables
const int V = 3;

// Params: [0]=lamb, [1]=n_exp
void model_func(const double *params, const double *x, double *dxdt) {
  double lamb = params[0];
  double n_exp = params[1];

  // The Goodwin model (or cyclic inhibition) typically forms a loop:
  // x1_dot = f(xn) - k*x1
  // xi_dot = x(i-1) - k*xi

  // In var3_Temp.m:
  // u_dot = 1/(1 + w^n) - lamb * u  (where w is x3, u is x1)
  // v_dot = u - lamb * v            (where v is x2)
  // w_dot = v - lamb * w            (where w is x3)

  // This generalizes to:
  dxdt[0] = 1.0 / (1.0 + std::pow(x[V - 1], n_exp)) - lamb * x[0];

  for (int i = 1; i < V; ++i) {
    dxdt[i] = x[i - 1] - lamb * x[i];
  }
}

// Function to expose the fixed variable count
int get_v() { return V; }
}
