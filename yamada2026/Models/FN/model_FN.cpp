#include "../../Lib/model_interface.h"
#include <cmath>

extern "C" {

const int V = 2;

// Model: FitzHugh-Nagumo
// Corresponds to Fig1_FN.m
// Params: [0]=a, [1]=b, [2]=c
void model_func(const double *params, const double *x, double *dxdt) {
  double a = params[0];
  double b = params[1];
  double c = params[2];

  // x_dot = a*(- x(2) + x(1) - (x(1)^3)/3);
  dxdt[0] = a * (-x[1] + x[0] - (x[0] * x[0] * x[0]) / 3.0);

  // y_dot = x(1) - b*x(2) + c;
  dxdt[1] = x[0] - b * x[1] + c;
}

int get_v() { return V; }
}
