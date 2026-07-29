#include "../../Lib/model_interface.h"
#include <cmath>

extern "C" {

const int V = 2;

// Model: Stuart-Landau (deterministic)
// Params: [0]=a, [1]=b, [2]=w
void model_func(const double *params, const double *x, double *dxdt) {
  double a = params[0];
  double b = params[1];
  double w = params[2];

  double r2 = x[0] * x[0] + x[1] * x[1];

  // x_dot = a*x(1) - w*x(2) - (x(1)^2+x(2)^2)*(x(1)-b*x(2))
  dxdt[0] = a * x[0] - w * x[1] - r2 * (x[0] - b * x[1]);

  // y_dot = a*x(2) + w*x(1) - (x(1)^2+x(2)^2)*(x(2)+b*x(1))
  dxdt[1] = a * x[1] + w * x[0] - r2 * (x[1] + b * x[0]);
}

int get_v() { return V; }
}
