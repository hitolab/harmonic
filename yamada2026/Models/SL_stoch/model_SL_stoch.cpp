#include "../../Lib/model_interface.h"
#include <cmath>
#include <random>

extern "C" {

const int V = 2;

// Stuart-Landau in normalized time tau = t/T, for use with
// EulerMaruyamaIterate (model_func called exactly once per step, so noise
// is sampled fresh every call -- no RK4 sub-stage freezing needed).
// Params: [0]=a, [1]=b, [2]=w, [3]=noise_std, [4]=dtau, [5]=T
// (1/T)dx/dt = F(x) + sqrt(D)*xi  ->  noise per tau-step = noise_std/sqrt(dtau)
static thread_local std::mt19937 generator(12345);
static thread_local std::normal_distribution<double> dist(0.0, 1.0);

void model_func(const double *params, const double *x, double *dxdt) {
  double a          = params[0];
  double b          = params[1];
  double w          = params[2];
  double noise_std  = params[3];
  double dtau       = params[4];
  double T          = params[5];

  double noise_x = 0.0, noise_y = 0.0;
  if (noise_std > 0.0 && dtau > 0.0) {
      double noise_factor = noise_std / std::sqrt(dtau);
      noise_x = dist(generator) * noise_factor;
      noise_y = dist(generator) * noise_factor;
  }

  double r2 = x[0] * x[0] + x[1] * x[1];

  dxdt[0] = T * (a * x[0] - w * x[1] - r2 * (x[0] - b * x[1])) + noise_x;
  dxdt[1] = T * (a * x[1] + w * x[0] - r2 * (x[1] + b * x[0])) + noise_y;
}

int get_v() { return V; }
}
