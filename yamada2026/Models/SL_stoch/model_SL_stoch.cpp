#include "../../Lib/model_interface.h"
#include <cmath>
#include <random>

extern "C" {

const int V = 2;

// Model: Stuart-Landau with noise (stochastic)
// Params: [0]=a, [1]=b, [2]=w, [3]=noise_std, [4]=dt
static thread_local std::mt19937 generator(12345);
static thread_local std::normal_distribution<double> dist(0.0, 1.0);

static thread_local int call_count = 0;
static thread_local double current_noise_x = 0.0;
static thread_local double current_noise_y = 0.0;

void model_func(const double *params, const double *x, double *dxdt) {
  double a = params[0];
  double b = params[1];
  double w = params[2];
  double noise_std = params[3];
  double dt = params[4];

  if (call_count == 0) {
      if (noise_std > 0.0 && dt > 0.0) {
          double noise_factor = noise_std / std::sqrt(dt);
          current_noise_x = dist(generator) * noise_factor;
          current_noise_y = dist(generator) * noise_factor;
      } else {
          current_noise_x = 0.0;
          current_noise_y = 0.0;
      }
  }

  double r2 = x[0] * x[0] + x[1] * x[1];

  // x_dot = a*x(1) - w*x(2) - (x(1)^2+x(2)^2)*(x(1)-b*x(2))
  dxdt[0] = a * x[0] - w * x[1] - r2 * (x[0] - b * x[1]) + current_noise_x;

  // y_dot = a*x(2) + w*x(1) - (x(1)^2+x(2)^2)*(x(2)+b*x(1))
  dxdt[1] = a * x[1] + w * x[0] - r2 * (x[1] + b * x[0]) + current_noise_y;

  call_count = (call_count + 1) % 4;
}

int get_v() { return V; }
}
