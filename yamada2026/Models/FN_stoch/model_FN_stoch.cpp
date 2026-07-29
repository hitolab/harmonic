#include "../../Lib/model_interface.h"
#include <cmath>
#include <random>

extern "C" {

const int V = 2;

// Use thread_local to avoid issues if run in parallel (though harmonicity is mostly single threaded)
static thread_local std::mt19937 generator(12345);
static thread_local std::normal_distribution<double> dist(0.0, 1.0);

static thread_local int call_count = 0;
static thread_local double current_noise_x = 0.0;
static thread_local double current_noise_y = 0.0;

// Model: FitzHugh-Nagumo with noise
// Params: [0]=a, [1]=b, [2]=c, [3]=noise_std, [4]=dt
void model_func(const double *params, const double *x, double *dxdt) {
  double a = params[0];
  double b = params[1];
  double c = params[2];
  double noise_std = params[3];
  double dt = params[4];

  // RK4 makes 4 calls per time step (k1, k2, k3, k4).
  // To implement Euler-Maruyama consistently, we generate noise once per step.
  if (call_count == 0) {
      if (noise_std > 0.0 && dt > 0.0) {
          // RK4 integration multiplies dxdt by dt.
          // For dW, variance is dt, so std is sqrt(dt).
          // We divide by dt here so that when RK4 multiplies by dt, it becomes sqrt(dt).
          double noise_factor = noise_std / std::sqrt(dt);
          current_noise_x = dist(generator) * noise_factor;
          current_noise_y = dist(generator) * noise_factor;
      } else {
          current_noise_x = 0.0;
          current_noise_y = 0.0;
      }
  }

  // x_dot = a*(- x[1] + x[0] - (x[0]^3)/3) + noise_x
  dxdt[0] = a * (-x[1] + x[0] - (x[0] * x[0] * x[0]) / 3.0) + current_noise_x;

  // y_dot = x[0] - b*x[1] + c + noise_y
  dxdt[1] = x[0] - b * x[1] + c + current_noise_y;

  // Increment call count, wrap around at 4
  call_count = (call_count + 1) % 4;
}

int get_v() { return V; }
}
