#include "MagnusMatrix.hpp"
#include <complex>
#include <cmath>
#include <iostream>

#define alpha expansion_coefficients[0]
#define beta expansion_coefficients[1]
#define gamma expansion_coefficients[2]
#define delta expansion_coefficients[3]

HHG::Systems::Magnus::Magnus(h_float _k, h_float _kappa, h_float _k_z, h_float delta_t)
    : k(_k * delta_t), k2(k * k), k3(k * k2), k4(k2 * k2),
      kappa(_kappa * delta_t), kappa2(kappa * kappa), kappa4(kappa2 * kappa2),
      k_z(_k_z * delta_t), k_z2(k_z * k_z), k_z3(k_z * k_z2), k_z4(k_z2 * k_z2)
{ }

HHG::Systems::Magnus::u_matrix HHG::Systems::Magnus::Omega(std::array<h_float, 4> const& expansion_coefficients) const
{
    const h_float alpha2 = alpha * alpha;
    const h_float alpha3 = alpha2 * alpha;
    const h_float alpha4 = alpha2 * alpha2;

    const h_float beta2 = beta * beta;
    const h_float beta3 = beta2 * beta;

    const h_float gamma2 = gamma * gamma;

    const h_float a = -alpha3*gamma*k*k_z2*kappa2/2520 - alpha3*gamma*k*kappa4/2520 + alpha2*beta2*k*k_z2*kappa2/2520 + alpha2*beta2*k*kappa4/2520 + alpha2*gamma*k2*k_z*kappa2/1260 - alpha*beta2*k2*k_z*kappa2/1080 - alpha*gamma*k3*kappa2/2520 + alpha*k_z + beta2*k3*kappa2/1890 + beta2*k*kappa2/60 - beta*delta*k*kappa2/420 + gamma2*k*kappa2/420 - k;
    const h_float b = k*kappa*(-alpha4*beta*k_z4 - 2*alpha4*beta*k_z2*kappa2 - alpha4*beta*kappa4 + 4*alpha3*beta*k*k_z3 + 4*alpha3*beta*k*k_z*kappa2 - 6*alpha2*beta*k2*k_z2 - 2*alpha2*beta*k2*kappa2 - 42*alpha2*beta*k_z2 - 42*alpha2*beta*kappa2 + 18*alpha2*delta*k_z2 + 18*alpha2*delta*kappa2 - 36*alpha*beta*gamma*k_z2 - 36*alpha*beta*gamma*kappa2 + 4*alpha*beta*k3*k_z + 84*alpha*beta*k*k_z - 36*alpha*delta*k*k_z + 18*beta3*k_z2 + 18*beta3*kappa2 + 36*beta*gamma*k*k_z - beta*k4 - 42*beta*k2 - 2520*beta + 18*delta*k2)/15120;
    const h_float c = kappa*(-3*alpha3*gamma*k*k_z3 - 3*alpha3*gamma*k*k_z*kappa2 + 3*alpha2*beta2*k*k_z3 + 3*alpha2*beta2*k*k_z*kappa2 + 9*alpha2*gamma*k2*k_z2 + 3*alpha2*gamma*k2*kappa2 - 6*alpha*beta2*k2*k_z2 + alpha*beta2*k2*kappa2 - 9*alpha*gamma*k3*k_z - 7560*alpha + 3*beta2*k3*k_z + 126*beta2*k*k_z - 18*beta*delta*k*k_z + 18*gamma2*k*k_z + 3*gamma*k4)/7560;


    /* Omega = 
     *  0,  a, b,
     * -a,  0, c,
     * -b, -c, 0;
     */
    const h_float norm_squared = a * a + b * b + c * c;
    const h_float ev = sqrt(norm_squared); // to be multiplied by +/- i. The third eigenvalue is 0.
    
    const h_float x = std::cos(ev);
    const h_float y = std::sin(ev);

    u_matrix U;
    U << (a*a + b*b)*x + c*c,   (b*c*(x-1) + a*ev*y), (-a*c*(x-1) + b*ev*y),
         (b*c*(x-1) - a*ev*y),  (a*a + c*c)*x + b*b,  (a*b*(x-1) + c*ev*y),
         (-a*c*(x-1) - b*ev*y), (a*b*(x-1) - c*ev*y), (b*b + c*c)*x + a*a;
    U /= norm_squared;
    return U;
}
