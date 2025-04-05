#include "MagnusMatrix.hpp"
#include <complex>
#include <cmath>
#include <iostream>

HHG::Magnus::Magnus(h_float _k, h_float _kappa, h_float _k_z, h_float delta_t)
    : k(_k * delta_t), k2(k * k), k3(k * k2), k4(k2 * k2),
      kappa(_kappa * delta_t), kappa2(kappa * kappa), kappa4(kappa2 * kappa2),
      k_z(_k_z * delta_t), k_z2(k_z * k_z), k_z3(k_z * k_z2), k_z4(k_z2 * k_z2)
{ }

HHG::Magnus::u_matrix HHG::Magnus::Omega(h_float alpha, h_float beta, h_float gamma, h_float delta) const
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
     * 0, a, b,
     * -a, 0, c,
     * -b, -c, 0;
     */
    const h_float b2c2 = b * b + c * c;
    const h_float norm_squared = a * a + b2c2;
    const h_complex ev = imaginary_unit * sqrt(norm_squared); // to be multiplied by +/- 1. The third eigenvalue is 0.
    
    m_matrix V;
    m_matrix V_inv;

    V << c / a, -(b*ev + a*c) / b2c2, (b*ev - a*c) / b2c2,
        -b / a, (a*b - c*ev) / b2c2, (a*b + c*ev) / b2c2,
        1, 1, 1;
    
    V_inv << a*c / norm_squared, - a*b / norm_squared, a*a / norm_squared,
        (b*ev - a*c) / (2*norm_squared), (a*b + c*ev) / (2*norm_squared), b2c2 / (2*norm_squared),
        -(b*ev + a*c) / (2*norm_squared), (a*b - c*ev) / (2*norm_squared), b2c2 / (2*norm_squared);

    // Test was successfull on 05.04.2025
    // std::cout << (V * V_inv - m_matrix::Identity(3, 3)).norm() << std::endl;

    const m_matrix U = V * Eigen::DiagonalMatrix<h_complex, 3>(1., std::exp(ev), std::exp(-ev)) * V_inv;
    //assert(is_zero(U.imag().squaredNorm()));

    return U.real();
}
