## Continues the electric fields measured in the experiments
## so that the pulses end with E(t_end) = 0 = E'(t_end)
## while maintaining continuity up to the second derivative
## Ouput:
# E_A(0) = 17.39232
# E_A(1) = 9.735221250000002
# E_A(2) = 2.9677800000000056
# E_A(3) = 0.06778124999999591
# E_A(4) = 1.4210854715202004e-14
# E_B(0) = 2.26608
# E_B(1) = -1.6794843749999981
# E_B(2) = -3.229109999999997
# E_B(3) = -1.720374374999997
# E_B(4) = 0.0

dt = 0.03318960199004975

E_A = 17.39232
E_B = 2.26608

derivative_A = (17.39232 - 23.4984) / dt
derivative_B = (2.26608 - 6.31152) / dt

second_derivative_A = (17.39232 - 2*23.4984 + 26.72112) / (dt*dt)
second_derivative_B = (2.26608 - 2*6.31152 + 9.61536) / (dt*dt)

def third_order_coeff(E, Eprime, Eprimeprime, x_end):
    return - 4 * (Eprimeprime / (2 * x_end) + (3*Eprime)/(4*x_end**2) + E / (x_end**3))

def fourth_order_coeff(E, Eprime, Eprimeprime, x_end, a):
    return - 0.25 * (3 * a / x_end + (2 * Eprimeprime) / (x_end**2) + Eprime / (x_end**3))

N_extra = 4
x_end = N_extra*dt

third_order_A  =  third_order_coeff(E_A, derivative_A, second_derivative_A, x_end)
fourth_order_A = fourth_order_coeff(E_A, derivative_A, second_derivative_A, x_end, third_order_A)

third_order_B  =  third_order_coeff(E_B, derivative_B, second_derivative_B, x_end)
fourth_order_B = fourth_order_coeff(E_B, derivative_B, second_derivative_B, x_end, third_order_B)

def func_A(x):
    return E_A + derivative_A * x + second_derivative_A * x**2 + third_order_A * x**3 + fourth_order_A * x**4

def func_B(x):
    return E_B + derivative_B * x + second_derivative_B * x**2 + third_order_B * x**3 + fourth_order_B * x**4

for i in range(N_extra + 1):
    print(f"E_A({i}) = {func_A(i*dt)}")

for i in range(N_extra + 1):
    print(f"E_B({i}) = {func_B(i*dt)}")
