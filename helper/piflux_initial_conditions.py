from sympy import Matrix, symbols, I, simplify, sqrt, expand, latex

# Define real symbols for phi components
alpha, beta = symbols('alpha beta', real=True)

# Define phi as a column vector
phi = Matrix([alpha, beta])

# Define Pauli matrices
sigma_x = Matrix([[0, 1], [1, 0]])
sigma_y = Matrix([[0, -I], [I, 0]])
sigma_z = Matrix([[1, 0], [0, -1]])


x, y, z = symbols('x y z', real=True)
epsilon = sqrt(x**2 + y**2 + z**2)
normalization = 1 / (2 * epsilon * (z + epsilon))
V = Matrix([[z + epsilon, -x + I * y], [x + I * y, z + epsilon]])

# Compute phi^dagger
phi_dagger = phi.H  # Hermitian (conjugate transpose)

# Compute V^dagger
V_dagger = V.H  # Hermitian (conjugate transpose)

print("Test: ", simplify(normalization * V_dagger * V))

# Compute the expression for each sigma_i
for sigma_i, name in zip([sigma_x, sigma_y, sigma_z], ['sigma_x', 'sigma_y', 'sigma_z']):
    as_matrix = phi_dagger * V_dagger * sigma_i * V * phi
    expression = simplify(expand(normalization * as_matrix[0]))
    print(f"{name} result:")
    print(latex(expression))
    print()