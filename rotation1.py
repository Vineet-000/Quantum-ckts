import numpy as np

# Use a single complex dtype for numpy everywhere.
DTYPE = np.complex128

INV_SQRT2 = 1.0 / np.sqrt(2.0)
H = INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=DTYPE)

# LAMBDA_PI is the base rotation angle realized by the H/T building blocks:
# cos(LAMBDA_PI) = cos^2(pi/8) = (1 + 1/sqrt2)/2. Because LAMBDA_PI / (2 pi) is
# irrational, the multiples {k * LAMBDA_PI mod 2 pi} densely fill [0, 2 pi).
LAMBDA_PI = np.arccos((1.0 + INV_SQRT2) / 2.0)
TWO_PI = 2.0 * np.pi


class Bloch:
    """Axis-angle (Bloch) form of a 2x2 unitary G:

        G = e^{i alpha} (cos(theta/2) I - i sin(theta/2) (n . sigma))

    i.e. a global phase e^{i alpha} times a rotation by angle `theta` about the
    Bloch-sphere axis `n`. Here (n . sigma) = n_x X + n_y Y + n_z Z.
    """

    alpha: float  # global phase
    n: np.ndarray  # unit rotation axis, shape (3,): [n_x, n_y, n_z]
    theta: float  # rotation angle


def to_bloch(g: np.ndarray) -> Bloch:
    """Recover the Bloch form (alpha, n, theta) of a 2x2 unitary `g`."""
    b = Bloch()
    
    det_g = np.linalg.det(g)
    b.alpha = 0.5 * np.angle(det_g)
    
    g_tilde = np.exp(-1j * b.alpha) * g
    
    trace_val = np.real(np.trace(g_tilde))
    cos_theta_half = np.clip(trace_val / 2.0, -1.0, 1.0)
    b.theta = 2.0 * np.arccos(cos_theta_half)
    
    sin_theta_half = np.sin(b.theta / 2.0)
    
    if np.isclose(sin_theta_half, 0.0):
        b.n = np.array([0.0, 0.0, 1.0])
    else:
        X = np.array([[0, 1], [1, 0]], dtype=DTYPE)
        Y = np.array([[0, -1j], [1j, 0]], dtype=DTYPE)
        Z = np.array([[1, 0], [0, -1]], dtype=DTYPE)
        
        nx = np.real(0.5j * np.trace(X @ g_tilde)) / sin_theta_half
        ny = np.real(0.5j * np.trace(Y @ g_tilde)) / sin_theta_half
        nz = np.real(0.5j * np.trace(Z @ g_tilde)) / sin_theta_half
        
        n_unnorm = np.array([nx, ny, nz])
        b.n = n_unnorm / np.linalg.norm(n_unnorm)
        
    return b


# n1, n2 are two orthogonal Bloch-sphere axes (n1 . n2 == 0)
cot_pi_8 = 1.0 / np.tan(np.pi / 8.0)

n1_raw = np.array([-cot_pi_8, 1.0, cot_pi_8])
n1 = n1_raw / np.linalg.norm(n1_raw)

n2_raw = np.array([1.0/np.sqrt(2.0), np.sqrt(2.0)*cot_pi_8, -1.0/np.sqrt(2.0)])
n2 = n2_raw / np.linalg.norm(n2_raw)

# frame derived from the axes (given)
# take the dot product of the Bloch axis with these
# the minus sign arises from the double cover issue
a1 = -n1
a2 = -n2
a3 = np.cross(a1, a2)


def n1n2n1_angles(b: Bloch) -> tuple[float, float, float, float]:
    """Factor the rotation part of a unitary (given as its Bloch form `b`) as
        u = e^{i global_phase} * Rn1(alpha) * Rn2(beta) * Rn1(gamma)

    where Ra(angle) is a rotation by `angle` about axis a, and {a1, a2, a3} is
    the orthonormal frame defined above. Returns (alpha, beta, gamma, global_phase).
    """
    theta_half = b.theta / 2.0
    
    A_val = np.cos(theta_half)
    B_val = np.dot(b.n, a1) * np.sin(theta_half)
    C_val = np.dot(b.n, a2) * np.sin(theta_half)
    D_val = np.dot(b.n, a3) * np.sin(theta_half)
    
    sum_half = np.arctan2(B_val, A_val)
    diff_half = np.arctan2(D_val, C_val)
    
    gamma_half = (sum_half + diff_half) / 2.0
    alpha_half = (sum_half - diff_half) / 2.0
    beta_half = np.arctan2(np.sqrt(C_val**2 + D_val**2), np.sqrt(A_val**2 + B_val**2))
    
    alpha = 2.0 * alpha_half
    beta = 2.0 * beta_half
    gamma = 2.0 * gamma_half
    
    return alpha, beta, gamma, b.alpha


def approx_angle_with_tolerance(angle: float, tolerance: float) -> int:
    """Find an integer multiple k such that
        (k * LAMBDA_PI) mod 2*pi  ~=  angle   (within `tolerance`)  
    Since LAMBDA_PI / (2 pi) is irrational, such a k always exists; search
    k = 1, 2, 3, ... and return the first one whose wrapped multiple lands within
    `tolerance` of `angle` (compare both as angles in [0, 2 pi)).

    Hint:
      * wrap an angle into [0, 2 pi)
      * the angular distance between two wrapped angles a, b is
        min(|a - b|, TWO_PI - |a - b|) (so 0.01 and 2*pi - 0.01 count as close).
    """
    k = 1
    target = angle % TWO_PI
    
    while True:
        current_angle = (k * LAMBDA_PI) % TWO_PI
        dist = abs(current_angle - target)
        dist = min(dist, TWO_PI - dist)
        
        if dist <= tolerance:
            return k
            
        k += 1


def decompose_2x2(u: np.ndarray, tolerance: float) -> tuple[int, int, int]:
    """Approximate a 2x2 unitary `u` as a product of powers of M1 and M2:

        u  ~=  M1^k * M2^l * M1^m     (up to a global phase)

    where M1 is a rotation about axis a1 and M2 a rotation about axis a2, each by
    the base angle realized by the H/T building blocks. Returns the powers
    (k, l, m).

    Steps (combine the two functions above):

      1. Get the Bloch form of u (to_bloch), then factor its rotation into the
         three frame angles with n1n2n1_angles:
             alpha, beta, gamma, _global_phase = n1n2n1_angles(to_bloch(u))
         alpha and gamma are rotations about a1 (realized by powers of M1);
         beta is a rotation about a2 (realized by powers of M2).

      2. Convert each angle to an integer power with approx_angle_with_tolerance:
             k = approx_angle_with_tolerance(alpha, tolerance)   # power of M1
             l = approx_angle_with_tolerance(beta,  tolerance)   # power of M2
             m = approx_angle_with_tolerance(gamma, tolerance)   # power of M1
         (Mind the relationship between a target rotation angle and the base
         angle each application of M1/M2 adds.)

      3. Return (k, l, m).
    """
    b = to_bloch(u)
    alpha, beta, gamma, _global_phase = n1n2n1_angles(b)
    
    k = approx_angle_with_tolerance(alpha, tolerance)
    l = approx_angle_with_tolerance(beta, tolerance)
    m = approx_angle_with_tolerance(gamma, tolerance)
    
    return (k, l, m)


def from_axis_angle(b: Bloch) -> np.ndarray:
    """Build a 2x2 unitary from its Bloch form."""
    I = np.array([[1, 0], [0, 1]], dtype=DTYPE)
    X = np.array([[0, 1], [1, 0]], dtype=DTYPE)
    Y = np.array([[0, -1j], [1j, 0]], dtype=DTYPE)
    Z = np.array([[1, 0], [0, -1]], dtype=DTYPE)
    
    n_dot = b.n[0]*X + b.n[1]*Y + b.n[2]*Z
    th = b.theta / 2.0
    
    mat = np.cos(th)*I - 1j*np.sin(th)*n_dot
    return np.exp(1j * b.alpha) * mat


def Rz(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j * theta / 2.0), 0],
        [0, np.exp(1j * theta / 2.0)]
    ], dtype=DTYPE)


def Ry(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
        [np.sin(theta / 2.0),  np.cos(theta / 2.0)]
    ], dtype=DTYPE)


def unitary2_sqrt(u: np.ndarray) -> np.ndarray:
    b = to_bloch(u)
    b_sqrt = Bloch()
    b_sqrt.alpha = b.alpha / 2.0
    b_sqrt.theta = b.theta / 2.0
    b_sqrt.n = b.n 
    return from_axis_angle(b_sqrt)


def euler_angles_zyz(u: np.ndarray) -> tuple[float, float, float, float]:
    det_u = np.linalg.det(u)
    alpha = 0.5 * np.angle(det_u)
    
    u_tilde = np.exp(-1j * alpha) * u
    cos_gamma_half = np.clip(np.abs(u_tilde[0, 0]), 0.0, 1.0)
    gamma = 2.0 * np.arccos(cos_gamma_half)
    
    if np.isclose(cos_gamma_half, 1.0):
        gamma = 0.0
        beta_plus_delta = 2.0 * np.angle(u_tilde[1, 1])
        beta = beta_plus_delta / 2.0
        delta = beta_plus_delta / 2.0
    elif np.isclose(cos_gamma_half, 0.0):
        gamma = np.pi
        beta_minus_delta = 2.0 * np.angle(u_tilde[1, 0])
        beta = beta_minus_delta / 2.0
        delta = -beta_minus_delta / 2.0
    else:
        b_plus_d = 2.0 * np.angle(u_tilde[1, 1])
        b_minus_d = 2.0 * np.angle(u_tilde[1, 0])
        beta = (b_plus_d + b_minus_d) / 2.0
        delta = (b_plus_d - b_minus_d) / 2.0
        
    return alpha, beta, gamma, delta


T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=DTYPE)


def expand_word(exponent_list: list[tuple[str, int]]) -> str:
    s = ""
    for g, p in exponent_list:
        if g == 'H':
            if p % 2 != 0:
                s += 'H'
        elif g == 'T':
            s += 'T' * (p % 8)
    return s


def invert_gates(word: str) -> str:
    inv = ""
    for c in word[::-1]:
        if c == 'H':
            inv += 'H'
        elif c == 'T':
            inv += 'T' * 7
    return inv


def power_gates(word: str, k: int) -> str:
    if k == 0:
        return ""
    if k > 0:
        return word * k
    return invert_gates(word) * abs(k)


def gates_to_unitary(word: str) -> np.ndarray:
    mat = np.eye(2, dtype=DTYPE)
    for c in word:  
        if c == 'H':
            mat = mat @ H
        elif c == 'T':
            mat = mat @ T_GATE
    return mat


# FIXED: The correct Week 2 generator bases
M1_LIST = [('H', 1), ('T', 1)]
M2_LIST = [('T', 1), ('H', 1)]

m1_str = expand_word(M1_LIST)
m2_str = expand_word(M2_LIST)


def approximate_in_ht(u: np.ndarray, tolerance: float) -> str:
    k, l, m = decompose_2x2(u, tolerance)
    
    wk = power_gates(m1_str, k)
    wl = power_gates(m2_str, l)
    wm = power_gates(m1_str, m)
    
    
    return wk + wl + wm
