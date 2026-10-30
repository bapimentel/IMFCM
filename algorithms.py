import numpy as np


ALGORITHMS = ['FCM', 'GK', 'FCM-S', 'FCM-D', 'FCM-WD', 'MFCM', 'FCM-MFCM', 'IMFCM']


def execute(nRep, dataset, centersAll, method, parM=2):
    Jmin = 2147483647
    best_resp = 0

    for r in range(nRep):
        centers = list(map(int, centersAll[r,].tolist()))

        if method == 'FCM':
            resp = FCM(dataset, centers, parM)
        elif method == 'GK':
            resp = GK(dataset, centers, parM)
        elif method == 'FCM-S':
            resp = FCM_S(dataset, centers, parM)
        elif method == 'FCM-D':
            resp = FCM_D(dataset, centers, parM)
        elif method == 'FCM-WD':
            resp = FCM_WD(dataset, centers, parM)
        elif method == 'MFCM':
            resp = MFCM(dataset, centers, parM)
        elif method == 'FCM-MFCM':
            resp = FCM_MFCM(dataset, centers, parM, 0.5)
        elif method == 'IMFCM':
            resp = IMFCM(dataset, centers, parM)
        else:
            raise ValueError(f'Unknown method: {method}')

        J = resp[0]
        if Jmin > J:
            Jmin = J
            best_resp = resp

    return best_resp


def initialize_prototypes(data, centers):
    return data[np.asarray(centers), :].astype('float64')


def get_partition(memb):
    return np.argmax(memb, axis=1)


def FCM(data, centers, parM):
    maxIteration = 100
    J = 1000
    count = 0
    P = initializePrototypes(data, centers)
    Ubefore = 0
    Jbefore = J + 1.0

    while abs(Jbefore - J) > 0.0001 and count < maxIteration:
        count += 1
        D = updateDistances(data, P)
        U = updateMembership(D, parM)
        P = updatePrototypes(data, U, parM)
        Jbefore = J
        J = updateCriterion(U, D, parM)
        Ubefore = U

    L = getPartition(Ubefore)
    return [J, L, Ubefore, count, np.ones(P.shape)]


def initializePrototypes(data, centers):
    return np.asarray(data)[np.asarray(centers), :].astype('float64')


def updatePrototypes(data, memberships, parM):
    delta = np.power(memberships, parM)
    numerator = delta.T @ data
    denominator = np.sum(delta, axis=0)[:, np.newaxis]
    return numerator / denominator


def updateDistances(data, prototypes):
    nObj = data.shape[0]
    nProt = prototypes.shape[0]
    D = np.zeros((nObj, nProt), dtype=np.float64)
    for k in range(nProt):
        diff = data - prototypes[k, :]
        D[:, k] = np.einsum('ij,ij->i', diff, diff)
    return D


def updateMembership(distances, parM):
    epsilon = 1e-7
    D = distances + epsilon
    expo = 1.0 / (parM - 1.0)
    invD = D ** (-expo)
    s = invD.sum(axis=1, keepdims=True) * (D ** expo)
    U = 1.0 / s
    return U


def updateCriterion(memberships, distances, parM):
    delta = np.power(memberships, parM)
    return np.sum(delta * distances)


def getPartition(memberships):
    return (np.argmax(memberships, axis=1) + 1).tolist()


def GK(data, centers, parM, max_iter=100, tol=1e-6):
    n = data.shape[0]
    c = len(centers)

    P = data[centers, :].copy()
    U = initialize_membership_gk(n, c)
    J = np.inf
    count = 0

    while count < max_iter:
        count += 1
        M = update_matrix_covariance(data, P, U, parM)
        D = update_distances_mahalanobis(data, P, M)
        U_new = update_membership_gk(D, parM)
        P = update_prototypes_gk(data, U_new, parM)
        J_new = update_criterion_gk(U_new, D, parM)

        if abs(J_new - J) < tol:
            break

        U, J = U_new, J_new

    L = np.argmax(U, axis=1)
    return [J, L, U, count, gk_weights(M)]


def gk_weights(M):
    # diagonal of the norm-inducing matrix det(F)^(1/p) F^-1 of each cluster, as in update_distances_mahalanobis
    p = M[0].shape[0]
    rows = []
    for Mk in M:
        matrix = (Mk + Mk.T) / 2.0 + 1e-8 * np.eye(p)
        rows.append(np.diag(np.linalg.pinv(matrix)) * np.linalg.det(matrix) ** (1 / p))
    return np.array(rows)


def initialize_membership_gk(nObj, nProt):
    U = np.random.rand(nObj, nProt)
    U /= U.sum(axis=1, keepdims=True)
    return U


def update_prototypes_gk(data, U, m):
    um = U ** m
    return (um.T @ data) / um.sum(axis=0)[:, None]


def update_matrix_covariance(data, P, U, m):
    nProt = P.shape[0]
    um = U ** m
    M = []

    for k in range(nProt):
        dif = data - P[k]
        Mk = (um[:, k][:, None] * dif).T @ dif / um[:, k].sum()
        M.append(Mk)

    return M


def update_distances_mahalanobis(data, P, M):
    nObj, nVar = data.shape
    nProt = P.shape[0]
    D = np.zeros((nObj, nProt))

    for k in range(nProt):
        matrix = (M[k] + M[k].T) / 2.0
        epsilon = 1e-8
        matrix = matrix + epsilon * np.eye(matrix.shape[0])
        cov_inv = np.linalg.pinv(matrix)
        cov_det = np.linalg.det(matrix)
        dif = data - P[k]
        D[:, k] = np.einsum('ij,jk,ik->i', dif, cov_inv, dif) * (cov_det ** (1 / nVar))

    return D


def update_membership_gk(D, m):
    D = np.maximum(D, 1e-10)
    expo = 1.0 / (m - 1.0)
    invD = (1.0 / D) ** expo
    U = invD / invD.sum(axis=1, keepdims=True)
    return U


def update_criterion_gk(U, D, m):
    return np.sum((U ** m) * D)


def FCM_S(data, centers, parM):
    max_iteration = 100
    J = np.iinfo(np.int32).max
    count = 0
    P = np.array([data[c] for c in centers])
    W = np.ones(data.shape[1])
    Ubefore = None
    Jbefore = J + 1.0

    while abs(Jbefore - J) > 0.000001 and count < max_iteration:
        count += 1
        U = update_membership_fcm_s(data, P, parM, W)
        W = update_weights_fcm_s(data, P, U, parM)
        P = update_prototypes_fcm_s(data, U, parM)
        Jbefore = J
        J = update_criterion_fcm_s(U, data, W, P, parM)
        Ubefore = U

    L = get_partition(Ubefore)
    return [J, L, Ubefore, count, np.tile(W, (len(centers), 1))]


def update_prototypes_fcm_s(data, memberships, parM):
    delta = memberships ** parM
    numerator = delta.T @ data
    denom = delta.sum(axis=0)[:, np.newaxis]
    return numerator / denom


def update_weights_fcm_s(data, prototypes, memberships, parM):
    nVar = prototypes.shape[1]
    um = memberships ** parM
    diff2 = (data[:, np.newaxis, :] - prototypes[np.newaxis, :, :]) ** 2
    sum_val = np.einsum('oi,oih->h', um, diff2)
    prod = np.prod(sum_val) ** (1.0 / nVar)
    W = prod / (sum_val + 1e-10)
    return W


def update_membership_fcm_s(data, prototypes, parM, weight):
    diff = data[:, np.newaxis, :] - prototypes[np.newaxis, :, :]
    WD = np.sum(weight[np.newaxis, np.newaxis, :] * np.square(diff), axis=2)

    expo = 1.0 / (parM - 1.0)
    D = WD + 1e-7
    S = (D ** (-expo)).sum(axis=1, keepdims=True)
    U = 1.0 / (D * S)
    return U


def update_criterion_fcm_s(memberships, data, weight, prototypes, parM):
    delta = memberships ** parM
    diff = data[:, np.newaxis, :] - prototypes[np.newaxis, :, :]
    d = np.sum(weight[np.newaxis, np.newaxis, :] * diff ** 2, axis=2)
    return np.sum(delta * d)


def FCM_D(data, centers, parM):
    data = np.asarray(data, dtype=float)

    if not np.all(np.isfinite(data)):
        raise ValueError("FCM_D received data with NaN or infinite values.")

    maxIteration = 100
    J = np.iinfo(np.int32).max
    count = 0
    P = initialize_prototypes(data, centers)
    W = np.ones((len(centers), data.shape[1]))
    Ubefore = 0
    Jbefore = J + 1.0

    while abs(Jbefore - J) > 0.0001 and count < maxIteration:
        count += 1
        U = update_membership_fcm_d(data, P, parM, W)
        W = update_weights_fcm_d(data, P, U, parM)
        P = update_prototypes_fcm_d(data, U, parM)
        Jbefore = J
        J = update_criterion_fcm_d(U, data, W, P, parM)
        Ubefore = U

    L = get_partition(Ubefore)
    return [J, L, Ubefore, count, W]


def update_prototypes_fcm_d(data, memberships, parM):
    delta = np.power(memberships, parM)
    numerator = np.dot(delta.T, data)
    denominator = np.sum(delta, axis=0)[:, np.newaxis]
    denominator = np.maximum(denominator, np.finfo(float).tiny)
    return numerator / denominator


def update_weights_fcm_d(data, prototypes, memberships, parM):
    u = np.power(memberships, parM)
    diff = data[:, np.newaxis, :] - prototypes[np.newaxis, :, :]
    s = np.sum(u[:, :, np.newaxis] * np.square(diff), axis=0)

    tiny = np.finfo(float).tiny
    s_safe = np.maximum(s, tiny)

    log_geometric_mean = np.mean(np.log(s_safe), axis=1, keepdims=True)
    geometric_mean = np.exp(log_geometric_mean)

    W = geometric_mean / s_safe
    return W


def update_membership_fcm_d(data, prototypes, parM, weight):
    diff = data[:, np.newaxis, :] - prototypes[np.newaxis, :, :]
    WD = np.sum(weight[np.newaxis, :, :] * np.square(diff), axis=2)

    D = WD + 1e-7
    expo = 1.0 / (parM - 1.0)
    invD = D ** (-expo)
    U = invD / invD.sum(axis=1, keepdims=True)
    return U


def update_criterion_fcm_d(memberships, data, weight, prototypes, parM):
    delta = np.power(memberships, parM)
    diff = data[:, np.newaxis, :] - prototypes[np.newaxis, :, :]
    weighted_distances = np.sum(weight[np.newaxis, :, :] * np.square(diff), axis=2)
    return np.sum(delta * weighted_distances)


def FCM_WD(data, centers, parM):
    maxIteration = 100
    J = np.iinfo(np.int32).max
    count = 0
    P = initialize_prototypes(data, centers)
    W = np.ones((data.shape[0], len(centers)))
    W2 = np.ones((len(centers), data.shape[1]))
    Ubefore = 0
    Jbefore = J + 1.0

    while abs(Jbefore - J) > 0.0001 and count < maxIteration:
        count += 1
        U = update_membership_fcm_wd(data, P, parM, W, W2)
        W = update_weights_fcm_wd(data, P, U, parM, W2)
        W2 = update_weights2_fcm_wd(data, P, U, parM, W)
        P = update_prototypes_fcm_wd(data, U, parM, W)
        Jbefore = J
        J = update_criterion_fcm_wd(U, data, P, parM, W, W2)
        Ubefore = U

    L = get_partition(Ubefore)
    return [J, L, Ubefore, count, W2]


def weighted_distances_fcm_wd(data, prototypes, weight2):
    diff = data[:, np.newaxis, :] - prototypes[np.newaxis, :, :]
    return np.sum(weight2[np.newaxis, :, :] * diff ** 2, axis=2)


def update_prototypes_fcm_wd(data, memberships, parM, weight):
    delta = (memberships / (weight + 1e-7)) ** parM
    numerator = delta.T @ data
    denominator = delta.sum(axis=0)[:, np.newaxis]
    return numerator / denominator


def update_weights_fcm_wd(data, prototypes, memberships, parM, weight2):
    UM = memberships ** parM
    WD = weighted_distances_fcm_wd(data, prototypes, weight2)
    UD = UM * WD + 1e-7

    beta = -1.0 / (parM + 1.0)
    invterm = UD ** (-beta)
    S = invterm.sum(axis=1, keepdims=True)
    return invterm / S


def update_weights2_fcm_wd(data, prototypes, memberships, parM, weight):
    nVar = prototypes.shape[1]
    u = (memberships / (weight + 1e-7)) ** parM
    diff2 = (data[:, np.newaxis, :] - prototypes[np.newaxis, :, :]) ** 2
    s_ih = np.einsum('ki,kih->ih', u, diff2)

    product_i = np.prod(s_ih, axis=1) ** (1.0 / nVar)
    return product_i[:, np.newaxis] / s_ih


def update_membership_fcm_wd(data, prototypes, parM, weight, weight2):
    WD = weighted_distances_fcm_wd(data, prototypes, weight2)

    expo1 = 1.0 / (parM - 1.0)
    expo2 = parM / (parM - 1.0)

    Dw = WD + 1e-7
    Ww = weight + 1e-7

    B = (Dw ** (-expo1)) * (Ww ** expo2)
    S = B.sum(axis=1, keepdims=True)
    return B / S


def update_criterion_fcm_wd(memberships, data, prototypes, parM, weight, weight2):
    delta = (memberships / (weight + 1e-7)) ** parM
    distance = weighted_distances_fcm_wd(data, prototypes, weight2)
    return np.sum(delta * distance)


def MFCM(data, centers, parM):
    max_iteration = 100
    J = np.iinfo(np.int32).max
    count = 0
    P = np.array([data[c] for c in centers])
    Ubefore = None
    Jbefore = J + 1.0

    while Jbefore - J > 0.0001 and count < max_iteration:
        count += 1
        D = update_distances_mfcm(data, P)
        U = update_membership(D, parM)
        P = update_prototypes_mfcm(data, U, parM)
        Jbefore = J
        J = update_criterion(U, D, parM)
        Ubefore = U

    M = np.ones((len(centers), data.shape[1]))
    memb = aggregate_matrix(Ubefore, M)
    L = get_partition(memb)
    return [J, L, memb, count, compute_aij(Ubefore)]


def update_prototypes_mfcm(data, memberships, parM):
    nVar = data.shape[1]
    nProt = memberships[0].shape[1]
    P = np.zeros((nProt, nVar))
    for i in range(nVar):
        membership = memberships[i]
        weighted_memberships = membership ** parM
        numerator = np.sum(data[:, i, np.newaxis] * weighted_memberships, axis=0)
        denominator = np.sum(weighted_memberships, axis=0)
        P[:, i] = numerator / denominator
    return P


def update_distances_mfcm(data, prototypes):
    nObj = data.shape[0]
    nProt = prototypes.shape[0]
    nVar = data.shape[1]

    D = np.zeros((nVar, nObj, nProt))
    for i in range(nVar):
        diff = data[:, i, np.newaxis] - prototypes[np.newaxis, :, i]
        D[i] = diff ** 2.0

    return D


def update_membership(distances, parM):
    exponent = 1.0 / (parM - 1.0)
    eps = 1e-7

    Dstack = np.stack(distances) + eps
    invD = Dstack ** (-exponent)
    S = invD.sum(axis=(0, 2))

    U = invD / S[np.newaxis, :, np.newaxis]
    return [U[v] for v in range(U.shape[0])]


def update_criterion(memberships, distances, parM):
    J = 0
    nVar = len(distances)
    for i in range(nVar):
        J += np.sum((memberships[i] ** parM) * distances[i])
    return J


def aggregate_matrix(memberships, M):
    Ustack = np.stack(memberships)
    numerator = np.einsum('ki,ijk->jk', M, Ustack)
    total = numerator.sum(axis=1, keepdims=True)
    return numerator / total


def FCM_MFCM(data, centers, parM, eta):
    max_iteration = 100
    J = np.iinfo(np.int32).max
    count = 0
    P = data[centers, :].copy()
    Ubefore = None
    Jbefore = J + 1.0
    while abs(Jbefore - J) > 0.0001 and count < max_iteration:
        count += 1
        D = update_distances_fcm_mfcm(data, P)
        U = update_membership_fcm_mfcm(D, parM, eta)
        P = update_prototypes_fcm_mfcm(data, U, parM, eta)
        Jbefore = J
        J = update_criterion_fcm_mfcm(U, D, parM, eta)
        Ubefore = U
    memb = np.sum(np.stack(Ubefore), axis=0)
    L = get_partition(memb)
    return [J, L, memb, count, compute_aij(Ubefore)]


def update_distances_fcm_mfcm(data, prototypes):
    n_var = data.shape[1]
    D = []
    for i in range(n_var):
        diff = data[:, i, np.newaxis] - prototypes[np.newaxis, :, i]
        D.append(diff ** 2 + 1e-6)
    return D


def delta_weights_fcm_mfcm(memberships_pow, eta, n_var):
    sum_total = memberships_pow.sum(axis=0)
    coef = eta / (n_var - 1) if n_var > 1 else 0.0
    return (1.0 - eta) * memberships_pow + coef * (sum_total[np.newaxis, :, :] - memberships_pow)


def update_membership_fcm_mfcm(distances, parM, eta):
    n_var = len(distances)
    Dstack = np.stack(distances)
    sum_total = Dstack.sum(axis=0)
    coef = eta / (n_var - 1) if n_var > 1 else 0.0
    E = (1.0 - eta) * Dstack + coef * (sum_total[np.newaxis, :, :] - Dstack)

    expo = 1.0 / (parM - 1.0)
    invE = E ** (-expo)
    Z = invE.sum(axis=(0, 2))
    U = invE / Z[np.newaxis, :, np.newaxis]
    return [U[v] for v in range(n_var)]


def update_prototypes_fcm_mfcm(data, memberships, parM, eta):
    n_var = data.shape[1]
    Um = np.stack(memberships) ** parM
    delta = delta_weights_fcm_mfcm(Um, eta, n_var)

    numerator = np.einsum('ijk,ji->ik', delta, data)
    denominator = delta.sum(axis=1)
    return (numerator / denominator).T


def update_criterion_fcm_mfcm(memberships, distances, parM, eta):
    n_var = len(distances)
    Um = np.stack(memberships) ** parM
    Dstack = np.stack(distances)
    delta = delta_weights_fcm_mfcm(Um, eta, n_var)
    return np.sum(delta * Dstack)


def IMFCM(data, centers, parM):
    maxIteration = 100
    J = np.iinfo(np.int32).max
    count = 0
    P = initialize_prototypes(data, centers)
    A = np.ones((len(centers), data.shape[1]))
    Ubefore = 0
    Jbefore = J + 1.0

    while abs(Jbefore - J) > 0.0001 and count < maxIteration:
        count += 1
        D = update_distances(data, P, A)
        U = update_membership(D, parM)
        A = compute_aij(U)
        P = update_prototypes(data, U, A, parM)
        Jbefore = J
        J = update_criterion(U, D, parM)
        Ubefore = U

    M = np.ones((len(centers), data.shape[1]))
    memb = aggregate_matrix(Ubefore, M)
    L = get_partition(memb)
    return [J, L, memb, count, Ubefore, P, A]


def prototype_weights(memberships, A, parM):
    Um = np.stack(memberships) ** parM
    E = np.exp(1.0 - A[:, :, None] / A[:, None, :])
    return Um + np.einsum('ski,isj->jki', Um, E)


def update_prototypes(data, memberships, A, parM):
    W = prototype_weights(memberships, A, parM)
    numerator = np.einsum('jki,kj->ij', W, data)
    denominator = W.sum(axis=1).T
    return numerator / denominator


def update_distances(data, prototypes, A):
    nObj = data.shape[0]
    nProt = prototypes.shape[0]
    nVar = data.shape[1]

    D = np.zeros((nVar, nObj, nProt))
    weight_factors = np.exp(1.0 - A[:, :, np.newaxis] / A[:, np.newaxis, :])

    for i in range(nVar):
        diff = data[:, i, np.newaxis] - prototypes[np.newaxis, :, i]
        distance = diff ** 2.0
        weighted_sum = np.sum(weight_factors[:, i, :] * (data[:, np.newaxis, :] - prototypes[np.newaxis, :, :]) ** 2.0, axis=2)
        D[i] = distance + weighted_sum

    return D


def compute_aij(memberships):
    nVar = len(memberships)
    q = 2.0 / (nVar - 1.0)
    Ustack = np.stack(memberships) ** q
    CS = Ustack.sum(axis=1)
    total = CS.sum(axis=0)
    return (CS / total[np.newaxis, :]).T


def compute_overall_centroid(u, x, m):
    x = np.asarray(x, dtype=float)
    U = np.asarray(u, dtype=float)
    p = x.shape[1]
    if U.ndim != 3 or U.shape[0] != p or U.shape[1] != x.shape[0]:
        raise ValueError("Memberships must have shape (nVar, nObj, nProt).")

    W = prototype_weights(U, compute_aij(U), m)
    masses = W.sum(axis=(1, 2))
    if np.any(masses <= 0):
        raise ValueError("The centroid is undefined for a variable without mass.")

    return np.einsum('jki,kj->j', W, x) / masses


def dispersions(u, x, y, m):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    U = np.asarray(u, dtype=float)

    if x.ndim != 2 or y.ndim != 2 or U.ndim != 3:
        raise ValueError("x, y and the memberships must be arrays with 2, 2 and 3 dimensions.")

    n, p = x.shape
    c = y.shape[0]
    if y.shape != (c, p) or U.shape != (p, n, c):
        raise ValueError(f"Incompatible shapes: x={x.shape}, y={y.shape}, U={U.shape}.")
    if m <= 1:
        raise ValueError("The fuzzifier m must be greater than 1.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Data and prototypes must be finite.")
    if not np.all(np.isfinite(U)) or np.any(U < 0):
        raise ValueError("Memberships must be finite and non-negative.")

    A = np.asarray(compute_aij(u), dtype=float)
    if A.shape != (c, p) or not np.all(np.isfinite(A)) or np.any(A <= 0):
        raise ValueError("The weights a_ij must be finite and strictly positive.")

    z = compute_overall_centroid(u, x, m)

    W = prototype_weights(U, A, m)
    T_ij = np.einsum('jki,kj->ij', W, (x - z[None, :]) ** 2)
    J_ij = np.einsum('jki,kij->ij', W, (x[:, None, :] - y[None, :, :]) ** 2)
    B_ij = W.sum(axis=1).T * (y - z[None, :]) ** 2

    return T_ij, J_ij, B_ij


def ratio(numerator, denominator):
    a, b = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float)
    )
    return np.divide(a, b,
                     out=np.full(a.shape, np.nan, dtype=float),
                     where=np.abs(b) > 1e-15)


def compute_interpretation_indices(u, x, y, m):
    T_ij, J_ij, B_ij = dispersions(u, x, y, m)

    T_i, J_i, B_i = T_ij.sum(axis=1), J_ij.sum(axis=1), B_ij.sum(axis=1)
    T_j, B_j = T_ij.sum(axis=0), B_ij.sum(axis=0)
    T, J, B = T_i.sum(), J_i.sum(), B_i.sum()

    return {
        'R': float(ratio(B, T)),
        'COR_j': ratio(B_j, T_j),
        'CTR_j': ratio(B_j, B),
        'T_rel': ratio(T_i, T),
        'J_rel': ratio(J_i, J),
        'B_rel': ratio(B_i, B),
        'COR_ij': ratio(B_ij, T_j[None, :]),
        'CTR_ij': ratio(B_ij, B_i[:, None]),
        'CE_ij': ratio(B_ij, B),
    }


M_GRID = [1.1, 1.3, 1.5, 1.7, 2.0, 2.2, 2.5, 2.7, 3.0, 3.2, 3.5, 3.7, 4.0, 4.3, 4.5, 4.7, 5.0]


def first_peak(v, grid):
    for g in range(1, len(v) - 1):
        if v[g] > v[g - 1] and v[g] > v[g + 1]:
            return float(grid[g])
    return float(grid[int(np.argmax(v))])


def recommend_m(grid, Jbar):
    g = np.asarray(grid, float)
    d1 = np.gradient(Jbar, g)
    d2 = np.gradient(d1, g)
    d3 = np.gradient(d2, g)
    return dict(grid=g, Jbar=np.asarray(Jbar, float), d1=d1, d2=d2, d3=d3,
                m1=first_peak(d1, g), m2=first_peak(d2, g), m3=first_peak(d3, g))
