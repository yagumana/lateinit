import numpy as np
import torch
import random
import sys
import logging
import os
from datetime import datetime

# def compute_euclidean_distances(a, b):
#     dim = len(a)
#     dis = 0
#     for j in range(dim):
#         dis += (a[j] - b[j])**2
#     dis = np.sqrt(dis)
#     if dis < 1:
#         return True
#     else:
#         return False
#     return dis

def is_neighbour_2d(x):
    dim = len(x)
    dis = 0
    for i in range(dim):
        dis += x[i]**2
    dis /= 2 * np.sqrt(x[0]**2+x[1]**2)
    
    if dis < 1:
        return True
    else:
        return False

def is_neighbour_2d_inj1(x):
    """
    r=1, theta \in [pi/6, pi/3]の円の一部と、
    r=2, theta \in [7*pi/6, 4*pi/3]の円の一部
    を組み合わせた多様体の近傍判定
    """
    dim = len(x)
    dis = 0

    # atan2を使用して角度を計算
    angle = np.arctan2(x[1], x[0])
    # 角度が負の場合は2πを加えて0から2πの範囲に調整
    if angle < 0:
        angle += 2 * np.pi
    
    if np.pi/6 <= angle <= np.pi/3:
        d = np.abs(np.sqrt(x[0]**2 + x[1]**2) - 1)

    elif 7*np.pi/6 <= angle <= 4*np.pi/3:
        """
        正確ではない（r=1の部分も加味すべき）が、管状近傍判定には影響しない (どうせ1以上になる)
        """
        d = np.min(np.abs(np.sqrt(x[0]**2 + x[1]**2) - 2))

    elif np.pi/3 < angle < 7*np.pi/6:
        d1 = np.sqrt((x[0]-1/2)**2 + (x[1]-np.sqrt(3)/2)**2)
        d2 = np.sqrt((x[0]+np.sqrt(3))**2 + (x[1]+1)**2)
        d = min(d1, d2)
    
    else :
        d1 = np.sqrt((x[0]-np.sqrt(3)/2)**2 + (x[1]-1/2)**2)
        d2 = np.sqrt((x[0]+1)**2 + (x[1]+np.sqrt(3))**2)
        d = min(d1, d2)
    
    for i in range(2, dim):
        dis += x[i]**2
    
    dis = np.sqrt(dis + d**2)

    if dis < 1:
        return True
    else:
        return False

def is_neighbour_2d_inj2(x):
    """
    r=1, theta \in [pi/6, pi/3]の円の一部と、
    r=2, theta \in [7*pi/6, 4*pi/3]の円の一部
    を組み合わせた多様体の近傍判定
    """
    dim = len(x)
    dis = 0

    # atan2を使用して角度を計算
    angle = np.arctan2(x[1], x[0])
    # 角度が負の場合は2πを加えて0から2πの範囲に調整
    if angle < 0:
        angle += 2 * np.pi
    
    if np.pi/6 <= angle <= np.pi/3:
        d = np.abs(np.sqrt(x[0]**2 + x[1]**2) - 1)

    elif 7*np.pi/6 <= angle <= 4*np.pi/3:
        """
        正確ではない（r=1の部分も加味すべき）が、管状近傍判定には影響しない (どうせ1以上になる)
        """
        d = np.min(np.abs(np.sqrt(x[0]**2 + x[1]**2) - 2))

    elif np.pi/3 < angle < 7*np.pi/6:
        d1 = np.sqrt((x[0]-1/2)**2 + (x[1]-np.sqrt(3)/2)**2)
        d2 = np.sqrt((x[0]+np.sqrt(3))**2 + (x[1]+1)**2)
        d = min(d1, d2)
    
    else :
        d1 = np.sqrt((x[0]-np.sqrt(3)/2)**2 + (x[1]-1/2)**2)
        d2 = np.sqrt((x[0]+1)**2 + (x[1]+np.sqrt(3))**2)
        d = min(d1, d2)
    
    for i in range(2, dim):
        dis += x[i]**2
    
    dis = np.sqrt(dis + d**2)

    if dis < 2:
        return True
    else:
        return False


def is_neighbour_3d(x):
    dim = len(x)
    normal = 0 # 3次元の空間からの距離
    for i in range(3, dim):
        normal += x[i]**2

    # 2次元球面までの距離
    dis_3 = 0
    for i in range(3):
        dis_3 += x[i]**2
    dis_3 = abs(dis_3-1)
    
    dis = np.sqrt(dis_3 + normal)
    if dis < 1:
        return True
    else:
        return False


def is_neighbour_torus(x, R=2, r=1):
    """
    導出は、0517-symbreak noteを参照
    """
    dim = len(x)
    A1 = abs(np.sqrt(x[0]**2+x[1]**2)-R)
    A2 = np.sqrt(A1**2+x[2]**2)
    d2 = abs(A2-r)

    d3_n = 0
    for i in range(3, dim):
        d3_n += x[i] ** 2

    d6 = np.sqrt(d2**2 + d3_n)

    injectivity_r = min(R-r, r)

    if d6 < injectivity_r:
        return True
    else:
        return False



# 以下、楕円の近傍にあるかを判定するコード

import cmath
import math
import sys

def clamp(v):
    return max(-1.0, min(1.0, v))

def solv2Equation(a, b, c):
    discriminant = b * b - 4.0 * a * c
    return [(-b + cmath.sqrt(discriminant)) / (2.0 * a),
            (-b - cmath.sqrt(discriminant)) / (2.0 * a)]

def cubicRoot(val):
    d = abs(val) ** (1.0 / 3.0)
    r = cmath.phase(val) / 3.0
    return cmath.rect(d, r)

def calcMinDistPoint(x, y, a, b, solveNum, ts, eps):
    dist = sys.float_info.max
    px = py = 0

    for i in range(solveNum):
        t = ts[i]
        if abs(t.imag) > eps or abs(a + t.real * b) < eps or abs(b + t.real * a) < eps:
            continue

        cos_t = clamp(x / (a + t.real * b))
        sin_t = clamp(y / (b + t.real * a))

        if abs(cos_t) > abs(sin_t):
            sin_t = math.sqrt(1.0 - cos_t ** 2) * (1 if sin_t >= 0.0 else -1)
        else:
            cos_t = math.sqrt(1.0 - sin_t ** 2) * (1 if cos_t >= 0.0 else -1)

        nx = b * cos_t * t.real
        ny = a * sin_t * t.real
        L = math.sqrt(nx ** 2 + ny ** 2)

        if L < dist:
            dist = L
            px = a * cos_t
            py = b * sin_t

    return (dist != sys.float_info.max), px, py, dist

def distEllipsePoint(xLen, yLen, x, y):
    """
    xLen: [int]
        x軸側径
    yLen: [int]
        y軸側径
    x: [int]
        点のx座標
    y: [int]
        点のy座標

    Return: [int]
        点から楕円上の点までの最短距離

    # Example usage
    result, px, py, dist = distEllipsePoint(5, 3, 1, 1)
    print("Result:", result, "Point:", px, py, "Distance:", dist)
    """
    dist = px = py = 0
    xLen = max(0.0, xLen)
    yLen = max(0.0, yLen)

    if xLen == 0.0:
        dy = abs(y) - yLen
        if dy > 0.0:
            dist = math.sqrt(x * x + dy * dy)
            return True, px, py, dist
        dist = abs(x)
        return True, px, py, dist

    if yLen == 0.0:
        dx = abs(x) - xLen
        if dx > 0.0:
            dist = math.sqrt(y * y + dx * dx)
            return True, px, py, dist
        dist = abs(y)
        return True, px, py, dist

    a, b = xLen, yLen
    a2, b2 = a ** 2, b ** 2
    x2, y2 = x ** 2, y ** 2
    A = 2.0 / (a * b) * (a2 + b2)
    B = 4.0 + (a2 - x2) / b2 + (b2 - y2) / a2
    C = 2.0 / (a * b) * (a2 + b2 - x2 - y2)
    D = 1.0 - x2 / a2 - y2 / b2
    E = A / 4.0
    p = B - 6.0 * E ** 2
    q = C - 2.0 * B * E + 8.0 * E ** 3
    r = D - C * E + B * E ** 2 - 3.0 * E ** 4
    alpha = 2.0 * p
    beta = p ** 2 - 4.0 * r
    gamma = -q ** 2
    f = beta - alpha ** 2 / 3.0
    g = gamma - alpha * beta / 3.0 + 2.0 / 27.0 * alpha ** 3

    w1 = complex(-0.5, math.sqrt(3.0) / 2.0)
    w2 = complex(-0.5, -math.sqrt(3.0) / 2.0)
    eps = 1e-11

    if abs(q) < eps:
        solveNum = 4
        h = solv2Equation(1.0, p, r)
        ts = [cmath.sqrt(h[0]) - E, -cmath.sqrt(h[0]) - E, cmath.sqrt(h[1]) - E, -cmath.sqrt(h[1]) - E]
        return calcMinDistPoint(x, y, a, b, solveNum, ts, eps)

    v = [0, 0, 0]
    adjD = -4.0 * f ** 3 - 27.0 * g ** 2
    if adjD >= 0.0:
        m = math.sqrt(-f / 3.0)
        n = -g / (m * m)
        for j in range(3):
            np2m = clamp(n / (2.0 * m))
            v[j] = 2.0 * m * math.cos(1.0 / 3.0 * math.acos(np2m) + j * 2.0 / 3.0 * math.pi)
    else:
        rt = cmath.sqrt((g / 2.0) ** 2 + (f / 3.0) ** 3)
        L = complex(-g / 2.0, +0.0) + rt
        R = complex(-g / 2.0, -0.0) - rt
        CL = cubicRoot(L)
        CR = cubicRoot(R)
        v = [CL + CR, w1 * CL + w2 * CR, w2 * CL + w1 * CR]

    ts = [0] * 12
    solveNum = 0
    for i in range(3):
        solves = solv2Equation(1.0, -cmath.sqrt(v[i] - alpha / 3.0), (p + v[i] - alpha / 3.0) / 2.0 + q / (2.0 * cmath.sqrt(v[i] - alpha / 3.0)))
        solves += solv2Equation(1.0, cmath.sqrt(v[i] - alpha / 3.0), (p + v[i] - alpha / 3.0) / 2.0 - q / (2.0 * cmath.sqrt(v[i] - alpha / 3.0)))
        for s in solves:
            ts[solveNum] = s - E
            solveNum += 1

    return calcMinDistPoint(x, y, a, b, solveNum, ts, eps)

# Example usage
# result, px, py, dist = distEllipsePoint(5, 3, 1, 1)
# print("Result:", result, "Point:", px, py, "Distance:", dist)


def is_neighbour_ellipse(x, a=1, b=1):
    dim = len(x)

    if a >= b:
        injective_R = b*b/a
    elif a <= b:
        injective_R = a*a/b
    
    result, px, py, dist = distEllipsePoint(a, b, x[0], x[1])
    # distの値をクリップする
    dist = max(dist, 1e-5)
    dist = min(dist, 1e5)

    # 管状近傍の内か外かの判定
    total_dis = 0
    # 3次元目からn次元目までの距離の2乗を足す
    for i in range(3, dim):
        total_dis += x[i]**2
    # 2次元平面内での距離の2乗を足す
    total_dis += dist**2 
    # 2乗して計算しているので、平方根をとる
    total_dis = np.sqrt(total_dis)

    if total_dis < injective_R:
        return True
    else:
        return False

def is_neighbour_hypersphere(x, dim_z=21):
    dim = len(x)
    injective_R = 1
    total_dis = 0

    # 半径1のn次元球面からの距離の2乗を計算
    for i in range(dim_z+1):
        total_dis += x[i]**2
    total_dis = np.abs(total_dis-1)

    # n次元球面からの距離の2乗を計算
    for i in range(dim_z+1, dim):
        total_dis += x[i]**2
    total_dis = np.sqrt(total_dis)

    if total_dis < injective_R:
        return True
    else:
        return False
    

def neighbourhood_cnt(Us, dataset_name, R=2, r=1, dim_z = 21, cnt2_flag=False):
    """
    粒子が管状近傍の外にある確率を計算する. cnt_probが管状近傍の外にある確率に対応する. cnt_prob2は、データ多様体からの距離が2以上の確率に対応する.
    """
    Time_step, Data_size, dim = Us.shape
    cnt_prob = []
    cnt_prob2 = []
    logging.info(f"dataset_name: {dataset_name}")

    for t in range(Time_step):
        cnt = 0
        cnt2 = 0
        for i in range(Data_size):
            if dataset_name == "circle":
                if not is_neighbour_2d(Us[t][i]):
                    cnt += 1
            elif dataset_name == "circle_half":
                if not is_neighbour_2d(Us[t][i]):
                    cnt += 1
            elif dataset_name == "circle_two_injectivity":
                if not is_neighbour_2d_inj1(Us[t][i]):
                    cnt += 1
                if cnt2_flag:
                    if not is_neighbour_2d_inj2(Us[t][i]):
                        cnt2 += 1

            elif dataset_name == "sphere" or dataset_name == "sphere_notuniform":
                if not is_neighbour_3d(Us[t][i]):
                    cnt += 1
            elif dataset_name == "torus":
                if not is_neighbour_torus(Us[t][i], R=R, r=r):
                    cnt += 1
            elif dataset_name == "ellipse":
                if not is_neighbour_ellipse(Us[t][i], a=R, b=r):
                    cnt += 1
            elif dataset_name == "embed_sphere":
                if not is_neighbour_hypersphere(Us[t][i], dim_z):
                    cnt += 1
                pass
            elif dataset_name == "hyper_sphere":
                if not is_neighbour_hypersphere(Us[t][i], dim_z):
                    cnt += 1
            else:
                logging.info("Not implemented! check the dataset_name again!")     
                sys.exit()       
        
        cnt_prob.append(cnt/Data_size)
        cnt_prob2.append(cnt2/Data_size)

    return cnt_prob, cnt_prob2

    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_filename):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # タイムスタンプを取得してファイル名に追加
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename_with_timestamp = f"{log_filename}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/{log_filename_with_timestamp}"), # タイムスタンプ付きファイル名
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to file: {log_filename_with_timestamp}")