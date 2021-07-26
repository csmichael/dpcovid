import pandas as pd
import numpy as np
np.set_printoptions(precision=4, suppress=True)

from math import radians
from math import cos
from scipy.stats import gamma
import random
import math
from matplotlib import pyplot as plt

import argparse
import time


DEBUG = False
R_UNIT = 50

X_MAX = 10
DX = 1000
Y_MAX = 16
DY = 500


def convertToXY(p_gis, lat_mean):
    lat = p_gis[0]
    longi = p_gis[1]
    r = 6371000
    x = r * longi * cos(lat_mean)
    y = r * lat
    return x, y


def adjustXY(p, pmin):
    return p[0] - pmin[0], p[1] - pmin[1]


def get_reference_xy(df):
    longs = df['longt'].apply(radians)
    lats = df['lat'].apply(radians)
    # get the lower left and upper right points
    pmax_gis = (max(lats), max(longs))
    pmin_gis = (min(lats), min(longs))
    # get the mean point for GIS to X,Y coordinates mapping
    lat_mean = (pmax_gis[0] + pmin_gis[0]) / 2
    pmax = convertToXY(pmax_gis, lat_mean)
    pmin = convertToXY(pmin_gis, lat_mean)
    if DEBUG:
        print(pmax, pmin)
    pmax_adj = adjustXY(pmax, pmin)
    pmin_adj = adjustXY(pmin, pmin)
    if DEBUG:
        print(pmax_adj, pmin_adj)
    return lat_mean, pmin


def get_min_max_xy(df):
    longs = df['longt'].apply(radians)
    lats = df['lat'].apply(radians)
    # get the lower left and upper right points
    pmax_gis = (max(lats), max(longs))
    pmin_gis = (min(lats), min(longs))
    # get the mean point for GIS to X,Y coordinates mapping
    lat_mean = (pmax_gis[0] + pmin_gis[0]) / 2
    pmax = convertToXY(pmax_gis, lat_mean)
    pmin = convertToXY(pmin_gis, lat_mean)
    if DEBUG:
        print(pmax, pmin)
    pmax_adj = adjustXY(pmax, pmin)
    pmin_adj = adjustXY(pmin, pmin)
    return pmin_adj, pmax_adj


def load_data(num_users=100):
    df = pd.read_csv('loc-gowalla_totalCheckins.txt', sep='\t', header=None)
    df.columns = ['id', 'time', 'lat', 'longt', 'locId']
    df = df[(df['lat'] > 37.735) & (df['lat'] < 37.81151) & (df['longt'] < -122.38) & (df['longt'] > -122.5)]
    uids = df.id.unique()

    uids = uids[:num_users]
    df = df[df['id'].isin(uids)]
    return df, uids


def load_data_syn(num_users=100000):
    df = pd.read_csv('loc-gowalla_totalCheckins.txt', sep='\t', header=None)
    df.columns = ['id', 'time', 'lat', 'longt', 'locId']
    # df = df[(df['lat'] > 37.735) & (df['lat'] < 37.81151) & (df['longt'] < -122.38) & (df['longt'] > -122.5)]
    uids = df.id.unique()

    uids = uids[:num_users]
    df = df[df['id'].isin(uids)]
    return df, uids



def pick_patients(uids, num_patients=5, randomly=True, random_seed=-1, non_random_start_index=50):
    if randomly:
        if random_seed != -1:
            random.seed(random_seed)
        idxs = random.sample(range(0, len(uids)), num_patients)
        idxs.sort()
        pids = [uids[idx] for idx in idxs]
        non_patients_ids = []
        idx_pre = 0
        for idx in idxs:
            non_patients_ids = np.concatenate((non_patients_ids, uids[idx_pre: idx]))
            idx_pre = idx + 1
        non_patients_ids = np.concatenate((non_patients_ids, uids[idx_pre:]))
        print("random idxs", idxs)
    else:
        pids = uids[non_random_start_index:non_random_start_index + num_patients]
        non_patients_ids = np.concatenate((uids[:non_random_start_index], uids[non_random_start_index + num_patients:]))

    print("num of patients: ", len(pids), "num of test users:", len(non_patients_ids))
    print("patients: ", pids)

    if DEBUG:
        print("non_patients_ids", non_patients_ids)

    return pids, non_patients_ids


# convert dataframe records of GIS to X,Y coordinates
# lat_mean is the mean latitude of all records (not necessarily df_input)
# pmin is the lower-left corner absolutely X,Y points
def dataframeToXY(df_input, lat_mean, pmin):
    num_points = len(df_input)
    p_list = []
    for i in range(num_points):
        p_gis = (radians(df_input.iloc[i]['lat']), radians(df_input.iloc[i]['longt']))
        p = convertToXY(p_gis, lat_mean)
        p = adjustXY(p, pmin)
        # print(p)
        p_list.append(p)
    p_list = np.array(p_list)
    return p_list


def get_XY_from_df(df_input):
    p_list = []
    for i, row in df_input.iterrows():
        p = (row['x'], row['y'])
        p_list.append(p)
    p_list = np.array(p_list)
    return p_list


def eula_dist(p_a, p_b):
    return np.sqrt(np.square(p_a[0] - p_b[0]) + np.square(p_a[1] - p_b[1]))


def check_points(p_list, p_patients, radius=5):
    for idx, p in enumerate(p_list):
        for p_t in p_patients:
            if eula_dist(p, p_t) < radius:
                return True, idx

    return False, idx


def uniform_gen_noisy_points(p_list, eps_u):
    num_points = len(p_list)
    eps_divided = eps_u / num_points
    r = gamma.rvs(a=2, scale=1 / eps_divided, size=num_points) * R_UNIT
    theta = np.random.rand(num_points) * 2 * np.pi
    noise = np.c_[r * np.cos(theta), r * np.sin(theta)]
    noisy_points = p_list + noise
    return noisy_points


def my_method_gen_noise(p_list, idx, eps_u):
    num_points = len(p_list)
    # print(num_points)
    if DEBUG:
        print(eps_u)

    eps_idx = (eps_u) * (0.5 + 0.5 / num_points)
    eps_divided = eps_u * 0.5 / num_points
    r = gamma.rvs(a=2, scale=1 / eps_divided, size=num_points - 1) * R_UNIT
    r_idx = gamma.rvs(a=2, scale=1 / eps_idx, size=1) * R_UNIT
    if DEBUG:
        print(eps_idx)
        print(eps_divided)
        print(r)
        print(r_idx)

    r = np.insert(r, idx, r_idx)
    if DEBUG:
        print(r)
    theta = np.random.rand(num_points) * 2 * np.pi
    noise = np.c_[r * np.cos(theta), r * np.sin(theta)]
    # print(r)
    noisy_points_new = p_list + noise
    return noisy_points_new


def gen_noise_with_noisy_count(p_list, noisy_count, eps_u):
    num_points = len(p_list)
    allocated_eps = np.zeros(num_points)
    p_noisy_count = np.zeros(num_points)

    x_num = 10
    dx = 1000
    y_num = 16
    dy = 500

    for idx, p in enumerate(p_list):
        x, y = p
        x_index = int(x / dx)
        y_index = int(y / dy)
        p_noisy_count[idx] = noisy_count[x_index][y_index]
    allocated_eps = p_noisy_count / p_noisy_count.sum() * eps_u
    allocated_eps[allocated_eps < 0] = 0.00001
    print("the privacy budget allocation by noisy count: ")
    print(allocated_eps)
    r = [gamma.rvs(a=2, scale=1 / eps_idx ) * R_UNIT for eps_idx in allocated_eps]
    # print(r)
    theta = np.random.rand(num_points) * 2 * np.pi
    noise = np.c_[r * np.cos(theta), r * np.sin(theta)]
    noisy_points = p_list + noise
    return noisy_points


def prob_dist(eps, dist):
    return eps * eps * math.exp(-eps * dist) / (2 * math.pi)


def test_visualize_optimization(eps_set, diff):
    # code snippet to draw the eps * eps plots of user 24
    from numpy import exp, arange
    from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show

    x = eps_set
    y = eps_set
    X, Y = meshgrid(x, y)  # grid of point
    Z = np.zeros((len(x), len(y)))
    for idx, eps_x in enumerate(x):
        for idy, eps_y in enumerate(y):
            Z[idx][idy] = diff[0][idx] + diff[1][idy]

    # Z = z_func(X, Y)  # evaluation of the function on the grid

    im = imshow(Z, cmap=cm.RdBu)  # drawing the function
    # adding the Contour lines with labels
    cset = contour(Z, arange(0, 13000, 500), linewidths=2, cmap=cm.Set2)
    clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
    colorbar(im)  # adding the colobar on the right
    # latex fashion title
    title('$z=eps vs. eps')
    show()


def derivative(eps, d_const):
    eps_exp = math.exp(-eps * d_const)
    return 2 * eps * eps_exp - eps * eps * d_const * eps_exp


def eval_objective_derivative(pt, eps, noisy_count):
    x, y = pt
    x_i = int(x / DX)
    y_i = int(y / DY)
    p_noisy_count = noisy_count[x_i][y_i]

    objective_t = 0
    derivative_t = 0

    for t_x in range(X_MAX):
        for t_y in range(Y_MAX):
            temp_p = ((t_x+0.5)*DX, (t_y+0.5)*DY)
            count_diff = abs(noisy_count[x_i][y_i] - noisy_count[t_x][t_y])
            distance = eula_dist(pt, temp_p) / R_UNIT
            prob = prob_dist(eps, distance) * DX * DY
            objective_t += count_diff * prob
            constant = DX * DY / (2 * math.pi)
            derivative_t += constant * derivative(eps, distance) * count_diff

    return objective_t, derivative_t


def allocate_noisy_count(p_list, noisy_count, eps_u):
    p_noisy_count = np.zeros(len(p_list))
    for idx, p in enumerate(p_list):
        x, y = p
        x_index = int(x / DX)
        y_index = int(y / DY)
        p_noisy_count[idx] = noisy_count[x_index][y_index]
    allocated_eps = p_noisy_count / p_noisy_count.sum() * eps_u
    allocated_eps[allocated_eps < 0] = 0.00001
    return allocated_eps


def find_optimal(p_list, noisy_count, eps_u):
    num_points = len(p_list)
    # allocated_eps = np.full(num_points, eps_u / num_points)
    allocated_eps = np.random.rand(num_points)
    sum = allocated_eps.sum()
    allocated_eps = allocated_eps * eps_u / sum

    if num_points == 1:
        return allocated_eps

    objective = 0
    eta = 0.01
    early_stopping_count = 0
    num_iters = 0

    while True:
        new_objective = 0
        num_iters += 1
        if num_iters > 500:
            break
        for idx, pt in enumerate(p_list):
            objective_t, derivative_t = eval_objective_derivative(pt, allocated_eps[idx], noisy_count)
            new_objective += objective_t
            allocated_eps[idx] -= eta * derivative_t
            if allocated_eps[idx] < 0.01:
                allocated_eps[idx] = 0.01

        # map it back to the simplex
        sum = allocated_eps.sum()
        allocated_eps = allocated_eps * eps_u / sum

        print('descent step %d. new objective: %.2f. old objective: %.2f' % (num_iters, new_objective, objective))


        if new_objective < 1000:
            objective = new_objective
            break
        if abs(new_objective - objective) < 1000:
            early_stopping_count += 1
            if early_stopping_count > 5:
                if new_objective > 10000:
                    early_stopping_count = 0
                    allocated_eps = np.random.rand(num_points)
                    sum = allocated_eps.sum()
                    allocated_eps = allocated_eps * eps_u / sum
                    objective = new_objective
                    continue
                else:
                    break
        else:
            early_stopping_count = 0
        objective = new_objective

    if objective > 10000:
        allocated_eps = allocate_noisy_count(p_list, noisy_count, eps_u)
        # allocated_eps = np.full(num_points, eps_u / num_points)
    print("the privacy budget allocation by gradient descent: ")
    print(allocated_eps)
    return allocated_eps


def gen_noise_with_gradient(p_list, noisy_count, eps_u):
    num_points = len(p_list)
    allocated_eps = find_optimal(p_list, noisy_count, eps_u)
    r = [gamma.rvs(a=2, scale=1 / eps_p) * R_UNIT for eps_p in allocated_eps]

    theta = np.random.rand(num_points) * 2 * np.pi
    noise = np.c_[r * np.cos(theta), r * np.sin(theta)]
    noisy_points = p_list + noise
    return noisy_points


def gen_noise_patients(p_patients, eps_patient):
    num_points = len(p_patients)
    if num_points == 0:
        return p_patients
    eps_divided = eps_patient / num_points
    r = gamma.rvs(a=2, scale=1 / eps_divided, size=num_points) * R_UNIT
    theta = np.random.rand(num_points) * 2 * np.pi
    noise = np.c_[r * np.cos(theta), r * np.sin(theta)]
    noisy_points = p_patients + noise
    return noisy_points


def gen_noise_with_optimization(p_list, noisy_count, eps_u):
    x_num = 10
    dx = 1000
    y_num = 16
    dy = 500

    # eps_set = [0.01, 0.03, 0.1, 0.3, 0.5]
    eps_delta = 0.02
    eps_set = np.arange(eps_delta, eps_u, eps_delta)
    num_eps = len(eps_set)

    num_points = len(p_list)

    p_noisy_count = np.zeros(num_points)
    diff = np.zeros((num_points, num_eps))

    for idx, p in enumerate(p_list):
        x, y = p
        x_i = int(x/dx)
        y_i = int(y/dy)
        p_noisy_count[idx] = noisy_count[x_i][y_i]

        for eps_idx, eps_t in enumerate(eps_set):
            diff_t = 0
            for id_x in range(x_num):
                for id_y in range(y_num):
                    cur_p = ((id_x+0.5) * dx, (id_y+0.5) * dy)
                    count_diff = abs(noisy_count[x_i][y_i] - noisy_count[id_x][id_y])
                    prob = prob_dist(eps_t, eula_dist(p, cur_p) / R_UNIT) * dx * dy
                    diff_t += count_diff * prob
            diff[idx][eps_idx] = diff_t
        if DEBUG:
            print(" for point # : ", idx, p, "diff is ", diff[idx], "noisy count is", p_noisy_count[idx])

    # test_visualize(eps_set, diff)

    opt = np.zeros((num_points+1, 2*num_eps))
    opt_idx = np.full((num_points+1, 2*num_eps), -1)
    for i in range(num_eps):
        opt[0][i] = 0
    for i in range(num_points):
        for prev in range(num_eps)[i:]:
            if i == 0 and prev > 0:
                break
            for cur in range(num_eps):
                if opt_idx[i+1][prev+cur+1] == -1 or opt[i][prev] + diff[i][cur] < opt[i+1][prev+cur+1]:
                    opt[i+1][prev+cur+1] = opt[i][prev] + diff[i][cur]
                    opt_idx[i+1][prev+cur+1] = cur+1

    if DEBUG:
        print('the smallest cost function', opt[num_points][num_eps])
    remain = num_eps
    allocated_eps = np.zeros(num_points)

    for i in range(num_points):
        p = num_points - i
        if DEBUG:
            print('allocated for point ', p, 'budget' , opt_idx[p][remain] )
        allocated_eps[p-1] = opt_idx[p][remain] * eps_delta
        remain -= opt_idx[p][remain]

    if DEBUG:
        print("the privacy budget allocation by discrete dynamic programming: ")
    print(allocated_eps)

    r = [gamma.rvs(a=2, scale=1 / eps_p ) * R_UNIT for eps_p in allocated_eps]

    theta = np.random.rand(num_points) * 2 * np.pi
    noise = np.c_[r * np.cos(theta), r * np.sin(theta)]
    noisy_points = p_list + noise
    return noisy_points


def check_close_contacts_by_id(df, pids, non_patients_ids):
    df_patients = df[df['id'].isin(pids)]
    df_others = df[df['id'].isin(non_patients_ids)]
    patients_pois = df_patients.locId.unique()

    # check based on exact locId
    df_close_contacts = df_others[df_others['locId'].isin(patients_pois)]
    close_contacts = df_close_contacts.id.unique()
    print("num of close contacts: ", len(close_contacts))
    print("non-close contacts: ", len(non_patients_ids) - len(close_contacts))

    if DEBUG:
        print(close_contacts)
        print(np.setdiff1d(non_patients_ids, close_contacts))

    return df_patients


def compare_methods(df, df_patients, non_patients_ids, eps_u, eps_patient, synthetic=False):
    if synthetic:
        p_patients = get_XY_from_df(df_patients)
    else:
        lat_mean, pmin = get_reference_xy(df)
        p_patients = dataframeToXY(df_patients, lat_mean, pmin)

    noisy_count = get_noisy_count(df, df_patients, eps_patient=eps_patient, synthetic=synthetic)

    row = {}
    test_methods = ['dynamic', 'po', 'uo']
    for method in test_methods:
        t_p_method = f_p_method = 0
        f_n_method = t_n_method = 0
        pos = neg = 0
        avg_response = 0
        for i, uid in enumerate(non_patients_ids):
            df_id = df[df["id"] == uid]
            if synthetic:
                p_id = get_XY_from_df(df_id)
            else:
                p_id = dataframeToXY(df_id, lat_mean, pmin)
            p_list = p_id
            if i % 20 == 0:
                print("finished %.2f" % (i / len(non_patients_ids) * 100), "% with uid: ", uid)

            if len(p_list) > 25:
                # print('passed point i, with uid', i, uid)
                continue

            flag, _ = check_points(p_list, p_patients, 5)
            if flag:
                pos += 1
            else:
                neg += 1

            if method == 'dynamic':
                noisy_points = gen_noise_with_optimization(p_list, noisy_count, eps_u)
                t_start = time.time()
                flag_my, _ = check_points(noisy_points, p_patients, 80)
                avg_response += time.time() - t_start
            elif method == 'po':
                noisy_points = uniform_gen_noisy_points(p_list, eps_u)
                t_start = time.time()
                flag_my, _ = check_points(noisy_points, p_patients, 80)
                avg_response += time.time() - t_start
            elif method == 'uo':
                noisy_patient_points = gen_noise_patients(p_patients, eps_patient=eps_patient)
                t_start = time.time()
                flag_my, _ = check_points(p_list, noisy_patient_points, 80)
                avg_response += time.time() - t_start

            if flag and flag_my:
                t_p_method += 1
            elif not flag and flag_my:
                f_p_method += 1
            if flag and not flag_my:
                f_n_method += 1
            if not flag and not flag_my:
                t_n_method += 1
            # end of loop of checking all points
        if pos == 0:
            recall = 0
        else:
            recall = t_p_method / pos
        if t_p_method + f_p_method == 0:
            precision = 0
        else:
            precision = t_p_method / (t_p_method + f_p_method)


        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2 * ((precision * recall)/(precision + recall))

        if pos + neg == 0:
            avg_response = 0
            acc = 0
        else:
            avg_response = avg_response / (pos + neg)
            acc = (t_p_method + t_n_method) / (pos + neg)

        row['acc_' + method] = acc
        row['recall_' + method] = recall
        row['precision_' + method] = precision
        row['f1_' + method] = f1
        row['time_' + method] = avg_response

            # noisy_points = my_method_gen_noise(p_list, idx, eps_u)
            # noisy_points = gen_noise_with_noisy_count(p_list, noisy_count, eps_u)
            # noisy_points = gen_noise_with_gradient(p_list, noisy_count, eps_u)
            # noisy_points = p_list
            # noisy_points_3 = gen_noise_with_noisy_count(p_list, noisy_count, eps_u)

            # print("positive uid %d checked by discrete method:" % uid, flag2)
            # t_p_discre += int(flag2)
            # print("positive uid %d checked by noisy count:" % uid, flag3)
            # t_p_noisy_c += int(flag3)

    print("# of close contact: ", pos, "# of non-close contact: ", neg)
    print(row)
    return row


def get_noisy_count(df, df_patients, eps_patient, synthetic=False):
    if synthetic:
        p_patients = get_XY_from_df(df_patients)
    else:
        lat_mean, pmin = get_reference_xy(df)
        p_patients = dataframeToXY(df_patients, lat_mean, pmin)
    x_num = 12
    dx = 1000
    y_num = 18
    dy = 500
    count = np.zeros((x_num, y_num))
    for p in p_patients:
        x, y = p
        x_index = int(x / dx)
        y_index = int(y / dy)
        count[x_index, y_index] += 1

    if DEBUG:
        print(count)

    loc, scale = 0., 1.0 / eps_patient
    s = np.random.laplace(loc, scale, x_num * y_num).reshape(x_num, y_num)
    count = count + s
    return count


def visualize(df, df_patients):
    plt.figure(figsize=(15, 15), dpi=150)

    lat_mean, pmin = get_reference_xy(df)
    p_patients = dataframeToXY(df_patients, lat_mean, pmin)
    x, y = p_patients.T
    plt.scatter(x, y, s=100, color='orange')

    plt.grid()

    plt.show()

    return

    uid = 24
    df_id = df[df["id"] == uid]
    eps_u = 2.0
    p_id = dataframeToXY(df_id, lat_mean, pmin)
    flag, idx = check_points(p_id, p_patients, 5)
    x, y = p_id.T
    plt.scatter(x, y, color='black')

    print("is user: ", uid, " a close contact?: ", flag)
    print("now idx: ", idx)

    df_id_39 = df[df["id"] == 39]
    p_id_39 = dataframeToXY(df_id_39, lat_mean, pmin)
    x, y = p_id_39.T
    # plt.scatter(x, y, color='black')

    if False:
        noisy_count = get_noisy_count(df, df_patients)
        noisy_points = gen_noise_with_noisy_count(p_id, noisy_count, eps_u)
        print(noisy_points)
        x, y = noisy_points.T
        plt.scatter(x, y, color='orange')

        flag, idx = check_points(noisy_points, p_patients, 80)
        print("tested close contact by my method?: ", flag)

    for i in range(len(p_id)-1):
        x = [p_id[i][0], p_id[i+1][0]]
        y = [p_id[i][1], p_id[i+1][1]]
        plt.arrow(x[0], y[0], x[1] - x[0], y[1] - y[0], width=2, head_width=40, head_length=50, color='black', length_includes_head=True)
        # plt.plot(x, y, color='blue')

    for i in range(len(p_id_39)-5):
        x = [p_id_39[i][0], p_id_39[i+1][0]]
        y = [p_id_39[i][1], p_id_39[i+1][1]]
        plt.arrow(x[0], y[0], x[1] - x[0], y[1] - y[0], head_width=40, head_length=50, width=2, length_includes_head=True)
        plt.scatter(x, y, color='black')

    for p in p_id:
        for p_t in p_patients:
            if eula_dist(p, p_t) < 5:
                x, y = p.T
                plt.scatter(x, y, color='red')

    for p in p_id_39:
        for p_t in p_patients:
            if eula_dist(p, p_t) < 5:
                x, y = p.T
                plt.scatter(x, y, color='yellow')

    plt.grid()

    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps_P', type=float, default=2.0,
                        help='eps_P')
    parser.add_argument('--eps_u', type=float, default=2.0,
                        help='eps_u')
    parser.add_argument('--num_patients', type=int, default=2, help='Number of patients.')
    parser.add_argument('--num_users', type=int, default=200, help='Number of users.')
    parser.add_argument('--randomly', action='store_true', default=False,
                        help='Random select patients.')
    parser.add_argument('--gen_synthetic', action='store_true', default=False,
                        help='Generate synthetic dataset.')
    parser.add_argument('--random_start', type=int, default=150, help='random start.')
    parser.add_argument('--num_rounds', type=int, default=1, help='Number of rounds.')


    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    df, uids = load_data(num_users=args.num_users)
    num_experiments = args.num_rounds
    result = []
    for i in range(num_experiments):
        print("experiment round: ", i)
        pids, non_patients_ids = pick_patients(uids, num_patients=args.num_patients, randomly=args.randomly, non_random_start_index=args.random_start, random_seed=-1)
        df_patients = check_close_contacts_by_id(df, pids, non_patients_ids)
        # non_patients_ids = [24]
        # visualize(df, df_patients)
        # return
        row = compare_methods(df, df_patients, non_patients_ids, eps_u=args.eps_u, eps_patient=args.eps_P)
        if row != -1:
            result.append(row)
    result = pd.DataFrame(result)
    test_methods = ['uo', 'po', 'dynamic']
    final = []
    for method in test_methods:
        averaged = [method, result['recall_' + method].mean(), result['precision_' + method].mean(),
                    result['f1_' + method].mean(), result['time_' + method].mean()*1000, result['acc_' + method].mean()]
        final.append(averaged)

        #final['recall_' + method] = result['recall_' + method].mean()
        #final['precision_' + method] = result['precision_' + method].mean()
        #final['f1_' + method] = result['f1_' + method].mean()
        #final['time_' + method] = result['time_' + method].mean()*1000 # showing ms

    df_result = pd.DataFrame(final, columns=['METHOD', 'RECALL', 'PRECISION', 'F1', 'TIME', 'ACC'])
    df_result = df_result.set_index('METHOD')
    print(args)
    print(df_result)
    df_result.to_csv('result_T_{:d}_num_patiets_{:d}_eps_u_{:.2f}_eps_P_{:.2f}_{:s}.csv'.format(
        args.num_users, args.num_patients, args.eps_u, args.eps_P, str(time.time())), sep='\t')


def gen_synthetic():
    args = parse_arguments()
    df, _ = load_data(num_users=1600)
    pmin, pmax = get_min_max_xy(df)

    df_data, uids = load_data_syn(num_users=100000)
    print('pmin, pmax', pmin, pmax)
    df_data.columns = ['id', 'time', 'x', 'y', 'locId']

    for i, row in df_data.iterrows():
        if i % 1000 == 0:
            print("progress: ", i, i/len(df_data))
        p_x = random.uniform(pmin[0], pmax[0])
        p_y = random.uniform(pmin[1], pmax[1])
        df_data.at[i, 'x'] = p_x
        df_data.at[i, 'y'] = p_y

    df_data.to_csv('df_synthetic_'+str(100000)+'.csv', sep='\t', header=None)


def compare_synthetic(args):

    df = pd.read_csv('df_synthetic_'+str(100000)+'.csv', sep='\t', header=None)
    df.columns = ['rid', 'id', 'time', 'x', 'y', 'locId']
    uids = df.id.unique()

    uids = uids[:args.num_users]
    df = df[df['id'].isin(uids)]

    result = []
    num_experiments = args.num_rounds

    for i in range(num_experiments):
        print("experiment round: ", i)

        pids, non_patients_ids = pick_patients(uids, num_patients=args.num_patients, randomly=args.randomly,
                                           non_random_start_index=args.random_start, random_seed=-1)
        df_patients = df[df['id'].isin(pids)]

        row = compare_methods(df, df_patients, non_patients_ids, eps_u=args.eps_u, eps_patient=args.eps_P, synthetic=True)
        print(row)

        if row != -1:
            result.append(row)
    result = pd.DataFrame(result)
    test_methods = ['uo', 'po', 'dynamic']
    final = []
    for method in test_methods:
        averaged = [method, result['recall_' + method].mean(), result['precision_' + method].mean(),
                    result['f1_' + method].mean(), result['time_' + method].mean() * 1000]
        final.append(averaged)

        # final['recall_' + method] = result['recall_' + method].mean()
        # final['precision_' + method] = result['precision_' + method].mean()
        # final['f1_' + method] = result['f1_' + method].mean()
        # final['time_' + method] = result['time_' + method].mean()*1000 # showing ms

    df_result = pd.DataFrame(final, columns=['METHOD', 'RECALL', 'PRECISION', 'F1', 'TIME'])
    df_result = df_result.set_index('METHOD')
    print(args)
    print(df_result)
    df_result.to_csv('syn_result_T_{:d}_num_patiets_{:d}_eps_u_{:.2f}_eps_P_{:.2f}_{:s}.csv'.format(
        args.num_users, args.num_patients, args.eps_u, args.eps_P, str(time.time())), sep='\t')


if __name__ == '__main__':
    args = parse_arguments()
    if args.gen_synthetic:
        gen_synthetic()
    else:
        main(args)
    #compare_synthetic()

