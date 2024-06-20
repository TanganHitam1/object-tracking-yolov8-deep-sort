"""
@Author: Du Yunhao
@Filename: GSI.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/1 9:18
@Discription: Gaussian-smoothed interpolation
"""
import os
import numpy as np
from os.path import join
from collections import defaultdict
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF


# Linear Interpolation
def LinearInterpolation(input_, interval): # input: [f, id, x, y, w, h, s, o, c, d]
    input_ = input_[np.lexsort([input_[:, 0], input_[:, 1]])]  # Sort by frame number and ID (Ascending order) -> (np.array) [f, i, x, y, w, h, s, o, c, d] (np.array)
    output_ = input_.copy() # Copy input data
    # Linear interpolation
    id_pre, f_pre, row_pre = -1, -1, np.zeros((10,)) # Initialize ID, frame number, and row data (10 dimensions) -> (int, int, np.array) (-1, -1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for row in input_: # Iterate through each row (np.array) [f, i, x, y, w, h, s, o, c, d]
        f_curr, id_curr = row[:2].astype(int) # Frame number, ID
        if id_curr == id_pre:  # (Same ID)
            if f_pre + 1 < f_curr < f_pre + interval: # (Interval less than threshold)
                for i, f in enumerate(range(f_pre + 1, f_curr), start=1):  # Linear interpolation
                    step = (row - row_pre) / (f_curr - f_pre) * i # Linear interpolation step size (np.array) [x, y, w, h, s, o, c, d] (np.array) [x, y, w, h, s, o, c, d] (np.array)
                    row_new = row_pre + step # Linear interpolation result (np.array) [x, y, w, h, s, o, c, d] (np.array) [x, y, w, h, s, o, c, d] (np.array) [x, y, w, h, s, o, c, d] (np.array)
                    output_ = np.append(output_, row_new[np.newaxis, :], axis=0) # Append linear interpolation result to output (np.array) [f, i, x, y, w, h, s, o, c, d] (np.array)
        else:  # 不同ID
            id_pre = id_curr # Update ID (ID)
        row_pre = row # Update row data (np.array) [f, i, x, y, w, h, s, o, c, d] (np.array)
        f_pre = f_curr # Update frame number (Frame number)
    output_ = output_[np.lexsort([output_[:, 0], output_[:, 1]])] # Sort by frame number and ID (Ascending order) -> (np.array) [f, i, x, y, w, h, s, o, c, d] (np.array)
    return output_

# (Gaussian Smooth)
def GaussianSmooth(input_, tau): # input: [f, id, x, y, w, h, s, o, c, d] 
    output_ = list() # Initialize output data (list)
    ids = set(input_[:, 1]) # Get unique IDs (set) -> (set) {ID1, ID2, ...}
    for id_ in ids: # Iterate through each ID (ID)
        # print("ID:",id_)
        tracks = input_[input_[:, 1] == id_] # Get trajectory information of the current ID (np.array) [f, i, x, y, w, h, s, o, c, d] (np.array)
        len_scale = np.clip(tau * np.log(tau ** 3 / len(tracks)), tau ** -1, tau ** 2) # Calculate length scale (float)
        # print(tau)
        # print(tau ** -1)
        # print(tau ** 2)
        # print(tau * np.log(tau ** 3 / len(tracks)))
        gpr = GPR(RBF(len_scale, 'fixed')) # Gaussian Process Regressor (GPR) -> (GPR) GPR(RBF(len_scale, 'fixed')) 
        # print("Tracks length: ",len(tracks))
        # print("Length Scale: ",len_scale)
        t = tracks[:, 0].reshape(-1, 1) # Frame number (np.array) [f] (np.array)
        x = tracks[:, 2].reshape(-1, 1) # x-coordinate (np.array) [x] (np.array)
        y = tracks[:, 3].reshape(-1, 1) # y-coordinate (np.array) [y] (np.array)
        w = tracks[:, 4].reshape(-1, 1) # width (np.array) [w] (np.array)
        h = tracks[:, 5].reshape(-1, 1) # height (np.array) [h] (np.array)
        # print("t: ",t)
        gpr.fit(t, x) # Fit x-coordinate (np.array) [x] (np.array) -> (GPR) GPR
        # print("predict: ", gpr.predict(t))
        # print("length ", len(gpr.predict(t)))
        xx = gpr.predict(t) # Predict x-coordinate (np.array) [x] (np.array) -> (np.array) [x] (np.array)
        # print("xx: ",xx)
        gpr.fit(t, y) # Fit y-coordinate (np.array) [y] (np.array) -> (GPR) GPR
        yy = gpr.predict(t) # Predict y-coordinate (np.array) [y] (np.array) -> (np.array) [y] (np.array)
        gpr.fit(t, w) # Fit width (np.array) [w] (np.array) -> (GPR) GPR
        ww = gpr.predict(t) # Predict width (np.array) [w] (np.array) -> (np.array) [w] (np.array)
        gpr.fit(t, h) # Fit height (np.array) [h] (np.array) -> (GPR) GPR
        hh = gpr.predict(t) # Predict height (np.array) [h] (np.array) -> (np.array) [h] (np.array)
        # print(tracks[:,2])
        # print(x)
        # print(xx)
        output_.extend([
            [int(t[i, 0]), int(id_), float(xx[i]), float(yy[i]), float(ww[i]), float(hh[i]), 1, -1, -1 , -1] for i in range(len(t)) # Append Gaussian smoothed interpolation result to output (list) [f, i, x, y, w, h, s, o, c, d] (list)
        ])
    print("Output: ",output_)
    # output_ = output_[np.lexsort([output_[:, 1], output_[:, 0]])]
    output_ = sorted(output_, key=lambda x: x[0])
    return output_ # Return Gaussian smoothed interpolation result (list)

# GSI
def GSInterpolation(path_in, path_out, interval, tau): # path_in: input file path (str), path_out: output file path (str), interval: interpolation interval (int), tau: Gaussian smooth parameter (float)
    input_ = np.loadtxt(path_in, delimiter=',') # Load input data (np.array) [f, i, x, y, w, h, s, o, c, d] (np.array)
    li = LinearInterpolation(input_, interval) # Linear interpolation (np.array) [f, i, x, y, w, h, s, o, c, d] (np.array)
    gsi = GaussianSmooth(li, tau) # Gaussian smooth (list) [f, i, x, y, w, h, s, o, c, d] (list)
    np.savetxt(path_out, gsi, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d') # Save Gaussian smoothed interpolation result (np.array) [f, i, x, y, w, h, s, o, c, d] (np.array)
    
if __name__ == "__main__":
    pathin = "./dataspace/StrongSORT/tmp\MOT17-02-FRCNN.txt"
    pathoutLI = "./LinearInterpolation.txt"
    pathoutGS = "./GS.txt"
    pathoutGSI = "./GSI.txt"
    interval = 20
    tau = 10
    input_ = np.loadtxt(pathin, delimiter=",")
    GSInterpolation(pathin, pathoutGSI, interval, tau)