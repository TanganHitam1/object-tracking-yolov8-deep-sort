"""
@Author: Du Yunhao
@Filename: AFLink.py
@Contact: dyh_bupt@163.com
@Time: 2021/12/28 19:55
@Discription: Appearance-Free Post Link
"""
import datetime
import os
import glob
import torch
import numpy as np
from os.path import join, exists
from collections import defaultdict
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from AFLink import config as cfg
from AFLink.train import train
from AFLink.dataset import LinkData
from AFLink.model import PostLinker
INFINITY = 1e5

class AFLink:
    def __init__(self, path_in, path_out, model, dataset, thrT: tuple, thrS: int, thrP: float):
        self.thrP = thrP          # prediction threshold (0.05 or 0.10)
        self.thrT = thrT          # time domain threshold (0,30) or (-10,30)
        self.thrS = thrS          # airspace threshold 75
        self.model = model        # Predictive model (PostLinker)
        self.dataset = dataset    # Dataset class (LinkData)
        self.path_out = path_out  # Results save path (path)
        self.track = np.loadtxt(path_in, delimiter=',') # Trajectory information (np.array)
        self.model.cuda()        # Model placed on GPU
        self.model.eval()       # Model set to evaluation mode (Do not use | Dropout)

    # Get trajectory information (Frame number, ID, x, y, w, h) -> {ID: [[f, x, y, w, h], ...]} (dict)
    def gather_info(self):
        id2info = defaultdict(list)
        self.track = self.track[np.argsort(self.track[:, 0])]  # Sort by frame (Ascending order) -> (np.array) [f, i, x, y, w, h, s, o, c, d] (np.array)
        for row in self.track: # Iterate through each row (np.array) [f, i, x, y, w, h, s, o, c, d]
            f, i, x, y, w, h = row[:6] # Frame number, ID, x, y, w, h
            id2info[i].append([f, x, y, w, h])
        self.track = np.array(self.track)
        id2info = {k: np.array(v) for k, v in id2info.items()}
        return id2info

    # loss matrix compression
    def compression(self, cost_matrix, ids):
        # row compression
        mask_row = cost_matrix.min(axis=1) < self.thrP
        matrix = cost_matrix[mask_row, :]
        ids_row = ids[mask_row]
        # column compression
        mask_col = cost_matrix.min(axis=0) < self.thrP
        matrix = matrix[:, mask_col]
        ids_col = ids[mask_col]
        # matrix compression
        return matrix, ids_row, ids_col

    # Connection loss prediction
    def predict(self, track1, track2):
        track1, track2 = self.dataset.transform(track1, track2)
        track1, track2 = track1.unsqueeze(0).cuda(), track2.unsqueeze(0).cuda()
        score = self.model(track1, track2)[0, 1].detach().cpu().numpy()
        return 1 - score

    # De-duplication: remove multiple boxes with the same ID in the same frame
    @staticmethod
    def deduplicate(tracks):
        _, index = np.unique(tracks[:, :2], return_index=True, axis=0)  # Ensure the uniqueness of frame number and ID number
        return tracks[index]

    # main function
    def link(self):
        id2info = self.gather_info()
        num = len(id2info)  # target quantity
        ids = np.array(list(id2info))  # Target ID
        fn_l2 = lambda x, y: np.sqrt(x ** 2 + y ** 2)  # L2 distance
        cost_matrix = np.ones((num, num)) * INFINITY  # loss matrix
        '''Calculate loss matrix'''
        for i, id_i in enumerate(ids):      # Previous track
            for j, id_j in enumerate(ids):  # The latter trajectory
                if id_i == id_j: continue   # Self-entertainment is prohibited
                info_i, info_j = id2info[id_i], id2info[id_j] # Previous trajectory information, the latter trajectory information
                fi, bi = info_i[-1][0], info_i[-1][1:3] # Previous frame number, previous trajectory information
                fj, bj = info_j[0][0], info_j[0][1:3]  # The latter frame number, the latter trajectory information
                if not self.thrT[0] <= fj - fi < self.thrT[1]: continue # Time domain threshold
                if self.thrS < fn_l2(bi[0] - bj[0], bi[1] - bj[1]): continue # Airspace threshold
                cost = self.predict(info_i, info_j) # Connection loss prediction
                if cost <= self.thrP: cost_matrix[i, j] = cost # Update loss matrix (if the loss is less than the threshold)
        '''Optimal matching of bipartite graphs'''
        id2id = dict()  # Store temporary matching results
        ID2ID = dict()  # Store final matching results
        cost_matrix, ids_row, ids_col = self.compression(cost_matrix, ids) # loss matrix compression (row, column) -> (matrix, ids_row, ids_col)
        indices = linear_sum_assignment(cost_matrix) # Optimal matching of bipartite graphs (Hungarian algorithm) -> (row, column)
        for i, j in zip(indices[0], indices[1]): # Iterate through the optimal matching results (row, column)
            if cost_matrix[i, j] < self.thrP:   # If the loss is less than the threshold (valid matching)
                id2id[ids_row[i]] = ids_col[j] # Store temporary matching results (Previous ID: The latter ID)
        for k, v in id2id.items(): # Iterate through the temporary matching results (Previous ID: The latter ID)
            if k in ID2ID: # If the previous ID is already in the final matching result
                ID2ID[v] = ID2ID[k] # The latter ID is the same as the previous ID
            else: # If the previous ID is not in the final matching result
                ID2ID[v] = k # The latter ID is the same as the previous ID
        # print('  ', ID2ID.items())
        '''Result storage'''
        res = self.track.copy() # Copy the original trajectory information (np.array) [f, i, x, y, w, h, s, o, c, d]
        for k, v in ID2ID.items(): # Iterate through the final matching results (Previous ID: The latter ID) -> (k, v) (Previous ID, The latter ID)
            res[res[:, 1] == k, 1] = v # Update the trajectory information (Previous ID -> The latter ID) -> (np.array) [f, i, x, y, w, h, s, o, c, d]
        res = self.deduplicate(res) # De-duplication: remove multiple boxes with the same ID in the same frame -> (np.array) [f, i, x, y, w, h, s, o, c, d]
        np.savetxt(self.path_out, res, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d') # Save the results to the specified path


if __name__ == '__main__':
    print(datetime.now())
    # dir_in = '/data/dyh/results/StrongSORT/TEST/MOT20_StrongSORT'
    dir_in = '/data/dyh/results/StrongSORT/ABLATION/CenterTrack'
    dir_out = dir_in + '_tmp'
    # dir_out = '/data/dyh/results/StrongSORT/ABLATION/CenterTrack_AFLink'
    if not exists(dir_out): os.mkdir(dir_out)
    model = PostLinker()
    model.load_state_dict(torch.load(join(cfg.model_savedir, 'newmodel_epoch20.pth')))
    dataset = LinkData(cfg.root_train, 'train')
    for path_in in sorted(glob.glob(dir_in + '/*.txt')):
        print('processing the file {}'.format(path_in))
        linker = AFLink(
            path_in=path_in,
            path_out=path_in.replace(dir_in, dir_out),
            model=model,
            dataset=dataset,
            thrT=(-10,30),  # (0,30) or (-10,30)
            thrS=75,  # 75
            thrP=0.10,  # 0.05 or 0.10
        )
        linker.link()
    print(datetime.now())
    eval(dir_out, flag=0)


