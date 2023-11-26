"""
A box generator class to spawn random boxes in the UNITY simulator
It allows to define the range where the boxes will be spawned in the workspace of the robot
"""

import os

import scipy
import scipy.spatial

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

class RandomBoxesGenerator:
    def __init__(self, box_mode="train", box_samples=400, box_split=0.1, box_save_val=False, box_load_val=False,
                 box_radius_val=0.01, box_min_distance_base=0.475, box_max_distance_base=0.67, box_folder="./agent/logs/dataset/",
                 box_x_min=-0.67, box_x_max=0.67, box_x_active=True, box_z_min=0.42, box_z_max=0.67, box_z_active=True, 
                 box_ry_min=-np.inf, box_ry_max=np.inf, box_ry_active=False, box_debug=[0.48159, 0.05, 0.4406864, 0.0, 0.0, 0.0]):

        self.box_mode = box_mode                            # Train, test, debug
        self.box_samples = box_samples
        self.index_train = 0                                # Current index
        self.index_val = 0                                  # Starting index of val boxes
        self.val_size = 0
        self.train_size = 0
        self.box_split = box_split                          # Val split -> 0.1, 0.2, etc
        self.box_load_val = box_load_val                    # Load validation boxes from a saved dataset
        self.box_folder = box_folder                        # Folder dataset path to load (val), and/or save the generated dataset (train, val)
        self.box_radius_val = box_radius_val                # Exclude spawned train boxes that are close to the val boxes within this radius (threshold) in meters. e.g. 0.01
        self.box_min_distance_base = box_min_distance_base
        self.box_max_distance_base = box_max_distance_base
        self.box_save_val = box_save_val                    # Save generated dataset 
        self.box_debug = box_debug

        ##############
        # Set ranges #
        ##############
        self.box_x_min = box_x_min
        self.box_x_max = box_x_max
        self.box_x_active = box_x_active
        self.box_z_min = box_z_min
        self.box_z_max = box_z_max
        self.box_z_active = box_z_active
        self.box_ry_min = box_ry_min
        self.box_ry_max = box_ry_max
        self.box_ry_active = box_ry_active

        ######################
        # Set-up the Dataset #
        ######################

        # Train split #
        self.objects_X, self.objects_Y, self.objects_Z, self.objects_RY = [], [], [], []

        # Val split #
        self.objects_X_val, self.objects_Y_val, self.objects_Z_val, self.objects_RY_val = [], [], [], []

        # Create the directory to save the dataset #
        Path(self.box_folder).mkdir(parents=True, exist_ok=True)

        # Set-up the train and val splits #
        if (self.box_mode == "debug"):
            self.val_size = 1
            self.train_size = 1
        else:
            self.val_size = int(self.box_samples * self.box_split)
            self.train_size = self.box_samples - self.val_size

        # Useful struct for computing fast euclidean distances. Refer to self._get_samples() #
        self.val_set = np.column_stack((self.objects_X_val, self.objects_Z_val))

        # Generate always a random train set #
        for i in range(self.box_samples):
            X, Y, Z, _, RY, _ = self._get_samples(self.objects_X_val, self.objects_Z_val)

            self.objects_X.append(X)
            self.objects_Y.append(Y)
            self.objects_Z.append(Z)
            self.objects_RY.append(RY)

        self.objects_X_train = self.objects_X[:self.train_size]
        self.objects_Y_train = self.objects_Y[:self.train_size]
        self.objects_Z_train = self.objects_Z[:self.train_size]
        self.objects_RY_train = self.objects_RY[:self.train_size]

        # Load validation boxes #
        if (self.box_load_val == True): 
            self._load_val()
        else: # Do not load - use random (newly generated)
            self.objects_X_val = self.objects_X[self.train_size:]
            self.objects_Y_val = self.objects_Y[self.train_size:]
            self.objects_Z_val = self.objects_Z[self.train_size:]
            self.objects_RY_val = self.objects_RY[self.train_size:]

        # Save the box dataset #
        if (self.box_save_val == True):
            self._save_dataset()

    def __call__(self):
        """
            return the next box from the dataset

            :return: x, y, z, 0, ry, 0 box coords - in unity coords system - orientation in degrees
        """

        if (self.box_mode == "train"):
            box = [
                self.objects_X_train[self.index_train], self.objects_Y_train[self.index_train],
                self.objects_Z_train[self.index_train], 0.0, self.objects_RY_train[self.index_train], 0.0
            ]

            self.index_train += 1
            if (self.index_train == self.train_size): # Important: start from the beginning
                self.index_train = 0

        elif (self.box_mode == "val"):
            box = [
                self.objects_X_val[self.index_val], self.objects_Y_val[self.index_val],
                self.objects_Z_val[self.index_val], 0.0,  self.objects_RY_val[self.index_val], 0.0
            ]

            self.index_val += 1
            if (self.index_val == self.val_size):
                self.index_val = 0

        elif (self.box_mode == "debug"):
            box = self.box_debug

        else:
            raise Exception("Not valid box mode")

        return box

    def _load_val(self):
        """
            load the validation split from a saved dataset file
        """

        val_set_file = np.loadtxt(os.path.join(self.box_folder, "val_data.txt"), delimiter=",")

        for box in val_set_file:
            self.objects_X_val.append(box[0])
            self.objects_Y_val.append(box[1])
            self.objects_Z_val.append(box[2])
            self.objects_RY_val.append(box[3])

        self.objects_X_val = np.asarray(self.objects_X_val)
        self.objects_Y_val = np.asarray(self.objects_Y_val)
        self.objects_Z_val = np.asarray(self.objects_Z_val)
        self.objects_RY_val = np.asarray(self.objects_Z_val)

        self.val_size = len(self.objects_X_val)

    def _get_samples(self, objects_X_val, objects_Z_val):
        """
            generate one random box with coordinates that complies with the user-defined constraints

            if 'box_radius_val' is not 0.0: keep spawning until the generated box is far apart from all the val boxes by the defined threshold

            :param objects_X_val: x validation boxes coords

            :return: x, y, z, 0, ry, 0 box coords - in unity coords system - orientation in degrees
        """
        while True:
            if(self.box_x_active):
                X = np.random.uniform(self.box_x_min, self.box_x_max, size=1)[0]
            else:
                X = 0

            Y = 0.05 # Hard-coded box height. Adapt to your task

            if(self.box_z_active):
                Z = np.random.uniform(self.box_z_min, self.box_z_max, size=1)[0]
            else:
                Z = 0

            if(self.box_ry_active):
                RY = np.random.uniform(self.box_ry_min, self.box_ry_max, size=1)[0]
            else:
                RY = 0

            distance_to_base = np.sqrt(X**2 + Z**2)
            if (distance_to_base > self.box_min_distance_base and distance_to_base < self.box_max_distance_base):

                # Train boxes must be far apart from the validation boxes #
                if(self.box_radius_val >= 0.001):                         # Threshold is active 
                    point_set = np.asarray([[X, Z]])                      # Current generated box

                    # Calculate the distances to all other validation boxes from the generated box #
                    dist_radius = scipy.spatial.distance.cdist(self.val_set, point_set, 'euclidean')

                    # Check violation #
                    violation = False
                    for val_dist in dist_radius:
                        if (val_dist <= self.box_radius_val):
                            violation = True

                    if violation: # Box too close to a validation box - spawn another box
                        continue

                return X, Y, Z, 0.0, RY, 0.0

    def _save_dataset(self):
        """
            save the generated dataset:
                - .txt formant: x,y,z,ry
                - saves also some .png plots
        """

        # Train split #
        x_np_train = np.asarray(self.objects_X_train)
        x_np_train = x_np_train.reshape(x_np_train.shape[0], 1)
        z_np_train = np.asarray(self.objects_Z_train)
        z_np_train = z_np_train.reshape(z_np_train.shape[0], 1)
        ry_np_train = np.asarray(self.objects_RY_train)
        ry_np_train = ry_np_train.reshape(ry_np_train.shape[0], 1)
        y_np_train = np.ones_like(x_np_train) * 0.05
        train_data = np.concatenate((x_np_train, y_np_train, z_np_train, ry_np_train), axis=1)

        # Val split #
        x_np_val = np.asarray(self.objects_X_val)
        x_np_val = x_np_val.reshape(x_np_val.shape[0], 1)
        z_np_val = np.asarray(self.objects_Z_val)
        z_np_val = z_np_val.reshape(z_np_val.shape[0], 1)
        ry_np_val = np.asarray(self.objects_RY_val)
        ry_np_val = ry_np_val.reshape(ry_np_val.shape[0], 1)
        y_np_val = np.ones_like(x_np_val) * 0.05
        val_data = np.concatenate((x_np_val, y_np_val, z_np_val, ry_np_val), axis=1)

        # Save to .txt files #
        np.savetxt(os.path.join(self.box_folder, "train_data.txt"), train_data, delimiter=",")
        np.savetxt(os.path.join(self.box_folder, "val_data.txt"), val_data, delimiter=",")

        #########
        # Plots #
        #########

        # x - z coords #
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.scatter(x_np_train, z_np_train, s=10, c='#726bff', marker="s", label='train')
        ax1.scatter(x_np_val, z_np_val, s=10, c='#ff6e66', marker="o", label='val')
        plt.legend(loc='upper left')
        plt.xlabel("x")
        plt.ylabel("z")
        plt.title("dataset")
        plt.savefig(os.path.join(self.box_folder, 'box_dataset_x_z_axis.png'))

        # ry - y coords #
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        # Adapt box height to your task #
        ax1.scatter(ry_np_train, 0.05 * np.ones_like(ry_np_train), s=10, c='b', marker="s", label='train')
        ax1.scatter(ry_np_val, 0.05 * np.ones_like(ry_np_val), s=10, c='r', marker="o", label='val')
        plt.legend(loc='upper left')
        plt.xlabel("ry (deg)")
        plt.ylabel("y")
        plt.title("dataset")
        plt.savefig(os.path.join(self.box_folder, 'box_dataset_ry_y_axis.png'))

        # x - z - ry coords #
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')

        ax1.scatter(x_np_train, z_np_train, ry_np_train, s=10, c='b', marker="s", label='train')
        ax1.scatter(x_np_val, z_np_val, ry_np_val, s=10, c='r', marker="o", label='val')
        plt.legend(loc='upper left')
        ax1.set_xlabel("x")
        ax1.set_ylabel("z")
        ax1.set_zlabel("ry (deg)")

        plt.title("dataset")
        plt.savefig(os.path.join(self.box_folder, 'box_dataset.png'))
