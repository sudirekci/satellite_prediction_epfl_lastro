import numpy as np


def find_range(ellipse_parameters):
    """
    Finds range of rotation angles in ellipse_parameters.
    :param ellipse_parameters: parameter list
    :return: range (mean-std, mean+std)
    """
    a = [x[3] for x in ellipse_parameters]
    thetas = [x[5] + (a[ind] != 0) * np.pi / 2 for ind, x in enumerate(ellipse_parameters)]
    mean, std = np.mean(thetas), np.std(thetas)
    return mean - std, mean + std


def calc_all_angles(centers, focus):
    """
    Calculate the angles between combinations of coordinates in the given array (centers). If focus == -1, calculate all
    possible angles. Otherwise, select the "focus"th element in centers, calculate angle combinations between that and
    the rest of the array elements.
    :param centers: given array
    :param focus: index of array element or -1
    :return: calculated angles (rad)
    """
    angles = []
    if focus == -1:
        start, end = 0, len(centers)
    else:
        start, end = focus, focus + 1
    for m in range(start, end):
        x1, y1 = centers[m]
        for n in range(0, len(centers)):
            if m != n:
                x2, y2 = centers[n]
                angles.append(np.arctan((y2 - y1) / (x2 - x1)))
    return np.unique(angles)


class Cutout_Connecter:

    def __init__(self, satellite_centers, all_thetas):
        self.satellite_centers = satellite_centers
        self.all_thetas = all_thetas
        self.cutout_no = len(self.satellite_centers)
        self.path_arr = []
        self.removed_points = []
        self.path_orientations = []

    def connect_cutouts(self):
        """
        Connect the cutouts given to the object while initialization and double check the paths.
        :return: self.path_arr: found paths, self.path_orientations: orientations of the paths
        """
        Cutout_Connecter.find_connected_cutouts(self)
        Cutout_Connecter.double_check_paths(self)
        return self.path_arr, self.path_orientations

    def find_connected_cutouts(self):
        """
        Finds connected tracks segments.
        """
        np.sort(self.all_thetas, axis=1)
        connected_arr = []
        final_arr = []
        tolerance = 5 * np.pi / 180
        # for each cutout,
        for k in range(0, self.cutout_no):
            x_k, y_k = self.satellite_centers[k]
            for m in range(0, self.cutout_no):
                # for each cutout other than the cutout we started with
                if k != m:
                    x_m, y_m = self.satellite_centers[m]
                    # calculate the angle between their centers
                    theta = np.arctan((y_m - y_k) / (x_m - x_k))
                    # if the angle is in the range of orientation angles of the first cutout (with some tolerance
                    # added), (Roll over cases are considered too, as the angles are constraint to be in (-pi/2, pi/2).
                    # That is the reason of the extra lines in the if statement)
                    if ((self.all_thetas[k, 0] - tolerance) < theta < (self.all_thetas[k, 1] + tolerance)) or (
                            (self.all_thetas[k, 0] - tolerance) <
                            np.pi / 2 and self.all_thetas[k, 0] - tolerance + np.pi < theta) \
                            or ((self.all_thetas[k, 1] + tolerance) > np.pi / 2 and theta < self.all_thetas[k, 1] +
                                tolerance - np.pi):
                        # if the first cutout is in the range of the second cutout already, they are connected. Add
                        # them to final_arr.
                        if [m, k] in connected_arr:
                            final_arr.append([m, k])
                        else:
                            connected_arr.append([k, m])
        # connect the pairs ([1,2], [2,3] --> [1, 2, 3]) and return the path array.
        Cutout_Connecter.find_connected(self, final_arr)

    def find_connected(self, arr):
        """
        Finds all paths in given array. Example: if arr = [[1,2], [2,4], [5,7], []5, 9] then path_arr = [[1,2,4],
        [5,7,9]].
        :param arr: given array with elements [i, j].
        """
        if not arr:
            return []
        arr.sort()
        while arr:
            self.path_arr.append(arr[0])
            arr.remove(arr[0])
            path = self.path_arr[-1]
            i = 0
            while i < len(arr):
                element = arr[i]
                if element[0] in path or element[1] in path:
                    if element[1] not in path:
                        self.path_arr[-1].append(element[1])
                    if element[0] not in path:
                        self.path_arr[-1].append(element[0])
                    path = self.path_arr[-1]
                    arr.remove(element)
                    i = 0
                else:
                    i += 1

    def double_check_paths(self):
        """
        Check the path array to correct it in case of missing or incorrect elements in paths.
        """
        if not self.path_arr:
            print('No paths found.')
            return [], []
        print('Initial version of path array:', self.path_arr)
        # clean paths from incorrect elements
        Cutout_Connecter.remove_wrong_points(self)
        self.path_arr = [x for x in self.path_arr if x]
        for m in range(0, len(self.path_arr)):
            angles = calc_all_angles(self.satellite_centers[self.path_arr[m]], focus=-1)
            self.path_orientations.append([np.mean(angles), np.std(angles)])

        print('check_missed_points starts....')
        # check missed points
        Cutout_Connecter.check_missed_points(self)
        print('Adjusted path array: ', self.path_arr)
        present_points = np.concatenate(self.path_arr)
        for m in range(0, (len(self.satellite_centers) - len(present_points))):
            self.path_orientations.append([0, 0])
        self.path_orientations = np.asarray(self.path_orientations)

    def check_missed_points(self):
        """
        For each paths in path_array, check if any other point in satellite_centers belongs to it but was not added.
        """
        present_points = np.concatenate(self.path_arr)
        cutout_no = len(self.satellite_centers)
        for p in range(0, len(self.path_arr)):
            mean, std = self.path_orientations[p]
            # For each point in satellite_centers that is not in removed_points or in other paths, check if it can be
            # included in the path
            std = max(std, 0.01*np.pi/180)  # in case the standard deviation is too small
            for k in range(0, cutout_no):
                if k not in present_points and k not in self.removed_points:
                    path = self.path_arr[p].copy()
                    path.append(k)
                    angles = calc_all_angles(self.satellite_centers[path], focus=len(path) - 1)
                    # if the mean of the angles calculated with the added element belongs to the distribution, add the
                    # element to the path
                    if mean - std * 3 < np.mean(angles) < mean + 3 * std and np.std(angles) < 5 * np.pi / 180:
                        self.path_arr[p].append(k)
                        self.path_arr[p].sort()
                        self.path_orientations[p] = np.mean(angles), np.std(angles)

    def remove_wrong_points(self):
        """
        Remove incorrect points from paths in path_arr by calculating angle combinations in the path with and without
        each point.
        """
        for m in range(0, len(self.path_arr)):
            self.path_arr[m].sort()
            path = self.path_arr[m]
            if len(path) <= 3:
                continue
            index_list = np.zeros(len(path))
            # for each coordinate in path, check if it's included in the range of angles of the path calculated without
            # it
            for ind, p in enumerate(path):
                path_copy = path.copy()
                path_copy.remove(p)
                angles = calc_all_angles(self.satellite_centers[path_copy], focus=-1)
                angles2 = calc_all_angles(self.satellite_centers[path], focus=ind)
                too_small = np.std(angles) < 0.005
                # if it cannot be included to the distribution of angles calculated without it, it does not belong to
                # the path. Remove it.
                if not np.mean(angles) - np.std(angles) * 3 - too_small * 0.005 < np.mean(angles2) < np.mean(angles) + \
                   np.std(angles) * 3 + too_small * 0.005:
                    index_list[ind] = 1
                    self.removed_points.append(p)

            self.path_arr[m] = [x for ind, x in enumerate(self.path_arr[m]) if index_list[ind] != 1]
