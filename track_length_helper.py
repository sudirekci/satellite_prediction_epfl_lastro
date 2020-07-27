import numpy as np
from utils import adjust_angle, find_ellipse_parameters, pre_process_cutout


class Track_Length_Helper:
    def __init__(self, diff_img, path_arr, path_orientations, satellite_centers, all_thetas, step_size, sizes, model,
                 seg, ellipse_path):
        self.diff_img = diff_img  # difference image
        self.path_arr = path_arr  # path array
        self.path_orientations = path_orientations  # orientations of tracks
        self.satellite_centers = satellite_centers  # center coordinates of track segments
        self.all_thetas = all_thetas  # ranges of orientations of each point in satellite_centers
        self.step_size = step_size  # step size used in taking consecutive cutouts (for length estimation)
        self.half_size, self.bin_size, self.im_size_x, self.im_size_y = sizes[0], sizes[1], sizes[2], sizes[3]
        self.model = model  # neural network model
        self.seg = seg  # segmentation map of the difference image
        self.ellipse_path = ellipse_path  # path to save cutouts of track segments
        self.half_half_size = int(self.half_size/2) # half_size = cutout size / 2, half_half_size = cutout size / 4

    def find_track_lengths(self):
        """
        Estimate lengths of each track in the difference image.
        :return: path_arr: path array (with individual cutouts added as [i]), path_orientations: path orientations,
        path_lengths: path lengths of each track
        """
        # find track lengths of paths with multiple cutouts
        path_lengths = Track_Length_Helper.find_track_lengths_path(self)
        # construct present_points from the cutouts in the path array, to focus on the cutouts that do not belong to
        # any path
        if self.path_arr:
            present_points = np.concatenate(self.path_arr)
        else:
            present_points = []
            self.path_orientations = np.zeros((len(self.satellite_centers), 2))
        cutout_no = len(self.satellite_centers)
        count = len(self.path_arr)
        # for each cutout,
        for ind in range(0, cutout_no):
            # if it is not included in any other path,
            if ind not in present_points:
                self.path_arr.append([ind])
                # estimate its length
                path_orientation, path_length = Track_Length_Helper.find_individual_track(self,
                                                                                          self.satellite_centers[ind],
                                                                                          self.all_thetas[ind])
                # add its orientation to path_orientations
                self.path_orientations[count] = path_orientation
                path_lengths.append(path_length)
                count += 1
        return self.path_arr, self.path_orientations, path_lengths

    def find_individual_track(self, center, thetas):
        """
        Estimate the length of tracks with a single cutout detected on them
        :param center: center coordinates of the cutout
        :param thetas: orientation range (min, max) of the track segment
        :return: path_orientation: corrected orientation, path_length: length of the track
        """
        # calculate the average of min and max orientations
        mean_theta = np.mean(thetas)
        # find the outermost coordinates on the track with accumulate = True (last parameter) because we want to tune
        # the orientation as we iterate along the path
        right_sat_coords, orientation1 = Track_Length_Helper.iterate_over_path(self, center.copy(), mean_theta, 1,
                                                                               center.copy(), True)
        left_sat_coords, orientation2 = Track_Length_Helper.iterate_over_path(self, center.copy(), mean_theta, -1,
                                                                              center.copy(), True)
        # calculate total length of track based on right_sat_coords and left_sat_coords
        path_length = Track_Length_Helper.total_track_length(self, right_sat_coords, left_sat_coords,
                                                             np.mean([orientation1, orientation2]))

        # update orientation as the mean of orientations obtained from iterating to the left and right
        path_orientation = [np.mean([orientation1, orientation2]), 0]

        return path_orientation, path_length

    def find_track_lengths_path(self):
        """
        Estimate the length of tracks with multiple cutouts
        :return: estimated lengths
        """
        path_lengths = []
        if self.path_arr:
            # for each path in path array,
            for ind, p in enumerate(self.path_arr):
                # find the coordinates of the outermost cutouts
                x_max, x_min = np.argmax(self.satellite_centers[p, 0]), np.argmax(-1*self.satellite_centers[p, 0])
                max_coords = self.satellite_centers[p[x_max]]
                min_coords = self.satellite_centers[p[x_min]]
                # find the average of coordinates in path
                means = [np.mean(self.satellite_centers[p, 0]), np.mean(self.satellite_centers[p, 1])]

                # calculate the distance of the average with the outermost cutouts
                r_right = (np.sqrt((max_coords[0]-means[0])**2+(max_coords[1]-means[1])**2))
                r_left = (np.sqrt((min_coords[0]-means[0])**2+(min_coords[1]-means[1])**2))

                # calculate the right and left outermost cutouts by going right and left by an amount r_right +
                # step_size and r_left + step_size and
                right_coords = [means[0] + r_right*np.cos(self.path_orientations[ind, 0]) + self.step_size, means[1] +
                                r_right*np.sin(self.path_orientations[ind, 0])]

                left_coords = [means[0] - r_left * np.cos(self.path_orientations[ind, 0]) + self.step_size, means[1] -
                               r_left * np.sin(self.path_orientations[ind, 0])]
                # find the outermost coordinates on the track
                right_sat_coords, _ = Track_Length_Helper.iterate_over_path(self, right_coords,
                                                                            self.path_orientations[ind, 0], 1,
                                                                            max_coords, False)
                left_sat_coords, _ = Track_Length_Helper.iterate_over_path(self, left_coords,
                                                                           self.path_orientations[ind, 0], -1,
                                                                           min_coords, False)
                # calculate total length of track based on right_sat_coords and left_sat_coords
                length = Track_Length_Helper.total_track_length(self, right_sat_coords, left_sat_coords,
                                                                self.path_orientations[ind, 0])

                path_lengths.append(length)

        return path_lengths

    def iterate_over_path(self, coords, orientation, right, last_coords_on_track, accumulate):
        """
        Starting from coords, take new cutouts in the direction of orientation until reached to an edge.
        :param coords: starting coordinates
        :param orientation: orientation of the track
        :param right: true if going to the right, false otherwise
        :param last_coords_on_track: coordinates of the outermost cutout on the track (right or left)
        :param accumulate: if true, orientation will be tuned during iteration. If false, orientation is not adjusted.
        :return: last_satellite_coords: outermost coordinates on the track, orientation: orientation of the track
        """
        tolerance1 = (np.pi/180)*5
        tolerance2 = (np.pi/180)*5
        last_satellite_coords = None
        count = 0

        # while an edge is not hit,
        while self.half_size < coords[0] < self.im_size_y - self.half_size and self.half_size < coords[1] < \
                self.im_size_x - self.half_size:
            x, y = coords
            x, y = int(x), int(y)
            # take a cutout
            cutout_data = pre_process_cutout(self.diff_img[y - self.half_size:y + self.half_size,
                                             x - self.half_size:x + self.half_size], self.half_half_size, self.bin_size)
            pred = self.model.predict(np.expand_dims(np.expand_dims(cutout_data, axis=-1), axis=0), batch_size=1)
            # if cutout contains a track,
            if round(pred[0][0]) == 1:
                # find its orientation
                _, _, theta_moment, M_x, M_y = find_ellipse_parameters(self.seg[y - self.half_size:y + self.half_size,
                                                                                x - self.half_size:x + self.half_size])
                if theta_moment is not None:
                    theta_moment = adjust_angle(-np.pi / 2 - theta_moment)
                    # if angle is close to the orientation of the track,
                    if np.abs(adjust_angle(theta_moment-orientation)) < \
                            (tolerance1+tolerance2*accumulate*(tolerance2 > 0)):
                        # update the last coordinates on track
                        last_satellite_coords = coords.copy()

                        # decrease tolerance gradually, with every new point on track (if accumulate is True)
                        if accumulate and count > 0:
                            orientation = (orientation*count+theta_moment)/(count+1)
                            tolerance2 -= np.pi/180*0.5
                        count += 1

            # calculate the new coordinates in the direction of orientation
            coords[0] += self.step_size * np.cos(orientation) * right
            coords[1] += self.step_size * np.sin(orientation) * right
        # if last coordinates on track has never been updated, then no point was found in that direction
        if last_satellite_coords is None:
            # then, take the coordinates of the center of the outermost cutout as the last coordinates on track
            last_satellite_coords = last_coords_on_track
        return last_satellite_coords, orientation

    def total_track_length(self, right_sat_coords, left_sat_coords, angle):
        """
        Estimate the track length
        :param right_sat_coords: center of the outermost cutout with a track, to the right
        :param left_sat_coords: center of the outermost cutout with a track, to the left
        :param angle: orientation of the track
        :return: estimated length
        """
        n = max(np.sin(angle), np.cos(angle))

        # find parameters of ellipses in the outermost cutouts
        length1, _, _, M_x1, M_y1 = find_ellipse_parameters(
            self.seg[int(right_sat_coords[1]) - self.half_half_size:int(right_sat_coords[1]) + self.half_half_size,
                     int(right_sat_coords[0]) - self.half_half_size:int(right_sat_coords[0]) + self.half_half_size])

        length2, _, _, M_x2, M_y2 = find_ellipse_parameters(
            self.seg[int(left_sat_coords[1]) - self.half_half_size:int(left_sat_coords[1]) + self.half_half_size,
                     int(left_sat_coords[0]) - self.half_half_size:int(left_sat_coords[0]) + self.half_half_size])

        length = 0
        # calculate the distance between centers
        l = np.sqrt((right_sat_coords[0] - left_sat_coords[0]) ** 2 + (right_sat_coords[1] - left_sat_coords[1]) ** 2)
        length += l
        # add the semi-major axes of ellipses in the outermost cutouts to the length
        if M_x1 is not None:
            length3 = np.sqrt((self.half_half_size - M_x1) ** 2 + (self.half_half_size - M_y1) ** 2)
            length += length1 - length3*(-1)**(M_y1 > self.half_half_size)
        else:
            length -= self.half_half_size/n
        if M_x2 is not None:
            length4 = np.sqrt((self.half_half_size - M_x2) ** 2 + (self.half_half_size - M_y2) ** 2)
            length += length2 - length4*(-1)**(M_y2 < self.half_half_size)
        else:
            length -= self.half_half_size/n

        return length
