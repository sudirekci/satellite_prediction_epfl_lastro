import numpy as np

N = 100  # threshold for removing cutouts with low total fluxes
stationary_th = 6  # threshold for removing stationary points
half_size = 80  # threshold for removing detections that are too close to borders or empty sections
slow_sat_th = 10  # distance tolerance for slow satellites
slope_error_th = 3 * np.pi / 180  # angle tolerance for slow satellites
colors = ['k', 'b', 'r', 'y']
hsize_cutout = 1000  # distance parameter
half_hsize_cutout = int(hsize_cutout/2)
x_step = 5  # number of steps in x direction while finding detections and removing cosmic rays
y_step = 5  # number of steps in y direction while finding detections and removing cosmic rays
fast_arr_points = 16  # number of points a track needs to have to be counted as a fast track
fast_sat_distance = 5  # distance tolerance for fast satellites

filename1 = 'segmentation_map.fits'
filename2 = 'segmentation_catalog.cat'

fields, all_data, all_ellipse_params = [], [], []
no_epochs, slow_sats_final, all_fast_sats, def_sex_path_2, def_sex_path_1, time_diffs = None, None, None, None, None, \
                                                                                        None


def open_catalog(path):
    """
    Open the catalog of detections in the given directory
    :param path: given directory
    :return: data: fluxes and coordinates of detections, ellipse_params: lengths and uncertainities of semi-major axes
    of the detections, ellipticities
    """
    catalog = open(path + filename2, 'r+')
    catalog = catalog.readlines()
    catalog = [x.strip() for x in catalog]
    catalog = [x.split() for x in catalog[11:len(catalog)]]
    data = np.transpose(np.asarray([[x[0] for x in catalog], [x[1] for x in catalog], [x[2] for x in catalog]])) \
        .astype(float)
    ellipse_params = np.transpose(np.asarray([[x[6] for x in catalog], [x[7] for x in catalog],
                                              [x[10] for x in catalog]])).astype(float)

    return data, ellipse_params


def find_closest_pt(arr, coords, th):
    """
    Find the closest point in the given array to given coordinates. Return it if the distance is smaller than the given
    threshold.
    :param arr: given array
    :param coords: given coordinates
    :param th: given threshold
    :return: the index of the closest point and the distance, if the distance is smaller than th. -1, -1 otherwise.
    """
    dist_arr = np.sum(np.square(arr - coords), axis=1)
    ind = np.argmin(dist_arr)
    if th == -1 or dist_arr[ind] < th ** 2:
        return ind, np.sqrt(dist_arr[ind])
    return -1, -1


def remove_stationary_pts():
    """
    Remove stationary points in the catalogs (detected objects) of the single epoch tiles
    """
    global all_data, all_ellipse_params
    new_all_arr = [[] for _ in range(0, no_epochs)]
    new_ellipse_params = [[] for _ in range(0, no_epochs)]
    do_not_add_list = [[] for _ in range(0, no_epochs)]
    for ep in range(0, no_epochs):
        # for each point in all_arr[ep],
        for m in range(0, len(all_data[ep])):
            pt = all_data[ep][m]
            found = False
            # iterate over the points in other epochs
            for epp in range(ep + 1, no_epochs):
                # find the closest point to the initial point
                closest, _ = find_closest_pt(all_data[epp][:, 1:3], pt[1:3], stationary_th)
                # if the flux is not too different and it is closer than stationary_th, add them to do_not_add_list.
                # Do not eliminate them immediately to check matches between other epochs as well.
                if closest != -1 and np.abs(all_data[epp][closest, 0] - pt[0]) / pt[0] < 0.8:
                    found = True
                    do_not_add_list[epp].append(closest)
            if not found and m not in do_not_add_list[ep]:
                new_all_arr[ep].append(pt)
                new_ellipse_params[ep].append(all_ellipse_params[ep][m])

    # array operations to get them into proper shape ([array(), array() ...] where array()s are numpy arrays with size
    # (m, 3) where m is the number of detections.
    new_all_arr = [np.concatenate(new_all_arr[ep]) for ep in range(0, no_epochs)]
    new_ellipse_params = [np.concatenate(new_ellipse_params[ep]) for ep in range(0, no_epochs)]

    all_data = [new_all_arr[ep].reshape(int(len(new_all_arr[ep]) / 3), 3) for ep in range(0, no_epochs)]
    all_ellipse_params = [new_ellipse_params[ep].reshape(int(len(new_ellipse_params[ep]) / 3), 3) for ep in
                          range(0, no_epochs)]


def find_slow_sats(im_size_x, im_size_y, reverse=False):
    """
    Find the slow satellites across epochs.
    :param im_size_x: length of the x axis of the full field image.
    :param im_size_y: length of the y axis of the full field image.
    :param reverse: the time direction of the search. If False, the search starts from epoch 1 and continues with
    epoch 2 (for example). If True, the opposite happens.
    :return: a list of 2D arrays. Each array is in the form [[m, n], [m + 1, k], [t, closest]], where m, m+1 and t are
    the epochs of the 1st, 2nd and 3rd point; n, k, closest are the indices of the points in all_data.
    """
    possible_sats = []
    construct_time_array()
    for m in range(0, no_epochs - 2):
        # take the detections of the 2 consecutive epochs
        it_array = np.array(np.meshgrid(np.arange(0, len(all_data[m])), np.arange(0, len(all_data[m + 1])))).\
            T.reshape(-1, 2)
        for it in it_array:
            n, k = it
            # for each point in 2 epochs
            point1, point2 = all_data[m][n, 1:3], all_data[m + 1][k, 1:3]
            # limit the distance between the points to eliminate unnecessary calculation
            if -1 * half_hsize_cutout < point1[0] - point2[0] < half_hsize_cutout and -1 * half_hsize_cutout < \
                    point1[1] - point2[1] < half_hsize_cutout:
                for t in range(m + 2, no_epochs):
                    # find the next point based on the 2 points
                    _, dr, pt = find_next_pt(t - (m + 1), point1, point2, [m, m + 1, t])
                    # find the closest point to the calculated next point
                    closest, dist = find_closest_pt(all_data[t][:, 1:3], pt, 1000)
                    if closest == -1:
                        break
                    point3 = all_data[t][closest, 1:3]
                    # calculate the difference between inclination angles of the 3 points
                    slope_err, _ = slope_error(point1, point2, point3)
                    # if there exists a close point to the calculated next point, the calculated next point is
                    # inside the tile, slope error and the distance between the calculated next point and point3
                    # is not too large
                    if 0 < pt[0] < im_size_y and 0 < pt[1] < im_size_x and slope_err < slope_error_th and \
                            dist < 2 * (all_ellipse_params[t][closest, 0]) + slow_sat_th:
                        # add it to slow satellite list (reverse = True if the search is backwards in time)
                        if reverse:
                            possible_sats.append(
                                [[no_epochs - t - 1, closest], [no_epochs - m - 2, k], [no_epochs - m - 1, n]])
                        else:
                            possible_sats.append([[m, n], [m + 1, k], [t, closest]])
    return possible_sats


def find_next_pt(rank, point1, point2, inds):
    """
    Given 2 points, find the "rank"th next point in their direction.
    :param rank: rank of the point whose coordinates are calculated
    :param point1: first point
    :param point2: second point
    :param inds: indices of the given points in all_data
    :return: v: velocity calculated from the distance and time elapsed between point1 & point2, dr: distance between
    point1 & point2, next_pt: coordinates of the "rank"th next point
    """
    dt = time_diffs[inds[0], inds[1]]
    dr = euclidean(point1, point2)
    v = dr / dt
    delta_t = time_diffs[inds[1], inds[2]]
    next_pt = [point2[0] + rank * v * delta_t * (point2[0] - point1[0]) / dr,
               point2[1] + rank * v * delta_t * (point2[1] - point1[1]) / dr]
    return v, dr, next_pt


def construct_time_array():
    """
    Construct the time_diffs array, where time_diffs[i, j] is the time interval between the ith and jth epoch in
    seconds. If j > i, time_diffs[i, j] > 0. time_diffs[i, j] = - time_diffs[j, i] for easy use in reverse direction
    """
    global time_diffs
    time_diffs = np.zeros((no_epochs, no_epochs))
    for m in range(0, no_epochs):
        t11 = time.Time(str(fields[m].header['STARTMJD']), format='mjd')
        t12 = time.Time(str(fields[m].header['ENDMJD']), format='mjd')
        t1 = t11+(t12-t11)/2
        for n in range(m+1, no_epochs):
            t21 = time.Time(str(fields[n].header['STARTMJD']), format='mjd')
            t22 = time.Time(str(fields[n].header['ENDMJD']), format='mjd')
            t2 = t21 + (t22 - t21) / 2
            time_delta = (t2-t1).to_value('sec')
            time_diffs[m, n] = time_delta
            time_diffs[n, m] = -1*time_delta


def slope_error(point1, point2, point3):
    """
    Find the difference between the inclination angles of the combinations point2-point3 and point2-point1
    :param point1: point
    :param point2: point
    :param point3: point
    :return: absolute value of the inclination angle difference, inclination angles
    """
    x1, x2 = (point3[1] - point2[1]) / (point3[0] - point2[0]), (point2[1] - point1[1]) / (point2[0] - point1[0])
    diff = adjust_angle(np.arctan(x1) - np.arctan(x2))
    return np.abs(diff), (x1, x2)


def find_fast_sats_3(ep):
    """
    Find the fast tracks in the given epoch.
    :param ep: given epoch
    :return: found tracks
    """
    global all_data, all_ellipse_params

    length = len(all_data[ep])
    present_pts = []
    fast_sats = []
    # for each point, from the left side of the array (small xs)
    for m in range(0, length):
        # take another point from the right side (large x)
        for n in range(length - 1, m, -1):
            # find the line equation connecting the points
            a, b = get_line_params(ep, m, n)
            factor = np.cos(np.arctan(a))
            index_arr = [m, n]
            pts = all_data[ep][m + 1:n, 1:3]  # the points between the starting and ending points
            extras = pts[:, 1] - a * pts[:, 0] - b  # put them in the line equation
            # if the distance of the points are smaller than fast_sat_distance (threshold), they are matched
            matched_pts = np.where(np.abs(extras * factor) < fast_sat_distance)[0]
            matched_pts = matched_pts + m + 1
            matched_pts = np.setdiff1d(matched_pts, present_pts)
            index_arr = np.concatenate((index_arr, matched_pts))
            # if the number of points on the track is greater than fast_arr_points and the line is not too steep or
            # flat, add it to the fast track list. Lines with large or small slopes tend to belong to the points on the
            # edges of CCDs and they do not represent real tracks.
            if len(index_arr) > fast_arr_points and 100 > np.abs(a) > 0.01:
                fast_sats.append(index_arr)
                present_pts = np.concatenate((present_pts, index_arr))
    return fast_sats


def get_line_params(ep, ind1, ind2):
    """
    Find the parameters of the line passing through point1 and point2, where point1 = all_data[ep][ind1, 1:3] and
    point2 = all_data[ep][ind2, 1:3].
    :param ep: given epoch
    :param ind1: first index
    :param ind2: second index
    :return: m, c: line parameters when it's written as y = mx + c
    """
    pt1, pt2 = all_data[ep][ind1, 1:3], all_data[ep][ind2, 1:3]
    m = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    c = pt1[1]-m*pt1[0]
    return m, c


def plot_fast_sats():
    """
    Plot fast tracks in np_epochs figures.
    """
    for m in range(0, no_epochs):
        plt.figure()
        plt.plot(all_data[m][:, 1], all_data[m][:, 2], 'k*', markersize=4)

        for n in range(0, len(all_fast_sats[m])):
            indices = np.asarray(all_fast_sats[m][n])
            xs = all_data[m][indices, 1]
            min_ind, max_ind = np.argmin(xs), np.argmax(xs)
            p1, p2 = all_data[m][indices[min_ind], 1:3], all_data[m][indices[max_ind], 1:3]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]])
            # make the points on tracks colored
            plt.plot(all_data[m][indices, 1], all_data[m][indices, 2], '*', markersize=4)
        ek = 'th'
        if m == 0:
            ek = 'st'
        elif m == 1:
            ek = 'nd'
        elif m == 2:
            ek = 'rd'
        plt.title(str(m + 1) + ek + ' Epoch')


def plot_slow_sats():
    """
    Plot slow tracks
    """
    if len(slow_sats_final) == 0:
        print('No slow satellites were found.')
        return
    plt.figure()
    for m in range(0, no_epochs):
        plt.plot(all_data[m][:, 1], all_data[m][:, 2], colors[m] + '*', markersize=4)

    for n in range(0, len(slow_sats_final)):
        indices = slow_sats_final[n]
        p1, p2 = all_data[indices[0][0]][indices[0][1], 1:3], all_data[indices[-1][0]][indices[-1][1], 1:3]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=3)
    plt.title('Slow Satellites')


def find_detections(path, filenames, im_size_x, im_size_y):
    """
    Find and calibrate all the detections in the full field images. The calibrations are made by taking the first epoch
    as reference, so an astrometric calibration is needed if the first epoch is not calibrated.
    :param path: directory where full-field images are
    :param filenames: file names of full-field images
    :param im_size_x: size of the image in x axis
    :param im_size_y: size of the image in y axis
    """
    global all_data, all_ellipse_params

    step_size_x = im_size_x // x_step
    step_size_y = im_size_y // y_step
    print('Length of patch in x axis:', step_size_x)
    print('Length of patch in y axis', step_size_y)

    all_data = [[] for _ in range(0, no_epochs)]
    all_ellipse_params = [[] for _ in range(0, no_epochs)]
    nan_arr = None
    ref_stars = None

    for x_c in range(0, x_step):
        for y_c in range(0, y_step):
            x_end = (x_c + 1) * step_size_x
            y_end = (y_c + 1) * step_size_y
            if x_c == x_step - 1:
                x_end = im_size_x
            if y_c == y_step - 1:
                y_end = im_size_y
            for ep in range(0, no_epochs):
                print('Epoch:', str(ep))
                print('starts:', str(x_c * step_size_x), str(y_c * step_size_y))
                print('ends:', str(x_end), str(y_end))
                img = fits.open(path + filenames[ep])[1].data[x_c * step_size_x:x_end, y_c * step_size_y:y_end]
                # find the reference stars in the segment
                if ep == 0:
                    # find the places where img is nan
                    nan_arr = np.argwhere(np.isnan(img))
                    ref_stars = find_reference_stars(path, img)
                    best_bias = 0
                else:
                    stars = find_reference_stars(path, img)
                    best_bias = calibrate_with_stars(ref_stars[:, 1:3], stars[:, 1:3])

                print('Removing cosmic rays...')
                _, img = lacosmicx.lacosmicx(img.copy(order='C').astype(np.float32))

                median, noise_level = clip(img, nsigma=5)
                print('Median & Noise Level: ', median, noise_level)
                img -= median
                # filter the image segment
                img = ndimage.convolve(img, gaussian_kernel(5, 8))

                hdu = fits.PrimaryHDU(img)
                hdu.writeto(path + 'cutout_pp.fits', overwrite=True)

                os.system('sex ' + path + 'cutout_pp.fits' + '[0] -CHECKIMAGE_TYPE APERTURES -CHECKIMAGE_NAME '
                          + path + filename1 + ' -CATALOG_NAME ' + path + filename2 + ' -c ' + def_sex_path_2 +
                          'default.sex -PARAMETERS_NAME ' + def_sex_path_2 + 'default.param')
                # calibrate the new detections by taking the first epoch as reference
                new_detections, new_ellipse_params = open_catalog(path)
                new_detections[:, 1:3] += best_bias

                # print('len before', str(len(new_detections)))
                # remove detections close to artifact regions
                new_detections = remove_borders(nan_arr, new_detections)
                # print('len after', str(len(new_detections)))
                # add the coordinates of the origin of the image segment to the detections
                new_detections[:, 1:3] = new_detections[:, 1:3] + np.asarray([y_c * step_size_y, x_c * step_size_x])
                all_data[ep].append(new_detections)
                all_ellipse_params[ep].append(new_ellipse_params)

    all_data = [np.concatenate(all_data[ep]) for ep in range(0, no_epochs)]
    all_ellipse_params = [np.concatenate(all_ellipse_params[ep]) for ep in range(0, no_epochs)]

    os.remove(path + 'cutout_pp.fits')
    for k in range(0, no_epochs):
        os.remove(path+filenames[k])


def find_reference_stars(path, img):
    """
    Find the first 20 stars with the highest fluxes in the given image.
    :param path: directory of the given image
    :param img: given image
    :return: the fluxes, x and y coordinates of the first 20 stars.
    """

    hdu = fits.PrimaryHDU(img)
    hdu.writeto(path + 'cutout.fits', overwrite=True)

    os.system('sex ' + path + 'cutout.fits' + '[0] -CHECKIMAGE_TYPE APERTURES -CHECKIMAGE_NAME '
              + path + filename1 + ' -CATALOG_NAME ' + path + filename2 + ' -c ' + def_sex_path_1 +
              'default.sex -PARAMETERS_NAME ' + def_sex_path_1 + 'default.param')

    objects, params = open_catalog(path)
    # sort the detections according to their fluxes
    mask = np.argsort(objects[:, 0])
    objects = objects[mask]
    params = params[mask]
    # discard detections whose ellipticities are greater than 0.6 (not to include very bright tracks in reference
    # stars).
    mask2 = params[:, 2] < 0.6
    best_stars = objects[mask2][0:20]
    return best_stars[np.argsort(best_stars[:, 1])]


def calibrate_with_stars(ref_objects, new_objects):
    """
    Find the best shift parameters to calibrate new_objects to ref_objects. These arrays contain bright stars only.
    :param ref_objects: Reference coordinates
    :param new_objects: Detection coordinates
    :return: the best shift parameters
    """
    # take 3 stars from ref_ojects and new_objects
    ref_comb_list = list(combinations(ref_objects, 3))
    new_comb_list = list(combinations(new_objects, 3))

    min_error = 100000
    best_bias = 0
    it_array = np.array(np.meshgrid(np.arange(0, len(ref_comb_list)), np.arange(0, len(new_comb_list)))).\
        T.reshape(-1, 2)
    for it in it_array:
        ind1, ind2 = it
        error = np.sum(np.std(np.asarray(ref_comb_list[ind1])-np.asarray(new_comb_list[ind2]), axis=0))
        # find the bias with the minimum sum of standard deviations
        if error < min_error:
            min_error = error
            best_bias = np.mean(np.asarray(ref_comb_list[ind1])-np.asarray(new_comb_list[ind2]), axis=0)

    print('Minimum error:', min_error)
    print('Best bias:', best_bias)
    return best_bias


def remove_borders(nan_arr, new_detections):
    """
    Remove the points that are close to the empty sections of the image less than half_size.
    :param nan_arr: array denoting the coordinates that correspond to the nan regions
    :param new_detections: detections in the image
    :return: the detections that are not too close to the empty sections
    """
    if nan_arr.size == 0:
        return new_detections
    new_detections_cleaned = []
    for _, pt in enumerate(new_detections):
        arr = np.linalg.norm(nan_arr - pt[1:3], axis=1)
        if np.min(arr) > half_size:
            new_detections_cleaned.append(pt)

    return np.asarray(new_detections_cleaned)


def remove_edges(im_size_x, im_size_y):
    """
    Remove the detections close to the edges of the image.
    :param im_size_x: size of the full-field image in x axis
    :param im_size_y: size of the full-field image in y axis
    """
    global all_data, all_ellipse_params
    x_max, y_max = im_size_x - half_size, im_size_y - half_size
    for ep in range(0, no_epochs):
        mask = (all_data[ep][:, 1] > half_size) & (all_data[ep][:, 1] < y_max) & (all_data[ep][:, 2] > half_size) & \
               (all_data[ep][:, 2] < x_max)
        all_data[ep] = all_data[ep][mask]
        all_ellipse_params[ep] = all_ellipse_params[ep][mask]


def print_info():
    print('ARRAY SHAPES:')
    print('Slow track array has a length of n, where n is the number of slow tracks. slow_sats_final[i] has the shape '
          '[ep1, p1], [ep2, p2], [ep3, p3] where ep1, ep2, ep3 are the epochs of the points p1, p2, p3. Note that '
          'p1, p2, p3 are indices and to find the corresponding points, all_data[ep1][p1] (for the first one) '
          'should be used.')
    print('Fast track array has the same length as the number of epochs. For each epoch, (all_fast_sats[ep]), it has a '
          'length of n where n is the number of fast tracks detected in that epoch. all_fast_sats[ep][i] has length m, '
          'where m is the number of points on that track. Each element in all_fast_sats[ep][i] are indices. Again, the '
          'corresponding points can be found with all_data[ep][all_fast_sats[ep][i][j]].')


def main(field_name, path, obs_date, no_plots):

    global fields, all_data, all_ellipse_params, no_epochs, slow_sats_final, all_fast_sats, \
        def_sex_path_2, def_sex_path_1

    # first, create the directory if it does not exist
    if not os.path.exists(path):
        os.mkdir(path)

    script_path = os.path.dirname(os.path.realpath(__file__))
    def_sex_path_2 = script_path + '/single_ep_sextractor_files/'
    def_sex_path_1 = script_path + '/ref_sextractor_files/'

    filenames = create_mosaic(path, field_name, obs_date)
    no_epochs = len(filenames)

    im_size_x, im_size_y = fits.open(path+filenames[0])[1].data.shape

    for ind, file in enumerate(filenames):
        field = fits.open(path + file)[0]
        fields.append(field)

    fields = np.asarray(fields)
    print('Finding detections...')
    find_detections(path, filenames, im_size_x, im_size_y)

    print('Number of detections before removal of stationary points: ',
         len(all_data[0]), len(all_data[1]), len(all_data[2]), len(all_data[3]))

    print('Removing stationary points...')
    remove_stationary_pts()
    remove_edges(im_size_x, im_size_y)

    print('Number of detections after removal of stationary points: ',
          len(all_data[0]), len(all_data[1]), len(all_data[2]), len(all_data[3]))

    # sort the detection array
    for ep in range(0, no_epochs):
        mask = np.argsort(all_data[ep][:, 1])
        all_data[ep] = all_data[ep][mask]
        all_ellipse_params[ep] = all_ellipse_params[ep][mask]

    print('Saving detections and ellipse parameters...')
    np.save(path + 'all_data_moving.npy', all_data)
    np.save(path + 'all_ellipse_params_moving.npy', all_ellipse_params)

    if not no_plots:
        plt.figure()
        plt.title('After Removal of Stationary Points \n 1 = black, 2 = blue, 3 = red, 4 = yellow')
        for k in range(0, no_epochs):
            plt.plot(all_data[k][:, 1], all_data[k][:, 2], colors[k] + '*', markersize=4)

    print('Finding slow satellites...')

    slow_sats = find_slow_sats(im_size_x, im_size_y, reverse=False)
    print('Slow sats forward direction: ', slow_sats)
    all_data = all_data[::-1]
    slow_sats2 = find_slow_sats(im_size_x, im_size_y, reverse=True)
    all_data = all_data[::-1]
    print('Slow sats backward direction: ', slow_sats2)
    slow_sats = slow_sats + slow_sats2
    used = []
    # eliminate the detections present twice
    slow_sats_final = [x for x in slow_sats if x not in used and (used.append(x) or True)]
    print('Slow sats final:', slow_sats_final)
    print('# of 3 point sets = ', len(slow_sats_final))

    if not no_plots:
        plot_slow_sats()

    print('Finding fast satellites...')

    all_fast_sats = []
    for n in range(0, no_epochs):
        print('Epoch: ', n)
        fast_sats = find_fast_sats_3(n)
        print('Fast sats: ', fast_sats)
        print('# of fast sats ', len(fast_sats))
        all_fast_sats.append(fast_sats)

    if not no_plots:
        plot_fast_sats()

    print('Saving fast and slow satellite info...')
    print_info()
    np.save(path + field_name + '_' + obs_date + '_all_slow_sats.npy', slow_sats_final)
    np.save(path + field_name + '_' + obs_date + '_all_fast_sats.npy', all_fast_sats)

    if not no_plots:
        plt.show()


if __name__ == '__main__':
    import numpy as np
    import astropy.io.fits as fits
    import matplotlib.pyplot as plt
    import os
    import sys
    from utils import *
    from scipy import ndimage
    from scipy.spatial.distance import euclidean
    import astropy.time as time
    from itertools import combinations
    from create_se_mosaic import create_mosaic
    import lacosmicx
    from numpy import ones, vstack
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--field_name', metavar='path', required=True,
                        help='name of the field')
    parser.add_argument('--path', metavar='path', required=True,
                        help='path for creating & deleting necessary files')
    parser.add_argument('--obs_date', metavar='path', required=True,
                        help='observation date in the format yyyy-mm-dd')
    parser.add_argument('--no_plots', required=False, default=False, action='store_true',
                        help='Add if no plots should be drawn')
    args = parser.parse_args()
    main(field_name=args.field_name, path=fix_path(args.path), obs_date=args.obs_date,
         no_plots=args.no_plots)
