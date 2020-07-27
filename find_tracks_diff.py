path, pre_processed_path, blurred_ref_path, ellipse_path = None, None, None, None
cluster_thresh = 80  # distance threshold for clustering
L = 128  # cutout size
N = 100  # threshold for removing cutouts with low total fluxes
ellipse_cutout_size = 56  # cutout size for orientation estimation
half_size = int(L/2)  # half of the cutout size
half_half_size = int(L/4)
artefact_th = 100  # threshold for removing cutouts close to artifact regions
bin_size_2 = 2
bin_size = 4
im_size = int(L/bin_size)  # binned cutout size
im_size_x, im_size_y = None, None

model, data, diff_img, objects = None, None, None, None


def load_model():
    """
    Load the neural network.
    :return: neural network model
    """
    global model
    model = tf.keras.models.load_model('model_0.967.h5', compile=False)


def predict_cutouts(cutout_coord_x, cutout_coord_y, index_list):
    """
    Feed each cutout to the neural network, get the results, discard the data of the cutouts that do not contain tracks.
    :param cutout_coord_x: x coordinates of the cutouts
    :param cutout_coord_y: y coordinates of the cutouts
    :param index_list: clusters
    :return: predictions: output of the network to cutouts with tracks, ones_list: indices of clusters with tracks
    """
    global model
    predictions = []
    ones_list = []
    # for each cutout,
    for m in range(0, len(cutout_coord_x)):
        x_coord = cutout_coord_x[m]
        y_coord = cutout_coord_y[m]
        img = ndimage.convolve(diff_img[y_coord-half_size:y_coord+half_size, x_coord-half_size:x_coord+half_size],
                               gaussian_kernel(5, 5))
        img = np.reshape(img, (im_size, bin_size, im_size, bin_size)).mean((-1, 1))
        img = img + np.abs(np.min(img))
        img = img / np.max(img)
        # feed it into the neural network
        pred = model.predict(np.expand_dims(np.expand_dims(img, axis=-1), axis=0), batch_size=1)
        predictions.append(pred[0][0])
        if round(pred[0][0]) == 1:
            # if it's output is greater than 0.5, append it to the list of indices of clusters with tracks
            ones_list.append(index_list[m])

    print('Found ' + str(len(ones_list)) + ' cutout(s) with satellites.')

    return predictions, ones_list


def find_satellite_centers(ones_list, clusters):
    """
    For the clusters that contain tracks, find the best point that represents the track by taking the point with the
    highest output from the neural network.
    :param ones_list: indices of clusters with tracks
    :param clusters: cluster list
    :return: centers of track segments
    """
    global data, diff_img
    one_cutout_no = len(ones_list)
    satellite_centers = np.zeros((one_cutout_no, 2))
    half = int(ellipse_cutout_size/2)
    # for each cluster with a track,
    for k, cls in enumerate(ones_list):
        inds = np.where(clusters == cls)[0]
        if len(inds) == 1:
            # if cluster contains 1 point only, assign it as the center
            satellite_centers[k] = data[inds, 1:3]
        else:
            preds = []
            # otherwise, for each point in the cluster
            for m, ind in enumerate(inds):
                x, y = int(data[ind, 1]), int(data[ind, 2])
                # if detection is not close to the edges
                if half_size < y < im_size_x - half_size and half_size < x < im_size_y - half_size:
                    # take a smaller cutout around the point
                    cutout_data = pre_process_cutout(
                        diff_img[y - half_half_size:y + half_half_size, x - half_half_size: x + half_half_size],
                        half_half_size, bin_size_2)
                    pred = model.predict(np.expand_dims(np.expand_dims(cutout_data, axis=-1), axis=0),
                                         batch_size=1)
                    # save the output of the neural network to the smaller cutout
                    preds.append(pred[0][0])
                else:
                    preds.append(0)
            # take the point with the highest output as the center
            satellite_centers[k] = data[inds[np.argmax(preds)], 1:3]
        x = int(round(satellite_centers[k, 0]))
        y = int(round(satellite_centers[k, 1]))
        # save cutout around the satellite center for ellipse parameter calculations
        if not os.path.exists(ellipse_path):
            os.mkdir(ellipse_path)
        np.save(ellipse_path + str(k) + '.npy', objects[int(y) - half:int(y) + half, int(x) - half:int(x) + half])

    print('Satellite centers found.')
    return satellite_centers


def hough_ellipse_orientation(im):
    """
    Find the orientation range of the track in the given cutout with Hough transform
    :param im: given cutout
    :return: minimum and maximum angles in rad
    """
    im = im / np.max(im)
    # filter cutout with Canny edge filter
    edges = canny(im, sigma=2.0)
    # find the best fitting ellipses with Hough transform
    result = hough_ellipse(edges, accuracy=20, threshold=10,
                           min_size=3, max_size=50)
    result.sort(order='accumulator')
    # take orientations of the first 10 best ellipses
    min_th, max_th = find_range(list(result[-10:]))
    # find range of angles
    min_th, max_th = adjust_angle(min_th), adjust_angle(max_th)
    return min_th, max_th


def find_orientations(satellite_centers):
    """
    Find orientations of the track segments.
    :param satellite_centers: center coordinates of the track segments
    :return: orientations
    """
    one_cutout_no = len(satellite_centers)
    all_thetas = np.zeros((one_cutout_no, 2))
    half = int(ellipse_cutout_size/2)

    for k in range(0, one_cutout_no):
        x, y = satellite_centers[k]
        im = objects[int(y) - half:int(y) + half, int(x) - half:int(x) + half]

        # find the minimum and maximum orientation from the ellipses found by hough transform
        min_th, max_th = hough_ellipse_orientation(im)
        # find the orientation from image moments
        _, _, theta_moment, _, _ = find_ellipse_parameters(im)
        theta_moment = adjust_angle(-np.pi / 2 - theta_moment)

        # if the orientation calculated from image moments is not too different from the min / max angles from hough
        # transform,
        if not np.abs(theta_moment - min_th) > np.pi / 6 or not np.abs(theta_moment - max_th) > np.pi / 6:
            # update minimum and maximum orientations as:
            min_th, max_th = min(min_th, max_th, theta_moment), max(min_th, max_th, theta_moment)

        all_thetas[k] = min_th, max_th

    print('Orientations calculated.')
    return all_thetas


def show_plots(cutout_coord_x, cutout_coord_y, all_thetas, satellite_centers, index_list, ones_list):
    """
    Show results with 2 plots.
    :param cutout_coord_x: x coordinates of the cutouts
    :param cutout_coord_y: y coordinates of the cutouts
    :param all_thetas: orientations of the track segments, min and max
    :param satellite_centers: center coordinates of the track segments
    :param index_list: indices of clusters that are not too close to the edges
    :param ones_list: indices of clusters in index_list that contain tracks
    """
    plt.figure(1)
    ax = plt.gca()
    plt.imshow(diff_img, interpolation='nearest', origin='lower', vmin=np.percentile(diff_img, 1.),
               vmax=np.percentile(diff_img, 99.))
    for m in range(0, len(cutout_coord_x)):
        x, y = cutout_coord_x[m], cutout_coord_y[m]
        if index_list[m] in ones_list:
            ax.add_patch(matplotlib.patches.Rectangle((x - half_size, y - half_size), L, L,
                                                      alpha=1, edgecolor='r', linewidth=2,
                                                      fill=False))
        else:
            ax.add_patch(matplotlib.patches.Rectangle((x - half_size, y - half_size), L, L,
                                                      alpha=1, edgecolor='k', linewidth=2,
                                                      fill=False))
    plt.title('Found Cutouts\nRed = with a track, Black = without a track')

    if ones_list:
        plt.figure(2)
        plt.imshow(diff_img, interpolation='nearest', origin='lower', vmin=np.percentile(diff_img, 1.),
                   vmax=np.percentile(diff_img, 99.))
        one_count = 0
        for m in range(0, len(cutout_coord_x)):
            if m in ones_list:
                min_th, max_th = all_thetas[one_count]
                x, y = satellite_centers[one_count]
                plt.plot([x - 70 * np.cos(min_th), x + 70 * np.cos(min_th)],
                         [y - 70 * np.sin(min_th), y + 70 * np.sin(min_th)], linewidth=2)

                plt.plot([x - 70 * np.cos(max_th), x + 70 * np.cos(max_th)],
                         [y - 70 * np.sin(max_th), y + 70 * np.sin(max_th)], linewidth=2)
                plt.text(x - 25, y - 25, str(one_count), fontsize=18, fontweight='bold',
                         color='w')
                one_count += 1
        plt.title('Orientations of Cutouts With Tracks')

    plt.show()


def save_results(path_arr, path_orientations, path_lengths, satellite_centers, diff_filename):
    """
    Save the results as numpy arrays and a txt document that shows them in a table.
    :param path_arr: path array
    :param path_orientations: orientations of tracks
    :param path_lengths: lengths of tracks
    :param satellite_centers: center coordinates of the track segments
    :param diff_filename: filename of the difference image
    """
    global path
    diff_fn = '_'.join(diff_filename.split('_')[:-1])
    np.save(path + diff_fn + '_path_array.npy', path_arr)
    np.save(path + diff_fn + '_path_orientations.npy', path_orientations)
    np.save(path + diff_fn + '_path_lengths.npy', path_lengths)
    np.save(path + diff_fn + '_satellite_centers.npy', satellite_centers)
    file = open(path + diff_fn+'_results.txt', 'w+')
    file.write("{:^30} | {:^30} | {:^30} | {:^30} |\n".format('Track Indices', 'Orientation of Track (deg)',
                                                            'Std of Orientation (deg)', 'Track Length'))
    file.write('-' * 131 + '\n')
    for ind, path in enumerate(path_arr):
        file.write("{:^30} | {:^30.1f} | ".format(str(path), path_orientations[ind, 0]*180/np.pi))
        if len(path) == 1:
            file.write("{:^30} | ".format("-"))
        else:
            file.write("{:^30.1f} | ".format(path_orientations[ind, 1]*180/np.pi))
        file.write("{:^30.1f} |\n".format(path_lengths[ind]))
        file.write('-' * 131 + '\n')
    file.write('\n\n')

    for ind, center in enumerate(satellite_centers):
        file.write(str(ind) + ': ' + str(center) + '\n')
    file.write('\n\nNOTE: Coordinates are given as (x, y). To find the point in the array, use array[y, x].\n')


def main(diff_image, ref_image, meta_path, no_plots):

    script_path = os.path.dirname(os.path.realpath(__file__))

    global path, pre_processed_path, blurred_ref_path, ellipse_path, data, diff_img, objects, im_size_x, im_size_y

    path = meta_path

    # first, create the directory if it does not exist
    if not os.path.exists(path):
        os.mkdir(path)

    pre_processed_path = path + 'pre_processed_fits/'  # path to save pre processed difference images
    blurred_ref_path = path + 'blurred_ref_fits/'  # path to save blurred reference images
    ellipse_path = path + 'track_cutouts/'  # path to save cutouts

    # load neural network
    load_model()

    # pre-process difference and reference images
    diff_filename = pre_process_diff(diff_image, ref_image, pre_processed_path)
    ref_filename = pre_process_ref(ref_image, blurred_ref_path)

    # sextract processed difference image & blurred reference image
    os.system('sex ' + pre_processed_path + diff_filename + '[0] -CHECKIMAGE_TYPE OBJECTS -CHECKIMAGE_NAME '
              + path + 'diff_seg_map.fits' + ' -CATALOG_NAME ' + path + 'diff_catalog.cat -c ' + script_path +
              '/diff_img_sextractor_files/default.sex -PARAMETERS_NAME ' + script_path +
              '/diff_img_sextractor_files/default.param')

    os.system('sex ' + blurred_ref_path + ref_filename + '[0] -CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME '
              + path + 'ref_seg_map.fits' + ' -CATALOG_NAME ' + path + 'ref_catalog.cat -c ' + script_path +
              '/diff_img_sextractor_files/default.sex -PARAMETERS_NAME ' + script_path +
              '/diff_img_sextractor_files/default.param')

    # find the cutouts in the image
    cutout_coord_x, cutout_coord_y, index_list, clusters, data, diff_img, objects, im_size_x, im_size_y = \
        find_cutouts(diff_filename, path, pre_processed_path, cluster_thresh, half_size, N, artefact_th)

    # feed the cutouts to the neural network to see if they contain tracks
    predictions, ones_list = predict_cutouts(cutout_coord_x, cutout_coord_y, index_list)

    if ones_list:
        # find center points of the detected track segments in cutouts
        satellite_centers = find_satellite_centers(ones_list, clusters)
        # find orientations of the track segments
        all_thetas = find_orientations(satellite_centers)

        # find the connected cutouts
        cutout_connecter = Cutout_Connecter(satellite_centers, all_thetas)
        path_arr, path_orientations = cutout_connecter.connect_cutouts()

        print('Estimating track lengths...')
        track_length_helper = Track_Length_Helper(diff_img, path_arr, path_orientations, satellite_centers, all_thetas,
                                                  5, [half_size, bin_size, im_size_x, im_size_y], model, objects,
                                                  ellipse_path)

        # find track lengths
        path_arr, path_orientations, path_lengths = track_length_helper.find_track_lengths()
        # save the final track orientations, lengths and coordinates as both numpy arrays and in a txt file
        save_results(path_arr, path_orientations, path_lengths, satellite_centers, diff_filename)
    else:
        all_thetas, satellite_centers = None, None

    # show plots of cutouts that contain track segments and their orientations, if draw_plots is True
    if not no_plots:
        print('Plotting the results...')
        show_plots(cutout_coord_x, cutout_coord_y, all_thetas, satellite_centers, index_list, ones_list)


if __name__ == '__main__':
    import argparse
    import numpy as np
    from astropy.io import fits
    import os
    import scipy.cluster.hierarchy as hcluster
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import matplotlib
    from utils import *
    from skimage.feature import canny
    from skimage.transform import hough_ellipse
    from track_length_helper import Track_Length_Helper
    from cutout_connecter import *
    from cutout_finder_methods import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--diff_image', metavar='path', required=True,
                        help='path to difference image')
    parser.add_argument('--ref_image', metavar='path', required=True,
                        help='path to reference image')
    parser.add_argument('--path', metavar='path', required=True,
                        help='path where segmentation map and catalog meta-data and results will be at')
    parser.add_argument('--no_plots', required=False, default=False, action='store_true',
                        help='Add if no plots should be drawn')
    args = parser.parse_args()

    main(diff_image=args.diff_image, ref_image=args.ref_image, meta_path=fix_path(args.path), no_plots=args.no_plots)

"""
python3 find_tracks_diff.py --diff_image /home/su/Desktop/lastro/OMEGACAM_2020-01-29_SDSSJ0924+0219_mosaic_J092656+021826_hsize_864_diffimg.fits --ref_image /home/su/Desktop/lastro/J092656+021826_hsize_864_ref.fits --path /home/su/Desktop/lastro/predict_sat_test/
"""
