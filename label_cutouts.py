L = 128  # cutout size
bin_size = 4
bin_size_2 = 2
N = 100  # threshold for removing cutouts with low total fluxes
half_size = int(L / 2)  # half of the cutout size
half_half_size = int(L / 4)
artefact_th = 100  # threshold for removing cutouts close to artifact regions
cluster_thresh = 80  # distance threshold for clustering
ellipse_cutout_size = 56  # cutout size for orientation estimation

pp_path, ref_path, ones_cutout_path, zeros_cutout_path, def_sex_path, ref_img, diff_img = None, None, None, None, \
                                                                                          None, None, None
x_list, y_list = [], []


def onclick(event):
    """
    Save the coordinates clicked on the difference image, if they are valid.
    :param event: mouse click event
    """
    x_data, y_data = event.xdata, event.ydata
    if x_data is not None and y_data is not None:
        x_list.append(x_data)
        y_list.append(y_data)
        print('xdata=%f, ydata=%f' % (x_data, y_data))


def on_key(event):
    """
    Take the key press event and 1) plot the reference image on a separate window if 'r' is pressed, quit if 'x' is
    pressed and close figures and save cutouts if 'd' is pressed.
    :param event: key press event
    """
    key = event.key
    if key == 'r':
        # plot the reference image
        plt.figure()
        plt.imshow(ref_img, interpolation='nearest', origin='lower', vmin=np.percentile(ref_img, 1.),
                 vmax=np.percentile(ref_img, 99.))
        plt.title('Reference Image')
        plt.show()
    elif key == 'x':
        # exit
        print('Exiting...')
        sys.exit()
    elif key == 'd':
        # save cutouts, close the plots
        print('Saving cutouts...')
        plt.close('all')


def initialize_paths(path):
    """
    Initialize directories based on the given path and the path of the script.
    :param path: given path
    """
    global pp_path, ref_path, ones_cutout_path, zeros_cutout_path, def_sex_path
    script_path = os.path.dirname(os.path.realpath(__file__))
    pp_path = path + 'pre_processed_fits/'
    ref_path = path + 'blurred_ref_fits/'
    ones_cutout_path = path + 'cutouts/ones/'
    zeros_cutout_path = path + 'cutouts/zeros/'
    def_sex_path = script_path + '/diff_img_sextractor_files/'


def show_plots(cutout_coord_x, cutout_coord_y, ones_list, no_cutouts):
    """
    Show plots of cutouts.
    :param cutout_coord_x: x coordinates of cutout centers
    :param cutout_coord_y: y coordinates of cutout centers
    :param ones_list: array of 0s and 1s. ones_list[i] = 1 if the ith cutout contains a track, 0 otherwise.
    :param no_cutouts: total number of cutouts.
    """
    plt.figure(1)
    plt.figure(2)
    dim1 = int(math.ceil(np.sqrt(np.sum(ones_list))))
    dim2 = int(math.ceil(np.sqrt(no_cutouts-np.sum(ones_list))))
    one_count = 0
    zero_count = 0
    for ind in range(0, no_cutouts):
        x, y = cutout_coord_x[ind], cutout_coord_y[ind]
        if ones_list[ind] == 1:
            plt.figure(1)
            ax = plt.subplot(dim1, dim1, one_count + 1)
            one_count += 1
        else:
            plt.figure(2)
            ax = plt.subplot(dim2, dim2, zero_count + 1)
            zero_count += 1
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        imgg = diff_img[y - half_size:y + half_size, x - half_size:x + half_size]
        plt.imshow(imgg, interpolation='nearest', origin='lower', vmin=np.percentile(imgg, 1.),
                   vmax=np.percentile(imgg, 99.))
    plt.figure(1)
    plt.suptitle('Cutouts With Tracks')
    plt.figure(2)
    plt.suptitle('Cutouts Without Tracks')

    plt.show()


def create_dirs(path):
    """
    Create the directories where the cutouts will be saved. Cutout with and without satellites are saved to
    path/cutouts/ones/ and path/cutouts/zeros/ respectively.
    :param path: given path
    """
    if not os.path.exists(path+'cutouts/'):
        os.mkdir(path+'cutouts/')
    if not os.path.exists(ones_cutout_path):
        os.mkdir(ones_cutout_path)
        print('Directory to save cutouts with tracks created.')
    if not os.path.exists(zeros_cutout_path):
        os.mkdir(zeros_cutout_path)
        print('Directory to save cutouts without tracks created.')


def main(diff_image, ref_image, path, no_plots):
    """
    Find the cutouts in the image, let the user choose the ones with a track, classify and save cutouts based on the
    selections of the user.
    :param diff_image: path to difference image
    :param ref_image: path to reference image
    :param path: directory where catalogs, segmentation maps and directories to save cutouts will be created.
    :param no_plots: If True, no plots of cutouts will be shown.
    """
    global ref_img, diff_img

    # first, create the directory if it does not exist
    if not os.path.exists(path):
        os.mkdir(path)

    # initialize paths for processed difference images, processed reference images etc.
    initialize_paths(path)

    # pre-process difference and reference images, if not done already
    diff_filename = pre_process_diff(diff_image, ref_image, pp_path)
    ref_filename = pre_process_ref(ref_image, ref_path)

    ref_img = fits.open(ref_image)[1].data

    # sextract difference and reference images
    os.system('sex ' + pp_path + diff_filename + '[0] -CHECKIMAGE_TYPE OBJECTS -CHECKIMAGE_NAME ' + path +
              'diff_seg_map.fits' + ' -CATALOG_NAME ' + path + 'diff_catalog.cat -c ' + def_sex_path +
              'default.sex -PARAMETERS_NAME ' + def_sex_path + 'default.param')

    os.system('sex ' + ref_path + ref_filename + '[0] -CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME ' + path +
              'ref_seg_map.fits' + ' -CATALOG_NAME ' + path + 'ref_catalog.cat -c ' + def_sex_path +
              'default.sex -PARAMETERS_NAME ' + def_sex_path + 'default.param')

    # find the cutouts in the difference image
    cutout_coord_x, cutout_coord_y, index_list, clusters, data, diff_img, objects, im_size_x, im_size_y = \
        find_cutouts(diff_filename, path, pp_path, cluster_thresh, half_size, N, artefact_th)

    # plot the cutouts
    fig = plt.figure()
    plt.imshow(diff_img, interpolation='nearest', origin='lower', vmin=np.percentile(diff_img, 1.),
               vmax=np.percentile(diff_img, 99.))
    plt.title('Found Cutouts')
    ax = plt.gca()
    for m in range(0, len(cutout_coord_x)):
        x, y = cutout_coord_x[m], cutout_coord_y[m]
        ax.add_patch(matplotlib.patches.Rectangle((x - half_size, y - half_size), L, L,
                                                  alpha=1, edgecolor='k', linewidth=2,
                                                  fill=False))
    # associate button and key press events with the figure for labeling
    _ = fig.canvas.mpl_connect('button_press_event', onclick)
    _ = fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    len_x = len(x_list)
    no_cutouts = len(cutout_coord_x)
    ones_list = np.zeros(no_cutouts)

    # remove old cutouts of the image, to prevent saving multiple times
    create_dirs(path)
    for filename in glob.glob(ones_cutout_path + diff_filename+'*'):
        os.remove(filename)
    for filename in glob.glob(zeros_cutout_path + diff_filename+'*'):
        os.remove(filename)

    # match the points clicked by the user to the cutouts, save them to ones_cutout_path
    for m in range(0, len_x):
        ind, dist, x, y = find_nearest(cutout_coord_x, cutout_coord_y, x_list[m], y_list[m])
        if dist < half_size:
            ones_list[ind] = 1
            data = diff_img[y - half_size:y + half_size, x - half_size:x + half_size]
            np.save(ones_cutout_path + diff_filename + '_' + str(m) + '.npy', data)

    # save the rest (cutouts without tracks) to zeros_cutout_path
    for m in range(0, len(cutout_coord_x)):
        if ones_list[m] == 0:
            x, y = cutout_coord_x[m], cutout_coord_y[m]
            data = diff_img[y - half_size:y + half_size, x - half_size:x + half_size]
            np.save(zeros_cutout_path + diff_filename + '_' + str(m + len_x) + '.npy', data)

    # draw plots of cutouts if no_plots = False
    if not no_plots:
        show_plots(cutout_coord_x, cutout_coord_y, ones_list, no_cutouts)


if __name__ == '__main__':
    import sys
    import numpy as np
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import os
    import scipy.cluster.hierarchy as hcluster
    import matplotlib.patches
    from utils import *
    from cutout_finder_methods import *
    import math
    import glob
    import argparse

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

    main(diff_image=args.diff_image, ref_image=args.ref_image, path=fix_path(args.path), no_plots=args.no_plots)
