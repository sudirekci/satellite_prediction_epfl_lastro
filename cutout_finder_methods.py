import numpy as np
from utils import clip
from astropy.io import fits
import scipy.cluster.hierarchy as hcluster


def open_catalog(path, start, diff=True):
    """
    Open catalog of the difference or reference image, created by sextractor.
    :param start: the line number where catalog data starts
    :param path: path to catalog (except filename)
    :param diff: true if the catalog belongs to the difference image, false otherwise
    :return: catalog elements as list
    """
    if diff:
        type = 'diff'
    else:
        type = 'ref'

    c = open(path + type + '_catalog.cat', 'r+')
    c = c.readlines()
    c = [x.strip() for x in c]
    return [x.split() for x in c[start:len(c)]]


def load_catalog_data(catalog):
    """
    Construct arrays from catalog (list).
    :param catalog: list of elements in catalog (file)
    :return: data: fluxes and coordinates of detections, size (n, 3), cxx, cyy, cxy: cxx, cyy, cxy parameters of the
    detections taken from sextractor, size (n,) where n is the number of detections.
    """
    data = np.transpose(np.asarray([[x[0] for x in catalog], [x[1] for x in catalog], [x[2] for x in catalog]])).\
        astype(np.float)
    cxx = np.asarray([x[3] for x in catalog]).astype(np.float)
    cyy = np.asarray([x[4] for x in catalog]).astype(np.float)
    cxy = np.asarray([x[5] for x in catalog]).astype(np.float)
    return data, cxx, cyy, cxy


def remove_ref_points(data, data2, cxx, cyy, cxy):
    """
    Remove detections in the catalog of difference image that are too close to the detections in the catalog of
    reference image by using ellipse parameters of the detections obtained from sextractor.
    :param data: detections in the difference image
    :param data2: detections in the reference image
    :param cxx: ellipse parameter
    :param cyy: ellipse parameter
    :param cxy: ellipse parameter
    :return: detections in the difference image, cleaned
    """
    len_before = len(data)
    index_list = np.zeros(len_before)
    for k in range(0, len(data2)):
        x2, y2 = data2[k, 1:3]
        for l in range(0, len(data)):
            x1, y1 = data[l, 1:3]
            # if (x1, y1), point from the detections in the difference image is in the aperture of a point in the
            # reference image, remember to remove it by setting index_list[l] = 1
            if index_list[l] == 0 and cxx[k] * pow(x2 - x1, 2) + cyy[k] * pow(y2 - y1, 2) + cxy[k] * (x1 - x2) * (
                    y1 - y2) < 15:
                index_list[l] = 1

    data = data[index_list < 1]
    len_after = len(data)
    print('Points present in the reference image are removed. Length of catalog decreased from '
          + str(len_before) + ' to ' + str(len_after))
    return data


def estimate_noise(diff_img):
    """
    Estimate background noise in the given image.
    :param diff_img: given image
    :return: background noise
    """
    arr = diff_img[~np.isnan(diff_img)]
    arr = arr[np.nonzero(arr)]
    _, noise_level = clip(arr, nsigma=5)
    return noise_level


def find_cutouts(diff_filename, path, pre_processed_path, cluster_thresh, half_size, N, artefact_th):
    """
    Find the cutouts in the difference image from its catalog.
    :param diff_filename: filename of the processed difference image
    :param path: directory of segmentation maps and catalogs
    :param pre_processed_path: directory of pre-processed difference images
    :param cluster_thresh: distance threshold for clustering
    :param half_size: half of the cutout size
    :param N: threshold for removing cutouts with low total fluxes
    :param artefact_th: threshold for removing cutouts close to artifact regions
    :return: cutout_coord_x, cutout_coord_y: x and y coordinates of the cutouts generated from the centers of clusters,
     index_list: indices of clusters which fit the criterion described below, clusters: cluster list, data: coordinate
     list of detected objects in difference image, diff_img: difference image (as 2D array), objects: segmentation map
     of the difference image, im_size_x, im_size_y: sizes of the difference image for x and y axis.
    """
    diff_cat = open_catalog(path, 8, diff=True)
    ref_cat = open_catalog(path, 8, diff=False)

    diff_img = fits.open(pre_processed_path + diff_filename)[0].data
    objects = fits.open(path + 'diff_seg_map.fits')[0].data
    im_size_x, im_size_y = diff_img.shape
    # load catalogs of difference and reference image
    data, _, _, _ = load_catalog_data(diff_cat)
    data2, cxx, cyy, cxy = load_catalog_data(ref_cat)
    # remove detections in the catalog of difference image that are too close to the detections in the catalog of
    # reference image.
    data = remove_ref_points(data, data2, cxx, cyy, cxy)
    # cluster the detections based on distance.
    clusters = hcluster.fclusterdata(data[:, 1:3], cluster_thresh, criterion="distance")
    classes = np.max(clusters)
    noise_level = estimate_noise(diff_img)

    cutout_coord_x, cutout_coord_y, index_list = [], [], []

    # for each cluster,
    for m in range(0, classes):
        # find the center by a weighted average, weights are the fluxes of the detections
        inds = np.where(clusters == (m + 1))[0]
        x_coord, y_coord = np.average(data[inds, 1:3], axis=0, weights=data[inds, 0])
        x_coord, y_coord = int(x_coord), int(y_coord)
        # calculate the total flux in the cluster
        total_flux = np.sum(data[inds, 0])
        # if the center is not too close to the edges and total flux is greater than N*background noise
        if half_size < x_coord < im_size_x - half_size and half_size < y_coord < im_size_y - half_size and \
                total_flux > N * noise_level:
            arr = diff_img[
                  max(y_coord - (artefact_th + half_size), 20):min(y_coord + artefact_th + half_size, im_size_y - 20),
                  max(x_coord - (artefact_th + half_size), 20):min(x_coord + artefact_th + half_size, im_size_x - 20)]
            # if the center is not too close to an empty band (artifacts)
            if arr.any(axis=0).all() and arr.any(axis=1).all():
                # append the cluster to the list
                cutout_coord_x.append(x_coord), cutout_coord_y.append(y_coord)
                index_list.append(m+1)

    print('Cutout coordinates extracted. Total # of cutouts = ' + str(len(cutout_coord_y)))
    return cutout_coord_x, cutout_coord_y, index_list, clusters, data, diff_img, objects, im_size_x, im_size_y
