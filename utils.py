from scipy import ndimage
import numpy as np
import os
from astropy.io import fits


def gaussian_kernel(size, sigma=1):
    """
    Returns a 2D Gaussian kernel with dimensions (size, size)
    :param size: dimension of the kernel
    :param sigma: standard deviation of the kernel
    :return: Gaussian kernel
    """
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def circular_mask(size):
    """
    Returns a circular mask with elements 1 inside the circle, 0 outside.
    :param size: diameter of the circle
    :return: mask as a square numpy array
    """
    radius = int(size/2)
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1))
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask2 = x ** 2 + y ** 2 <= radius ** 2
    kernel[mask2] = 1
    return kernel


def create_mask(arr):
    """
    Creates a mask from the given array where mask[i] = 1 if mean(arr)-15*noise(arr) < arr[i] < mean(arr)+15*noise(arr),
    0 otherwise. Noise denotes the background noise.
    :param arr: given array.
    :return: mask
    """
    _, noise = clip(arr, 5)
    mean = np.mean(arr)
    min,  max = mean-15*noise, mean+15*noise
    mask = (arr > min) & (arr < max)
    return mask


def preprocess(img, ref):
    """
    Pre-process a difference image based on its reference image.
    :param img: difference image
    :param ref: reference image
    :return: pre-processed difference image
    """
    mask = create_mask(ref)
    ref = ndimage.maximum_filter(ref, footprint=circular_mask(13))
    ref = ndimage.convolve(ref, gaussian_kernel(13, sigma=3))
    ref[mask] = 0
    img = img / np.amax(img) * 1000
    ref = ref / np.amax(ref) * 1000
    img = ndimage.convolve(img, gaussian_kernel(5))
    img = img / (5 * ref + 0.5 * np.std(ref))
    img = ndimage.convolve(img, gaussian_kernel(5))
    img = img / np.std(img)
    return img


def pre_process_cutout(arr, im_dim, bin_dim):
    """
    Bin a given cutout with size (s, s) and scale it to [0, 1].
    :param arr: given cutout
    :param im_dim: final dimensions of the cutout.
    :param bin_dim: s/im_dim, integer
    :return: processed cutout
    """
    arr = np.reshape(arr, (im_dim, bin_dim, im_dim, bin_dim)).mean((-1, 1))
    arr = arr + np.abs(np.min(arr))
    arr = arr / np.max(arr)
    return arr


def clip(data, nsigma):
    """iteratively removes data until all is within nsigma of the median, then returns the median and std"""
    lennewdata = 0
    lenolddata = len(data)
    while lenolddata > lennewdata:
        lenolddata = data.size
        data = data[np.where((data < np.nanmedian(data)+nsigma*np.nanstd(data)) &
                             (data > np.nanmedian(data)-nsigma*np.nanstd(data)))]
        lennewdata = data.size
    return np.median(data), np.std(data)


def save_data(new_data, f):
    """Saves new_data to file f"""
    for x, y in new_data.items():
        f.write(x+' '+y+'\n')
    f.close()


def find_ellipse_parameters(data):
    """
    Finds ellipse parameters of the given image (edge filtered) from image moments.
    :param data: given image
    :return: length: semi-major axis length, width: semi-minor axis length,
    theta_moment: rotation angle (rad), M_x: mean along x axis, M_y: mean along y axis.
    """
    binary = np.zeros(data.shape)
    binary[data > 0] = 1
    inds = np.where(binary > 0)
    M00 = np.sum(np.sum(binary))
    if M00 == 0:
        return None, None, None, None, None
    M_x = np.sum(inds[0])/M00
    M_y = np.sum(inds[1])/M00
    var_x = np.sum(np.square(inds[0]))/M00 - M_x**2
    var_y = np.sum(np.square(inds[1])) / M00 - M_y ** 2
    cor_xy = np.sum(np.multiply(inds[0], inds[1])) / M00 - M_x*M_y
    theta_moment = 1/2 * np.arctan(2*cor_xy/(var_x-var_y)) + (var_x < var_y)*np.pi/2
    length = np.sqrt(8*(var_x+var_y+np.sqrt(4*cor_xy**2+(var_x-var_y)**2))) / 2
    width = np.sqrt(8*(var_x+var_y-np.sqrt(4*cor_xy**2+(var_x-var_y)**2))) / 2

    return length, width, theta_moment, M_x, M_y


def adjust_angle(angle):
    """
    Adjusts given angle (in rad) so that it is in the interval (-pi/2, pi/2).
    :param angle: given angle
    :return: adjusted angle
    """
    if angle is None:
        return None
    if -np.pi/2 < angle < np.pi/2:
        return angle
    angle = np.mod(angle, 2*np.pi)
    if np.pi/2 < angle < 3*np.pi/2:
        angle -= np.pi
    elif 3*np.pi/2 < angle:
        angle -= 2*np.pi
    return angle


def pre_process_ref(ref_image, ref_path):
    """
    Check if reference image is blurred, blur it if it is not the case (original file is not altered)
    :param ref_image: path to original reference image
    :param ref_path: directory where the processed reference image will be saved
    :return: path to processed reference image
    """
    ref_filename = 'blurred_' + ref_image.split('/')[-1] + '_ref.fits'
    # check if path for blurred reference images exists
    if os.path.exists(ref_path):
        # check if reference image is processed
        if os.path.exists(ref_path + ref_filename):
            print('Reference image is processed already. Continuing...')
            return ref_filename
    else:
        # create path for blurred reference images
        os.mkdir(ref_path)

    # blur and save reference image
    data = fits.open(ref_image)[1].data
    data = ndimage.convolve(data, gaussian_kernel(5))
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(ref_path + ref_filename, overwrite=True)
    print('Reference image processed.')
    return ref_filename


def pre_process_diff(diff_image, ref_image, diff_path):
    """
    Check if given difference image is pre-processed, process it if it is not the case. Searchs in pre_processed_path
    i.e. meta_path (given) + pre_processed_fits/.
    :param diff_image: path to unprocessed difference image
    :param ref_image: path to reference image
    :param diff_path: directory where the processed difference image will be saved
    :return: path to processed difference image
    """
    diff_filename = diff_image.split('/')[-1] + '_pp.fits'
    # check if path for pre-processed difference images exists
    if os.path.exists(diff_path):
        # check if difference image is pre-processed
        if os.path.exists(diff_path + diff_filename):
            print('Difference image is processed already. Continuing...')
            return diff_filename
    else:
        # create path for pre-processed difference images
        os.mkdir(diff_path)

    # pre-process difference image
    ref = fits.open(ref_image)[1].data
    img = fits.open(diff_image)[1].data

    processed_diff = preprocess(img, ref)
    hdu = fits.PrimaryHDU(processed_diff)
    hdu.writeto(diff_path + diff_filename, overwrite=True)
    print('Difference image processed.')
    return diff_filename


def find_nearest(cutout_coord_x, cutout_coord_y, x_coord, y_coord):
    """
    Find the closest point to (x_coord, y_coord) in the coordinate set with xs in cutout_coord_x and ys in
    cutout_coord_y.
    :param cutout_coord_x: xs of the coordinate set
    :param cutout_coord_y: ys of the coordinate set
    :param x_coord: x coordinate
    :param y_coord: y coordinate
    :return: ind: the index of the closest point, np.sqrt(dists[ind]): the distance between the closest point and the
    actual point, cutout_coord_x[ind]: x coordinate of the closest point, cutout_coord_y[ind]: y coordinate of the
    closest point
    """
    dists = np.square(cutout_coord_x-x_coord)+np.square(cutout_coord_y-y_coord)
    ind = np.argmin(dists)
    return ind, np.sqrt(dists[ind]), cutout_coord_x[ind], cutout_coord_y[ind]


def fix_path(path):
    """
    Make sure that the given path ends with '/'.
    :param path: given path
    :return: path with '/'
    """
    if path[-1] != '/':
        return path + '/'
    return path
