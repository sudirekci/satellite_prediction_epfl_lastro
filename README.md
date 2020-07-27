Required packages to run the codes: scipy, skimage, astropy, matplotlib, tensorflow, numpy. These can be installed via pip.
Additionally, sextractor and lacosmicx should be installed as well.

This project includes python scripts, default sextractor files and a neural network model. The model is pre-trained and has an accuracy of 96.7%. However, the script to train the network from scratch is label_track_nn.py (in case a better network is needed). To train the model, the user should have a dataset of labeled cutouts. For labeling, a useful script is label_cutouts.py. This script takes a difference image and a reference image as input and finds the cutouts in the difference image. The user can label the cutouts via the interactive plot of the difference image, produced by the script, and the cutouts are saved automatically to their respective directories.

find_tracks_diff.py takes a difference image and a reference image as input and finds the tracks in the difference image, with the help of the pre-trained model. It estimates the orientations and the lengths of the tracks as well. The results are saved as numpy arrays and a txt file automatically.

find_tracks_single_ep.py takes a field name and an observation date as input. It first constructs full-field images of the field taken on the observation date. Then, the cosmic rays are removed from the images with lacosmicx and the objects are found with sextractor. Finally, all the fast and slow satellite tracks are found, plotted and saved.

# satellite_prediction_epfl_lastro
