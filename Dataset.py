from sklearn.model_selection import train_test_split
from functions import to_coefs, normalize

import numpy as np
import h5py


class Dataset:
    """Class is used for storing variables and functions related to 
    storing, augmenting and loading of the dataset. This class has been
    tested using the metal oxide dataset but may be able to used for 
    a varity of datasets if they are .h5 files."""

    def __init__(self, file_name='dataset/dataset_raw.h5'):
        # Variables used for triggering dataset changes.
        self.current_split_percentage = 0
        self.current_poly_degree = 0
        self.current_poly_aug = 0

        # Orginal unsplit data.
        hf = h5py.File(file_name, 'r')
        self.images = hf['images']
        self.labels = hf['spectra']

        # Current data that is used in the model.
        self.training_images, self.training_labels = None, None
        self.testing_images, self.testing_labels = None, None

        # A record is kept of the original split image so when polynomial
        # augmentation is applied it uses the original data and not the
        # already modified data.
        self.original_training_images, self.original_training_labels = None, None
        self.original_testing_images, self.original_testing_labels = None, None

    def check_split(self, percentage: float, id: int):
        """Check to see if the split variable has changed. If it has
        change then reset the training data variables."""
        if self.current_split_percentage is not percentage:

            # Update the self.training and self.testing variables with new training split
            print("Splitting Dataset")
            self.training_images, self.testing_images, self.training_labels, self.testing_labels = train_test_split(
                self.images,
                self.labels,
                shuffle=False,
                test_size=percentage
            )
            if id == 0:
                # Set the orginal data variables.
                self.original_training_images = self.training_images
                self.original_training_labels = self.training_labels
                self.original_testing_images = self.testing_images
                self.original_testing_labels = self.testing_labels

            if self.current_poly_aug != 0:
                # Aplly polynomial augmentation if it needed.
                self.apply_polynomial_augmentation(self.current_poly_degree)
                self.normilize_labels()

            # Apply normilization of the labels (images soon).
            self.normilize_labels()

            # Update the trigger value
            self.current_split_percentage = percentage

    def normilize_labels(self):
        """Normilize the labels to remove the mean. If not
        normilized it will fit to the average curve."""

        self.training_labels, self.labels_mean, self.labels_deviation = normalize(
            self.training_labels)

    def check_poly_aug(self, aug: int):
        """Check to see if the polynomial augmentation trigger has 
        changed. If it has changed then take the orginal data and apply
        polynomial augmentation to it and set that to the current data."""

        if self.current_poly_aug is not aug:
            # If the augmentagtion trigger is swapped then apply
            # poolynomial augmentation useing the degree variable.
            self.apply_polynomial_augmentation(self.current_poly_degree)
            # Update the trigger value
            self.current_poly_aug = aug

    def check_poly_degree(self, degree: int, poly_aug: bool):
        """Check to see if the polynomial degree has changed.  If it has
        changed then take the orginal data and apply polynomial augmentation
        to it and set that to the current data."""

        if poly_aug != 0:
            if self.current_poly_degree is not degree:
                # Update the self augmented data with the new degree
                self.apply_polynomial_augmentation(degree)
                # Update the trigger value
                self.current_poly_degree = degree

    def apply_polynomial_augmentation(self, degree):
        """Apply polynomial augmentation to the label data."""

        print(f"Applying polynomial Augmentation with degree: {degree}")
        self.training_labels = np.vstack(
            [to_coefs(label, degree) for label in self.original_training_labels])
        self.testing_labels = np.vstack(
            [to_coefs(label, degree) for label in self.original_testing_labels])
