from __future__ import (print_function, division)
__authors__ = 'fernandes, vilimstich, helmbrecht'

import numpy as np
import scipy.signal
import matplotlib.cm as cmap
from skimage import transform
import skimage as sk
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


class Experiment(object):
    """ Class for processing experiments from already parsed data
    """


    def __init__(self, images, stimuli, dt, metadata, upsampling=1,
                 begin_parameter=None):
        """ Loads the experiment from prepared data

        :param images: 3D array of calcium images in time [time, x, y]
        :param stimuli: pandas dataframe or dictionary of lists describing what
            was presented. time [t] ha
        :param dt: sampling interval
        :param metadata: metadata dictionary, for all that is not in the
            stimulus dataframe and dictionary
        """
        self.stimuli = stimuli
        self.dt = dt
        self.orig_images = images
        self.steps = images.shape[0]
        self.images = self.dFoverF(images)
        self.time = np.arange(self.steps) * dt
        self.metadata = metadata
        self.mean_images = np.mean(self.images, axis=0)
        self.upsampling = upsampling
        self.u_steps = self.steps * upsampling
        self.u_time = np.arange(self.u_steps) * (dt/upsampling)

        if begin_parameter is None:
            self.images = self.df_over_f(images)
        else:
            self.images = self.df_over_f_no_stimulus(images, begin_parameter)


    def ds_image(self, images, resize_factor):
        frames_ds = sk.transform.downscale_local_mean(images,
                                                      (1, resize_factor,
                                                       resize_factor))
        return frames_ds

    def display_scores(self, scores, axis):
        axis.imshow(self.mean_images, cmap=cmap.Greys_r)
        disp_scores = scores.copy()
        disp_scores[disp_scores == 0] = np.nan
        axis.imshow(disp_scores, cmap=cmap.Reds)


    def display_scores_c(self, scores, axis, color, alphax=0.9):
        # axis.imshow(self.mean_images, cmap=cmap.Greys_r)
        disp_scores = scores.copy()
        disp_scores[disp_scores == 0] = np.nan
        img = axis.imshow(disp_scores, cmap=color, alpha=alphax)
        plt.colorbar(img, ax=axis)

        # axis.set_clim(vmin=0, vmax=1)

    def display_scores_new(self, scores, axis, color, alphax=0.9):
        # axis.imshow(self.mean_images, cmap=cmap.Greys_r)
        disp_scores = scores.copy()
        disp_scores[disp_scores == 0] = np.nan
        axis.imshow(disp_scores, cmap=color, alpha=alphax)



    def median_filter(self, images, frames=10):
        new_image = ndimage.filters.median_filter(images, size=(frames,1,1), footprint=None, output=None, mode='reflect', cval=0.0, origin=0)
        return new_image

    def med_filter(self, images, frames=10):
        win = frames # frames
        newer_image = np.zeros(np.shape(images))

        for j in range(np.shape(images)[1]):
            for k in range(np.shape(images)[1]):
                newer_image[:, j, k] = np.convolve(np.ones(win), images[:,j,k], mode='same')/win
        return newer_image

    def df_over_f(self, images, take_lowest_percentile=5):
        """ Calculates the delta F over F for a image series

        :param images: the images to process
        :param take_lowest_percentile: the lowest percentile of brightness to
        take as F0
        :return: processed images
        """
        F0 = np.percentile(images, take_lowest_percentile, 0)
        return (images*1.0 - F0)/F0


    def df_over_f_no_stimulus(self, images, begin_parameter):
        """ Takes F0 as the activity when there is no stimulus

        :param images: images to process
        :param begin_parameter: the parameter which is 0 while there is
            no stimulus
        :return: processed images
        """
        start_time = 0
        i = 0
        while self.stimuli[begin_parameter][i] == 0:
            i += 1
            start_time = self.stimuli['t'][i]

        start_idx = np.searchsorted(self.time, start_time)
        F0 = np.mean(images[:start_idx,:,:], 0)
        return (images*1.0 - F0)/F0

    def dFoverF(self, images, take_lowest_percentile=5):
        """ Calculates the delta F over F for a image series

        :param images: the images to process
        :param take_lowest_percentile: the lowes percentile of brightness to
        take as F0
        :return: processed images
        """
        F0 = np.percentile(images, take_lowest_percentile, 0)
        return (images*1.0 - F0)/F0

    def df_over_f_THEIA(self, images, take_lowest_percentile=5):
        """ Calculates the delta F over F for a image series

        :param images: the images to process
        :param take_lowest_percentile: the lowest percentile of brightness to
        take as F0
        :return: processed images
        """
        F0 = np.percentile(images, take_lowest_percentile, 0)
        return (np.astype(images, np.float32) - F0)/F0

    def dFoverF_frames(self, images, frames=100, take_lowest_percentile=5):
        """ Calculates the delta F over F for a image series

        :param images: the images to process
        :param take_lowest_percentile: the lowes percentile of brightness to
        take as F0
        :return: processed images
        """
        F0 = np.percentile(images[frames:,:,:], take_lowest_percentile, 0)
        return (images*1.0 - F0)/(np.absolute(F0)+1)

    def dFoverF_tp(self, images, t1=0, t2=300):
        """ Calculates the delta F over F for a image series
        :param images: the images to process
        :param take_lowest_percentile: the lowes percentile of brightness to
        take as F0
        :return: processed images
        """
        F0 = np.mean(images[t1:t2, :, :], 0)
        # F0 = np.percentile(images, take_lowest_percentile, 0)
        return (images*1.0 - F0)/F0

    def dFoverF_s_st(self, images):
        """ Calculates the delta F over F for a image series
        :param images: the images to process
        :param take_lowest_percentile: the lowes percentile of brightness to
        take as F0
        :return: processed images
        """
        s_start = np.where(self.stimuli['ON'] == 1)[0][0]
        framex = int(np.round(self.stimuli['t'][s_start]*1/self.dt))
        F0 = np.mean(images[framex-55:framex-5, :, :], 0)
        # F0 = np.percentile(images, take_lowest_percentile, 0)
        return (images*1.0 - F0)/F0



    def correct_first_frames(self, images, number_of_frames=99, percentile=40):
        percentile_image = np.percentile(images[number_of_frames:, :, :], percentile, 0)
        images[0:number_of_frames, :,:] = percentile_image
        return images

    def correct_first_frames_0(self, images, number_of_frames=99):
        images[0:number_of_frames, :,:] = 0
        return images

    def set_baseline_to_zero(self, images):
        base = np.amin(images, axis=0)
        return images-base


    def regression_matrix(self):
        """ Makes a 2D matrix of the images, first dimension time,
         second flattened pixels

        :return: partially flattened images
        """
        return self.images.reshape((self.images.shape[0], -1))

    def regression_matrix_raw_data(self):
        """ Makes a 2D matrix of the images, first dimension time,
         second flattened pixels

        :return: partially flattened images
        """
        return self.orig_images.reshape((self.orig_images.shape[0], -1))

    def reshape_to_image(self, img):
        """ Returns the shape (original width and height) to a flattened image
        or sequence of images

        :param img: the image to reshape
        :return: reshaped image
        """
        if len(img.shape) == 2:
            return img.reshape(-1, self.images.shape[1], self.images.shape[2])
        else:
            return img.reshape(self.images.shape[1], self.images.shape[2])
    def reshape_to_image_raw_data(self, img):
        """ Returns the shape (original width and height) to a flattened image
        or sequence of images

        :param img: the image to reshape
        :return: reshaped image
        """
        if len(img.shape) == 2:
            return img.reshape(-1, self.orig_images.shape[1], self.orig_images.shape[2])
        else:
            return img.reshape(self.orig_images.shape[1], self.orig_images.shape[2])

    def exp_decay_kernel_g6s(self, tau=1.8):
        """ Exponential decay kernel to model the calcium response function

        :param tau: decay time constant
        :return: the normalised kernel
        """
        decay = np.exp(-self.time / tau)
        return decay / np.sum(decay)

    def exp_decay_kernel_g6f(self, tau=0.8):
        """ Exponential decay kernel to model the calcium response function

        :param tau: decay time constant
        :return: the normalised kernel
        """
        decay = np.exp(-self.time / tau)
        return decay / np.sum(decay)

    def convolve_regressors(self, regressor, kernel):
        """ Convolves the regressor with a kernel function

        :param regressor: the regressor, or regressor matrix
        :param kernel:
        :return: the convolved kernel
        """
        if regressor.shape[0] != self.steps:
            raise Exception('The regressor is not interpolated')

        # if convolving multiple regressors, do it as a 2D convolution
        if len(regressor.shape) > 1:
            return scipy.signal.convolve2d(regressor,
                                           kernel.reshape(-1, 1)
                                           )[0:self.steps, :]
        else:
            return np.convolve(regressor, kernel)[0:self.steps]

    def interpolate_regressor(self, reg_field, type='visual'):
        """ Interpolates and convolves the regressor

        :param reg_field: either a string pointing to the stimulus field
            or and array of with the same number of elements as stimuli
        :return: the interpolated and convolved regressor
        """
        # these conditions process all the case, first if an array or a
        # stimulus parameter name is given
        if type == 'visual':
            if isinstance(reg_field, np.ndarray):
                # then, whether it is the proper shape
                if reg_field.shape[0] == self.steps:
                    reg_i = reg_field
                # or it has to be interpolated
                else:
                    # and if so, do it for every regressor if there are multiple
                    if len(reg_field.shape) > 1:
                        reg_i = []
                        for i in range(reg_field.shape[1]):
                            reg_i.append(np.interp(self.time, self.stimuli['t'],
                                                   reg_field[:, i], left=0,
                                                   right=0))
                        reg_i = np.vstack(reg_i).T

                    else:
                        reg_i = np.interp(self.time, self.stimuli['t'], reg_field,
                                  left=0, right=0)
            elif isinstance(reg_field, basestring):
                reg_raw = self.stimuli[reg_field]
                reg_i = np.interp(self.time, self.stimuli['t'], reg_raw)

        elif type == 'behavioral':
            if isinstance(reg_field, basestring):
                reg_raw = self.behavior[reg_field]
                reg_i = np.interp(self.time, self.behavior['t'], reg_raw)
        return reg_i

    def differentiate_parameter(self, parameter):
        """ Differentiate a stimulus parameter

        :param parameter: the name of the column to be differentiated
        :return: differentiated stimulus parameter, with points at image
                 sampling times
        """
        resampling = 10  # it has to be an integer bigger than 1, otherwise the
        # velocity will be incorrectly determined
        resampled_time = np.linspace(0, self.time[-1],
                                     len(self.time) * resampling)
        reg = np.interp(resampled_time, self.stimuli['t'],
                        self.stimuli[parameter], left=0, right=0)
        dreg = np.append(0, np.ediff1d(reg) /
                         np.ediff1d(resampled_time))
        return dreg[::resampling]

    @staticmethod
    def gram_schmidt_columns(m):
        """ Performs a gram-schmidt procedure to orthonormalize the columns of
        a matrix m

        :param m:
        :return:
        """
        result = m.copy()
        for i in range(m.shape[1]):
            result[:, i] = (result[:, i]) / np.linalg.norm(result[:, i])
            if i < m.shape[1] - 1:
                result[:, i + 1:] = result[:, i + 1:] - np.outer(result[:, i],
                                                                 np.dot(
                                                                     result[:,
                                                                     i + 1:].T,
                                                                     result[:,
                                                                     i]).T)
        return result

    def t_scores(self, regressors, orthonormal=False):
        """
        Generates the T scores for given regressors

        :param regressors: arrays for regression (time in rows, different
                           regressors in columns
        :return: T scores

        Based on Miri et al 2011 citing T statistics (Hoel 1984)
        """
        G = regressors
        X = self.regression_matrix()
        if len(G.shape) == 1:
            b = np.dot(G.T, X) / np.dot(G.T, G)
            eps = X - np.outer(G, b)
        else:
            # if the regressor matrix is orthonormalised, the pseudoinverse for
            # linear regression is I
            if orthonormal:
                b = np.dot(G.T, X)
            else:
                b = np.dot(np.dot(np.linalg.pinv(np.dot(G.T, G)), G.T), X)
            eps = X - np.dot(G, b)

        T = b / np.sqrt(np.sum(eps.T ** 2, 1) / (eps.shape[0] - 3.0))
        return self.reshape_to_image(T)

    def t_scores_raw_data(self, regressors, orthonormal=False):
        """
        Generates the T scores for given regressors

        :param regressors: arrays for regression (time in rows, different
                           regressors in columns
        :return: T scores
        """
        G = regressors
        X = self.regression_matrix_raw_data()
        if len(G.shape) == 1:
            b = np.dot(G.T, X) / np.dot(G.T, G)
            eps = X - np.outer(G, b)
        else:
            # if the regressor matrix is orthonormalised, the pseudoinverse for
            # linear regression is I
            if orthonormal:
                b = np.dot(G.T, X)
            else:
                b = np.dot(np.dot(np.linalg.pinv(np.dot(G.T, G)), G.T), X)
            eps = X - np.dot(G, b)

        T = b / np.sqrt(np.sum(eps.T ** 2, 1) / (eps.shape[0] - 3.0))
        return self.reshape_to_image_raw_data(T)

    def orthonomal_t_scores(self, regressors):
        """ gives the T scores for each regressor

        :return: T scores, dimensions [image.height, image.width, num. of regs]
        """
        positions = regressors.shape[1]
        T_scores = np.zeros((positions, self.images.shape[1],
                             self.images.shape[2]))

        for i in range(positions):
            order = np.roll(np.arange(positions), -i)
            pos_kernels = self.gram_schmidt_columns(regressors[:, order])
            T = self.t_scores(pos_kernels, True)
            T_scores[i, :, :] = T[0, :, :]

        return T_scores

    def pearson(self, regressors):
        """ Gives the pearson correlation coefficient

        :param regressors: the regressors, with time in rows
        :return: the pearson correlation coefficient
        """
        # two versions, depending whether there is one or multiple regressors
        X = self.regression_matrix()
        Y = regressors
        if len(Y.shape) == 1:
            numerator = np.dot(X.T, Y) - self.steps * np.mean(X, 0) * np.mean(Y)
            denominator = (self.steps - 1) * np.std(X, 0) * np.std(Y)
            result = numerator / denominator
        else:
            numerator = np.dot(X.T, Y) - self.steps * np.outer(np.mean(X, 0),
                                                               np.mean(Y, 0))
            denominator = (self.steps - 1) * np.outer(np.std(X, 0),
                                                      np.std(Y, 0))
            result = (numerator / denominator).T

        return self.reshape_to_image(result)

    def pearson_raw_data(self, regressors):
            """ Gives the pearson correlation coefficient

            :param regressors: the regressors, with time in rows
            :return: the pearson correlation coefficient
            """
            # two versions, depending whether there is one or multiple regressors
            X = self.regression_matrix_raw_data()
            Y = regressors
            if len(Y.shape) == 1:
                numerator = np.dot(X.T, Y) - self.steps * np.mean(X, 0) * np.mean(Y)
                denominator = (self.steps - 1) * np.std(X, 0) * np.std(Y)
                result = numerator / denominator
            else:
                numerator = np.dot(X.T, Y) - self.steps * np.outer(np.mean(X, 0),
                                                                   np.mean(Y, 0))
                denominator = (self.steps - 1) * np.outer(np.std(X, 0),
                                                          np.std(Y, 0))
                result = (numerator / denominator).T

            return self.reshape_to_image_raw_data(result)

    @staticmethod
    def stimulus_space_regressor(regressor, points=3,
                                 distance_f=lambda x, y, sd: np.exp(
                                     -(x-y)**2/sd**2), sd=None):
        """ Creates a set of regressors spanning the values of a stimulus
        parameter

        :param regressor: the stimulus parameter which the new regressors span
        :param points: the centres of tuning functions, or the number of
                       equidistant tuning functions
        :param distance_f: the distance function, by default a Gaussian
        :param sd: standard deviation of the distance function, if applicable
            if not given, calculated as the distance between the first two
            means of tuning functions
        :return: the set of regressors
        """
        if isinstance(points, np.ndarray):
            test_points = points
        else:
            test_points = np.linspace(np.min(regressor), np.max(regressor),
                                      points)
        if sd is None:
            sd = (test_points[1]-test_points[0])

        return distance_f(regressor[:, None], test_points[None, :], sd)
