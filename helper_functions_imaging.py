__authors__ = 'fernandes, donovan'
import numpy as np
from sklearn import linear_model, metrics
def make_reg(regressor_to_make, steps, t,frame_rate):
    '''Builds a regressor prediction based on given visual stimulus metadata
    :param regressor_to_make: the chosen regressor
    :param steps: how long is the regressor in frames
    :param t: real time
    :param frame_rate: imaging frame rate
    :return: resulting regressor based on prediction'''
    zero_reg=np.zeros(steps)
    regressor_to_make_reg=np.copy(zero_reg)
    where_regressor_to_make=t[regressor_to_make[0]]
    where_regressor_to_make_frame=where_regressor_to_make*frame_rate
    where_regressor_to_make_frame_int=(where_regressor_to_make_frame.values.astype(int))#change for python3
    make_ones_regressor_to_make=regressor_to_make_reg[where_regressor_to_make_frame_int]=1
    regressor_to_make_reg_ones=regressor_to_make_reg
    regressor_to_make_result=regressor_to_make_reg_ones
    return regressor_to_make_result

def find_timepoints_reg_high(regressor_to_get_high):
    '''Finds timepoints where regressor is more than zero
    :return:timepoints'''
    regressor_high = np.asarray(np.where(regressor_to_get_high > 0))
    return regressor_high


def dFoverF_ROIs(traces_ROIs, take_lowest_percentile=5):
    """ Calculates the delta F over F for an image series

    :param images: the images to process
    :param take_lowest_percentile: the lowes percentile of brightness to take as F0
    :return: processed images
    """
    F0 = np.percentile(traces_ROIs, take_lowest_percentile, 0)
    return (traces_ROIs - F0)/F0


def mean_for_timepoints_with_ROIs(ROIs_seed_deltaF_F0, regs, how_long):
    '''Averages ROIs average for periods whenão regressors are bigger than zer

    :param ROIs_seed_deltaF_F0: traces to
    :param regs: reGressors/prediction
    :param how_long: how many frames to average. Change based on GCaMP version used (empirically determined)
    :return: return mean across timepoints for each regressor'''
    assert how_long >= 1

    roi_arr = np.asarray(ROIs_seed_deltaF_F0) #rois by frames
    num_reg_timepoints = [len(r[0]) for r in regs]
    roi_res = np.zeros((roi_arr.shape[0], len(regs), num_reg_timepoints[0]))
    for ridx, reg in enumerate(regs):
        for tidx, timepoint in enumerate(reg[0]):
            roi_res[:, ridx, tidx] = roi_arr[:, timepoint:timepoint+how_long].mean(-1)
    return roi_res.mean(-1)


def filter_rois(regressors, dff, r_threshold,reg):
    '''fit for all regressors and then take the r2, remove all ROIs that are not highly correlated
    :param regressors: used regressors for the experiment
    :param dff: DeltaF/F0 traces
    :param r_threshold: r2 threshold for linear regression
    :param reg: regression
    :return: idx of ROIs that pass the threshold, ndarray of r2 scores, estimated coefficients for the regression'''
    reg.fit(regressors.T, dff.T)
    r2 = metrics.r2_score(dff.T, reg.predict(regressors.T), multioutput='raw_values')
    coefs=reg.coef_
    return r2 > r_threshold, r2, coefs

def max_correlation_values(list1,list2,list3):
    '''def returns maximum value from 3 given lists'''
    list4 = [max(value) for value in zip(list1, list2,list3)]
    return list4
