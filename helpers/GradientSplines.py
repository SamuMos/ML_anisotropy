import numpy as np
import multiprocessing as mp
import xarray as xr
from scipy.interpolate import UnivariateSpline


def Spline_fit(X, Y, k):
    """callable for multiprocessing, handles parameter number to avoid errors"""
    if len(X) > k:
        return UnivariateSpline(X, Y, k=k)
    else:
        return np.nan


def Spline_fit_double_wind(X, Y, k):
    """callable for multiprocessing that fits two splines.
    Used for wind in LLJ cases"""
    if len(X) > k + 1:
        return UnivariateSpline(X, Y, k=k), UnivariateSpline(X[1:], Y[1:], k=k)
    else:
        return np.nan


def Spline_fit_double_temp(X, Y):
    """callable for multiprocessing that fits two splines.
    Used for temp in LLJ cases"""
    if len(X) < 5:
        return np.nan
    elif len(X) == 5:
        return UnivariateSpline(X, Y, k=4)
    else:
        # choose the spline with the least sum of square error
        spl = [UnivariateSpline(X, Y, k=4), UnivariateSpline(X, Y, k=5)]
        SSE = [
            np.array([(spl[i](X[j]) - Y[j]) ** 2 for j in range(len(X))]).sum()
            for i in [0, 1]
        ]
        return spl[np.argmin(SSE)]


def Gradients_spline(Z, profiles, k=3, z0=0, LLJ=None,  model="wind"):
    """fits all of the profiles with splines according to the prescribed model
        The fit results are splines as a function of log(z)
    """

    length = len(profiles)

    # prepare mdel according to variable
    if model == "wind":
        # prepare data: list of arrays without nans, each with its own length
        Y = [[] for i in range(length)]
        X = [[] for i in range(length)]
        for i in range(length):
            row = profiles[i, :]
            notnan = ~np.isnan(row)
            if z0 > 0:
                Y[i] = np.append(0, row[notnan])
                X[i] = np.append(z0, Z[notnan])
            else:
                Y[i] = np.array(row[notnan])
                X[i] = np.array(Z[notnan])

        # compute splines interpolation
        # a spline of log(Z) with u(z0)=0 is used                                           
        fit_res = [Spline_fit(np.log(X[i]), Y[i], k=k) for i in range(length)]

        # extract gradients
        # in case of LLJ use the spline without Z0 for all levels except the first one
        gradient = np.empty(np.shape(profiles)) * np.nan
        for i in range(length):
            if fit_res[i] is not np.nan:
                gradient[i] = fit_res[i].derivative()(np.log(Z)) / Z

    # Model for temperature
    elif model == "temp":
        # prepare data: list of arrays without nans, each with its own length
        Y = [[] for i in range(length)]
        X = [[] for i in range(length)]
        for i in range(length):
            row = profiles[i, :]
            notnan = ~np.isnan(row)
            Y[i] = np.array(row[notnan])
            X[i] = np.array(Z[notnan])
            
        fit_res = [Spline_fit(np.log(X[i]), Y[i], k=k) for i in range(length)]
    
    # extract gradients
    gradient = np.empty(np.shape(profiles)) * np.nan
    for i in range(length):
        if fit_res[i] is not np.nan:
            gradient[i] = fit_res[i].derivative()(np.log(Z))/Z

    return fit_res, np.array(gradient)


def Gradients_spline_old(Z, profiles, z0=0, LLJ=None, model="wind"):
    """Old version, from 2022"""
    if model == "wind" and z0 == 0 and LLJ is not None:
        raise ValueError(
            "Function Gradient_spline is not implemented for Low Level Jet cases without providing roughness length"
        )

    length = len(profiles)

    # prepare mdel according to variable
    if model == "wind":
        # prepare data: list of arrays without nans, each with its own length
        Y = [[] for i in range(length)]
        X = [[] for i in range(length)]
        for i in range(length):
            row = profiles[i, :]
            notnan = ~np.isnan(row)
            if z0 > 0:
                Y[i] = np.append(0, row[notnan])
                X[i] = np.append(z0, Z[notnan])
            else:
                Y[i] = np.array(row[notnan])
                X[i] = np.array(Z[notnan])

        # compute splines interpolation
        # a spline of log(Z) with u(z0)=0 is used except for LLJ cases where a simple spline is used
        if LLJ is not None and z0 > 0:
            fit_res = [
                [
                    Spline_fit(np.log(X[i]), Y[i], k=3),
                    Spline_fit_double_wind(X[i], Y[i], k=4),
                ][LLJ[i]]
                for i in range(length)
            ]
        else:
            fit_res = [Spline_fit(np.log(X[i]), Y[i], k=3) for i in range(length)]

        # extract gradients
        # in case of LLJ use the spline without Z0 for all levels except the first one
        gradient = np.empty(np.shape(profiles)) * np.nan
        if LLJ is None:
            for i in range(length):
                if fit_res[i] is not np.nan:
                    gradient[i] = fit_res[i].derivative()(np.log(Z)) / Z
        else:
            for i in range(length):
                if fit_res[i] is not np.nan:
                    if not LLJ[i]:
                        gradient[i] = fit_res[i].derivative()(np.log(Z)) / Z
                    else:
                        gradient[i] = np.append(
                            fit_res[i][1].derivative()(Z[0]),
                            fit_res[i][0].derivative()(Z[1:]),
                        )
                        # leave only one spline to the output
                        fit_res[i] = fit_res[i][0]

    # Model for temperature
    elif model == "temp":
        # prepare data: list of arrays without nans, each with its own length
        Y = [[] for i in range(length)]
        X = [[] for i in range(length)]
        for i in range(length):
            row = profiles[i, :]
            notnan = ~np.isnan(row)
            Y[i] = np.array(row[notnan])
            X[i] = np.array(Z[notnan])

        # compute splines interpolation
        # if LLJ two models, one to get the first level
        if LLJ is not None:
            fit_res = [
                [Spline_fit(X[i], Y[i], k=4), Spline_fit_double_temp(X[i], Y[i])][
                    LLJ[i]
                ]
                for i in range(length)
            ]
        else:
            fit_res = [Spline_fit(X[i], Y[i], k=4) for i in range(length)]

        # extract gradients
        gradient = np.empty(np.shape(profiles)) * np.nan
        for i in range(length):
            if fit_res[i] is not np.nan:
                gradient[i] = fit_res[i].derivative()(Z)

    return fit_res, np.array(gradient)