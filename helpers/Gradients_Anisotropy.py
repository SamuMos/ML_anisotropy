import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from numpy import ma
import xarray as xr
from . import Gradient as G
from . import Anisotropy as A
from . import GradientSplines as GS
from functools import partial
from scipy.interpolate import UnivariateSpline
import warnings


# PROCESS 0 - Quality criterias and reaveraging of stable data to 10 minute
def QC_reaverage(data_s, data_s_low, data_un, data_un_low, T_flux_threshold=-1e-3):
    """
    Works on xarray datasets of stable and unstable, drops wrong stability profiles, applies quality criterias and reaverages stable to 10 minute
    """

    # Drops profiles with nans and changes in stabilty
    data = data_s
    data = data.where(data.wT < 0).dropna(dim="time", how="any")
    data_s = data_s.sel(time=data.time)

    data = data_un
    data = data.where(data.wT > 0).dropna(dim="time", how="any")
    data_un = data_un.sel(time=data.time)

    # Quality criterias
    stationarity = (data_s.statU < 30) & (data_s.statUW < 30) & (data_s.statWT < 30)
    uncertainty = (
        (data_s.WynU < 0.8) & (data_s.WynUW < 0.8) & (data_s.WynWT < 0.8)
    )  # currently not used
    minimum_T_flux = data_s.wT < T_flux_threshold

    mask = stationarity & minimum_T_flux
    data_s = data_s.where(mask)

    # Fraction of non-filtered data_s
    fraction = np.sum(mask) / (np.sum(mask) + np.sum(np.invert(mask)))
    print(
        "Surviving the stationarity and wT QC for STABLE -> {:4.1f}%    ({:6.0f} datapoints)".format(
            fraction.data * 100, np.sum(mask.data)
        )
    )

    # 10 min average for stable data_s
    data_s = data_s.resample(time="10min", skipna=True).mean()
    data_s_low = data_s_low.resample(time="10min", skipna=True).mean()

    # Have the same time indexes for high and low freq
    data_s_low = data_s_low.where(data_s_low.time.isin(data_s.time), drop=True)
    data_s = data_s.where(data_s.time.isin(data_s_low.time), drop=True)
    data_un_low = data_un_low.where(data_un_low.time.isin(data_un.time), drop=True)
    data_un = data_un.where(data_un.time.isin(data_un_low.time), drop=True)

    return data_s, data_s_low, data_un, data_un_low


# PROCESS 1 - Gradients calculation
def Gradients_calculation(
    data_s,
    data_s_low,
    data_un,
    data_un_low,
    roughness_length=0,
    return_fit=False,
    keep_hf_T=False,
):
    """
    Calculates gradients for stable and unstable data, using low frequency temperature measurements
    """

    # CHOOSE FUNCTIONS for stable and unstable wind and temperature
    # function to use in case of low level Jet (or just more complex) has index one
    if roughness_length > 0:
        wind_stable_function_0 = partial(G.businger_rough, z0=roughness_length)
        wind_stable_derivative_0 = partial(G.D_businger_rough, z0=roughness_length)
        wind_stable_function_1 = partial(G.quadratic_rough, z0=roughness_length)
        wind_stable_derivative_1 = partial(G.D_quadratic_rough, z0=roughness_length)
        wind_bounds = [([-np.inf, 0], np.inf), ([-np.inf, -np.inf, 0], np.inf)]
    else:
        wind_stable_function_0 = G.businger
        wind_stable_derivative_0 = G.D_businger
        wind_stable_function_1 = G.quadratic
        wind_stable_derivative_1 = G.D_quadratic
        wind_bounds = [
            ([-np.inf, -np.inf, 0], np.inf),
            ([-np.inf, -np.inf, -np.inf, 0], np.inf),
        ]

    temp_stable_function_0 = G.businger
    temp_stable_derivative_0 = G.D_businger
    temp_stable_function_1 = G.full
    temp_stable_derivative_1 = G.D_full

    if roughness_length > 0:
        wind_unstable_function_0 = partial(G.businger_rough, z0=roughness_length)
        wind_unstable_derivative_0 = partial(G.D_businger_rough, z0=roughness_length)
        wind_unstable_function_1 = partial(G.quadratic_rough, z0=roughness_length)
        wind_unstable_derivative_1 = partial(G.D_quadratic_rough, z0=roughness_length)
        # wind_bounds = [ ([-np.inf, 0], np.inf) , ([-np.inf, -np.inf, 0], np.inf) ]
    else:
        wind_unstable_function_0 = G.businger
        wind_unstable_derivative_0 = G.D_businger
        wind_unstable_function_1 = G.quadratic
        wind_unstable_derivative_1 = G.D_quadratic
        # wind_bounds = [ ([-np.inf, -np.inf, 0], np.inf) , ([-np.inf, -np.inf, -np.inf, 0], np.inf) ]

    temp_unstable_function_0 = G.businger
    temp_unstable_derivative_0 = G.D_businger

    # ------------------------------------------

    # STABLE PART
    # check for low level jets
    LLJ_present, LLJ_index = check_LLJ(data_s.meanU.data)

    # Calculate gradients
    if LLJ_present:
        fitresU_s, gradU_s = G.MultiFunc_Gradient(
            data_s.meanU.data,
            data_s.heights.data,
            Functions=[wind_stable_function_0, wind_stable_function_1],
            Derivatives=[wind_stable_derivative_0, wind_stable_derivative_1],
            func_index=LLJ_index,
            bounds=wind_bounds,
            return_fit=True,
        )

        fitresT_s, gradT_s = G.MultiFunc_Gradient(
            data_s_low.T.data,
            data_s_low.heights.data,
            Functions=[temp_stable_function_0, temp_stable_function_1],
            Derivatives=[temp_stable_derivative_0, temp_stable_derivative_1],
            func_index=LLJ_index,
            return_fit=True,
        )
    else:
        fitresU_s, gradU_s = G.Gradient(
            data_s.meanU.data,
            data_s.heights.data,
            Function=wind_stable_function_0,
            Derivative=wind_stable_derivative_0,
            bounds=wind_bounds,
            return_fit=True,
        )
        fitresT_s, gradT_s = G.Gradient(
            data_s_low.T.data,
            data_s_low.heights.data,
            Function=temp_stable_function_0,
            Derivative=temp_stable_derivative_0,
            return_fit=True,
        )

    # FIX TEMPERATURE data with low freq measurements
    # time index cross reference
    ind0 = np.nonzero(np.in1d(data_s.time, data_s_low.time))[0]
    ind1 = np.nonzero(np.in1d(data_s_low.time, data_s.time))[0]
    time_index_s = np.column_stack((ind0, ind1))

    # fill temperature with low freq data
    shape = (len(data_s.time), len(data_s.heights))
    gradT_s_new = np.empty(shape)
    gradT_s_new[:] = np.nan
    meanT_s_new = np.empty(shape)
    meanT_s_new[:] = np.nan

    heights = data_s.heights.data
    # if heights of low freq and high freq are not the same you have to recall the functions at the right height
    # use a bool variable to avoid code repetition
    ricalculate = True
    if np.size(heights) == np.size(data_s_low.heights):
        if (heights == data_s_low.heights).all():
            ricalculate = True

    if ricalculate:
        if LLJ_present:
            for i in range(len(time_index_s)):
                Functions = [temp_stable_function_0, temp_stable_function_1]
                Derivatives = [temp_stable_derivative_0, temp_stable_derivative_1]
                gradT_s_new[time_index_s[i][0], :] = Derivatives[
                    LLJ_index[time_index_s[i][1]]
                ](heights, *fitresT_s[time_index_s[i][1]])
                meanT_s_new[time_index_s[i][0], :] = Functions[
                    LLJ_index[time_index_s[i][1]]
                ](heights, *fitresT_s[time_index_s[i][1]])
        else:
            for i in range(len(time_index_s)):
                gradT_s_new[time_index_s[i][0], :] = temp_stable_derivative_0(
                    heights, *fitresT_s[time_index_s[i][1]]
                )
                meanT_s_new[time_index_s[i][0], :] = temp_stable_function_0(
                    heights, *fitresT_s[time_index_s[i][1]]
                )
    else:
        for i in range(len(time_index_s)):
            gradT_s_new[time_index_s[i][0]] = gradT_s[time_index_s[i][1]]
            meanT_s_new[time_index_s[i][0], :] = data_s_low.T.data[
                time_index_s[i][1], :
            ]

    # Richardson numbers
    Ri_s = 9.81 / meanT_s_new * gradT_s_new / gradU_s**2
    Rif_s = 9.81 / meanT_s_new * data_s.wT.data / gradU_s / data_s.uw.data

    # assign data to dataset
    data_s = data_s.assign(
        LLJ=(["time"], LLJ_index),
        gradU=(["time", "heights"], gradU_s),
        gradT=(["time", "heights"], gradT_s_new),
        Ri=(["time", "heights"], Ri_s),
        Rif=(["time", "heights"], Rif_s),
    )
    if not keep_hf_T:
        data_s = data_s.assign(meanT=(["time", "heights"], meanT_s_new))
    # ------------------------------------------------------------------

    # UNSTABLE PART
    # check for low level jets
    LLJ_present, LLJ_index = check_LLJ(data_un.meanU.data)

    # Calculate gradients
    if LLJ_present:
        fitresU_un, gradU_un = G.MultiFunc_Gradient(
            data_un.meanU.data,
            data_un.heights.data,
            Functions=[wind_unstable_function_0, wind_unstable_function_1],
            Derivatives=[wind_unstable_derivative_0, wind_unstable_derivative_1],
            func_index=LLJ_index,
            bounds=wind_bounds,
            return_fit=True,
        )
    else:
        fitresU_un, gradU_un = G.Gradient(
            data_un.meanU.data,
            data_un.heights.data,
            Function=wind_unstable_function_0,
            Derivative=wind_unstable_derivative_0,
            bounds=wind_bounds,
            return_fit=True,
        )

    fitresT_un, gradT_un = G.Gradient(
        data_un_low.T.data,
        data_un_low.heights.data,
        Function=temp_unstable_function_0,
        Derivative=temp_unstable_derivative_0,
        return_fit=True,
    )

    # FIX TEMPERATURE data with low freq measurements
    # time index cross reference
    ind0 = np.nonzero(np.in1d(data_un.time, data_un_low.time))[0]
    ind1 = np.nonzero(np.in1d(data_un_low.time, data_un.time))[0]
    time_index_un = np.column_stack((ind0, ind1))

    # fill temperature with low freq data
    shape = (len(data_un.time), len(data_un.heights))
    gradT_un_new = np.empty(shape)
    gradT_un_new[:] = np.nan
    meanT_un_new = np.empty(shape)
    meanT_un_new[:] = np.nan
    heights = data_un.heights.data
    # if heights of low freq and high freq are not the same you have to recall the functions at the right height
    # use a bool variable to avoid code repetition
    ricalculate = True
    if np.size(heights) == np.size(data_s_low.heights):
        if (heights == data_s_low.heights).all():
            ricalculate = True

    if ricalculate:
        """if LLJ_present: #use two different functions for unstable temp profile
            for i in range(len(time_index_un)):
                Functions = [temp_unstable_function_0, temp_unstable_function_0]
                Derivatives = [temp_unstable_derivative_0, temp_unstable_derivative_1]
                gradT_un_new[time_index_un[i][0], :] = Derivatives[LLJ_index[time_index_un[i][1]]] \
                    (heights, *fitresT_un[time_index_un[i][1]])
                meanT_un_new[time_index_un[i][0], :] = Functions[LLJ_index[time_index_un[i][1]]] \
                    (heights, *fitresT_un[time_index_un[i][1]])
        else:"""
        for i in range(len(time_index_un)):
            gradT_un_new[time_index_un[i][0], :] = temp_unstable_derivative_0(
                heights, *fitresT_un[time_index_un[i][1]]
            )
            meanT_un_new[time_index_un[i][0], :] = temp_unstable_function_0(
                heights, *fitresT_un[time_index_un[i][1]]
            )
    else:
        for i in range(len(time_index_un)):
            gradT_un_new[time_index_un[i][0], :] = gradT_un[time_index_un[i][1], :]
            meanT_un_new[time_index_un[i][0], :] = data_un_low.T.data[
                time_index_un[i][1], :
            ]

    # Richardson numbers
    Ri_un = 9.81 / meanT_un_new * gradT_un_new / gradU_un**2
    Rif_un = 9.81 / meanT_un_new * data_un.wT.data / gradU_un / data_un.uw.data

    # assign data to dataset
    data_un = data_un.assign(
        LLJ=(["time"], LLJ_index),
        meanT=(["time", "heights"], meanT_un_new),
        gradU=(["time", "heights"], gradU_un),
        gradT=(["time", "heights"], gradT_un_new),
        Ri=(["time", "heights"], Ri_un),
        Rif=(["time", "heights"], Rif_un),
    )

    if not keep_hf_T:
        data_un = data_un.assign(meanT=(["time", "heights"], meanT_un_new))

    if return_fit:
        return (
            data_s,
            data_un,
            [wind_stable_function_0, wind_stable_function_1],
            [wind_unstable_function_0, wind_unstable_function_1],
            fitresU_s,
            fitresU_un,
            [temp_stable_function_0, temp_stable_function_1],
            temp_unstable_function_0,
            fitresT_s,
            fitresT_un,
        )
    else:
        return data_s, data_un


def Gradients_calculation_splines(
    data_s,
    data_s_low,
    data_un,
    data_un_low,
    roughness_length=0,
    return_fit=False,
    keep_hf_T=False,
):
    """
    Calculates gradients for stable and unstable data, using low frequency temperature measurements
    """

    # Identify LLJ cases
    LLJ_present, LLJ_index = check_LLJ(data_s.meanU.data)

    # Call gradient function
    fitres_U_s, gradU_s = GS.Gradients_spline_old(
        data_s.heights.data,
        data_s.meanU.data,
        z0=roughness_length,
        LLJ=LLJ_index,
        model="wind",
    )
    fitres_T_s, gradT_s = GS.Gradients_spline_old(
        data_s_low.heights.data, data_s_low.T.data, LLJ=LLJ_index, model="temp"
    )
    fitres_U_un, gradU_un = GS.Gradients_spline_old(
        data_un.heights.data, data_un.meanU.data, z0=roughness_length, model="wind"
    )
    fitres_T_un, gradT_un = GS.Gradients_spline_old(
        data_un_low.heights.data, data_un_low.T.data, model="temp"
    )

    # recalculate temperature profile
    if np.array_equal(data_s.heights, data_s_low.heights):
        meanT_s = data_s_low.T.data
    else:
        meanT_s = np.empty(data_s.meanT.shape) * np.nan
        gradT_s = np.empty(data_s.meanT.shape) * np.nan
        for i in range(len(fitres_T_s)):
            if fitres_T_s[i] is not np.nan:
                meanT_s[i] = fitres_T_s[i](data_s.heights.data)
                gradT_s[i] = fitres_T_s[i].derivative()(data_s.heights.data)

    if np.array_equal(data_un.heights, data_un_low.heights):
        meanT_un = data_un_low.T.data
    else:
        meanT_un = np.empty(data_un.meanT.shape) * np.nan
        gradT_un = np.empty(data_un.meanT.shape) * np.nan
        for i in range(len(fitres_T_un)):
            if fitres_T_un[i] is not np.nan:
                meanT_un[i] = fitres_T_un[i](data_un.heights.data)
                gradT_un[i] = fitres_T_un[i].derivative()(data_un.heights.data)

    # Richardson numbers
    Ri_s = 9.81 / meanT_s * gradT_s / gradU_s**2
    Rif_s = 9.81 / meanT_s * data_s.wT.data / gradU_s / data_s.uw.data
    Ri_un = 9.81 / meanT_un * gradT_un / gradU_un**2
    Rif_un = 9.81 / meanT_un * data_un.wT.data / gradU_un / data_un.uw.data

    # assign data to dataset
    data_s = data_s.assign(
        LLJ=(["time"], LLJ_index),
        gradU=(["time", "heights"], gradU_s),
        gradT=(["time", "heights"], gradT_s),
        Ri=(["time", "heights"], Ri_s),
        Rif=(["time", "heights"], Rif_s),
    )

    if not keep_hf_T:
        data_s = data_s.assign(meanT=(["time", "heights"], meanT_s))

    data_un = data_un.assign(
        meanT=(["time", "heights"], meanT_un),
        gradU=(["time", "heights"], gradU_un),
        gradT=(["time", "heights"], gradT_un),
        Ri=(["time", "heights"], Ri_un),
        Rif=(["time", "heights"], Rif_un),
    )

    if not keep_hf_T:
        data_un = data_un.assign(meanT=(["time", "heights"], meanT_un))

    if return_fit:
        return data_s, data_un, fitres_U_s, fitres_U_un, fitres_T_s, fitres_T_un
    else:
        return data_s, data_un


def Gradients_calculation_multimethod(
    data_s,
    data_s_low,
    data_un,
    data_un_low,
    roughness_length=0,
    return_fit=False,
    keep_hf_T=False,
):
    """
    Calculates gradients for stable and unstable data, using analytical fitting for wind and splines for temperature
    """
    # Identify LLJ cases
    LLJ_present, LLJ_index = check_LLJ(data_s.meanU.data)

    # GRADIENTS
    # Analytical fitting for wind
    # select functions
    if roughness_length > 0:
        wind_functions = [
            partial(G.businger_rough, z0=roughness_length),
            partial(G.quadratic_rough, z0=roughness_length),
        ]
        wind_derivatives = [
            partial(G.D_businger_rough, z0=roughness_length),
            partial(G.D_quadratic_rough, z0=roughness_length),
        ]
        wind_bounds = [([-np.inf, 0], np.inf), ([-np.inf, -np.inf, 0], np.inf)]
    else:
        wind_functions = [G.businger, G.quadratic]
        wind_derivatives = [G.D_businger, G.D_quadratic]
        wind_bounds = [
            ([-np.inf, -np.inf, 0], np.inf),
            ([-np.inf, -np.inf, -np.inf, 0], np.inf),
        ]

    # Calculate gradients - stable
    if LLJ_present:
        fitres_U_s, gradU_s = G.MultiFunc_Gradient(
            data_s.meanU.data,
            data_s.heights.data,
            Functions=wind_functions,
            Derivatives=wind_derivatives,
            func_index=LLJ_index,
            bounds=wind_bounds,
            return_fit=True,
        )
    else:
        fitres_U_s, gradU_s = G.Gradient(
            data_s.meanU.data,
            data_s.heights.data,
            Function=wind_functions[0],
            Derivative=wind_derivatives[0],
            bounds=wind_bounds[0],
            return_fit=True,
        )

    # unstable
    fitres_U_un, gradU_un = G.Gradient(
        data_un.meanU.data,
        data_un.heights.data,
        Function=wind_functions[0],
        Derivative=wind_derivatives[0],
        bounds=wind_bounds[0],
        return_fit=True,
    )

    # splines fitting for temperature
    fitres_T_s, gradT_s = GS.Gradients_spline_old(
        data_s_low.heights.data, data_s_low.T.data, LLJ=LLJ_index, model="temp"
    )
    fitres_T_un, gradT_un = GS.Gradients_spline_old(
        data_un_low.heights.data, data_un_low.T.data, model="temp"
    )

    # recalculate temperature profile
    if np.array_equal(data_s.heights, data_s_low.heights):
        meanT_s = data_s_low.T.data
    else:
        meanT_s = np.empty(data_s.meanT.shape) * np.nan
        gradT_s = np.empty(data_s.meanT.shape) * np.nan
        for i in range(len(fitres_T_s)):
            if fitres_T_s[i] is not np.nan:
                meanT_s[i] = fitres_T_s[i](data_s.heights.data)
                gradT_s[i] = fitres_T_s[i].derivative()(data_s.heights.data)

    if np.array_equal(data_un.heights, data_un_low.heights):
        meanT_un = data_un_low.T.data
    else:
        meanT_un = np.empty(data_un.meanT.shape) * np.nan
        gradT_un = np.empty(data_un.meanT.shape) * np.nan
        for i in range(len(fitres_T_un)):
            if fitres_T_un[i] is not np.nan:
                meanT_un[i] = fitres_T_un[i](data_un.heights.data)
                gradT_un[i] = fitres_T_un[i].derivative()(data_un.heights.data)

    # Richardson numbers
    Ri_s = 9.81 / meanT_s * gradT_s / gradU_s**2
    Rif_s = 9.81 / meanT_s * data_s.wT.data / gradU_s / data_s.uw.data
    Ri_un = 9.81 / meanT_un * gradT_un / gradU_un**2
    Rif_un = 9.81 / meanT_un * data_un.wT.data / gradU_un / data_un.uw.data

    # assign data to dataset
    data_s = data_s.assign(
        LLJ=(["time"], LLJ_index),
        gradU=(["time", "heights"], gradU_s),
        gradT=(["time", "heights"], gradT_s),
        Ri=(["time", "heights"], Ri_s),
        Rif=(["time", "heights"], Rif_s),
    )

    if not keep_hf_T:
        data_s = data_s.assign(meanT=(["time", "heights"], meanT_s))

    data_un = data_un.assign(
        meanT=(["time", "heights"], meanT_un),
        gradU=(["time", "heights"], gradU_un),
        gradT=(["time", "heights"], gradT_un),
        Ri=(["time", "heights"], Ri_un),
        Rif=(["time", "heights"], Rif_un),
    )

    if not keep_hf_T:
        data_un = data_un.assign(meanT=(["time", "heights"], meanT_un))

    if return_fit:
        return (
            data_s,
            data_un,
            wind_functions,
            fitres_U_s,
            fitres_U_un,
            fitres_T_s,
            fitres_T_un,
        )
    else:
        return data_s, data_un


def Gradients_calculation_multimethod_1ds(
    data, data_low, roughness_length=0, return_fit=False, keep_hf_T=False
):
    # Identify LLJ cases
    LLJ_present, LLJ_index = check_LLJ(data.meanU.data)

    # GRADIENTS
    # Analytical fitting for wind
    # select functions
    if roughness_length > 0:
        wind_functions = [
            partial(G.businger_rough, z0=roughness_length),
            partial(G.quadratic_rough, z0=roughness_length),
        ]
        wind_derivatives = [
            partial(G.D_businger_rough, z0=roughness_length),
            partial(G.D_quadratic_rough, z0=roughness_length),
        ]
        wind_bounds = [([-np.inf, 0], np.inf), ([-np.inf, -np.inf, 0], np.inf)]
    else:
        wind_functions = [G.businger, G.quadratic]
        wind_derivatives = [G.D_businger, G.D_quadratic]
        wind_bounds = [
            ([-np.inf, -np.inf, 0], np.inf),
            ([-np.inf, -np.inf, -np.inf, 0], np.inf),
        ]

    # Calculate gradients - stable
    if LLJ_present:
        fitres_U, gradU = G.MultiFunc_Gradient(
            data.meanU.data,
            data.heights.data,
            Functions=wind_functions,
            Derivatives=wind_derivatives,
            func_index=LLJ_index,
            bounds=wind_bounds,
            return_fit=True,
        )
    else:
        fitres_U, gradU = G.Gradient(
            data.meanU.data,
            data.heights.data,
            Function=wind_functions[0],
            Derivative=wind_derivatives[0],
            bounds=wind_bounds[0],
            return_fit=True,
        )

    # splines fitting for temperature
    fitres_T, gradT = GS.Gradients_spline_old(
        data_low.heights.data, data_low.T.data, LLJ=LLJ_index, model="temp"
    )

    # recalculate temperature profile
    if np.array_equal(data.heights, data_low.heights):
        meanT = data_low.T.data
    else:
        meanT = np.empty(data.meanT.shape) * np.nan
        gradT = np.empty(data.meanT.shape) * np.nan
        for i in range(len(fitres_T)):
            if fitres_T[i] is not np.nan:
                meanT[i] = fitres_T[i](data.heights.data)
                gradT[i] = fitres_T[i].derivative()(data.heights.data)

    # Richardson numbers
    Ri = 9.81 / meanT * gradT / gradU**2
    Rif = 9.81 / meanT * data.wT.data / gradU / data.uw.data

    # assign data to dataset
    data = data.assign(
        LLJ=(["time"], LLJ_index),
        gradU=(["time", "heights"], gradU),
        gradT=(["time", "heights"], gradT),
        Ri=(["time", "heights"], Ri),
        Rif=(["time", "heights"], Rif),
    )

    if not keep_hf_T:
        data = data.assign(meanT=(["time", "heights"], meanT))

    if return_fit:
        return data, wind_functions, fitres_U, fitres_T
    else:
        return data

def Gradients_calculation_splines_1ds(
    data, data_low, roughness_length=0, return_fit=False, keep_hf_T=False, kw=4, kt=3
):
    # Identify LLJ cases
    LLJ_present, LLJ_index = check_LLJ(data.meanU.data)

    # GRADIENTS
    # Call gradient function
    fitres_U, gradU = GS.Gradients_spline(
        data.heights.data,
        data.meanU.data,
        z0=roughness_length,
        model="wind",
        k=kw
    )

    # splines fitting for temperature
    fitres_T, gradT = GS.Gradients_spline(
        data_low.heights.data, data_low.T.data, LLJ=LLJ_index, model="temp", k=kt
    )

    # recalculate temperature profile
    if np.array_equal(data.heights, data_low.heights):
        meanT = data_low.T.data
    else:
        meanT = np.empty(data.meanT.shape) * np.nan
        gradT = np.empty(data.meanT.shape) * np.nan
        for i in range(len(fitres_T)):
            if fitres_T[i] is not np.nan:
                meanT[i] = fitres_T[i](np.log(data.heights.data))
                gradT[i] = fitres_T[i].derivative()(np.log(data.heights.data))/data.heights.data

    # Richardson numbers
    Ri = 9.81 / meanT * gradT / gradU**2
    Rif = 9.81 / meanT * data.wT.data / gradU / data.uw.data

    # assign data to dataset
    data = data.assign(
        LLJ=(["time"], LLJ_index),
        gradU=(["time", "heights"], gradU),
        gradT=(["time", "heights"], gradT),
        Ri=(["time", "heights"], Ri),
        Rif=(["time", "heights"], Rif),
    )

    if not keep_hf_T:
        data = data.assign(meanT=(["time", "heights"], meanT))

    if return_fit:
        return data, fitres_U, fitres_T
    else:
        return data


def check_LLJ(wind_profiles):
    """
    Checks if a low level jet is present in the dataset
    returns boolean and where it is found (0 = noLLJ, 1 = LLJ)
    """
    # check type of data to avoid dataarray from xarray that produces fake result for lack of alignment
    if type(wind_profiles) == xr.core.dataarray.DataArray:
        raise TypeError("Xarray format not supported, pass attribute using .data")

    LLJ_index = np.zeros(len(wind_profiles))
    for i in range(len(wind_profiles)):
        # LLJ condition: the difference between one level and the one 2 levels below to be negative
        gapped_difference = wind_profiles[i, 2:] - wind_profiles[i, :-2]
        if (gapped_difference < 0).sum() > 1:
            LLJ_index[i] = 1

    LLJ_present = False
    if (LLJ_index == 1).any():
        LLJ_present = True

    return LLJ_present, LLJ_index.astype(int)


def Anisotropy_calculation(data_s, data_un=None, RGB=True):
    """Calculates the component of the barycentric map and the RGB color values"""

    barycentric_s, RGB_s = A.Anisotropy(data_s)


    if RGB:
        data_s = data_s.assign(
            yb=(["time", "heights"], barycentric_s[1]),
            xb=(["time", "heights"], barycentric_s[0]),
            RGB=(["time", "heights", "rgb"], RGB_s),
        )
    else:
        data_s = data_s.assign(
            yb=(["time", "heights"], barycentric_s[1]),
            xb=(["time", "heights"], barycentric_s[0])
        )
    if data_un is not None:
        barycentric_un, RGB_un = A.Anisotropy(data_un)
        
        if RGB:
            data_un = data_un.assign(
                yb=(["time", "heights"], barycentric_un[1]),
                xb=(["time", "heights"], barycentric_un[0]),
                RGB=(["time", "heights", "rgb"], RGB_un),
            )
        else:
            data_un = data_un.assign(
                yb=(["time", "heights"], barycentric_un[1]),
                xb=(["time", "heights"], barycentric_un[0])
            )
        return data_s, data_un
    else:
        return data_s


def Apply_final_QCs(data_s, data_un):

    # Stable Richardson
    stability = data_s.wT < 0
    richardson = (data_s.Rif < 0.25) & (data_s.Ri < 0.25)

    mask = stability & richardson
    data_s = data_s.where(mask, drop=True)

    fraction = np.sum(mask) / np.sum(stability)
    print(
        "Stable Richardson -> {:4.1f}%    ({:6.0f})".format(
            fraction.data * 100, np.sum(mask.data)
        )
    )

    # Unstable stationarity
    stability = data_un.wT > 0
    # daytime = day
    stationarity = (data_un.statU < 30) & (
        data_un.statWT < 30
    )  # & (data_un.statUW < 30)
    uncertainty = (data_un.WynU < 0.8) & (data_un.WynUW < 0.8) & (data_un.WynWT < 0.8)

    mask = stability & stationarity  # & daytime

    data_un = data_un.where(mask, drop=True)

    fraction = np.sum(mask) / np.sum(stability)
    print(
        "UNSTABLE stationarity -> {:4.1f}%    ({:6.0f})".format(
            fraction.data * 100, np.sum(mask.data)
        )
    )

    # COUNTERGRADIENTS
    # stable
    QC_T = data_s.gradT > 0
    QC_U = (data_s.gradU > 0) & (data_s.Rif > 0)

    fraction = np.sum(QC_T) / (np.sum(QC_T) + np.sum(np.invert(QC_T)))
    print(
        "Stable TCG -> {:4.1f}%    ({:6.0f})".format(
            fraction.data * 100, np.sum(QC_T).data
        )
    )

    fraction = np.sum(QC_U) / (np.sum(QC_U) + np.sum(np.invert(QC_U)))
    print(
        "Stable WCG -> {:4.1f}%    ({:6.0f})".format(
            fraction.data * 100, np.sum(QC_U).data
        )
    )

    data_s = data_s.assign(
        gradT=(["time", "heights"], data_s.where(QC_T).gradT.data),
        gradU=(["time", "heights"], data_s.where(QC_U).gradU.data),
    )

    # unstable
    QC_T = data_un.gradT < 0
    QC_U = (data_un.gradU > 0) & (data_un.Rif < 0)

    fraction = np.sum(QC_T) / (np.sum(QC_T) + np.sum(np.invert(QC_T)))
    print(
        "Unstable TCG -> {:4.1f}%    ({:6.0f})".format(
            fraction.data * 100, np.sum(QC_T).data
        )
    )

    fraction = np.sum(QC_U) / (np.sum(QC_U) + np.sum(np.invert(QC_U)))
    print(
        "Unstable WCG -> {:4.1f}%    ({:6.0f})".format(
            fraction.data * 100, np.sum(QC_U).data
        )
    )

    data_un = data_un.assign(
        gradT=(["time", "heights"], data_un.where(QC_T).gradT.data),
        gradU=(["time", "heights"], data_un.where(QC_U).gradU.data),
    )

    return data_s, data_un
