"""Underlying simulator for the YYG/C19Pro SEIR model.

Learn more at: https://github.com/youyanggu/yyg-seir-simulator. Developed by Youyang Gu.
"""

import datetime

import numpy as np

from fixed_params import *
from utils import inv_sigmoid


def get_daily_imports(region_model, i):
    """Returns the number of new daily imported cases based on day index i (out of N days).

    - beginning_days_flat is how many days at the beginning we maintain a constant import.
    - end_days_offset is the number of days from the end of the projections
        before we get 0 new imports.
    - The number of daily imports is initially region_model.DAILY_IMPORTS, and
        decreases linearly until day N-end_days_offset.
    """

    N = region_model.N
    assert i < N, 'day index must be less than total days'

    if hasattr(region_model, 'beginning_days_flat'):
        beginning_days_flat = region_model.beginning_days_flat
    else:
        beginning_days_flat = 10
    assert beginning_days_flat >= 0

    if hasattr(region_model, 'end_days_offset'):
        end_days_offset = region_model.end_days_offset
    else:
        end_days_offset = int(N - min(N, DAYS_WITH_IMPORTS))
    assert beginning_days_flat + end_days_offset <= N
    n_ = N - beginning_days_flat - end_days_offset + 1

    daily_imports = region_model.DAILY_IMPORTS * \
        (1 - min(1, max(0, (i-beginning_days_flat+1)) / n_))

    if region_model.country_str not in ['China', 'South Korea', 'Australia'] and not \
            hasattr(region_model, 'end_days_offset'):
        # we want to maintain ~10 min daily imports a day
        daily_imports = max(daily_imports, min(10, 0.1 * region_model.DAILY_IMPORTS))

    return daily_imports


def get_immunity_mult(region_model):
    """Returns the immunity multiplier, a measure of the immunity in a region.

    The greater the immunity multiplier, the greater the effect of immunity.
    The more populous a region, the greater the effect of immunity (since outbreaks
        are usually localized)
    Later on, use this multiplier by multiplying the transmission rate by:
        effective_r = R_t * (1-perc_population_infected_thus_far)**immunity_mult
    """

    population = region_model.region_params['population']
    if region_model.country_str == 'US':
        if region_model.subregion_str:
            immunity_mult = IMMUNITY_MULTIPLIER_US_SUBREGION
        else:
            immunity_mult = IMMUNITY_MULTIPLIER
    elif region_model.subregion_str:
        immunity_mult = IMMUNITY_MULTIPLIER
    elif population < 20000000:
        immunity_mult = inv_sigmoid(
            10000000, 0.0000003, IMMUNITY_MULTIPLIER-1, 1)(population)
    else:
        immunity_mult = inv_sigmoid(
            20000000, 0.00000003, IMMUNITY_MULTIPLIER-1.25, 1.25)(population)

    if region_model.country_str not in EARLY_IMPACTED_COUNTRIES:
        # these countries may not have comprehensive early testing, hence we
        # correct for that by increasing the immunity mult
        immunity_mult *= 1.1

    return immunity_mult


def run(region_model):
    """Given a RegionModel object, runs the SEIR simulation."""
    dates = np.array([region_model.first_date + datetime.timedelta(days=i) \
        for i in range(region_model.N)])
    infections = np.array([0.] * region_model.N)
    hospitalizations = np.zeros(region_model.N) * np.nan
    deaths = np.array([0.] * region_model.N)
    mortaility_rates = np.array([region_model.MORTALITY_RATE] * region_model.N)

    assert infections.dtype == hospitalizations.dtype == \
        deaths.dtype == mortaility_rates.dtype == np.float64

    """
    We compute a normalized version of the infections and deaths probability distribution.
    We invert the infections and deaths norm to simplify the convolutions we will take later.
        Aka the beginning of the array is the farther days out in the convolution.
    """
    deaths_norm = DEATHS_DAYS_ARR[::-1] / DEATHS_DAYS_ARR.sum()
    infections_norm = INFECTIOUS_DAYS_ARR[::-1] / INFECTIOUS_DAYS_ARR.sum()
    if hasattr(region_model, 'quarantine_fraction'):
        # reduce infections in the latter end of the infectious period, based on reduction_idx
        infections_norm[:region_model.reduction_idx] = \
            infections_norm[:region_model.reduction_idx] * (1 - region_model.quarantine_fraction)
        infections_norm[region_model.reduction_idx] = \
            (infections_norm[region_model.reduction_idx] * 0.5) + \
            (infections_norm[region_model.reduction_idx] * 0.5 * \
                (1 - region_model.quarantine_fraction))

    # the greater the immunity mult, the greater the effect of immunity
    immunity_mult = get_immunity_mult(region_model)
    assert 0 <= IMMUNITY_MULTIPLIER <= 1, IMMUNITY_MULTIPLIER
    assert 0 <= IMMUNITY_MULTIPLIER_US_SUBREGION <= 1, IMMUNITY_MULTIPLIER_US_SUBREGION
    assert 0 <= immunity_mult <= 2, immunity_mult

    ########################################
    # Compute infections
    ########################################
    effective_r_arr = []
    for i in range(region_model.N):
        if i < INCUBATION_DAYS+len(infections_norm):
            # initialize infections
            infections[i] = region_model.DAILY_IMPORTS
            effective_r_arr.append(region_model.R_0_ARR[i])
            continue

        perc_population_infected_thus_far = \
            min(1., infections[:i-1].sum() / region_model.population)
        assert 0 <= perc_population_infected_thus_far <= 1, perc_population_infected_thus_far

        r_immunity_perc = (1. - perc_population_infected_thus_far)**immunity_mult
        effective_r = region_model.R_0_ARR[i] * r_immunity_perc
        # we apply a convolution on the infections norm array
        s = (infections[i-INCUBATION_DAYS-len(infections_norm)+1:i-INCUBATION_DAYS+1] * \
            infections_norm).sum() * effective_r
        infections[i] = s + get_daily_imports(region_model, i)
        effective_r_arr.append(effective_r)

    region_model.perc_population_infected_final = perc_population_infected_thus_far
    assert len(region_model.R_0_ARR) == len(effective_r_arr) == region_model.N
    region_model.effective_r_arr = effective_r_arr

    ########################################
    # Compute hospitalizations
    ########################################
    if region_model.compute_hospitalizations:
        """
        Simple estimation of hospitalizations by taking the sum of a
            window of n days of new infections * hospitalization rate
        Note: this represents hospital beds used on on day _i, not new hospitalizations
        """
        for _i in range(region_model.N):
            start_idx = max(0, _i-DAYS_UNTIL_HOSPITALIZATION-DAYS_IN_HOSPITAL)
            end_idx = max(0, _i-DAYS_UNTIL_HOSPITALIZATION)
            hospitalizations[_i] = int(HOSPITALIZATION_RATE * infections[start_idx:end_idx].sum())

    ########################################
    # Compute deaths
    ########################################
    assert len(deaths_norm) % 2 == 1, 'deaths arr must be odd length'
    deaths_offset = len(deaths_norm) // 2
    for _i in range(-deaths_offset, region_model.N-DAYS_BEFORE_DEATH):
        # we apply a convolution on the deaths norm array
        infections_subject_to_death = (infections[max(0, _i-deaths_offset):_i+deaths_offset+1] * \
            deaths_norm[:min(len(deaths_norm), deaths_offset+_i+1)]).sum()
        true_deaths = infections_subject_to_death * region_model.ifr_arr[_i + DAYS_BEFORE_DEATH]
        reported_deaths = true_deaths * \
            (1 - region_model.undetected_deaths_ratio_arr[_i + DAYS_BEFORE_DEATH])
        deaths[_i + DAYS_BEFORE_DEATH] = reported_deaths

    return dates, infections, hospitalizations, deaths

