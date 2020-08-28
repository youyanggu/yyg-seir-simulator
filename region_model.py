import datetime

import numpy as np

from fixed_params import *
import utils


def get_transition_sigmoid(inflection_idx, inflection_rate, low_value, high_value,
        check_values=True):
    """Returns a sigmoid function based on the specified parameters.

    A sigmoid helps smooth the transition between low_value and high_value,
        with the midpoint being inflection_idx.
    inflection_rate is typically a value between 0-1, with 1 being a very steep
        transition. We typically use 0.2-0.5 in our projections.
    """
    if check_values:
        assert 0 < inflection_rate <= 1, inflection_rate
        assert 0 < low_value <= 10, low_value
        assert 0 <= high_value <= 10, high_value
    shift = inflection_idx
    a = inflection_rate
    b = low_value - high_value
    c = high_value
    return utils.inv_sigmoid(shift, a, b, c)


class RegionModel:
    """
    The main class to capture a region and its single set of parameters.

    This object is instantiated and then passed to our SEIR simulator to simulate
        infections, hospitalizations and deaths based on the internal parameters.
    """

    def __init__(self, country_str, region_str, subregion_str,
            first_date, projection_create_date,
            projection_end_date,
            region_params=dict(),
            actual_deaths_smooth=None,
            randomize_params=False,
            compute_hospitalizations=False):
        """
        Parameters
        ----------
        country_str : str
            Name of the country (e.g. US, Canada)
        region_str : str
            Name of the region (e.g. CA, DC)
        subregion_str : str
            Name of the subregion - county for US, provinces/states for international.
            (e.g. Los Angeles County, Alberta)
        first_date : datetime.date
            First date of the simulation
        projection_create_date : datetime.date
            The date when the projection is being generated.
            This date is usually present day, unless we are doing validation testing,
            in which case we use a day in the past so we can compare projections to OOS data.
        region_params : dict, optional
            Additional metadata for a region, such as population and hospital beds.
        actual_deaths_smooth : np.array, optional
            Smoothed version of the deaths.
        compute_hospitalizations : bool, optional
            Whether to compute hospitalization estimates (default False)
        randomize : bool, optional
            Given a parameter for certain inputs such as daily_imports, generate a random
                value from a uniform distribution with the value as the mean.
                This is used to add more variance when training the model.
        """

        self.country_str = country_str
        self.region_str = region_str
        self.subregion_str = subregion_str
        self.first_date = first_date
        self.projection_create_date = projection_create_date
        self.projection_end_date = projection_end_date
        self.region_params = region_params
        self.actual_deaths_smooth = actual_deaths_smooth
        self.randomize_params = randomize_params
        self.compute_hospitalizations = compute_hospitalizations

        self.country_holidays = None
        self.N = (self.projection_end_date - self.first_date).days + 1

        assert self.N > DAYS_BEFORE_DEATH, 'Need N to be at least DAYS_BEFORE_DEATH'
        if projection_create_date:
            assert first_date < projection_create_date, \
                'First date must be before projection create date'
            assert projection_create_date < projection_end_date, \
                'Projection create date must be before project end date'

    def init_params(self, params_tups):
        """Initializes the object by saving the parameters that are passed in.

        This function also builds the R values for each day of the simulation,
            as well as the IFR values for each day.
        Note: This must be called before running the simulation.

        Parameters
        ----------
        params_tups : tuple
            This is a tuple of (param_name, param_value) tuples.
            Example: (('INITIAL_R_0', 2.2), ('LOCKDOWN_R_0', 0.9), etc.)
        """

        assert isinstance(params_tups, tuple), 'must be a tuple of tuples'
        for k, v in params_tups:
            if k in DATE_PARAMS:
                assert v >= self.first_date, \
                    f'{k} {v} must be after first date {self.first_date}'
            setattr(self, k, v)
        assert self.REOPEN_DATE > self.INFLECTION_DAY, \
            f'reopen date {self.REOPEN_DATE} must be after inflection day {self.INFLECTION_DAY}'
        self.params_tups = params_tups
        assert set([i[0] for i in params_tups]).issubset(set(ALL_PARAMS)), 'Unknown params'

        # Set parameters, if not provided/randomized
        self.set_rate_of_inflection()
        self.set_daily_imports()
        self.set_post_reopen_equilibrium_r()
        self.set_fall_r_multiplier()

        # Set additional values necessary to run simulations
        self.immunity_mult = self.get_immunity_mult()
        self.R_0_ARR = self.build_r_0_arr()
        self.ifr_arr = self.build_ifr_arr()
        self.undetected_deaths_ratio_arr = self.build_undetected_deaths_ratio_arr()

    def all_param_tups(self):
        """Returns all parameters as a tuple of (param_name, param_value) tuples."""
        all_param_dict = dict(self.params_tups)
        for addl_param in RANDOMIZED_PARAMS + POTENTIAL_RANDOMIZE_PARAMS:
            all_param_dict[addl_param] = getattr(self, addl_param.lower())

        all_params = [(k, all_param_dict[k]) for k in ALL_PARAMS]
        return tuple(all_params)

    def get_reopen_r(self):
        """Compute the R value after reopening.

        We cap the post-reopen R value to prevent exponential growth skewing estimates.
        """
        if self.LOCKDOWN_R_0 < 1:
            return max(self.LOCKDOWN_R_0, self.REOPEN_R)
        return self.REOPEN_R

    def set_rate_of_inflection(self):
        """Calculate and set the rate of inflection for transition from R0 to lockdown R."""
        if self.randomize_params:
            low, high = self.RATE_OF_INFLECTION * 0.75, self.RATE_OF_INFLECTION * 1.25
            self.rate_of_inflection = np.random.uniform(low, high)
        else:
            self.rate_of_inflection = self.RATE_OF_INFLECTION

    def set_daily_imports(self):
        """Calculate and set daily imports to initialize a region's infections."""
        if self.randomize_params:
            low, high = self.DAILY_IMPORTS * 0.5, self.DAILY_IMPORTS * 1.5
            self.daily_imports = np.random.randint(low, high)
        else:
            self.daily_imports = self.DAILY_IMPORTS

    def set_post_reopen_equilibrium_r(self):
        """Calculate and set the post-reopen equilibrium R.

        This is the R value after reopening effects have stablized and infection rates
            reach an equilibrium. This does not take into account immunity,
            so the equilibrium R can be above 1 while true infection rates decrease.
        """
        if hasattr(self, 'POST_REOPEN_EQUILIBRIUM_R') and \
                not np.isnan(self.POST_REOPEN_EQUILIBRIUM_R):
            post_reopen_equilibrium_r = self.POST_REOPEN_EQUILIBRIUM_R
        else:
            if self.country_str == 'US':
                if self.region_str in ['AZ', 'FL', 'OH', 'HI']:
                    low, mode, high = 0.85, 0.95, 1.05
                elif self.REOPEN_R < 1.1:
                    low, mode, high = 0.85, 0.95, 1.05 # mean is 0.95
                else:
                    low, mode, high = 0.9, 1, 1.1 # mean is 1
            else:
                if self.country_str in ['Brazil', 'Mexico']:
                    low, mode, high = 1, 1.1, 1.2
                elif self.country_str in ['Australia']:
                    low, mode, high = 0.8, 0.9, 1 # mean is 0.9
                else:
                    low, mode, high = 0.85, 1, 1.15 # mean is 1
            post_reopen_equilibrium_r = np.random.triangular(low, mode, high)

        assert 0 < post_reopen_equilibrium_r < 10, post_reopen_equilibrium_r
        self.post_reopen_equilibrium_r = post_reopen_equilibrium_r

    def set_fall_r_multiplier(self):
        """Calculate and set the fall R multiplier.

        We currently assume a minor uptick in R in the fall for seasonal countries.
        Full description at https://covid19-projections.com/about/#fall-wave
        """

        if hasattr(self, 'FALL_R_MULTIPLIER') and not np.isnan(self.FALL_R_MULTIPLIER):
            fall_r_multiplier = self.FALL_R_MULTIPLIER
        elif not self.has_us_seasonality():
            fall_r_multiplier = 1
        else:
            low, mode, high = 0.997, 1.001, 1.005 # mean is 1.001
            fall_r_multiplier = np.random.triangular(low, mode, high)

        self.fall_r_multiplier = fall_r_multiplier

    def get_immunity_mult(self):
        """Returns the immunity multiplier, a measure of the immunity in a region.

        The greater the immunity multiplier, the greater the effect of immunity.
        We assume that the more populous a region, the greater the effect of immunity
            (since outbreaks are often localized)
        Later on, use this multiplier by multiplying the transmission rate by:
            effective_r = R_t * (1-perc_population_infected_thus_far)**immunity_mult
        """

        assert 0 <= IMMUNITY_MULTIPLIER <= 2, IMMUNITY_MULTIPLIER
        assert 0 <= IMMUNITY_MULTIPLIER_US_SUBREGION <= 2, IMMUNITY_MULTIPLIER_US_SUBREGION

        population = self.region_params['population']
        if self.country_str == 'US':
            if self.subregion_str:
                immunity_mult = IMMUNITY_MULTIPLIER_US_SUBREGION
            else:
                immunity_mult = IMMUNITY_MULTIPLIER
        elif self.subregion_str:
            immunity_mult = IMMUNITY_MULTIPLIER
        elif population < 20000000:
            immunity_mult = IMMUNITY_MULTIPLIER
        else:
            # immunity is between IMMUNITY_MULTIPLIER and 1
            immunity_mult = get_transition_sigmoid(
                50000000, 0.00000003, IMMUNITY_MULTIPLIER, 1, check_values=False)(population)

        return immunity_mult

    def build_r_0_arr(self):
        """Returns an array of the reproduction numbers (R) for each day.

        Each element in the array represents a single day in the simulation.
            For example, if self.first_date is 2020-03-01 and self.projection_end_date
            is 2020-09-01, then R_0_ARR[10] would be the R value on 2020-03-11.

        Full description at: https://covid19-projections.com/about/#effective-reproduction-number-r
            and https://covid19-projections.com/model-details/#modeling-the-r-value

        We use three different R values: R0, post-mitigation R, and reopen R.
            We use an inverse logistic/sigmoid function to smooth the transition between
            the three R values.
        """

        reopen_r = self.get_reopen_r()
        assert 0.5 <= self.LOCKDOWN_FATIGUE <= 1.5, self.LOCKDOWN_FATIGUE

        reopen_date_shift = self.REOPEN_DATE + \
            datetime.timedelta(days=int(self.REOPEN_SHIFT_DAYS) + DEFAULT_REOPEN_SHIFT_DAYS)
        fatigue_idx = self.inflection_day_idx + DAYS_UNTIL_LOCKDOWN_FATIGUE
        reopen_idx = self.get_day_idx_from_date(reopen_date_shift)
        lockdown_reopen_midpoint_idx = (self.inflection_day_idx + reopen_idx) // 2

        NUMERATOR_CONST = 6
        days_until_post_reopen = int(np.rint(NUMERATOR_CONST / self.REOPEN_INFLECTION))
        assert 10 <= days_until_post_reopen <= 80, days_until_post_reopen
        post_reopen_midpoint_idx = reopen_idx + days_until_post_reopen
        post_reopen_idx = reopen_idx + days_until_post_reopen * 2
        fall_start_idx = self.get_day_idx_from_date(FALL_START_DATE_NORTH) - 30

        sig_lockdown = get_transition_sigmoid(
            self.inflection_day_idx, self.rate_of_inflection, self.INITIAL_R_0, self.LOCKDOWN_R_0)
        sig_fatigue = get_transition_sigmoid(
            fatigue_idx, 0.2, 0, self.LOCKDOWN_FATIGUE-1, check_values=False)
        sig_reopen = get_transition_sigmoid(
            reopen_idx, self.REOPEN_INFLECTION, self.LOCKDOWN_R_0 * self.LOCKDOWN_FATIGUE, reopen_r)
        sig_post_reopen = get_transition_sigmoid(
            post_reopen_idx, self.REOPEN_INFLECTION, reopen_r, min(reopen_r, self.post_reopen_equilibrium_r))

        dates = utils.date_range(self.first_date, self.projection_end_date)
        assert len(dates) == self.N

        R_0_ARR = [self.INITIAL_R_0]
        for day_idx in range(1, self.N):
            if day_idx < lockdown_reopen_midpoint_idx:
                r_t = sig_lockdown(day_idx)
                if abs(self.LOCKDOWN_FATIGUE - 1) > 1e-9:
                    r_t *= 1 + sig_fatigue(day_idx)
            elif day_idx > post_reopen_midpoint_idx:
                r_t = sig_post_reopen(day_idx)
            else:
                r_t = sig_reopen(day_idx)

            if day_idx > fall_start_idx:
                fall_r_mult = max(0.9, min(
                    1.5, self.fall_r_multiplier**(day_idx-fall_start_idx)))
                assert 0.9 <= fall_r_mult <= 1.5, fall_r_mult
                r_t *= fall_r_mult

            # Make sure R is stable
            if day_idx > reopen_idx and abs(r_t / R_0_ARR[-1] - 1) > 0.2:
                assert False, \
                    f'{str(self)} - R changed too quickly: {day_idx} {R_0_ARR[-1]} -> {r_t} {R_0_ARR}'

            R_0_ARR.append(r_t)

        assert len(R_0_ARR) == self.N
        self.reopen_idx = reopen_idx

        return R_0_ARR

    def build_ifr_arr(self):
        """Returns an array of the infection fatality rates for each day.

        Each element in the array represents a single day in the simulation.
            For example, if self.first_date is 2020-03-01 and self.projection_end_date
            is 2020-09-01, then ifr_arr[10] would be the IFR on 2020-03-11.

        Full description at: https://covid19-projections.com/about/#infection-fatality-rate-ifr
        """
        assert 0.9 <= MORTALITY_MULTIPLIER <= 1.1, MORTALITY_MULTIPLIER
        assert 0 < self.MORTALITY_RATE < 0.2, self.MORTALITY_RATE

        min_mortality_multiplier = MIN_MORTALITY_MULTIPLIER
        if self.region_tuple in [] or \
                self.region_tuple[:2] in [('US', 'MA')]:
            min_mortality_multiplier = 0.5

        ifr_arr = []
        for idx in range(self.N):
            if self.country_str in EARLY_IMPACTED_COUNTRIES:
                # Begin lowering IFR after 30 days due to improving treatments/lower age distribution
                total_days_with_mult = max(0, idx - 30)
            else:
                # slower rise in other countries, so we use 120 days
                total_days_with_mult = max(0, idx - 120)

            if self.country_str in EARLY_IMPACTED_COUNTRIES:
                # Post-reopening has a greater reduction in the IFR
                days_after_reopening = max(0, min(30, idx - (self.reopen_idx + DAYS_BEFORE_DEATH // 2)))
                days_else = max(0, total_days_with_mult - days_after_reopening)

                ifr_mult = max(min_mortality_multiplier,
                    MORTALITY_MULTIPLIER**days_else * MORTALITY_MULTIPLIER_US_REOPEN**days_after_reopening)
            else:
                ifr_mult = max(min_mortality_multiplier, MORTALITY_MULTIPLIER**total_days_with_mult)
            assert 0 < min_mortality_multiplier < 1, min_mortality_multiplier
            assert min_mortality_multiplier <= ifr_mult <= 1, ifr_mult
            ifr = max(MIN_IFR, self.MORTALITY_RATE * ifr_mult)
            ifr_arr.append(ifr)

        return ifr_arr

    def build_undetected_deaths_ratio_arr(self):
        """Return an array of the percent of deaths that are undetected for each day.

        We assume the percentage of undetected deaths will be high in the initial days
            due to lack of testing, but will decrease until it reaches near 0. We assume
            a floor of 5-10% of undetected deaths. So if there are 100 true deaths and
            20% are undetected, then only 80 deaths will be reported/projected.
            While the true undetected deaths ratio can vary from region to region,
            note that the exact value does not signficantly affect our projections.

        You can customize this function to set a higher undetected ratio for
            different countries depending on their testing progress. For example,
            many developing countries not have widespread testing,
            hence they may have a higher undetected deaths ratio. The true
            undetected ratio may be even higher, but we leave that for a future study.

        For more info: https://covid19-projections.com/about/#undetected-deaths
        """
        if not USE_UNDETECTED_DEATHS_RATIO:
            return list(np.zeros(self.N))

        init_undetected_deaths_ratio = 1
        if self.country_str in HIGH_INCOME_COUNTRIES:
            days_until_min_undetected = 60
            min_undetected = 0.05
        elif self.country_str in ['Ecuador', 'India', 'Pakistan']:
            days_until_min_undetected = 120
            min_undetected = 0.5
        elif self.country_str in ['Indonesia', 'Peru', 'South Africa']:
            days_until_min_undetected = 120
            min_undetected = 0.25
        elif self.country_str in ['Brazil', 'Mexico', 'Russia']:
            days_until_min_undetected = 120
            min_undetected = 0.2
        else:
            days_until_min_undetected = 120
            min_undetected = 0.15

        daily_step = (init_undetected_deaths_ratio - min_undetected) / days_until_min_undetected
        assert daily_step >= 0, daily_step

        undetected_deaths_ratio_arr = []
        for idx in range(self.N):
            undetected_deaths_ratio = max(
                min_undetected, init_undetected_deaths_ratio - daily_step * idx)
            assert 0 <= undetected_deaths_ratio <= 1, undetected_deaths_ratio
            undetected_deaths_ratio_arr.append(undetected_deaths_ratio)

        return undetected_deaths_ratio_arr

    def get_reporting_delay_distribution(self):
        """Return a probabiliy distribution of the death reporting lag by day.

        For example, if index 3 is 0.1, it means that 10% of deaths are reported
            3 days after the date of death.
        """
        death_reporting_lag_arr = DEATH_REPORTING_LAG_ARR

        return death_reporting_lag_arr / death_reporting_lag_arr.sum()

    def get_day_idx_from_date(self, date):
        """Get the day index given a date.

        Parameters
        ----------
        date : datetime.date
        """
        return (date - self.first_date).days

    def get_date_from_day_idx(self, day_idx):
        """Get the date given the day index.

        Parameters
        ----------
        day_idx : int
        """
        return self.first_date + datetime.timedelta(days=day_idx)

    def is_holiday(self, date):
        """Determines if a date is a holiday.

        Parameters
        ----------
        date : datetime.date
        """
        if self.country_holidays is None:
            self.country_holidays = utils.get_holidays(self.country_str)

        if date in self.country_holidays:
            return True
        if self.country_str == 'US' and date in ADDL_US_HOLIDAYS:
            return True
        return False

    def has_us_seasonality(self):
        """Determines if the country has the same seasonality pattern as the US."""
        return self.country_str not in \
            SOUTHERN_HEMISPHERE_COUNTRIES + NON_SEASONAL_COUNTRIES

    @property
    def population(self):
        assert isinstance(self.region_params['population'], int), 'population must be an int'
        return self.region_params['population']

    @property
    def hospital_beds(self):
        return int(self.population / 1000 * self.region_params['hospital_beds_per_1000'])

    @property
    def inflection_day_idx(self):
        return self.get_day_idx_from_date(self.INFLECTION_DAY)

    @property
    def region_tuple(self):
        return (self.country_str, self.region_str, self.subregion_str)

    def __str__(self):
        return f'{self.country_str} | {self.region_str} | {self.subregion_str}'

