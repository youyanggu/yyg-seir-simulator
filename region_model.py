import datetime

import numpy as np

from fixed_params import *
import utils


def get_transition_sigmoid(inflection_day, rate_of_inflection, init_r_0, lockdown_r_0,
        check_values=True):
    """Returns a sigmoid function based on the specified parameters.

    A sigmoid helps smooth the transition between init_r_0 and lockdown_r_0,
        with the midpoint being inflection_day.
    rate_of_inflection is typically a value between 0-1, with 1 being a very steep
        transition. We typically use 0.2-0.5 in our projections.
    """
    if check_values:
        assert 0 < rate_of_inflection <= 1, rate_of_inflection
        assert 0 < init_r_0 <= 10, init_r_0
        assert 0 <= lockdown_r_0 <= 10, lockdown_r_0
    shift = inflection_day
    a = rate_of_inflection
    b = init_r_0 - lockdown_r_0
    c = lockdown_r_0
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
        """

        self.country_str = country_str
        self.region_str = region_str
        self.subregion_str = subregion_str
        self.first_date = first_date
        self.projection_create_date = projection_create_date
        self.projection_end_date = projection_end_date
        self.region_params = region_params
        self.actual_deaths_smooth = actual_deaths_smooth
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
            if k in ['INFLECTION_DAY', 'REOPEN_DATE']:
                assert v >= self.first_date, \
                    f'{k} {v} must be after first date {self.first_date}'
            setattr(self, k, v)
        assert self.REOPEN_DATE > self.INFLECTION_DAY, \
            f'reopen date {self.REOPEN_DATE} must be after inflection day {self.INFLECTION_DAY}'
        self.params_tups = params_tups
        self.post_reopening_r_decay = self.get_post_reopening_r_decay()
        self.fall_r_multiplier = self.get_fall_r_multiplier()
        self.R_0_ARR = self.build_r_0_arr()
        self.ifr_arr = self.build_ifr_arr()
        self.undetected_deaths_ratio_arr = self.build_undetected_deaths_ratio_arr()

    def get_post_reopening_r_decay(self):
        """Calculates the post-reopening R decay.

        Full description at https://covid19-projections.com/about/#post-reopening
            If there is no POST_REOPENING_R_DECAY parameter passed in, we use a random
            uniform distribution to generate the post-reopening ratio to model uncertainty.
        """

        if hasattr(self, 'POST_REOPENING_R_DECAY'):
            return self.POST_REOPENING_R_DECAY

        # we randomly sample from a triangular distribution to get the post_reopening_r_decay
        if hasattr(self, 'custom_post_reopening_r_decay_range'):
            low, mode, high = self.custom_post_reopening_r_decay_range
        elif self.country_str == 'US':
            low, mode, high = 0.993, 0.996, 0.999 # mean is 0.996
        elif self.country_str in EARLY_IMPACTED_COUNTRIES:
            low, mode, high = 0.995, 0.998, 0.999 # mean is ~0.9973
        elif self.has_us_seasonality():
            low, mode, high = 0.995, 0.9975, 1 # mean is ~0.9975
        else:
            low, mode, high = 0.996, 0.998, 1 # mean is ~0.998
        post_reopening_r_decay = np.random.triangular(low, mode, high)

        assert 0 < post_reopening_r_decay <= 1
        return post_reopening_r_decay

    def get_fall_r_multiplier(self):
        """We currently assume a minor uptick in R in the fall for seasonal countries.

        Full description at https://covid19-projections.com/about/#fall-wave
        """

        if not self.has_us_seasonality():
            return 1
        low, mode, high = 0.998, 1.001, 1.005 # mean is ~1.0013
        fall_r_mult = np.random.triangular(low, mode, high)

        return fall_r_mult

    def get_max_post_open_r(self):
        """Return the max post-open R depending on the region type.

        Country-wide projections have a lower post-open R due to lower variability.
        """

        if self.subregion_str:
            return MAX_POST_REOPEN_R + 0.1
        elif self.region_str != 'ALL' or self.country_str == 'US':
            return MAX_POST_REOPEN_R + 0.1
        elif self.country_str in COUNTRIES_WITH_NO_FIRST_WAVE:
            return MAX_POST_REOPEN_R * 2
        else:
            return MAX_POST_REOPEN_R

    def all_param_tups(self):
        """Returns all parameters as a tuple of (param_name, param_value) tuples."""
        all_param_tups = list(self.params_tups[:])
        for addl_param in ['post_reopening_r_decay', 'fall_r_multiplier']:
            all_param_tups.append((addl_param.upper(), getattr(self, addl_param)))
        return tuple(all_param_tups)

    def build_r_0_arr(self):
        """Returns an array of the reproduction numbers (R) for each day.

        Each element in the array represents a single day in the simulation.
            For example, if self.first_date is 2020-03-01 and self.projection_end_date
            is 2020-09-01, then R_0_ARR[10] would be the R value on 2020-03-11.

        Full description at: https://covid19-projections.com/about/#effective-reproduction-number-r
            and https://covid19-projections.com/model-details/#modeling-the-r-value

        We use three different R values: R0, post-mitigation R, and reopening R.
            We use an inverse logistic/sigmoid function to smooth the transition between
            the three R values.

        To compute the reopen R, we apply a multiplier REOPEN_R_MULT to the lockdown R.
            We map this multiplier to reopen_mult, which assumes greater growth if the
            initial lockdown R is effective.
            e.g. 10% growth for R=1->1.1, but 10% growth for R=0.7 -> (2-0.7)**0.5*1.1*.7 = 0.88
            reopen_mult becomes 1 at around R=1.17 (i.e. no increase on reopening)

            Sample code below to compare the difference:
                mult = 1.1
                for lockdown_r in np.arange(0.5,1.21,0.05):
                    orig_reopen_r = mult * lockdown_r
                    reopen_mult = max(1, (2-lockdown_r)**0.5*mult)
                    new_reopen_r = reopen_mult * lockdown_r
                    print(lockdown_r, orig_reopen_r, new_reopen_r)
        """

        assert 1 <= self.REOPEN_R_MULT <= 10, self.REOPEN_R_MULT
        reopen_mult = max(1, (2-self.LOCKDOWN_R_0)**0.5 * self.REOPEN_R_MULT)
        reopen_r = reopen_mult * self.LOCKDOWN_R_0
        max_post_open_r = self.get_max_post_open_r()
        post_reopening_r = min(max(max_post_open_r, self.LOCKDOWN_R_0), reopen_r)
        assert reopen_r >= self.LOCKDOWN_R_0, 'Reopen R must be >= lockdown R'
        assert 0.5 <= self.LOCKDOWN_FATIGUE <= 1.5, self.LOCKDOWN_FATIGUE

        reopen_date_shift = self.REOPEN_DATE + \
            datetime.timedelta(days=int(self.REOPEN_SHIFT_DAYS) + DEFAULT_REOPEN_SHIFT_DAYS)
        fatigue_idx = self.inflection_day_idx + DAYS_UNTIL_LOCKDOWN_FATIGUE
        reopen_idx = self.get_day_idx_from_date(reopen_date_shift)
        lockdown_reopen_midpoint_idx = (self.inflection_day_idx + reopen_idx) // 2

        if self.LOCKDOWN_R_0 <= 1:
            # we wait longer before applying the post-reopening decay to allow for
            # longer reopening time (since R_t <= 1)
            days_until_post_reopening = 30
        else:
            days_until_post_reopening = 15
        post_reopening_idx = reopen_idx + days_until_post_reopening
        fall_start_idx = self.get_day_idx_from_date(FALL_START_DATE_NORTH) - 30

        sig_lockdown = get_transition_sigmoid(
            self.inflection_day_idx, self.RATE_OF_INFLECTION, self.INITIAL_R_0, self.LOCKDOWN_R_0)
        sig_fatigue = get_transition_sigmoid(
            fatigue_idx, 0.2, 0, self.LOCKDOWN_FATIGUE-1, check_values=False)
        sig_reopen = get_transition_sigmoid(
            reopen_idx, 0.2, self.LOCKDOWN_R_0, post_reopening_r)

        dates = utils.date_range(self.first_date, self.projection_end_date)
        assert len(dates) == self.N

        # how much to drop post_reopening_r R to get to 1 (max 0.9)
        min_post_reopening_total_decay = min(0.9, 1 / post_reopening_r)

        R_0_ARR = [self.INITIAL_R_0]
        for day_idx in range(1, self.N):
            if day_idx < lockdown_reopen_midpoint_idx:
                r_t = sig_lockdown(day_idx)
            else:
                post_reopening_total_decay = fall_r_mult = 1

                if day_idx > post_reopening_idx:
                    assert day_idx > reopen_idx, day_idx
                    post_reopening_total_decay = max(
                        min_post_reopening_total_decay,
                        self.post_reopening_r_decay**(day_idx-post_reopening_idx))
                assert 0 < post_reopening_total_decay <= 1, post_reopening_total_decay

                if day_idx > fall_start_idx:
                    fall_r_mult = max(0.9, min(
                        1.2, self.fall_r_multiplier**(day_idx-fall_start_idx)))
                assert 0.9 <= fall_r_mult <= 1.2, fall_r_mult

                r_t = sig_reopen(day_idx) * post_reopening_total_decay * fall_r_mult

            r_t *= 1 + sig_fatigue(day_idx)

            # Make sure R is stable
            if day_idx > reopen_idx and abs(r_t / R_0_ARR[-1] - 1) > 0.1:
                assert False, f'R changed too quickly: {day_idx} {R_0_ARR[-1]} -> {r_t} {R_0_ARR}'

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

        ifr_arr = []
        for idx in range(self.N):
            if self.country_str in EARLY_IMPACTED_COUNTRIES:
                # Begin lowering IFR after 30 days due to improving treatments/lower age distribution
                total_days_with_mult = max(0, idx - 30)
            else:
                # slower rise in other countries, so we use 120 days
                total_days_with_mult = max(0, idx - 120)

            if self.country_str == 'US':
                # We differentiate between pre/post reopening for US
                # Post-reopening has a greater reduction in the IFR
                days_after_reopening = max(0, min(30, idx - (self.reopen_idx + DAYS_BEFORE_DEATH)))
                days_else = max(0, total_days_with_mult - days_after_reopening)

                ifr_mult = max(MIN_MORTALITY_MULTIPLIER,
                    MORTALITY_MULTIPLIER**days_else * MORTALITY_MULTIPLIER_US_REOPEN**days_after_reopening)
            else:
                ifr_mult = max(MIN_MORTALITY_MULTIPLIER, MORTALITY_MULTIPLIER**total_days_with_mult)
            assert 0 < MIN_MORTALITY_MULTIPLIER < 1, MIN_MORTALITY_MULTIPLIER
            assert MIN_MORTALITY_MULTIPLIER <= ifr_mult <= 1, ifr_mult
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
            many countries in Latin America and Africa do not have widespread testing,
            hence the may have a higher undetected deaths ratio.

        For more info: https://covid19-projections.com/about/#undetected-deaths
        """
        if not USE_UNDETECTED_DEATHS_RATIO:
            return list(np.zeros(self.N))

        init_undetected_deaths_ratio = 1
        if self.country_str in EARLY_IMPACTED_COUNTRIES:
            days_until_min_undetected = 45
            min_undetected = 0.05
        else:
            # slower testing ramp-up for later-impacted countries
            days_until_min_undetected = 120
            min_undetected = 0.1

        daily_step = (init_undetected_deaths_ratio - min_undetected) / days_until_min_undetected
        assert daily_step >= 0, daily_step

        undetected_deaths_ratio_arr = []
        for idx in range(self.N):
            undetected_deaths_ratio = max(
                min_undetected, init_undetected_deaths_ratio - daily_step * idx)
            assert 0 <= undetected_deaths_ratio <= 1, undetected_deaths_ratio
            undetected_deaths_ratio_arr.append(undetected_deaths_ratio)

        return undetected_deaths_ratio_arr

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

    def __str__(self):
        return f'{self.country_str} | {self.region_str} | {self.subregion_str}'

