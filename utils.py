import datetime

import numpy as np

from fixed_params import DATE_STR_FMT, DASH_REGIONS


def inv_sigmoid(shift=0, a=1, b=1, c=0):
    """Returns a inverse sigmoid function based on the parameters."""
    return lambda x: b * np.exp(-(a*(x-shift))) / (1 + np.exp(-(a*(x-shift)))) + c


def str_to_date(date_str, fmt=DATE_STR_FMT):
    """Convert string date to datetime object."""
    return datetime.datetime.strptime(date_str, fmt).date()


def date_range(start_date, end_date, interval=1, str_fmt=DATE_STR_FMT):
    """Returns range of datetime dates from start_date to end_date (inclusive).

    start_date/end_date can be either str (with format str_fmt) or datetime objects.
    """

    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, str_fmt).date()
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, str_fmt).date()
    return [start_date + datetime.timedelta(n) \
        for n in range(0, (end_date - start_date).days + 1, interval)]


def remove_space_region(region):
    return region.replace(' ', '-')


def add_space_region(region):
    if region in DASH_REGIONS:
        return region
