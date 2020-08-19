"""Script to run simulations using the YYG/C19Pro SEIR model.

Learn more at: https://github.com/youyanggu/yyg-seir-simulator. Developed by Youyang Gu.

Sample usage: `python run_simulation.py -v --best_params_dir best_params/latest --country US --region CA`
For help: `python run_simulation.py --help`
"""

import argparse
import datetime
import glob
import json
import os

import numpy as np

from region_model import RegionModel
from simulation import run
from utils import str_to_date, remove_space_region


def load_best_params_from_file(best_params_dir, country, region=None, subregion=None):
    """Returns a dictionary that contains parameters for a specified region.

    Parameters
    ----------
    best_params_dir : str
        The directory where best_params are located
    country, region, subregion : str
        Specify which region we want to get the best_params for. Examples:
            country=US, region=None, subregion=None -> US
            country=US, region=CA, subregion=None -> California, US
            country=US, region=CA, subregion=Los Angeles -> Los Angeles County, California, US
            country=Canada, region=ALL, subregion=Ontario -> Ontario, Canada
    """

    assert os.path.isdir(best_params_dir), f'best_params directory does not exist: {best_params_dir}'
    assert country, 'Need to specify country to load params from file (region/subregion optional)'

    country_ = remove_space_region(country)
    region_ = remove_space_region(region)
    subregion_ = remove_space_region(subregion)

    if subregion_:
        if country_ == 'US':
            assert region_, 'need to provide state for US subregion'
            best_params_fname_search = f'{best_params_dir}/subregion/*{country_}_{region_}_{subregion_}.json'
        else:
            best_params_fname_search = f'{best_params_dir}/subregion/*{country_}_{subregion_}.json'
    elif country_ != 'US':
        assert region_ == 'ALL', f'region not supported for non-US countries: {region_}'
        best_params_fname_search = f'{best_params_dir}/global/*{country_}_ALL.json'
    else:
        if region_:
            best_params_fname_search = f'{best_params_dir}/*US_{region_}.json'
        else:
            best_params_fname_search = f'{best_params_dir}/*US_ALL.json'
    best_params_fnames = glob.glob(best_params_fname_search)

    assert len(best_params_fnames) > 0, f'File not found: {best_params_fname_search}'
    assert len(best_params_fnames) == 1, f'Multiple files: {best_params_fnames}'
    best_params_fname = best_params_fnames[0]

    print(f'Loading params file: {best_params_fname}')
    with open(best_params_fname) as f:
        best_params = json.load(f)

    return best_params


def convert_mean_params_to_params_dict(mean_params):
    """Convert list of [param_name, param_value_raw] pairs to dict of param_name to param_value.

    We also convert string dates to datetime objects

    Parameters
    ----------
    mean_params : list
        list of [param_name, param_value_raw] pairs
    """

    params_dict = {}
    for param_name, param_value_raw in mean_params:
        try:
            # attempt to convert to datetime.date object if it is a date
            params_dict[param_name] = str_to_date(param_value_raw)
        except (TypeError, ValueError):
            params_dict[param_name] = param_value_raw

    return params_dict


def convert_str_value_to_correct_type(param_value, old_value, use_timedelta=False):
    """Convert param_value to the same type as old_value."""

    for primitive_type in [bool, int, float]:
        if isinstance(old_value, primitive_type):
            return primitive_type(param_value)

    if isinstance(old_value, datetime.date):
        if use_timedelta:
            return datetime.timedelta(days=int(param_value))
        return str_to_date(param_value)

    raise NotImplementedError(f'Unknown type for value: {type(old_value)}')


def main(args):
    country = args.country
    region = args.region
    subregion = args.subregion
    skip_hospitalizations = args.skip_hospitalizations
    quarantine_perc = args.quarantine_perc
    quarantine_effectiveness = args.quarantine_effectiveness
    verbose = args.verbose

    if country != 'US' and not region:
        region = 'ALL'

    best_params_type = args.best_params_type
    assert best_params_type in ['mean', 'median', 'top', 'top10'], best_params_type

    if args.best_params_dir:
        # Load parameters from file
        best_params = load_best_params_from_file(args.best_params_dir, country, region, subregion)
        simulation_start_date = str_to_date(best_params['first_date'])
        simulation_create_date = str_to_date(best_params['date'])
        simulation_end_date = str_to_date(best_params['projection_end_date'])

        region_params = {'population' : best_params['population']}
        # mean_params, median_params, top_params, or top10_params
        params_type_name = f'{best_params_type}_params'
        if verbose:
            print('best params type:', best_params_type)
        params_dict = convert_mean_params_to_params_dict(best_params[params_type_name])
    else:
        """
        You can hard code your own parameters if you do not want to use the preset parameters.

        This can be especially useful for regions/countries where we do not have projections.

        Then simply run `python run_simulation.py -v` to use these parameters.
        """

        simulation_start_date = datetime.date(2020,2,1)
        simulation_create_date = datetime.date.today() # not used so can also be None
        simulation_end_date = datetime.date(2020,10,1)

        region_params = {'population' : 332000000}
        params_dict = {
            'INITIAL_R_0' : 2.24,
            'LOCKDOWN_R_0' : 0.9,
            'INFLECTION_DAY' : datetime.date(2020,3,18),
            'RATE_OF_INFLECTION' : 0.25,
            'LOCKDOWN_FATIGUE' : 1.,
            'DAILY_IMPORTS' : 500,
            'MORTALITY_RATE' : 0.01,
            'REOPEN_DATE' : datetime.date(2020,5,20),
            'REOPEN_SHIFT_DAYS': 0,
            'REOPEN_R' : 1.2,
            'REOPEN_INFLECTION' : 0.3,
            'POST_REOPEN_EQUILIBRIUM_R' : 1.,
            'FALL_R_MULTIPLIER' : 1.001,
        }

    if args.simulation_start_date:
        simulation_start_date = str_to_date(args.simulation_start_date)
    if args.simulation_end_date:
        simulation_end_date = str_to_date(args.simulation_end_date)

    if args.set_param:
        print('---------------------------------------')
        print('Overwriting params from command line...')
        for param_name, param_value in args.set_param:
            assert param_name in params_dict, f'Unrecognized param: {param_name}'
            old_value = params_dict[param_name]
            new_value = convert_str_value_to_correct_type(param_value, old_value)
            print(f'Setting {param_name} to: {new_value}')
            params_dict[param_name] = new_value

    if args.change_param:
        print('---------------------------------------')
        print('Changing params from command line...')
        for param_name, value_change in args.change_param:
            assert param_name in params_dict, f'Unrecognized param: {param_name}'
            old_value = params_dict[param_name]
            new_value = old_value + convert_str_value_to_correct_type(
                value_change, old_value, use_timedelta=True)
            print(f'Changing {param_name} from {old_value} to {new_value}')
            params_dict[param_name] = new_value

    region_model = RegionModel(country, region, subregion,
        simulation_start_date, simulation_create_date, simulation_end_date, region_params,
        compute_hospitalizations=(not skip_hospitalizations))

    if quarantine_perc > 0:
        print(f'Quarantine percentage: {quarantine_perc:.0%}')
        print(f'Quarantine effectiveness: {quarantine_effectiveness:.0%}')
        assert quarantine_effectiveness in [0.025, 0.1, 0.25, 0.5], \
            ('must specify --quarantine_effectiveness percentage.'
                ' Possible values: [0.025, 0.1, 0.25, 0.5]')
        quarantine_effectiveness_to_reduction_idx = {0.025: 0, 0.1: 1, 0.25: 2, 0.5: 3}
        region_model.quarantine_fraction = quarantine_perc
        region_model.reduction_idx = \
            quarantine_effectiveness_to_reduction_idx[quarantine_effectiveness]

    if verbose:
        print('================================')
        print(region_model)
        print('================================')
        print('Parameters:')
        for param_name, param_value in params_dict.items():
            print(f'{param_name:<25s} : {param_value}')

    # Add params to region_model
    params_tups = tuple(params_dict.items())
    region_model.init_params(params_tups)

    if verbose:
        print('--------------------------')
        print('Running simulation...')
        print('--------------------------')

    # Run simulation
    dates, infections, hospitalizations, deaths = run(region_model)

    """
    The following are lists with length N, where N is the number of days from
        simulation_start_date to simulation_end_date.

    dates            : datetime.date objects representing day i
    infections       : number of new infections on day i
    hospitalizations : occupied hospital beds on day i
    deaths           : number of new deaths on day i
    """
    assert len(dates) == len(infections) == len(hospitalizations) == len(deaths)
    assert dates[0] == simulation_start_date
    assert dates[-1] == simulation_end_date

    if verbose:
        infections_total = infections.cumsum()
        deaths_total = deaths.cumsum()
        for i in range(len(dates)):
            hospitalization_str = ''
            if not skip_hospitalizations:
                hospitalization_str = f'Hospital beds in use: {hospitalizations[i]:,.0f} - '
            daily_str = (f'{i+1:<3} - {dates[i]} - '
                f'New / total infections: {infections[i]:,.0f} / {infections_total[i]:,.0f} - '
                f'{hospitalization_str}'
                f'New / total deaths: {deaths[i]:,.2f} / {deaths_total[i]:,.1f} - '
                f'Mean R: {region_model.effective_r_arr[i]:.3f} - '
                f'IFR: {region_model.ifr_arr[i]:.2%}')
            print(daily_str) # comment out to spare console buffer
    print('-------------------------------------')
    print(f'End of simulation       : {region_model.projection_end_date}')
    print(f'Total infections        : {infections.sum():,.0f}')
    if not skip_hospitalizations:
        print(f'Peak hospital beds used : {hospitalizations.max():,.0f}')
    print(f'Total deaths            : {deaths.sum():,.0f}')

    if args.save_csv_fname:
        dates_str = np.array(list(map(str, dates)))
        combined_arr = np.vstack((dates_str, infections, hospitalizations, deaths,
            region_model.effective_r_arr)).T
        headers = 'dates,infections,hospitalizations,deaths,mean_r_t'
        np.savetxt(args.save_csv_fname, combined_arr, '%s', delimiter=',', header=headers)
        print('----------\nSaved file to:', args.save_csv_fname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Script to run simulations using the YYG/C19Pro SEIR model. Example: '
        '`python run_simulation.py -v --best_params_dir best_params/latest --country US --region CA`'))
    parser.add_argument('--skip_hospitalizations', action='store_true',
        help=('Skip the calculation of the number of occupied hospital beds.'
            ' Note that we have a very basic hospitalization heuristic, so exercise caution if you use it.'
            ' We skip hospitalizations in our production model to improve performance.'))
    parser.add_argument('--quarantine_perc', type=float, default=0,
        help=('percentage of people we put in quarantine (e.g. 0.5 = 50%% quarantine) (default is 0).'
            ' We do not use this in production.'))
    parser.add_argument('--quarantine_effectiveness', type=float, default=-1,
        help=('if --quarantine_perc is set, this is the percent reduction in transmission after quarantine.'
            'For example, 0.5 means a 50%% reduction in transmission. Valid values: 0.025, 0.1, 0.25, 0.5.'))
    parser.add_argument('--save_csv_fname',
        help='output csv file to save data')

    parser.add_argument('--simulation_start_date',
        help=('Set the start date of the simulation.'
            'This will override any existing values (Format: YYYY-MM-DD)'))
    parser.add_argument('--simulation_end_date',
        help=('Set the end date of the simulation.'
            'This will override any existing values (Format: YYYY-MM-DD)'))

    parser.add_argument('--best_params_dir',
        help='if passed, will load parameters from file based on the country, region, subregions')
    parser.add_argument('--best_params_type', default='mean',
        choices=['mean', 'median', 'top', 'top10'],
        help='we save four types of params for each region (default mean)')
    parser.add_argument('--set_param', action='append', nargs=2,
        help=('Takes two inputs, the name of the parameter and its value'))
    parser.add_argument('--change_param', action='append', nargs=2,
        help=('Takes two inputs, the name of the parameter and the amount to increase/decrease'))
    parser.add_argument('--country',
        help='only necessary if loading params from --best_params_dir')
    parser.add_argument('--region', default='',
        help='only necessary if loading params from --best_params_dir')
    parser.add_argument('--subregion', default='',
        help='only necessary if loading params from --best_params_dir')

    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    np.random.seed(0) # make results reproducible

    print('====================================================')
    print('YYG/C19PRO Simulator')
    print('Current time:', datetime.datetime.now())
    print('====================================================')

    main(args)

    print('====================================================')
    print('Done - Current time:', datetime.datetime.now())

