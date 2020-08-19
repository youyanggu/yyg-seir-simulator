import datetime

import numpy as np

PROJECTION_END_DATE = datetime.date(2020,11,1)
INCUBATION_DAYS = 2 # minimal incubation period of 2 days
INFECTIOUS_DAYS_ARR = np.array([0.5,1,2,3,2,1,0.5]) # distribution of infections by days after INCUBATION_DAYS; mean serial interval of 5 days
DEATHS_DAYS_ARR = np.array([1,2,3,4,4,3,3,3,3,3,2,2,2,2,2]) # distribution of deaths by days after exposure, centered around DAYS_BEFORE_DEATH
DEATH_REPORTING_LAG_ARR = np.array([0] + [15*0.85**i for i in range(30)]) # ~55% of deaths are reported within 5 days and ~80% within 10 days
MORTALITY_MULTIPLIER = 0.995 # decreasing IFR over time: https://covid19-projections.com/about/#infection-fatality-rate-ifr
MORTALITY_MULTIPLIER_US_REOPEN = 0.975 # faster rate of IFR decrease in the US after reopening
MIN_MORTALITY_MULTIPLIER = 0.3 # for a 0.995 mortality mutliplier, this kicks in after ~3.5 months for 0.75, ~7.5 months for 0.4
MIN_IFR = 0.002
DAYS_UNTIL_HOSPITALIZATION = 12 # Hospitalization parameters from many sources, including: https://doi.org/10.1016/S0140-6736(20)30566-3
HOSPITALIZATION_RATE = 0.02 # based on CDC report as of 2020-05-20: https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html
DAYS_IN_HOSPITAL = 11
DAYS_BEFORE_DEATH = 22 # 22 after exposure, from https://doi.org/10.3390/jcm9020538
IMMUNITY_MULTIPLIER = 0.5
IMMUNITY_MULTIPLIER_US_SUBREGION = 0.3
DAYS_UNTIL_LOCKDOWN_FATIGUE = 28
DAYS_WITH_IMPORTS = 100
USE_UNDETECTED_DEATHS_RATIO = True
DEFAULT_REOPEN_SHIFT_DAYS = 15
DATE_PARAMS = ['INFLECTION_DAY', 'REOPEN_DATE']
RANDOMIZED_PARAMS = ['POST_REOPEN_EQUILIBRIUM_R', 'FALL_R_MULTIPLIER'] # does not randomize if value exists
POTENTIAL_RANDOMIZE_PARAMS = ['RATE_OF_INFLECTION', 'DAILY_IMPORTS'] # if randomize flag, then randomize even if value exists

DATE_STR_FMT = '%Y-%m-%d'
ADDL_US_HOLIDAYS = [datetime.date(2020,4,12), datetime.date(2020,5,10), datetime.date(2020,10,31)] # Eastern, Mother's Day, Halloween
FALL_START_DATE_NORTH = datetime.date(2020,9,22)

ALL_PARAMS = ['INITIAL_R_0', 'LOCKDOWN_R_0', 'INFLECTION_DAY', 'RATE_OF_INFLECTION', 'LOCKDOWN_FATIGUE', 'DAILY_IMPORTS',
    'MORTALITY_RATE', 'REOPEN_DATE', 'REOPEN_SHIFT_DAYS', 'REOPEN_R', 'REOPEN_INFLECTION', 'POST_REOPEN_EQUILIBRIUM_R', 'FALL_R_MULTIPLIER']

#################
# Countries
#################
EU_COUNTRIES = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark',
    'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
    'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands',
    'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden'
]
LATIN_AMERICA_COUNTRIES = [
    'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Cuba', 'Dominican Republic',
    'Ecuador', 'Honduras', 'Mexico', 'Panama', 'Peru',
]
AFRICAN_COUNTRIES = ['Algeria', 'Egypt', 'Morocco', 'Nigeria', 'South Africa']
ASIAN_COUNTRIES = ['Bangladesh', 'China', 'Iran', 'Israel', 'Japan', 'Indonesia', 'India', 'Kuwait',
    'Malaysia', 'Pakistan', 'Philippines', 'Russia', 'Saudi Arabia', 'South Korea', 'Turkey',
    'United Arab Emirates']
EUROPEAN_COUNTRIES = EU_COUNTRIES + [
    'United Kingdom', 'Switzerland', 'Norway',
    'Belarus', 'Iceland', 'Moldova', 'Serbia', 'Ukraine']
OTHER_COUNTRIES = ['Australia', 'Canada']

ADDL_COUNTRIES_SUPPORTED = EUROPEAN_COUNTRIES + LATIN_AMERICA_COUNTRIES + \
    AFRICAN_COUNTRIES + ASIAN_COUNTRIES + OTHER_COUNTRIES
ALL_COUNTRIES = ADDL_COUNTRIES_SUPPORTED + ['US']

DASH_REGIONS = ['Miami-Dade']
NON_SEASONAL_COUNTRIES = ['Indonesia', 'Philippines', 'India', 'Malaysia', 'Nigeria',
    'Bolivia', 'Colombia', 'Cuba', 'Dominican Republic', 'Ecuador', 'Honduras', 'Panama', 'Peru', 'Brazil']
SOUTHERN_HEMISPHERE_COUNTRIES = ['Argentina', 'Australia', 'Chile', 'South Africa']
HIGH_INCOME_EUROPEAN_COUNTRIES = ['Iceland', 'Norway', 'Switzerland', 'United Kingdom'] + \
    [c for c in EU_COUNTRIES if c not in ['Bulgaria']]
HIGH_INCOME_COUNTRIES = ['US', 'Australia', 'Canada', 'Chile', 'Israel', 'Japan', 'South Korea',
    'Kuwait', 'Panama', 'Saudi Arabia', 'United Arab Emirates'] + HIGH_INCOME_EUROPEAN_COUNTRIES
EARLY_IMPACTED_COUNTRIES = ['US', 'Canada', 'China', 'Japan', 'South Korea', 'Israel', 'Iran'] + EUROPEAN_COUNTRIES
