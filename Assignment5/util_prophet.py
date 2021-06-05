import wbdata
import copy
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import datetime as dt
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from fbprophet.plot import plot_plotly

def identify_freq_and_periods(df, years_to_predict):
    
    day_diff = (df['ds'].iloc[1]-df['ds'].iloc[0]).days
    
    if day_diff>0:
        if day_diff>6:
            if day_diff>25:
                if day_diff>85:
                    if day_diff > 360:
                        freq='y'
                    else:
                        freq='q'
                else:
                    freq='m'
            else:
                freq='w'
        else:
            freq = 'd'

    else:
        raise ValueError('Do Not Recognize Freq')
        
    periods = {
        'y':years_to_predict,
        'm':12*years_to_predict,
        'w':52*years_to_predict,
        'q':4*years_to_predict,
        'd':365*years_to_predict
    }[freq]
    
    return freq, periods

def convert_to_celsius(x):
    return (x-32) * 5.0/9
    
def convert_to_celsius_exp(x):
    return pd.np.exp((x-32) * 5.0/9)

def get_all_gw_data(ref_date=pd.datetime.today(), emissions=True, fp=None):
    '''
    Parameters
    ----------
    ref_date : str/datetime
        reference date to limit future data
    emissions : bool
        load emissions data
    fp : None/str
        if provided, load from filepath instead of from
        internet
    '''
    
    if fp != None:
        edf = pd.read_csv(os.path.join(fp, 'emissions.csv'))
        tempdf = pd.read_csv(os.path.join(fp, 'temp.csv'))
        co2df = pd.read_csv(os.path.join(fp, 'co2.csv'))
    else:
        edf = load_emissions(ref_date=ref_date)
        tempdf = load_temp(ref_date=ref_date)
        co2df = load_co2(ref_date=ref_date)
        
    comb=pd.DataFrame()
    
    if emissions:
        df_list = [edf, tempdf, co2df]
    else:
        df_list = [tempdf, co2df]
    
    for d in df_list:
        if comb.empty:
            
            d['ds'] = pd.to_datetime(d['ds'])
            
            comb = d
            min_date = d['ds'].min()
            max_date = d['ds'].max()
        else:
            d['ds'] = pd.to_datetime(d['ds'])
            
            max_date = min(pd.to_datetime(max_date), d['ds'].max())
            min_date = max(pd.to_datetime(min_date), d['ds'].min())            
            
            comb = pd.merge_asof(comb, d, on='ds')

    dr = pd.DataFrame(pd.date_range(comb['ds'].min(), comb['ds'].max(), freq='m'),columns=['ds'])
    comb = comb.merge(dr, on='ds', how='outer').sort_values('ds')
    
    if emissions:
        comb['emissions'] = pd.to_numeric(comb['emissions'])

    for col in comb.columns:
        if col != 'ds':
            comb[col] = pd.to_numeric(comb[col])

    comb['temp'] = comb['temp'].interpolate()
    comb['co2'] = comb['co2'].interpolate()
    comb = comb.dropna().reset_index(drop=True)
    comb = comb[(comb['ds']>=min_date)&(comb['ds']<=max_date)]
    return comb

def plot_plotly_modified(
    m, 
    fcst, 
    test_x=[], 
    test_y=[], 
    lbl_list=['Actual Test'], 
    color_list=[], 
    size_list=[], 
    transform=None,
    transform_test=False,
    uncertainty=True,
    plot_cap=True,
    trend=False,
    changepoints=False,
    changepoints_threshold=0.01,
    xlabel='ds',
    ylabel='y',
    figsize=(900, 600)
):
    '''
    Parameters
    ----------
    m : fbprophet model
        model
    fcst : fbprophet forecast
        forecast
    test_x: None/pandas series
        Test data to overlay x value
    test_y : None/pandas series
        Test data to overlay y value
    transform : None/function
        If provided, transform the y values using this function
    '''
    
    m = copy.deepcopy(m)
    fcst = copy.deepcopy(fcst)
    
    if transform != None:
        m.history['y'] = transform(m.history['y'])
        fcst['yhat_lower'] = transform(fcst['yhat_lower'])
        fcst['yhat'] = transform(fcst['yhat'])
        fcst['yhat_upper'] = transform(fcst['yhat_upper'])
        
        if 'cap' in fcst:
            fcst['cap'] = transform(fcst['cap'])
            
        if 'floor' in fcst:
            fcst['floor'] = transform(fcst['floor'])
            
        if 'trend' in fcst:
            fcst['trend'] = transform(fcst['trend'])
        
    
    py.init_notebook_mode()
    fig1 = plot_plotly(
        m, 
        fcst, 
        uncertainty=uncertainty,
        plot_cap=plot_cap,
        trend=trend,
        changepoints=changepoints,
        changepoints_threshold=changepoints_threshold,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize
    )
    default_colors = ['red', 'orange', 'blue','green','purple', 'brown','grey','black']
    
    if type(test_x) != list: test_x = [test_x]
    if type(test_y) != list: test_y = [test_y]
        
    if type(color_list) in [str]: color_list = [color_list]
    if type(lbl_list) in [str]: lbl_list = [lbl_list]
    if type(size_list) in [str, int, float]: size_list = [size_list]
    
    if len(test_x)>0 and len(test_y) > 0:
        fig2 = go.Figure()
        
        trace_list = []
        for i in range(0, len(test_x)):
            
            if len(lbl_list)<i:
                lbl = 'trace_{}'.format(i)
            else:
                lbl = lbl_list[i]
            
            if len(color_list)>i:
                clr = color_list[i]
                default_colors = [i for i in default_colors if i != clr]
            else:                
                clr = default_colors[0]
                default_colors = [i for i in default_colors if i != clr]
                
            if len(size_list)>i:
                sz = float(size_list[i])
            else:
                sz = 4
                
            x = test_x[i]
            y = test_y[i]
            
            if transform_test and transform != None:
                y = pd.Series(y).apply(lambda j: transform(j))
                
            
            trace = go.Scatter(
                x=x, 
                y=y,
                name=lbl,
                mode='markers', 
                marker=dict(color=clr, size=sz),
            )
            
            trace_list.append(trace)

        fig2.add_traces(trace_list+[i for i in fig1.data])
        fig2.layout.update(fig1.layout)
        py.iplot(fig2)
    else:
        py.iplot(fig1)

def smape(actual, prediction):
    '''
    Parameters
    ----------
    actual : list/pandas series
        list of actual results
    prediction : list/pandas series
        list of predictions
    
    Returns
    -------
    sMAPE : float
        value <= 200 where lower
        means that there is less symmetric
        mean absolute percentage error
    '''
    
    # to clean data and prevent div-0 errors
    if type(actual) is pd.Series:
        actual = actual.values
        
    if type(prediction) is pd.Series:
        prediction = prediction.values
    
    n = len(actual)
    actual = pd.Series(actual)
    prediction = pd.Series(prediction)
    bool_iid = actual == prediction
    
    actual = actual[~bool_iid]
    prediction = prediction[~bool_iid]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    return 200.0/n * ((actual - prediction).abs()/(actual+prediction)).sum()

def load_temp(ref_date=pd.datetime.today()):
    '''
    Parameters
    ----------
    ref_date : str/datetime/None
        Date to cutoff results at the tail.
    
    Returns
    -------
    pandas dataframe of temperature data from the NASA
    '''
    temp = pd.read_csv('https://data.giss.nasa.gov/gistemp/tabledata_v3/GLB.Ts.csv')
    temp = temp.reset_index()
    temp.columns = temp.iloc[0]
    temp = temp.iloc[1:].set_index('Year')
    temp = temp.T.iloc[0:12].T

    df_list = []
    for iid, i in temp.iteritems():
        temp_df = pd.DataFrame(i)
        temp_df.rename(columns={iid: 'temp'}, inplace=True)
        temp_df['month'] = iid
        df_list.append(temp_df)

    temp = pd.concat(df_list).replace('***','').replace('', pd.np.nan)
    temp['ds'] = pd.to_datetime(temp.index+'-'+temp['month']+'-01')
    temp.sort_values('ds', inplace=True)
    temp.set_index('ds', inplace=True)
    temp.drop(['month'], axis=1, inplace=True)
    temp['temp'] = pd.to_numeric(temp['temp'])
    temp.fillna(method='ffill', inplace=True)
    temp.index = pd.to_datetime(temp.index)
    temp = temp[temp.index<= pd.to_datetime(pd.datetime.today().date())].reset_index()
    
    # to get into actual units
    temp['temp']+= 14*9/5+32
    temp[temp['ds']<=pd.to_datetime(ref_date)]
    return temp

def load_emissions(ref_date=pd.datetime.today()):
    '''
    Parameters
    ----------
    ref_date : str/datetime/None
        Date to cutoff results at the tail.
    
    Returns
    -------
    pandas dataframe of emissions data from the world
    bank. https://data.worldbank.org/indicator/EN.ATM.CO2E.KT
    '''
    edf = wbdata.get_data('EN.ATM.CO2E.KT')
    edf = pd.DataFrame(
        [(i['date'], i['value']) for i in edf if i['country']['value']=='World'], 
        columns=['year','emissions']
    ).sort_values(['year'])
    edf['ds'] = pd.to_datetime(edf['year']+'-01-01')
    edf.drop(['year'], axis=1, inplace=True)
    edf['emissions'] = pd.to_numeric(edf['emissions'])
    edf[edf['ds']<=pd.to_datetime(ref_date)]
    return edf

def load_co2(ref_date=pd.datetime.today()):
    '''
    Parameters
    ----------
    ref_date : str/datetime/None
        Date to cutoff results at the tail.
    
    Returns
    -------
    pandas dataframe of CO2 data where ds
    is dates in months and co2 is float
    parts per million. Data is from
    http://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/in_situ_co2/monthly/
    '''
    
    
    ref_date = pd.to_datetime(ref_date)
    
    df_co2 = pd.read_table('http://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/in_situ_co2/monthly/monthly_in_situ_co2_mlo.csv', header=None)
    co2ser = df_co2[0]
    co2ser = co2ser[co2ser[co2ser.str.replace(' ','')==''].index.max()+1:]
    co2df = pd.DataFrame([tuple([j.strip() for j in i]) for i in co2ser.str.split(',')])

    try:
        header_col = co2df[(co2df=='').max(axis=1)].index.max()
    except:
        header_col = 0

    headers = co2df.iloc[0:header_col+1]
    cols = (headers+' ').sum().str.replace('  ',' ').str.strip().tolist()

    co2df = co2df.iloc[header_col+1:]
    co2df.columns= cols
    co2df = co2df.replace('-99.99',pd.np.nan)
    co2df['ds'] = pd.to_datetime(co2df['Yr']+'-'+co2df['Mn']+'-15')
    co2df = co2df[co2df['ds']<=ref_date]

    co2df.rename(columns={'CO2 [ppm]':'co2'}, inplace=True)
    co2df = co2df[['ds','co2']]
    co2df['co2'] = pd.to_numeric(co2df['co2'])
    co2df = co2df.interpolate().dropna()
    co2df['ds'] = pd.to_datetime(co2df['ds'])
    co2df.reset_index(drop=True, inplace=True)
    return co2df