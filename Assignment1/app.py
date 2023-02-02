from dash import Dash, html, dcc, dash_table, Input, Output, State
import refinitiv.dataplatform.eikon as ek
import pandas as pd
import numpy as np
from dash import dcc
from datetime import datetime
import plotly.express as px
import os

ek.set_app_key(os.getenv('eikon_api'))

dt_prc_div_splt = pd.read_csv('unadjusted_price_history.csv')


spacer = html.Div(style={'margin': '10px','display':'inline'})

app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.Label(children="benchmark: "),
        dcc.Input(id = 'benchmark-id', type = 'text', value="IVV",placeholder="eg.IVV"),
        spacer,
        html.Label(children="ticker symbol: "),
        dcc.Input(id = 'asset-id', type = 'text', value="AAPL.O",placeholder="eg.AAPL.O"),
        spacer,
        html.Label(children="start date: "),
        dcc.Input(id = 'start-date',type ='text',value ="2022-01-01",placeholder="YYYY-MM-DD"),
        spacer,
        html.Label(children="end date: "),
        dcc.Input(id = 'end-date',type='text',value= datetime.now().strftime("%Y-%m-%d"),placeholder="YYYY-MM-DD"),
        spacer
    ]),
    html.Button('QUERY Refinitiv', id = 'run-query', n_clicks = 0),
    html.H2('Raw Data from Refinitiv'),
    dash_table.DataTable(
        id = "history-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.H2('Historical Returns'),
    dash_table.DataTable(
        id = "returns-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.H2('Alpha & Beta Scatter Plot'),
    html.Div([
        html.Label(children="graph start date: "),
        dcc.Input(id='graph-start-date', type='text', value="2022-01-01", placeholder="graph start date"),
        spacer,
        html.Label(children="graph end date: "),
        dcc.Input(id='graph-end-date', type='text', value=datetime.now().strftime("%Y-%m-%d"), placeholder="graph end date"),


    ]),
    html.Button('RUN AB Plot', id='run-ab-plot', n_clicks=0),
    dcc.Graph(id="ab-plot"),
    dash_table.DataTable(
        id = "ab-tbl",
        page_action = 'none',
        style_table ={'heights':'100px','widths':'500px'}
    ),
    html.P(id='summary-text', children="")

])

@app.callback(
    Output("history-tbl", "data"),
    Input("run-query", "n_clicks"),
    [State('benchmark-id', 'value'), State('asset-id', 'value'),State('start-date','value'),State('end-date','value')],
    prevent_initial_call=True
)
def query_refinitiv(n_clicks, benchmark_id, asset_id,start_date,end_date):
    assets = [benchmark_id, asset_id,start_date,end_date]
    prices, prc_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            'SDate': start_date,
            'EDate': datetime.now().strftime("%Y-%m-%d"),
            'Frq': 'D'
        }
    )

    divs, div_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.DivExDate',
            'TR.DivUnadjustedGross',
            'TR.DivType',
            'TR.DivPaymentType'
        ],
        parameters={
            'SDate': start_date,
            'EDate': datetime.now().strftime("%Y-%m-%d"),
            'Frq': 'D'
        }
    )

    splits, splits_err = ek.get_data(
        instruments=assets,
        fields=['TR.CAEffectiveDate', 'TR.CAAdjustmentFactor'],
        parameters={
            "CAEventType": "SSP",
            'SDate': start_date,
            'EDate': datetime.now().strftime("%Y-%m-%d"),
            'Frq': 'D'
        }
    )

    prices.rename(
        columns={
            'Open Price': 'open',
            'High Price': 'high',
            'Low Price': 'low',
            'Close Price': 'close'
        },
        inplace=True
    )
    prices.dropna(inplace=True)
    prices['Date'] = pd.to_datetime(prices['Date']).dt.date

    divs.rename(
        columns={
            'Dividend Ex Date': 'Date',
            'Gross Dividend Amount': 'div_amt',
            'Dividend Type': 'div_type',
            'Dividend Payment Type': 'pay_type'
        },
        inplace=True
    )
    divs.dropna(inplace=True)
    divs['Date'] = pd.to_datetime(divs['Date']).dt.date
    divs = divs[(divs.Date.notnull()) & (divs.div_amt > 0)]

    splits.rename(
        columns={
            'Capital Change Effective Date': 'Date',
            'Adjustment Factor': 'split_rto'
        },
        inplace=True
    )
    splits.dropna(inplace=True)
    splits['Date'] = pd.to_datetime(splits['Date']).dt.date

    unadjusted_price_history = pd.merge(
        prices, divs[['Instrument', 'Date', 'div_amt']],
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['div_amt'].fillna(0, inplace=True)

    unadjusted_price_history = pd.merge(
        unadjusted_price_history, splits,
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['split_rto'].fillna(1, inplace=True)

    if unadjusted_price_history.isnull().values.any():
        raise Exception('missing values detected!')

    return(unadjusted_price_history.to_dict('records'))


@app.callback(
    Output("returns-tbl", "data"),
    Input("history-tbl", "data"),
    prevent_initial_call = True
)
def calculate_returns(history_tbl):

    dt_prc_div_splt = pd.DataFrame(history_tbl)

    # Define what columns contain the Identifier, date, price, div, & split info
    ins_col = 'Instrument'
    dte_col = 'Date'
    prc_col = 'close'
    div_col = 'div_amt'
    spt_col = 'split_rto'

    dt_prc_div_splt[dte_col] = pd.to_datetime(dt_prc_div_splt[dte_col])
    dt_prc_div_splt = dt_prc_div_splt.sort_values([ins_col, dte_col])[
        [ins_col, dte_col, prc_col, div_col, spt_col]].groupby(ins_col)
    numerator = dt_prc_div_splt[[dte_col, ins_col, prc_col, div_col]].tail(-1)
    denominator = dt_prc_div_splt[[prc_col, spt_col]].head(-1)

    pivot = pd.DataFrame({
        'Date': numerator[dte_col].reset_index(drop=True),
        'Instrument': numerator[ins_col].reset_index(drop=True),
        'rtn': np.log(
            (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                    denominator[prc_col] * denominator[spt_col]
            ).reset_index(drop=True)
        )
    }).pivot_table(
            values='rtn', index='Date', columns='Instrument'
        )
    pivot["Date"] = pd.to_datetime(pivot.index).strftime("%Y-%m-%d")#the original Date looks like this 2022-01-12T00:00:00, but we need 2022-01-12
    return(pivot.to_dict('records')
    )

@app.callback(
    Output("ab-plot", "figure"),
    Input("run-ab-plot","n_clicks"),
    Input("returns-tbl", "data"),
    [State('benchmark-id', 'value'), State('asset-id', 'value'), State('graph-start-date', 'value'),
     State('graph-end-date', 'value')],
    prevent_initial_call = True
)
def render_ab_plot(n_clicks,returns, benchmark_id, asset_id,start_date,end_date):

    filtered_returns = []
    returns_copy = returns.copy()
    for record in returns_copy:
        if (start_date<= record['Date']<=end_date):
            del record['Date']
            filtered_returns.append(record)
    return(
        px.scatter(filtered_returns, x=benchmark_id, y=asset_id, trendline='ols')
    )
@app.callback(
    Output("ab-tbl", "data"),
    Input("ab-plot","figure"),
    prevent_initial_call = True
)
def render_ab_tbl(ab_plot):
    # Ri = alpha - beta * Rm
    trend_line = ab_plot['data'][1]['hovertemplate']#string type,contains information of the beta and the risk free return rate
    regression_equation_elements_list = trend_line.split("<br>")[1].strip().split()#list type,contains[asset,'=',beta,'*',benchmark,'+',risk_free_return_rate]
    beta,alpha = float(regression_equation_elements_list[2]), float(regression_equation_elements_list[6])

    return (
        pd.DataFrame({'alpha':[alpha],'beta': [beta]}).to_dict('records')
    )


if __name__ == '__main__':
    app.run_server(debug=True)