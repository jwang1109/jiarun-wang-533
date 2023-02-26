from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, date, timedelta
import plotly.express as px
import pandas as pd
from datetime import datetime
import numpy as np
import os
import refinitiv.dataplatform.eikon as ek
import refinitiv.data as rd

#####################################################

ek.set_app_key(os.getenv('EIKON_API'))

img_path = 'reactive_graph.jpeg'
spacer = html.Div(style={'margin': '10px', 'display': 'inline'})

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

percentage = dash_table.FormatTemplate.percentage(3)


def create_table(id):
    dash_table.DataTable(
        id=id,
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'})
    return dash_table.DataTable

controls = dbc.Card(
    [
        dbc.Row(html.Button('QUERY Refinitiv', id='run-query', n_clicks=0)),

        dbc.Row([
            html.H5('Asset:',
                    style={'display': 'inline-block', 'margin-right': 20}),
            dcc.Input(id='asset', type='text', value="IVV",
                      style={'display': 'inline-block',
                             'border': '1px solid black'}),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("α1"), html.Th("n1")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha1',
                                    type='number',
                                    value=-0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n1',
                                    type='number',
                                    value=3,
                                    min=1,
                                    step=1
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            ),

            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("α2"), html.Th("n2")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha2',
                                    type='number',
                                    value=0.01,
                                    max=1,
                                    min=-1,
                                    step=0.01
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n2',
                                    type='number',
                                    value=5,
                                    min=1,
                                    step=1
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            )
        ]),
        dbc.Row([
            dcc.DatePickerRange(
                id='raw-data-date-picker',
                min_date_allowed = date(2015, 1, 1),
                max_date_allowed = datetime.now(),
                start_date= date(2023, 1, 30)
                #datetime.date(
                #    datetime.now() - timedelta(days=3 * 365)
                #)
            ,
                end_date=date(2023, 2, 8)
                #datetime.now().date()

            )
        ]),
        spacer,
        dbc.Row(html.Button('Submit', id='run-strategy', n_clicks=0))
    ],
    body=True
)

app.layout = dbc.Container(
    [   html.H2('Group Members: Jiarun Wang: jw822, Boyuan Zeng: bz100, Yu Yan: yy360',style={'color':'#4169E1'}),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(
                    #reactive graph
                    html.Img(src=app.get_asset_url(path=img_path), style={'width': '100%'}),
                    md = 8
                )
            ],
            align="center",
        ),
        html.H2('Historical Data'),
        dash_table.DataTable(
            id="history-tbl",
            page_action='none',
            style_table={'height': '300px', 'overflowY': 'auto'}
        ),
        html.H2('Trade Blotter:'),
        spacer,
        dash_table.DataTable(
            id="orders",
            page_action='none',
            style_table={'height': '300px', 'overflowY': 'auto'}
        ),

    ],
    fluid=True
)


@app.callback(
    Output("history-tbl", "data"),
    Input("run-query", "n_clicks"),
    [State('asset', 'value'), State('raw-data-date-picker', 'start_date'),
     State('raw-data-date-picker', 'end_date')],
    prevent_initial_call=True
)
def query_refinitiv(n_clicks, asset, start_date, end_date):
    assets = [start_date, end_date, asset]
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
            'EDate': end_date,
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
    prices.drop(columns='Instrument', inplace=True)

    return (prices.to_dict('records'))

@app.callback(
    Output("orders", "data"),
    Input("run-strategy", "n_clicks"),
    Input("history-tbl", "data"),
    [State('asset','value'),State('alpha1', 'value'), State('n1', 'value'),State('alpha2','value'),State('n2','value')],
    prevent_initial_call=True
)
def render_blotter(n_clicks,history_tbl,asset,alpha1,n1,alpha2,n2):
    prices = pd.DataFrame(history_tbl)
    #1.Get the next business day from Refinitiv!!!!!!!
    prices['Date'] = pd.to_datetime(prices['Date']).dt.date
    rd.open_session()
    next_business_day = rd.dates_and_calendars.add_periods(
        start_date=prices['Date'].iloc[-1].strftime("%Y-%m-%d"),
        period="1D",
        calendars=["USA"],
        date_moving_convention="NextBusinessDay",
    )

    rd.close_session()
    #2. submitted entry orders
    submitted_entry_orders = pd.DataFrame({
        "trade_id": range(1, prices.shape[0]),
        "date": list(pd.to_datetime(prices["Date"].iloc[1:]).dt.date),
        "asset": str(asset),
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(
            prices['close'].iloc[:-1] * (1 + alpha1),
            2
        ),
        'status': 'SUBMITTED'
    })


    # 3. cancelled entry orders
    # if the lowest traded price is still higher than the price you bid, then the
    # order never filled and was cancelled.

    with np.errstate(invalid='ignore'):
        cancelled_entry_orders = submitted_entry_orders[
            np.greater(
                prices['low'].iloc[1:][::-1].rolling(n1).min()[::-1].to_numpy(),
                submitted_entry_orders['price'].to_numpy()
            )
        ].copy()
    cancelled_entry_orders.reset_index(drop=True, inplace=True)
    cancelled_entry_orders['status'] = 'CANCELLED'

    cancelled_entry_orders['date'] = pd.DataFrame(#get the correct cancel date
        {'cancel_date': submitted_entry_orders['date'].iloc[(n1 - 1):].to_numpy()},
        index=submitted_entry_orders['date'].iloc[:(1 - n1 )].to_numpy()
    ).loc[cancelled_entry_orders['date']]['cancel_date'].to_list()

    #4.filled_entry_orders
    filled_entry_orders = submitted_entry_orders[
        submitted_entry_orders['trade_id'].isin(
            list(
                set(submitted_entry_orders['trade_id']) - set(
                    cancelled_entry_orders['trade_id']
                )
            )
        )
    ].copy()
    filled_entry_orders.reset_index(drop=True, inplace=True)
    filled_entry_orders['status'] = 'FILLED'
    for i in range(0, len(filled_entry_orders)):
        idx1 = np.flatnonzero(
            prices['Date'] == filled_entry_orders['date'].iloc[i]
        )[0]

        slice1 = prices.iloc[idx1:(idx1 + n1)]['low']

        fill_inds = slice1 <= filled_entry_orders['price'].iloc[i]

        if (len(fill_inds) < n1) & (not any(fill_inds)):
            filled_entry_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_entry_orders.at[i, 'date'] = prices['Date'].iloc[
                fill_inds.idxmax()
            ]

    #5 issue exit order upon filled orders
    submitted_exit_orders = filled_entry_orders[filled_entry_orders['status']!='LIVE'].copy()
    submitted_exit_orders['trip'] = 'EXIT'
    submitted_exit_orders['action'] = "SELL"
    submitted_exit_orders['status'] = "SUBMITTED"
    submitted_exit_orders['price'] = submitted_exit_orders['price'] * (1 + alpha2)

    #6 cancel exit orders
    #if the exit order price is greater than the close price of today
    # and greater than the high price of the next n2 days, the order is cancelled
    with np.errstate(invalid='ignore'):
        maxi = list(np.maximum(
            prices['high'][1:][::-1].rolling(n2-1).max()[
            ::-1].to_numpy(), prices[:-1]['close']))
        maxi.append(np.nan)
        iloc = []
        for i in range(len(submitted_exit_orders)):
            row = submitted_exit_orders.iloc[i]
            if row['price']>maxi[list(prices['Date']).index(row['date'])]:
                iloc.append(i)
        cancelled_exit_orders = submitted_exit_orders.iloc[iloc].copy()
    cancelled_exit_orders.reset_index(drop=True, inplace=True)
    cancelled_exit_orders['status'] = 'CANCELLED'
    time_dic = pd.DataFrame(  # get the correct cancel date
        {'cancel_date': prices['Date'].iloc[(n2 - 1):].to_numpy()},
        index=prices['Date'].iloc[:(1 - n2)].to_numpy()
    )
    for i in list(cancelled_exit_orders.index.values):
        cancelled_exit_orders.at[i,'date'] = time_dic.loc[cancelled_exit_orders.at[i,'date']]['cancel_date']

    #7 issue market orders if the order is cancelled
    submitted_market_orders = cancelled_exit_orders.copy()
    submitted_market_orders['trip'] = 'EXIT'
    submitted_market_orders['type'] = 'MARKET'
    submitted_market_orders['action'] = "SELL"
    for i in list(submitted_market_orders.index.values):
        submitted_market_orders.at[i,'price'] = prices[prices['Date']==submitted_market_orders.at[i,'date']]['close']
    submitted_market_orders['status'] = 'SUBMITTED'
    #8 assume the market order is always filled
    filled_market_orders = submitted_market_orders.copy()
    filled_market_orders['status'] = 'FILLED'

    #9 filled exit order
    filled_exit_orders = submitted_exit_orders[
        submitted_exit_orders['trade_id'].isin(
            list(
                set(submitted_exit_orders['trade_id']) - set(
                    cancelled_exit_orders['trade_id']
                )
            )
        )
    ].copy()
    filled_exit_orders.reset_index(drop=True, inplace=True)
    filled_exit_orders['status'] = 'FILLED'
    for i in range(0, len(filled_exit_orders)):

        idx2 = np.flatnonzero(
            prices['Date'] == filled_exit_orders['date'].iloc[i]
        )[0]
        slice2 = prices.iloc[idx2:(idx2 + n2)]['high']
        slice2.iloc[0] = prices.iloc[idx2]['close']
        fill_inds = slice2 >= filled_exit_orders['price'].iloc[i]
        if (len(fill_inds) < n2) & (not any(fill_inds)):
            filled_exit_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_exit_orders.at[i, 'date'] = prices['Date'].iloc[
                fill_inds.idxmax()
            ]

    #10.live entry orders & live exit orders
    live_entry_orders = pd.DataFrame({
        "trade_id": prices.shape[0],
        "date": pd.to_datetime(next_business_day).date(),
        "asset": str(asset),
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(prices['close'].iloc[-1] * (1 + alpha1), 2),
        'status': 'LIVE'
    },
        index=[0]
    )

    if any(filled_entry_orders['status'] == 'LIVE'):
        live_entry_orders = pd.concat([
            filled_entry_orders[filled_entry_orders['status'] == 'LIVE'],
            live_entry_orders
        ])
        live_entry_orders['date'] = pd.to_datetime(next_business_day).date()

    filled_entry_orders = filled_entry_orders[
        filled_entry_orders['status'] == 'FILLED'
        ]



    if any(filled_exit_orders['status'] == 'LIVE'):
        live_exit_orders = pd.concat([
            filled_exit_orders[filled_exit_orders['status'] == 'LIVE'],
        ])
        live_exit_orders['date'] = pd.to_datetime(next_business_day).date()
        live_entry_orders = pd.concat([live_entry_orders, live_exit_orders])

    filled_exit_orders = filled_exit_orders[
        filled_exit_orders['status'] == 'FILLED'
        ]



    #11. Complete Orders
    orders = pd.concat(
        [
            submitted_entry_orders,
            cancelled_entry_orders,
            filled_entry_orders,
            submitted_exit_orders,
            cancelled_exit_orders,
            submitted_market_orders,
            filled_market_orders,
            filled_exit_orders,
            live_entry_orders,
        ]
    ).sort_values(['trade_id','date'])

    return orders.to_dict('records')


if __name__ == '__main__':
    app.run_server(debug=True)
