from dash import html, dcc
import dash_bootstrap_components as dbc
from components.modal import create_modal

def create_layout(args):
    return html.Div([
        html.H1("Spectrogram Labeling Tool"),
        dcc.Loading(id='loading', type='default', children=[
            html.Div(id='page-content'),
        ]),
        dcc.Store(id='file-data', storage_type='session'),
        dcc.Store(id='current-page', data=0, storage_type='session'),
        dcc.Store(id='current-filename', storage_type='session'),
        dcc.Store(id='colormap-store', data='default', storage_type='session'),
        html.Div([
            dbc.Button("Previous", id='prev-page', n_clicks=0, color='primary'),
            dbc.Button("Next", id='next-page', n_clicks=0, color='primary', className='ml-2'),
            dbc.Input(
                id='page-input',
                type='number',
                min=1,
                step=1,
                style={'width': '80px', 'display': 'inline-block', 'margin': '0 10px'}
            ),
            dbc.Button("Go", id='go-to-page', n_clicks=0, color='primary', size='sm'),
            html.Span(id='page-number', style={'margin-left': '20px'})
        ], style={'text-align': 'center', 'margin-top': '20px'}),
        html.Div([
            dbc.Switch(
                id='global-colormap-toggle',
                label='Use Hydrophone Colormap',
                value=False,
                className='custom-control-input'
            ),
        ], style={
            'position': 'fixed',
            'top': '20px',
            'right': '20px',
            'zIndex': '1000',
            'backgroundColor': 'white',
            'padding': '10px',
            'borderRadius': '5px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),
        create_modal(),
        html.Div(id='dummy-output', style={'display': 'none'})
    ])