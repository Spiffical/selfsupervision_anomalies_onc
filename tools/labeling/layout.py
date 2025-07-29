from dash import html, dcc
import dash_bootstrap_components as dbc
from components.modal import create_modal

def create_layout(args):
    return html.Div([
        # Main container
        html.Div([
            # Header section
            html.Div([
                html.H1("Spectrogram Labeling Tool", className="main-title")
            ], className="header-section"),
            
            # Controls panel
            html.Div([
                html.Div([
                    dbc.Button("← Previous", id='prev-page', n_clicks=0, 
                             className='btn-custom btn-primary-custom me-2'),
                    dbc.Button("Next →", id='next-page', n_clicks=0, 
                             className='btn-custom btn-primary-custom me-3'),
                    html.Div([
                        html.Label("Go to page:", className="me-2", 
                                 style={'font-weight': '500', 'color': '#495057'}),
                        dbc.Input(
                            id='page-input',
                            type='number',
                            min=1,
                            step=1,
                            className='page-input me-2',
                            style={'width': '80px', 'display': 'inline-block'}
                        ),
                        dbc.Button("Go", id='go-to-page', n_clicks=0, 
                                 className='btn-custom btn-secondary-custom')
                    ], style={'display': 'inline-flex', 'align-items': 'center'}),
                    html.Span(id='page-number', className='page-info')
                ], className="navigation-controls")
            ], className="controls-panel"),
            
            # Content area with loading
            html.Div([
                dcc.Loading(
                    id='loading', 
                    type='dot',
                    color='#667eea',
                    className='loading-overlay',
                    children=[
                        html.Div(id='page-content', className='page-content'),
                    ]
                ),
            ], className="content-area"),
            
        ], className="main-container"),
        
        # Data stores
        dcc.Store(id='file-data', storage_type='session'),
        dcc.Store(id='current-page', data=0, storage_type='session'),
        dcc.Store(id='current-filename', storage_type='session'),
        dcc.Store(id='colormap-store', data='default', storage_type='session'),
        dcc.Store(id='y-axis-scale-store', data='linear', storage_type='session'),
        
        # Floating settings panel
        html.Div([
            html.H6("Display Settings", className="settings-title"),
            dbc.Switch(
                id='global-colormap-toggle',
                label='Oceans3.0 Colormap',
                value=False,
                style={'margin-bottom': '15px'}
            ),
            dbc.Switch(
                id='global-y-axis-toggle',
                label='Logarithmic Y-Axis',
                value=False,
            ),
        ], className="settings-panel", style={
            'position': 'fixed',
            'top': '30px',
            'right': '30px',
            'zIndex': '1000',
            'width': '200px'
        }),
        
        # debounce timers for display settings to prevent rapid updates
        dcc.Interval(
            id='colormap-debounce-timer',
            interval=200,  # 200ms delay
            max_intervals=1,
            disabled=True
        ),
        dcc.Interval(
            id='y-axis-debounce-timer', 
            interval=200,  # 200ms delay
            max_intervals=1,
            disabled=True
        ),
        
        # Modal
        create_modal(),
        
        # Hidden dummy output
        html.Div(id='dummy-output', style={'display': 'none'})
    ], style={'margin': '0', 'padding': '0'})