import dash_bootstrap_components as dbc
from dash import dcc, html

def create_modal():
    return dbc.Modal(
        [
            dbc.ModalHeader([
                html.H4(id='modal-header', style={
                    'color': '#495057',
                    'font-weight': '600',
                    'margin': '0'
                })
            ], style={
                'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
                'border-bottom': '1px solid #dee2e6'
            }),
            dbc.ModalBody([
                # Control panel for modal settings
                html.Div([
                    html.Div([
                        html.Div([
                            html.Label("Colormap", style={
                                'font-weight': '600',
                                'color': '#495057',
                                'margin-bottom': '8px',
                                'display': 'block',
                                'font-size': '14px'
                            }),
                            dcc.RadioItems(
                                id='modal-colormap-toggle',
                                options=[
                                    {'label': ' Default', 'value': 'default'},
                                    {'label': ' Oceans3.0', 'value': 'hydrophone'},
                                ],
                                value='default',
                                style={'margin-top': '5px'},
                                labelStyle={
                                    'display': 'flex',
                                    'align-items': 'center',
                                    'margin-bottom': '8px',
                                    'font-weight': '500',
                                    'color': '#6c757d'
                                },
                                inputStyle={'margin-right': '8px'}
                            )
                        ], style={'flex': '1'})
                    ], style={'display': 'flex', 'margin-bottom': '15px'}),
                    
                    html.Div([
                        html.Div([
                            html.Label("Y-Axis Scale", style={
                                'font-weight': '600',
                                'color': '#495057',
                                'margin-bottom': '8px',
                                'display': 'block',
                                'font-size': '14px'
                            }),
                            dcc.RadioItems(
                                id='modal-y-axis-toggle',
                                options=[
                                    {'label': ' Linear', 'value': 'linear'},
                                    {'label': ' Logarithmic', 'value': 'log'},
                                ],
                                value='linear',
                                style={'margin-top': '5px'},
                                labelStyle={
                                    'display': 'flex',
                                    'align-items': 'center',
                                    'margin-bottom': '8px',
                                    'font-weight': '500',
                                    'color': '#6c757d'
                                },
                                inputStyle={'margin-right': '8px'}
                            )
                        ], style={'flex': '1'})
                    ], style={'display': 'flex'})
                ], style={
                    'background': 'rgba(248, 249, 250, 0.8)',
                    'border-radius': '12px',
                    'padding': '20px',
                    'margin-bottom': '20px',
                    'border': '1px solid #e9ecef'
                }),
                
                # Spectrogram visualization
                html.Div([
                    dcc.Graph(
                        id='modal-image-graph',
                        style={'border-radius': '10px', 'overflow': 'hidden'},
                        config={'displayModeBar': True, 'displaylogo': False}
                    )
                ], style={
                    'background': 'white',
                    'border-radius': '12px',
                    'padding': '15px',
                    'box-shadow': '0 4px 15px rgba(0, 0, 0, 0.05)',
                    'border': '1px solid #e9ecef'
                }),
                
                # Audio player section
                html.Div(id='modal-audio-player', style={
                    'margin-top': '20px',
                    'background': 'rgba(248, 249, 250, 0.5)',
                    'border-radius': '12px',
                    'padding': '15px',
                    'border': '1px solid #e9ecef'
                }),
            ], style={'padding': '25px'}),
            
            dbc.ModalFooter([
                dbc.Button(
                    "Close", 
                    id='close-modal', 
                    className='btn-custom',
                    style={
                        'background': 'linear-gradient(135deg, #6c757d 0%, #495057 100%)',
                        'color': 'white',
                        'border': 'none',
                        'border-radius': '8px',
                        'padding': '10px 25px',
                        'font-weight': '500',
                        'transition': 'all 0.3s ease'
                    }
                )
            ], style={
                'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
                'border-top': '1px solid #dee2e6',
                'justify-content': 'center'
            }),
        ],
        id='image-modal',
        size='xl',
        is_open=False,
        backdrop='static',
        style={
            'max-width': '90vw',
            'margin': '1.75rem auto'
        }
    )