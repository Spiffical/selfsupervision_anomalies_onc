import dash_bootstrap_components as dbc
from dash import dcc, html

def create_modal():
    return dbc.Modal(
        [
            dbc.ModalHeader(id='modal-header'),
            dbc.ModalBody([
                dcc.RadioItems(
                    id='modal-colormap-toggle',
                    options=[
                        {'label': 'Default Colormap', 'value': 'default'},
                        {'label': 'Hydrophone Colormap', 'value': 'hydrophone'},
                    ],
                    value='default',
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                    style={'margin-bottom': '10px'}
                ),
                dcc.Graph(id='modal-image-graph'),
                # Audio player section
                html.Div(id='modal-audio-player', style={'margin-top': '15px'}),
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id='close-modal', className='ml-auto')
            ),
        ],
        id='image-modal',
        size='xl',
        is_open=False,
    )