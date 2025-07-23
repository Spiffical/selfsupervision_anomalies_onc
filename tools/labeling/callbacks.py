from dash.dependencies import Input, Output, State, ALL
from dash import callback_context, html
import dash
from utils.image_processing import generate_image_cached, create_spectrogram_figure
from utils.file_operations import load_labels, save_labels
from utils.caching import preload_page_images, preload_next_page_images, log_cache_usage
from utils.audio_matching import find_matching_audio_files, get_representative_audio_file
from components.audio_player import create_audio_player
import dash_bootstrap_components as dbc
import threading
import os
import glob
from config import FOLDER, OUTPUT_FILE, SPECS_PER_PAGE, AVAILABLE_LABELS, ENABLE_AUDIO, AUDIO_FOLDER
from utils.image_processing import load_spectrogram_cached

def register_callbacks(app):
    @app.callback(
    Output('page-content', 'children'),
    Output('page-number', 'children'),
    Output('current-page', 'data'),
    Input('prev-page', 'n_clicks'),
    Input('next-page', 'n_clicks'),
    Input('global-colormap-toggle', 'value'),
    Input('global-y-axis-toggle', 'value'),
    Input('go-to-page', 'n_clicks'),
    State('file-data', 'data'),
    State('current-page', 'data'),
    State('page-input', 'value'),
    )
    def update_page(prev_clicks, next_clicks, use_hydrophone_colormap, use_log_y_axis, go_to_page_clicks, file_data, current_page, page_input):
        # Initialize current_page if None
        if current_page is None:
            current_page = 0

        # Initialize click counts
        if prev_clicks is None:
            prev_clicks = 0
        if next_clicks is None:
            next_clicks = 0

        # Determine which button was clicked
        changed_id = [p['prop_id'] for p in callback_context.triggered][0]
        if 'prev-page' in changed_id:
            if current_page > 0:
                current_page -= 1
        elif 'next-page' in changed_id:
            if current_page is None:
                current_page = 0
            current_page += 1
        elif 'go-to-page' in changed_id:
            if page_input is not None and page_input > 0:
                current_page = page_input - 1  # Subtract 1 because pages are 0-indexed

        # Ensure current_page is within bounds
        # Initialize file data if not already present
        if not file_data:
            mat_files = sorted(glob.glob(os.path.join(FOLDER, '*.mat')))
            total_spectrograms = len(mat_files)
            total_pages = (total_spectrograms + SPECS_PER_PAGE - 1) // SPECS_PER_PAGE
            file_data = {
                'mat_files': mat_files,
                'total_pages': total_pages,
                'total_spectrograms': total_spectrograms
            }
        else:
            mat_files = file_data['mat_files']
            total_pages = file_data['total_pages']
        current_page = max(0, min(current_page, total_pages - 1))

        # Calculate indices for the current page
        start_idx = current_page * SPECS_PER_PAGE
        end_idx = min(start_idx + SPECS_PER_PAGE, len(mat_files))

        # Get filenames for the current page
        current_page_files = mat_files[start_idx:end_idx]
        current_page_filenames = [os.path.basename(f) for f in current_page_files]

        # Load labels directly from the JSON file
        label_data = load_labels(OUTPUT_FILE)

        page_info = f"Page {current_page + 1} of {total_pages}"

        # Build the grid layout
        rows = []
        items_per_row = 5  # Number of spectrograms per row

        # Preload images for the current page
        preload_page_images(mat_files, start_idx, end_idx)

        # Preload images for the next page in a background thread
        threading.Thread(target=preload_next_page_images, args=(current_page, total_pages, mat_files)).start()

        for i in range(0, len(current_page_filenames), items_per_row):
            row_children = []
            for filename in current_page_filenames[i:i+items_per_row]:
                # All combinations are preloaded, so this should be fast
                colormap = 'hydrophone' if use_hydrophone_colormap else 'default'
                y_scale = 'log' if use_log_y_axis else 'linear'
                image_src = generate_image_cached(filename, colormap, y_scale)
                if image_src is None:
                    continue

                # Filename display
                filename_display = html.Div([
                    html.H6(filename, style={
                        'font-size': '12px',
                        'font-weight': '600',
                        'color': '#495057',
                        'margin': '0',
                        'text-align': 'center',
                        'word-break': 'break-word'
                    })
                ], style={
                    'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
                    'padding': '8px 10px',
                    'border-radius': '8px 8px 0 0',
                    'border-bottom': '1px solid #dee2e6'
                })

                # Image with hover effect
                image_container = html.Div([
                    html.Img(
                        src=image_src,
                        id={'type': 'spectrogram-image', 'filename': filename},
                                                 className='spectrogram-image',
                         style={
                             'width': '100%',
                             'cursor': 'pointer',
                             'border-radius': '0'
                         }
                    )
                ], style={
                    'position': 'relative',
                    'overflow': 'hidden',
                    'background': '#f8f9fa'
                })

                labels_for_file = label_data.get(filename, [])

                # Enhanced checkboxes with better styling
                checkboxes = []
                for label in AVAILABLE_LABELS:
                    is_checked = label in labels_for_file
                    checkbox = html.Div([
                        dbc.Checkbox(
                            id={'type': 'label-checkbox', 'filename': filename, 'label': label},
                            label='',
                            value=is_checked,
                            style={'margin-right': '8px'}
                        ),
                        html.Label(label, style={
                            'font-size': '12px',
                            'font-weight': '500',
                            'color': '#495057',
                            'margin': '0',
                            'cursor': 'pointer'
                        })
                    ], style={
                        'display': 'flex',
                        'align-items': 'center',
                        'margin-bottom': '6px',
                        'padding': '2px 0'
                    })
                    checkboxes.append(checkbox)

                # Create audio player for this spectrogram
                audio_player = None
                if ENABLE_AUDIO and AUDIO_FOLDER:
                    matching_audio_files = find_matching_audio_files(filename, AUDIO_FOLDER)
                    if matching_audio_files:
                        representative_audio = get_representative_audio_file(matching_audio_files)
                        audio_player = html.Div([
                            html.Hr(style={'margin': '10px 0 8px 0', 'border-color': '#e9ecef'}),
                            create_audio_player(representative_audio, filename, 
                                               player_id=f"grid-{hash(filename) % 10000}")
                        ])

                # Labels section with better styling
                labels_section = html.Div([
                    html.Div(checkboxes, style={'padding': '0'})
                ], style={
                    'padding': '12px',
                    'background': 'white'
                })

                # Build card content with improved structure
                card_content = [
                    filename_display,
                    image_container,
                    labels_section
                ]
                
                if audio_player:
                    card_content.insert(-1, audio_player)

                card = dbc.Card(
                    card_content, 
                    style={
                        'margin-bottom': '20px',
                        'border-radius': '12px',
                        'box-shadow': '0 4px 15px rgba(0, 0, 0, 0.08)',
                        'border': '1px solid #e9ecef',
                        'transition': 'all 0.3s ease',
                        'overflow': 'hidden'
                    }
                )

                col = dbc.Col(card, width=2)
                row_children.append(col)
            # Add empty columns if needed to fill the row
            while len(row_children) < items_per_row:
                row_children.append(dbc.Col(width=2))

            row = dbc.Row(row_children, justify='start')
            rows.append(row)

        content = html.Div(rows)

        log_cache_usage()

        return content, page_info, current_page

    @app.callback(
    Output('image-modal', 'is_open'),
    Output('current-filename', 'data'),
    Output('modal-image-graph', 'figure'),
    Output('modal-colormap-toggle', 'value'),
    Output('modal-y-axis-toggle', 'value'),
    Output('modal-header', 'children'),
    Output('modal-audio-player', 'children'),
    Input({'type': 'spectrogram-image', 'filename': ALL}, 'n_clicks'),
    Input('close-modal', 'n_clicks'),
    State('global-colormap-toggle', 'value'),
    State('global-y-axis-toggle', 'value'),
    prevent_initial_call=True
    )
    def display_image_modal(n_clicks_list, close_clicks, global_colormap, global_y_axis):
        ctx = dash.callback_context
        if not ctx.triggered:
            return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        triggered_id = ctx.triggered_id
        
        if triggered_id == 'close-modal':
            return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # Check if any of the image clicks are non-zero
        if isinstance(triggered_id, dict) and triggered_id.get('type') == 'spectrogram-image':
            if any(click and click > 0 for click in n_clicks_list):
                filename = triggered_id['filename']
                print(f"Image clicked: {filename}")
                spectrogram = load_spectrogram_cached(filename)
                if spectrogram is not None:
                    colormap = 'hydrophone' if global_colormap else 'default'
                    y_axis_scale = 'log' if global_y_axis else 'linear'
                    fig = create_spectrogram_figure(spectrogram, colormap, y_axis_scale)
                    
                    # Create audio player for modal
                    modal_audio_player = html.Div()
                    if ENABLE_AUDIO and AUDIO_FOLDER:
                        matching_audio_files = find_matching_audio_files(filename, AUDIO_FOLDER)
                        if matching_audio_files:
                            representative_audio = get_representative_audio_file(matching_audio_files)
                            modal_audio_player = html.Div([
                                html.Hr(),
                                html.H5("Audio Playback", style={'color': '#333'}),
                                create_audio_player(representative_audio, filename, 
                                                   player_id=f"modal-{hash(filename) % 10000}"),
                                html.Div([
                                    html.Small(f"Matched {len(matching_audio_files)} audio file(s)", 
                                             style={'color': '#666', 'font-style': 'italic'})
                                ], style={'text-align': 'center', 'margin-top': '5px'})
                            ])
                        else:
                            modal_audio_player = html.Div([
                                html.Hr(),
                                html.H5("Audio Playback", style={'color': '#333'}),
                                html.Small("No matching audio files found", 
                                         style={'color': 'gray', 'font-style': 'italic'})
                            ])
                    
                    return True, filename, fig, colormap, y_axis_scale, f"Spectrogram: {filename}", modal_audio_player

        return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    @app.callback(
    Output('modal-image-graph', 'figure', allow_duplicate=True),
    Input('modal-colormap-toggle', 'value'),
    Input('modal-y-axis-toggle', 'value'),
    State('current-filename', 'data'),
    prevent_initial_call=True
    )
    def update_modal_figure_options(colormap_value, y_axis_value, current_filename):
        if not current_filename:
            return dash.no_update
        spectrogram = load_spectrogram_cached(current_filename)
        if spectrogram is None:
            return dash.no_update
        fig = create_spectrogram_figure(spectrogram, colormap_value, y_axis_value)
        return fig

    @app.callback(
    Output({'type': 'spectrogram-image', 'filename': ALL}, 'src'),
    Input('global-colormap-toggle', 'value'),
    Input('global-y-axis-toggle', 'value'),
    State({'type': 'spectrogram-image', 'filename': ALL}, 'id'),
    prevent_initial_call=True
    )
    def update_all_spectrograms(use_hydrophone_colormap, use_log_y_axis, image_ids):
        updated_images = []
        for image_id in image_ids:
            filename = image_id['filename']
            # All combinations are preloaded, so this should be fast
            colormap = 'hydrophone' if use_hydrophone_colormap else 'default'
            y_scale = 'log' if use_log_y_axis else 'linear'
            image_src = generate_image_cached(filename, colormap, y_scale)
            updated_images.append(image_src if image_src is not None else dash.no_update)
        return updated_images

    @app.callback(
    Output('dummy-output', 'children'),
    Input({'type': 'label-checkbox', 'filename': ALL, 'label': ALL}, 'value'),
    )
    def update_labels(_):
        ctx = dash.callback_context
        if not ctx.triggered:
            return ''
        trigger = ctx.triggered[0]
        prop_id = trigger['prop_id']
        value = trigger['value']

        if value is None:
            return ''
        import json
        id_str, prop_name = prop_id.rsplit('.', 1)
        id_dict = json.loads(id_str)
        filename = id_dict['filename']
        label = id_dict['label']
        checked = value

        # Create a single-entry dictionary for this update
        label_update = {filename: label}
        
        # Use the new remove parameter based on checkbox state
        save_labels(OUTPUT_FILE, label_update, remove=not checked)

        return ''

    # Initialize audio players when page content changes
    app.clientside_callback(
        """
        function(page_content) {
            if (window.dash_clientside && window.dash_clientside.namespace) {
                setTimeout(function() {
                    window.dash_clientside.namespace.initializeAudioPlayers();
                }, 100);
            }
            return '';
        }
        """,
        Output('dummy-output', 'children', allow_duplicate=True),
        [Input('page-content', 'children')],
        prevent_initial_call=True
    )

    # Initialize audio players when modal content changes
    app.clientside_callback(
        """
        function(modal_audio_content) {
            if (window.dash_clientside && window.dash_clientside.namespace) {
                setTimeout(function() {
                    window.dash_clientside.namespace.initializeAudioPlayers();
                }, 150);
            }
            return '';
        }
        """,
        Output('dummy-output', 'children', allow_duplicate=True),
        [Input('modal-audio-player', 'children')],
        prevent_initial_call=True
    )

    # Handle slider seeking - using a simple pattern-matching callback
    @app.callback(
        Output({'type': 'slider-dummy', 'id': ALL}, 'children'),
        [Input({'type': 'time-slider', 'id': ALL}, 'value')],
        prevent_initial_call=True
    )
    def handle_slider_seeking(slider_values):
        """Handle time slider seeking via clientside JavaScript"""
        return ['' for _ in slider_values]