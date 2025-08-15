import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH, callback_context
import dash_bootstrap_components as dbc
import os
import json
import glob
import threading
from dash.exceptions import PreventUpdate

from config import FOLDER, OUTPUT_FILE, SPECS_PER_PAGE, AVAILABLE_LABELS, ENABLE_AUDIO, AUDIO_FOLDER, LEGACY_LABELS
from utils.file_operations import load_labels, save_labels
from utils.image_processing import generate_image_cached, load_spectrogram_cached, create_spectrogram_figure
from utils.audio_matching import find_matching_audio_files, get_representative_audio_file
from utils.caching import preload_page_images, preload_next_page_images, log_cache_usage
from components.audio_player import create_audio_player
from components.hierarchical_selector import create_hierarchical_selector


# Remove duplicate - use the one from caching.py


# Remove duplicate - use the one from caching.py


# Remove duplicate - use the one from caching.py


def create_label_badges(labels, filename):
    """Create badge display for current labels"""
    if not labels:
        return html.Div(
            "No labels selected",
            style={'color': '#6c757d', 'font-style': 'italic', 'font-size': '0.8em'}
        )
    
    badges = []
    for label in labels:
        badge = dbc.Badge(
            label,
            color="primary", 
            className="me-1 mb-1",
            style={
                'font-size': '0.7em', 
                'padding': '3px 6px',
                'max-width': '100%',  # Prevent overflow
                'word-break': 'break-word',  # Break long words
                'white-space': 'normal',  # Allow wrapping
                'display': 'inline-block'  # Better wrapping behavior
            }
        )
        badges.append(badge)
    
    return html.Div(
        badges, 
        style={
            'display': 'flex', 
            'flex-wrap': 'wrap', 
            'gap': '2px',
            'max-width': '100%',  # Constrain to container
            'overflow': 'hidden',  # Hide any overflow
            'align-items': 'flex-start'  # Better alignment when wrapping
        }
    )


def register_callbacks(app):
    @app.callback(
    Output('page-content', 'children'),
    Output('page-number', 'children'),
    Output('current-page', 'data'),
    Input('prev-page', 'n_clicks'),
    Input('next-page', 'n_clicks'),
    Input('go-to-page', 'n_clicks'),
    State('global-colormap-toggle', 'value'),
    State('global-y-axis-toggle', 'value'),
    State('file-data', 'data'),
    State('current-page', 'data'),
    State('page-input', 'value'),
    )
    def update_page(prev_clicks, next_clicks, go_to_page_clicks, use_hydrophone_colormap, use_log_y_axis, file_data, current_page, page_input):
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

        # Load labels - prioritize legacy file if provided
        if LEGACY_LABELS and os.path.exists(LEGACY_LABELS):
            label_data = load_labels(LEGACY_LABELS, convert_to_hierarchical=True)
            print(f"Loaded legacy labels from {LEGACY_LABELS}")
        else:
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

                # Get labels for this file
                labels_for_file = label_data.get(filename, [])

                # Lightweight label selector - only show if clicked
                labels_section = html.Div([
                    # Dynamic labels display (will show badges)
                    html.Div(
                        id={'type': 'current-labels-display', 'filename': filename},
                        children=create_label_badges(labels_for_file, filename),
                        style={'margin': '0 0 8px 0', 'min-height': '24px'}
                    ),
                    # Toggle button for hierarchical selector
                    dbc.Button(
                        "Edit Labels", 
                        id={'type': 'expand-labels', 'filename': filename},
                        size="sm", 
                        color="outline-primary",
                        style={'font-size': '0.7em'}
                    ),
                    # Placeholder for expanded selector (will be populated on click)
                    html.Div(
                        id={'type': 'label-selector-container', 'filename': filename},
                        style={'margin-top': '8px'}
                    ),
                    # Hidden store to track if selector is expanded
                    dcc.Store(
                        id={'type': 'selector-expanded', 'filename': filename},
                        data=False
                    )
                ], style={
                    'padding': '12px',
                    'background': 'white',
                    'border-top': '1px solid #e9ecef'
                })

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
        ctx = callback_context
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

    # debounced callbacks for display settings
    @app.callback(
        Output('colormap-debounce-timer', 'n_intervals'),
        Output('colormap-debounce-timer', 'disabled'),
        Input('global-colormap-toggle', 'value'),
        prevent_initial_call=True
    )
    def reset_colormap_timer(colormap_value):
        return 0, False
    
    @app.callback(
        Output('y-axis-debounce-timer', 'n_intervals'),
        Output('y-axis-debounce-timer', 'disabled'),
        Input('global-y-axis-toggle', 'value'),
        prevent_initial_call=True
    )
    def reset_y_axis_timer(y_axis_value):
        return 0, False

    # simplified debounced update for debugging
    @app.callback(
        Output({'type': 'spectrogram-image', 'filename': ALL}, 'src'),
        Input('colormap-debounce-timer', 'n_intervals'),
        Input('y-axis-debounce-timer', 'n_intervals'),
        State('global-colormap-toggle', 'value'),
        State('global-y-axis-toggle', 'value'),
        State({'type': 'spectrogram-image', 'filename': ALL}, 'id'),
        prevent_initial_call=True
    )
    def update_all_spectrograms_simple(colormap_timer, y_axis_timer, use_hydrophone_colormap, use_log_y_axis, image_ids):
        import time as time_module
        start_time = time_module.time()
        
        # only update when at least one timer has actually fired
        if (colormap_timer is None or colormap_timer < 1) and (y_axis_timer is None or y_axis_timer < 1):
            return [dash.no_update] * len(image_ids) if image_ids else []
            
        print(f"🔄 Updating {len(image_ids)} spectrograms...")
        
        colormap = 'hydrophone' if use_hydrophone_colormap else 'default'
        y_scale = 'log' if use_log_y_axis else 'linear'
        
        updated_images = []
        for image_id in image_ids:
            filename = image_id['filename']
            image_src = generate_image_cached(filename, colormap, y_scale)
            updated_images.append(image_src if image_src is not None else dash.no_update)
        
        end_time = time_module.time()
        total_time = (end_time - start_time) * 1000
        print(f"✅ Updated {len(image_ids)} spectrograms in {total_time:.1f}ms")
        
        return updated_images

    # New callback for handling hierarchical label updates
    @app.callback(
        Output('dummy-output', 'children'),
        Input({'type': 'selected-labels-store', 'filename': ALL}, 'data'),
        State({'type': 'selected-labels-store', 'filename': ALL}, 'id'),
        prevent_initial_call=True
    )
    def update_hierarchical_labels(labels_data_list, store_ids):
        """Handle updates from the hierarchical label selector"""
        ctx = callback_context
        if not ctx.triggered:
            return ''
        
        # find which store was updated
        trigger_info = ctx.triggered[0]
        prop_id = trigger_info['prop_id']
        
        if not prop_id or 'selected-labels-store' not in prop_id:
            return ''
        
        # More robust parsing of the component ID
        try:
            # Extract the component ID part before the property name
            if '.data' in prop_id:
                id_str = prop_id.replace('.data', '')
            else:
                return ''
            
            # Find the matching store ID from the states
            filename = None
            for store_id in store_ids:
                if store_id and 'filename' in store_id:
                    # Compare the serialized ID
                    import json
                    serialized_id = json.dumps(store_id, separators=(',', ':'))
                    if serialized_id == id_str:
                        filename = store_id['filename']
                        break
            
            if not filename:
                return ''
            
        except Exception as e:
            print(f"Error parsing component ID: {e}")
            return ''
        
        # get the current labels for this file
        current_labels = trigger_info['value'] or []
        
        # load existing label data
        existing_data = load_labels(OUTPUT_FILE)
        existing_labels = existing_data.get(filename, [])
        
        # determine which labels to add and remove
        existing_set = set(existing_labels)
        current_set = set(current_labels)
        
        # add new labels
        labels_to_add = current_set - existing_set
        if labels_to_add:
            for label in labels_to_add:
                save_labels(OUTPUT_FILE, {filename: label}, remove=False)
        
        # remove old labels
        labels_to_remove = existing_set - current_set
        if labels_to_remove:
            for label in labels_to_remove:
                save_labels(OUTPUT_FILE, {filename: label}, remove=True)
        
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
    
    # Toggle hierarchical selector when "Edit Labels" is clicked
    @app.callback(
        [Output({'type': 'label-selector-container', 'filename': MATCH}, 'children'),
         Output({'type': 'selector-expanded', 'filename': MATCH}, 'data'),
         Output({'type': 'expand-labels', 'filename': MATCH}, 'children')],
        Input({'type': 'expand-labels', 'filename': MATCH}, 'n_clicks'),
        [State({'type': 'expand-labels', 'filename': MATCH}, 'id'),
         State({'type': 'selector-expanded', 'filename': MATCH}, 'data')],
        prevent_initial_call=True
    )
    def toggle_label_selector(n_clicks, button_id, is_expanded):
        if not n_clicks:
            raise PreventUpdate
            
        filename = button_id['filename']
        
        # Toggle the expanded state
        new_expanded_state = not is_expanded
        
        if new_expanded_state:
            # Load current labels for this file and show hierarchical selector
            if LEGACY_LABELS and os.path.exists(LEGACY_LABELS):
                label_data = load_labels(LEGACY_LABELS, convert_to_hierarchical=True)
            else:
                label_data = load_labels(OUTPUT_FILE)
            selected_labels = label_data.get(filename, [])
            return create_hierarchical_selector(filename, selected_labels), True, "Collapse"
        else:
                         # Hide hierarchical selector
             return [], False, "Edit Labels"

    # Update badge display when labels change
    @app.callback(
        Output({'type': 'current-labels-display', 'filename': MATCH}, 'children'),
        Input({'type': 'selected-labels-store', 'filename': MATCH}, 'data'),
        State({'type': 'current-labels-display', 'filename': MATCH}, 'id'),
        prevent_initial_call=False
    )
    def update_label_badges(selected_labels, display_id):
        filename = display_id['filename']
        labels = selected_labels or []
        return create_label_badges(labels, filename)

    # Removed redundant hierarchical selector audio callback - main page callback handles this

    # Handle slider seeking - using a simple pattern-matching callback
    @app.callback(
        Output({'type': 'slider-dummy', 'id': ALL}, 'children'),
        [Input({'type': 'time-slider', 'id': ALL}, 'value')],
        prevent_initial_call=True
    )
    def handle_slider_seeking(slider_values):
        """Handle time slider seeking via clientside JavaScript"""
        return ['' for _ in slider_values]