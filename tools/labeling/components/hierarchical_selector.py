import dash
from dash import html, dcc, Input, Output, State, callback, ALL, MATCH
import dash_bootstrap_components as dbc
from hierarchical_labels import HIERARCHICAL_LABELS, get_all_paths, get_label_display_name, path_to_string

def create_hierarchical_selector(filename, selected_labels=None):
    """
    Create a hierarchical label selector for a specific spectrogram file.
    
    Args:
        filename: Name of the spectrogram file
        selected_labels: List of currently selected label paths
    """
    if selected_labels is None:
        selected_labels = []
    
    # convert string labels back to paths if needed
    selected_paths = []
    for label in selected_labels:
        if isinstance(label, str) and " > " in label:
            selected_paths.append(tuple(label.split(" > ")))
        elif isinstance(label, (list, tuple)):
            selected_paths.append(tuple(label))
        else:
            # old format - try to map to new hierarchy
            selected_paths.append((label,))
    
    return html.Div([
        # search box
        dbc.InputGroup([
            dbc.Input(
                id={'type': 'label-search', 'filename': filename},
                type='text',
                placeholder="Search labels...",
                style={'flex': '1'},
                persistence=True,
                persistence_type='memory'
            ),
            dbc.Button(
                "Clear",
                id={'type': 'clear-search', 'filename': filename},
                color="outline-secondary",
                style={'border-radius': '0 8px 8px 0'}
            )
        ], style={'margin-bottom': '12px'}),
        
        # selected labels display
        html.Div(
            id={'type': 'selected-labels-display', 'filename': filename},
            children=create_selected_labels_display(selected_paths, filename),
            style={
                'margin-bottom': '12px', 
                'min-height': '30px',
                'overflow': 'hidden',  # Prevent overflow
                'word-wrap': 'break-word',  # Break long words
                'max-width': '100%'  # Ensure it doesn't exceed container
            }
        ),
        
        # hierarchical tree
        html.Div(
            id={'type': 'hierarchical-tree', 'filename': filename},
            children=create_tree_structure(HIERARCHICAL_LABELS, filename, selected_paths),
            style={
                'max-height': '400px',
                'overflow-y': 'auto',
                'border': '1px solid #e9ecef',
                'border-radius': '8px',
                'padding': '8px',
                'background': '#fdfdfd'
            }
        ),
        
        # hidden store for selected labels
        dcc.Store(
            id={'type': 'selected-labels-store', 'filename': filename},
            data=[path_to_string(path) for path in selected_paths]
        ),
        
        # debounce timer for search - starts disabled and gets reset on each keystroke
        dcc.Interval(
            id={'type': 'search-debounce-timer', 'filename': filename},
            interval=300,  # 300ms delay
            max_intervals=1,
            disabled=True
        ),
        
        # hidden dummy element for audio player reinitialization callback
        html.Div(
            id={'type': 'audio-reinit-dummy', 'filename': filename},
            style={'display': 'none'}
        )
    ], style={'padding': '8px'})

def create_selected_labels_display(selected_paths, filename):
    """Create display for currently selected labels"""
    if not selected_paths:
        return html.Div(
            "No labels selected",
            style={'color': '#6c757d', 'font-style': 'italic', 'font-size': '0.9em'}
        )
    
    badges = []
    for i, path in enumerate(selected_paths):
        display_name = get_label_display_name(path)
        badge = dbc.Badge([
            display_name,
            html.Span("×", 
                style={'margin-left': '8px', 'cursor': 'pointer', 'font-weight': 'bold'},
                id={'type': 'remove-label', 'filename': filename, 'index': i}
            )
        ], 
        color="primary", 
        className="me-2 mb-1",
        style={
            'font-size': '0.8em', 
            'padding': '4px 8px',
            'max-width': '100%',  # Prevent badge from exceeding container
            'word-break': 'break-word',  # Break long text
            'white-space': 'normal',  # Allow wrapping
            'display': 'inline-block',  # Better handling of overflow
            'vertical-align': 'top'  # Align properly when wrapping
        }
        )
        badges.append(badge)
    
    return html.Div(
        badges, 
        style={
            'display': 'flex',
            'flex-wrap': 'wrap',  # Allow badges to wrap to next line
            'gap': '4px',  # Space between badges
            'align-items': 'flex-start'  # Align to top when wrapping
        }
    )

def create_tree_structure(hierarchy, filename, selected_paths, current_path=None, level=0):
    """Recursively create tree structure"""
    if current_path is None:
        current_path = []
    
    tree_items = []
    
    for key, value in hierarchy.items():
        new_path = current_path + [key]
        path_tuple = tuple(new_path)
        path_string = path_to_string(path_tuple)  # Convert to string for ID
        is_selected = path_tuple in selected_paths
        has_children = isinstance(value, dict) and value
        
        # create the node
        node_id = f"node-{'-'.join(new_path)}"
        
        # node content
        node_content = []
        
        # expand/collapse button for nodes with children
        if has_children:
            expand_btn = html.Span(
                "▶",
                id={'type': 'expand-btn', 'filename': filename, 'path': path_string},
                style={
                    'cursor': 'pointer',
                    'margin-right': '6px',
                    'color': '#6c757d',
                    'font-size': '0.8em',
                    'transform': 'rotate(0deg)',
                    'transition': 'transform 0.2s'
                }
            )
            node_content.append(expand_btn)
        else:
            # spacer for leaf nodes
            node_content.append(html.Span(style={'width': '16px', 'display': 'inline-block'}))
        
        # checkbox
        checkbox = dbc.Checkbox(
            id={'type': 'hierarchical-checkbox', 'filename': filename, 'path': path_string},
            value=is_selected,
            style={'margin-right': '8px', 'margin-top': '2px'}
        )
        node_content.append(checkbox)
        
        # label text
        label_text = html.Span(
            key,
            style={
                'font-size': '0.9em',
                'color': '#495057' if not is_selected else '#0d6efd',
                'font-weight': '500' if is_selected else '400',
                'cursor': 'pointer'
            },
            id={'type': 'label-text', 'filename': filename, 'path': path_string}
        )
        node_content.append(label_text)
        
        # create the tree item
        tree_item = html.Div([
            html.Div(
                node_content,
                style={
                    'display': 'flex',
                    'align-items': 'center',
                    'padding': '2px 0',
                    'margin-left': f'{level * 20}px',
                    'background': '#f8f9fa' if is_selected else 'transparent',
                    'border-radius': '4px',
                    'padding-left': '6px' if is_selected else '0'
                }
            ),
            # children container (initially hidden)
            html.Div(
                id={'type': 'children-container', 'filename': filename, 'path': path_string},
                children=create_tree_structure(value, filename, selected_paths, new_path, level + 1) if has_children else [],
                style={'display': 'none'} if has_children else {}
            )
        ])
        
        tree_items.append(tree_item)
    
    return tree_items

# Callback for expanding/collapsing tree nodes
@callback(
    Output({'type': 'children-container', 'filename': MATCH, 'path': MATCH}, 'style'),
    Output({'type': 'expand-btn', 'filename': MATCH, 'path': MATCH}, 'style'),
    Input({'type': 'expand-btn', 'filename': MATCH, 'path': MATCH}, 'n_clicks'),
    State({'type': 'children-container', 'filename': MATCH, 'path': MATCH}, 'style'),
    prevent_initial_call=True
)
def toggle_tree_node(n_clicks, current_style):
    if not n_clicks:
        return current_style, {'cursor': 'pointer', 'margin-right': '6px', 'color': '#6c757d', 'font-size': '0.8em', 'transform': 'rotate(0deg)', 'transition': 'transform 0.2s'}
    
    is_hidden = current_style.get('display') == 'none'
    
    if is_hidden:
        # show children
        return {'display': 'block'}, {
            'cursor': 'pointer',
            'margin-right': '6px', 
            'color': '#6c757d',
            'font-size': '0.8em',
            'transform': 'rotate(90deg)',
            'transition': 'transform 0.2s'
        }
    else:
        # hide children
        return {'display': 'none'}, {
            'cursor': 'pointer',
            'margin-right': '6px',
            'color': '#6c757d', 
            'font-size': '0.8em',
            'transform': 'rotate(0deg)',
            'transition': 'transform 0.2s'
        }

# Callback for checkbox selection
@callback(
    Output({'type': 'selected-labels-store', 'filename': MATCH}, 'data'),
    Output({'type': 'selected-labels-display', 'filename': MATCH}, 'children'),
    Input({'type': 'hierarchical-checkbox', 'filename': MATCH, 'path': ALL}, 'value'),
    State({'type': 'hierarchical-checkbox', 'filename': MATCH, 'path': ALL}, 'id'),
    State({'type': 'selected-labels-store', 'filename': MATCH}, 'data'),
    prevent_initial_call=True
)
def update_selected_labels(checkbox_values, checkbox_ids, current_labels):
    if not checkbox_values or not checkbox_ids:
        return [], create_selected_labels_display([], checkbox_ids[0]['filename'] if checkbox_ids else '')
    
    # get the filename from the first checkbox id
    filename = checkbox_ids[0]['filename']
    
    # collect selected paths
    selected_paths = []
    for i, is_checked in enumerate(checkbox_values):
        if is_checked:
            path_string = checkbox_ids[i]['path']
            path_tuple = tuple(path_string.split(" > "))  # Convert back to tuple
            selected_paths.append(path_tuple)
    
    # update display
    display = create_selected_labels_display(selected_paths, filename)
    
    # convert paths to strings for storage
    selected_strings = [path_to_string(path) for path in selected_paths]
    
    # Save labels directly to file
    try:
        from utils.file_operations import save_labels, load_labels
        from config import OUTPUT_FILE
        
        # Get current state
        existing_data = load_labels(OUTPUT_FILE)
        existing_labels = set(existing_data.get(filename, []))
        current_set = set(selected_strings)
        
        # Add new labels
        labels_to_add = current_set - existing_labels
        for label in labels_to_add:
            save_labels(OUTPUT_FILE, {filename: label}, remove=False)
        
        # Remove old labels
        labels_to_remove = existing_labels - current_set
        for label in labels_to_remove:
            save_labels(OUTPUT_FILE, {filename: label}, remove=True)
            
        print(f"Saved labels for {filename}: {selected_strings}")
        
    except Exception as e:
        print(f"Error saving labels: {e}")
    
    return selected_strings, display

# Callback for removing labels
@callback(
    Output({'type': 'selected-labels-store', 'filename': MATCH}, 'data', allow_duplicate=True),
    Output({'type': 'selected-labels-display', 'filename': MATCH}, 'children', allow_duplicate=True),
    Output({'type': 'hierarchical-tree', 'filename': MATCH}, 'children', allow_duplicate=True),
    Input({'type': 'remove-label', 'filename': MATCH, 'index': ALL}, 'n_clicks'),
    State({'type': 'remove-label', 'filename': MATCH, 'index': ALL}, 'id'),
    State({'type': 'selected-labels-store', 'filename': MATCH}, 'data'),
    prevent_initial_call=True
)
def remove_label(n_clicks_list, remove_ids, current_labels):
    if not n_clicks_list or not any(n_clicks_list):
        return current_labels, dash.no_update, dash.no_update
    
    # find which remove button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_labels, dash.no_update
    
    try:
        # get the index of the clicked remove button
        clicked_id = ctx.triggered[0]['prop_id']
        
        # Find which button was actually clicked by checking n_clicks
        remove_index = None
        filename = None
        
        for i, n_clicks in enumerate(n_clicks_list):
            if n_clicks and n_clicks > 0:
                if i < len(remove_ids):
                    remove_index = remove_ids[i]['index']
                    filename = remove_ids[i]['filename']
                    break
        
        if remove_index is None or filename is None:
            return current_labels, dash.no_update, dash.no_update
        
        # remove the label at that index
        updated_labels = [label for i, label in enumerate(current_labels) if i != remove_index]
        
        # Save the updated labels directly to file
        try:
            from utils.file_operations import save_labels, load_labels
            from config import OUTPUT_FILE
            
            # Get current state and update
            existing_data = load_labels(OUTPUT_FILE)
            existing_labels = set(existing_data.get(filename, []))
            updated_set = set(updated_labels)
            
            # Remove the labels that are no longer selected
            labels_to_remove = existing_labels - updated_set
            for label in labels_to_remove:
                save_labels(OUTPUT_FILE, {filename: label}, remove=True)
                
            print(f"Removed label for {filename}, remaining: {updated_labels}")
            
        except Exception as e:
            print(f"Error removing labels: {e}")
        
        # update display and tree structure
        selected_paths = [tuple(label.split(" > ")) for label in updated_labels]
        display = create_selected_labels_display(selected_paths, filename)
        tree_structure = create_tree_structure(HIERARCHICAL_LABELS, filename, selected_paths)
        
        return updated_labels, display, tree_structure
        
    except Exception as e:
        print(f"Error in remove_label callback: {e}")
        return current_labels, dash.no_update, dash.no_update

# Clear search input callback
@callback(
    Output({'type': 'label-search', 'filename': MATCH}, 'value'),
    Input({'type': 'clear-search', 'filename': MATCH}, 'n_clicks'),
    prevent_initial_call=True
)
def clear_search_input(clear_clicks):
    if clear_clicks:
        return ""
    return dash.no_update

# Timer reset callback - resets the debounce timer on each keystroke without updating UI
@callback(
    Output({'type': 'search-debounce-timer', 'filename': MATCH}, 'n_intervals'),
    Output({'type': 'search-debounce-timer', 'filename': MATCH}, 'disabled'),
    Input({'type': 'label-search', 'filename': MATCH}, 'value'),
    prevent_initial_call=True
)
def reset_search_timer(search_value):
    # reset timer to 0 and enable it - this doesn't cause UI refresh
    return 0, False

# Search functionality - now properly debounced using timer
@callback(
    Output({'type': 'hierarchical-tree', 'filename': MATCH}, 'children'),
    Input({'type': 'search-debounce-timer', 'filename': MATCH}, 'n_intervals'),
    Input({'type': 'clear-search', 'filename': MATCH}, 'n_clicks'),
    State({'type': 'label-search', 'filename': MATCH}, 'value'),
    State({'type': 'selected-labels-store', 'filename': MATCH}, 'data'),
    State({'type': 'label-search', 'filename': MATCH}, 'id'),
    prevent_initial_call=True
)
def filter_tree(timer_intervals, clear_clicks, search_value, selected_labels, search_id):
    filename = search_id['filename']
    
    # get selected paths - ensure we preserve all selected labels
    selected_paths = []
    if selected_labels:
        for label in selected_labels:
            if isinstance(label, str) and label.strip():
                selected_paths.append(tuple(label.split(" > ")))
    
    # only update when timer actually fires (intervals == 1)
    if timer_intervals is None or timer_intervals < 1:
        return dash.no_update
    
    # Don't update if search value is None
    if search_value is None:
        return dash.no_update
    
    # Handle empty search
    if not search_value or search_value.strip() == "":
        # show full hierarchy with default collapsed state, preserving selections
        return create_tree_structure(HIERARCHICAL_LABELS, filename, selected_paths)
    
    # Require at least 3 characters to start filtering (reduces updates significantly)
    search_term = search_value.strip().lower()
    if len(search_term) < 3:
        # Show full hierarchy for 1-2 characters to avoid constant updates
        return create_tree_structure(HIERARCHICAL_LABELS, filename, selected_paths)
    
    # filter hierarchy based on search and auto-expand matching paths
    # IMPORTANT: Include previously selected items in the filtered view even if they don't match search
    filtered_hierarchy, expanded_paths = filter_hierarchy_by_search_with_expansion_and_preserve_selections(
        HIERARCHICAL_LABELS, search_term, selected_paths
    )
    return create_tree_structure_with_expansion(filtered_hierarchy, filename, selected_paths, expanded_paths)

def filter_hierarchy_by_search_with_expansion_and_preserve_selections(hierarchy, search_term, selected_paths, current_path=None):
    """Filter hierarchy, return filtered hierarchy and paths to expand, while preserving selected items"""
    if current_path is None:
        current_path = []
    
    filtered = {}
    paths_to_expand = set()
    
    for key, value in hierarchy.items():
        new_path = current_path + [key]
        path_tuple = tuple(new_path)
        
        # check if current key matches search
        key_matches = search_term in key.lower()
        
        # check if this path is currently selected (should always be included)
        is_selected = path_tuple in selected_paths
        
        # check if any children match
        children_match = False
        filtered_children = {}
        child_expand_paths = set()
        
        if isinstance(value, dict) and value:
            if key_matches:
                # If parent matches, include ALL children (don't filter them further)
                filtered_children = value
                children_match = True
                # Expand this entire branch since parent matched
                for i in range(1, len(new_path) + 1):
                    parent_path = tuple(new_path[:i])
                    paths_to_expand.add(path_to_string(parent_path))
            else:
                # If parent doesn't match, recursively filter children
                filtered_children, child_expand_paths = filter_hierarchy_by_search_with_expansion_and_preserve_selections(
                    value, search_term, selected_paths, new_path
                )
                children_match = bool(filtered_children)
                paths_to_expand.update(child_expand_paths)
        
        # include this item if it matches search, has matching children, OR is currently selected
        if key_matches or children_match or is_selected:
            if filtered_children:
                filtered[key] = filtered_children
            else:
                filtered[key] = value
            
            # if this or any child matches, or if it's selected, mark parent paths for expansion
            if key_matches or children_match or is_selected:
                # add all parent paths to expansion list
                for i in range(1, len(new_path) + 1):
                    parent_path = tuple(new_path[:i])
                    paths_to_expand.add(path_to_string(parent_path))
    
    return filtered, paths_to_expand

def filter_hierarchy_by_search_with_expansion(hierarchy, search_term, current_path=None):
    """Filter hierarchy and return both filtered hierarchy and paths to expand - legacy function"""
    if current_path is None:
        current_path = []
    
    filtered = {}
    paths_to_expand = set()
    
    for key, value in hierarchy.items():
        new_path = current_path + [key]
        
        # check if current key matches search
        key_matches = search_term in key.lower()
        
        # check if any children match
        children_match = False
        filtered_children = {}
        child_expand_paths = set()
        
        if isinstance(value, dict) and value:
            filtered_children, child_expand_paths = filter_hierarchy_by_search_with_expansion(value, search_term, new_path)
            children_match = bool(filtered_children)
            paths_to_expand.update(child_expand_paths)
        
        # include this item if it matches or has matching children
        if key_matches or children_match:
            if filtered_children:
                filtered[key] = filtered_children
            else:
                filtered[key] = value
            
            # if this or any child matches, mark parent paths for expansion
            if key_matches or children_match:
                # add all parent paths to expansion list
                for i in range(1, len(new_path) + 1):
                    parent_path = tuple(new_path[:i])
                    paths_to_expand.add(path_to_string(parent_path))
    
    return filtered, paths_to_expand

def create_tree_structure_with_expansion(hierarchy, filename, selected_paths, expanded_paths, current_path=None, level=0):
    """Create tree structure with specified paths auto-expanded"""
    if current_path is None:
        current_path = []
    
    tree_items = []
    
    for key, value in hierarchy.items():
        new_path = current_path + [key]
        path_tuple = tuple(new_path)
        path_string = path_to_string(path_tuple)
        is_selected = path_tuple in selected_paths
        has_children = isinstance(value, dict) and value
        
        # check if this path should be expanded
        should_expand = path_string in expanded_paths
        
        # create the node
        node_id = f"node-{'-'.join(new_path)}"
        
        # node content
        node_content = []
        
        # expand/collapse button for nodes with children
        if has_children:
            # set initial rotation based on expansion state
            initial_rotation = '90deg' if should_expand else '0deg'
            expand_btn = html.Span(
                "▶",
                id={'type': 'expand-btn', 'filename': filename, 'path': path_string},
                style={
                    'cursor': 'pointer',
                    'margin-right': '6px',
                    'color': '#6c757d',
                    'font-size': '0.8em',
                    'transform': f'rotate({initial_rotation})',
                    'transition': 'transform 0.2s'
                }
            )
            node_content.append(expand_btn)
        else:
            # spacer for leaf nodes
            node_content.append(html.Span(style={'width': '16px', 'display': 'inline-block'}))
        
        # checkbox
        checkbox = dbc.Checkbox(
            id={'type': 'hierarchical-checkbox', 'filename': filename, 'path': path_string},
            value=is_selected,
            style={'margin-right': '8px', 'margin-top': '2px'}
        )
        node_content.append(checkbox)
        
        # label text
        label_text = html.Span(
            key,
            style={
                'font-size': '0.9em',
                'color': '#495057' if not is_selected else '#0d6efd',
                'font-weight': '500' if is_selected else '400',
                'cursor': 'pointer'
            },
            id={'type': 'label-text', 'filename': filename, 'path': path_string}
        )
        node_content.append(label_text)
        
        # create the tree item
        tree_item = html.Div([
            html.Div(
                node_content,
                style={
                    'display': 'flex',
                    'align-items': 'center',
                    'padding': '2px 0',
                    'margin-left': f'{level * 20}px',
                    'background': '#f8f9fa' if is_selected else 'transparent',
                    'border-radius': '4px',
                    'padding-left': '6px' if is_selected else '0'
                }
            ),
            # children container - show if should be expanded
            html.Div(
                id={'type': 'children-container', 'filename': filename, 'path': path_string},
                children=create_tree_structure_with_expansion(value, filename, selected_paths, expanded_paths, new_path, level + 1) if has_children else [],
                style={'display': 'block' if should_expand else 'none'} if has_children else {}
            )
        ])
        
        tree_items.append(tree_item)
    
    return tree_items

def filter_hierarchy_by_search(hierarchy, search_term, current_path=None):
    """Filter hierarchy to only show matching items - legacy function"""
    if current_path is None:
        current_path = []
    
    filtered = {}
    
    for key, value in hierarchy.items():
        new_path = current_path + [key]
        
        # check if current key matches search
        key_matches = search_term in key.lower()
        
        # check if any children match
        children_match = False
        filtered_children = {}
        if isinstance(value, dict) and value:
            filtered_children = filter_hierarchy_by_search(value, search_term, new_path)
            children_match = bool(filtered_children)
        
        # include this item if it matches or has matching children
        if key_matches or children_match:
            if filtered_children:
                filtered[key] = filtered_children
            else:
                filtered[key] = value
    
    return filtered


# Add client-side callback to reinitialize audio players when tree updates
def register_audio_reinit_callback(app):
    """Register callback to reinitialize audio players when hierarchical selectors change"""
    app.clientside_callback(
        """
        function(tree_children) {
            if (window.dash_clientside && window.dash_clientside.namespace) {
                setTimeout(function() {
                    console.log('Reinitializing audio players after hierarchical tree update...');
                    window.dash_clientside.namespace.initializeAudioPlayers();
                }, 200);
            }
            return '';
        }
        """,
        dash.Output({'type': 'audio-reinit-dummy', 'filename': dash.MATCH}, 'children'),
        [dash.Input({'type': 'hierarchical-tree', 'filename': dash.MATCH}, 'children')],
        prevent_initial_call=True
    )