from dash import Dash, html, dcc, Input, Output
import flask 

# Initialize the variable
selected_city = "Montréal"

app = Dash(__name__)

app.layout = html.Div(
	[ html.Button("Close Server", id="close-button"),
	html.Div(children=[
		html.Label('Dropdown'),
		dcc.Dropdown(['New York City', 'Montréal', 'San Francisco'], 'Montréal', id='Dropdown'),

		html.Br(),
		html.Label('Multi-Select Dropdown'),
		dcc.Dropdown(['New York City', 'Montréal', 'San Francisco'],
					 ['Montréal', 'San Francisco'],
					 multi=True),

		html.Br(),
		html.Label('Radio Items'),
		dcc.RadioItems(['New York City', 'Montréal', 'San Francisco'], 'Montréal'),
	], style={'padding': 10, 'flex': 1}),

	html.Div(children=[
		html.Label('Checkboxes'),
		dcc.Checklist(['New York City', 'Montréal', 'San Francisco'],
					  ['Montréal', 'San Francisco']
		),

		html.Br(),
		html.Label('Text Input'),
		dcc.Input(value='MTL', type='text'),

		html.Br(),
		html.Label('Slider'),
		dcc.Slider(
			min=0,
			max=9,
			marks={i: f'Label {i}' if i == 1 else str(i) for i in range(1, 6)},
			value=5,
		),
	], style={'padding': 10, 'flex': 1}),
	
	html.Div(id= 'selected-city-display', style={'padding': 10, 'flex': 1}),
	], style={'display': 'flex', 'flex-direction': 'row'})

# Callback to update the selected_city variable and display it
@app.callback(
	Output('selected-city-display', 'children'),
	Input('Dropdown', 'value'))

def update_selected_city(city):
	global selected_city
	selected_city = city
	return f"Selected city: {selected_city}"

@app.callback(Output("close-button", "n_clicks"), Input("close-button", "n_clicks"))
def close_server(n_clicks):
    if n_clicks is not None and n_clicks > 0:
        print("Closing server...")
        flask.request.environ.get("werkzeug.server.shutdown")()
    
if __name__ == '__main__':
	print('selected_city: ', selected_city)
	app.run_server(debug= False)
	print('selected_city: ', selected_city)

