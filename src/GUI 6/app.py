#import dependecies
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px

import pandas as pd
import numpy as np
from numpy import array, cos, sin, pi, radians

import base64
import io
import json
import re

#create the app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = 'Sat Analysis'

#################### Main Layout Page #########################

app.layout = html.Div([
    dcc.Location(id='url'),
    html.Div(id='output-data-upload'),
    html.Div(id='page-content'),
    html.Div(id='orbitPlotInfo', style={'display': 'none'}),
])

#################### Index Page #########################

index_page = html.Div([
    html.Div([
        html.H1('Satellite Constellation Analysis', id='title'),
        dcc.Upload(id='upload-data', children=html.Div([
                'Upload Data -- Drag and Drop or ', html.A('Select File')
        ])), 
    ], id='home-screen'),
    html.Div(id='home-output-container'),
])

def splitData(df):
    # drop unnecessary columns and rows
    df = df.drop(['Number SV'], axis=1, errors='ignore')
    df = df.drop(['Units'], axis=0, errors='ignore') 

    #seperate results from inputs
    result_colnames = list(df.columns)[list(df.columns).index('Architecture Outputs') + 1:]
    output = df[result_colnames]

    #seperate inputs
    input_colnames = list(df.columns)[:list(df.columns).index('Architecture Outputs')]
    orbital = df[input_colnames]

    #change index names to constellation in data
    output.index = output.index.map(str)
    orbital.index = orbital.index.map(str)
    for i in range(len(output.index.values)):
        output.index.values[i] = 'Constellation ' + str(i+1)
        orbital.index.values[i] = 'Constellation ' + str(i+1)
        # output.index.values[i] = output.index.values[i].replace("Value", "Constellation")
        # orbital.index.values[i] = orbital.index.values[i].replace("Value", "Constelaltion")
    
    # return the two new dataframes
    return orbital, output

#parse csv file contents
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), 
                index_col=0).transpose()
            orbital, output = splitData(df)
            
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded), index_col=0).transpose()
            orbital, output = splitData(df)

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    return filename, orbital, output

#Load in data for app
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('upload-data', 'last_modified')])
def update_output(contents, filename, date):
    if contents is not None:
        filename, inp, out = parse_contents(contents, filename, date)
        return html.Div([
                html.Div([inp.to_json(orient = 'split')], 
                id='input_data',style={'display': 'none'}),
                html.Div([out.to_json(orient = 'split')], 
                id='output_data',style={'display': 'none'})
                ])

#Update graph and pass data
@app.callback(Output('home-output-container', 'children'),
              [Input('output_data', 'children')])
def update_home_graph(data):
    if data is not None:
        dff = pd.read_json(data[0], orient='split')

        return html.Div([
            html.Div([
                html.Div('X-axis', 
                        style={'fontSize': 20, 'width': '33%', 
                            'display': 'inline-block', 'text-align': 'center'}),
                html.Div('Y-axis', 
                        style={'fontSize': 20, 'width': '33%', 
                            'display': 'inline-block', 'text-align': 'center'}),
                html.Div('Color', 
                        style={'fontSize': 20, 'width': '33%', 
                            'display': 'inline-block', 'text-align': 'center'}),

                # x axis selection
                html.Div([
                    dcc.Dropdown(
                        id='crossfilter-xaxis-column',
                        options=[{'label': i, 'value': i} for i in dff.columns],
                        value=dff.columns[0],
                        className='dropdown'
                    ),
                ],
                style={'width': '33%', 'display': 'inline-block', 
                        'text-align': 'center'}),
                
                # y axis selection
                html.Div([
                    dcc.Dropdown(
                        id='crossfilter-yaxis-column',
                        options=[{'label': i, 'value': i} for i in dff.columns],
                        value=dff.columns[1],
                        className='dropdown'
                    ),
                ], style={'width': '33%', 'display': 'inline-block',
                        'text-align': 'center'}),
                
                # color selection
                html.Div([
                    dcc.Dropdown(
                        id='crossfilter-color',
                        options=[{'label': i, 'value': i} for i in dff.columns],
                        value=dff.columns[2],
                        className='dropdown'
                    ),
                ],
                style={'width': '33%', 'display': 'inline-block',
                        'text-align': 'center'}),

                #scatterplot component that will have performance metrics plotted
                dcc.Graph(
                    id='crossfilter-indicator-scatter',
                ),
            ], style={'padding':'2%','width':'90%','background-color':'#1F1B24','margin':'20px auto'}),

            #Display info of clicked constellation
            html.Div(id='click-data', style={'padding':'2%','width':'40%','background-color':'#1F1B24','margin':'20px auto'}),

            #button for navigation to next page
            dcc.Link('Inputs', href='/inputs', className='nav-button', 
                style={'margin-left':'45%'}),
        ])

#Update graph based off dropdowns
@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    [Input('crossfilter-xaxis-column', 'value'),
     Input('crossfilter-yaxis-column', 'value'),
     Input('crossfilter-color', 'value'),
     Input('output_data', 'children')])
def update_graph(xaxis_column_name, yaxis_column_name, color_column_name, 
                out_data):

    #read data from output
    dff = pd.read_json(out_data[0], orient='split')

    #craete dynamic plot
    new_figure = px.scatter(dff, x=xaxis_column_name, y=yaxis_column_name, 
        template='plotly_dark', color=color_column_name)

    #Update marker layout
    new_figure.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
    return new_figure

#displays info on clicked constellation
@app.callback(
    Output('click-data', 'children'),
    [Input('crossfilter-indicator-scatter', 'clickData'),
     Input('crossfilter-xaxis-column', 'value'),
     Input('crossfilter-yaxis-column', 'value'),
     Input('output_data', 'children')])
def display_click_data(clickData, x_col, y_col, data):
    if clickData:

        #find the clicked data point within the
        dff = pd.read_json(data[0], orient='split').select_dtypes(exclude=[object])
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']
        entry = dff[dff[x_col] == x][dff[y_col] == y]
        result = [
            html.H1(entry.index[0]),
        ]

        #add all values
        for col in dff.columns:
            metric = entry[col]

            #if series object get the first entry
            if len(metric) > 1:
                metric = metric[0]
            
            #create string for display
            string = col + ": " + str(float(metric))
            result.append(html.P(string))

        return result

    return ""


#################### Input Page #########################

#base HTML for page
input_page = html.Div([
    html.H1('Input Parameter Weights',
    style = {'width': '80%', 'display': 'flex', 'align-items': 'center',
            'justify-content': 'center', 'background-color':'#1F1B24',
            'margin':'auto', 'padding':'2%'}),
    html.Div(id='input-output-container'),
        html.Div([
        dcc.Link('Back', href='/', className='nav-button'),
        dcc.Link('Plot', href='/plots', className='nav-button'),
    ], className='nav-container', style={'margin-left':'50%'})
])

# Build weight input page with initial content (dropdown, checklist, button)
@app.callback(Output('input-output-container', 'children'),
              [Input('output_data', 'children')])
def update_input_page(output_data):
    if output_data is not None:
        output_data = pd.read_json(output_data[0], orient='split').select_dtypes(exclude=[object])
        metricNames = output_data.columns
        numMetrics = len(metricNames)

        return html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Div('Select metrics to assign weights to. ' +
                            'If none are selected, all metrics are given equal ' +
                            'weight: '),
                        dcc.Dropdown(
                            id='metric-dropdown',
                            options=[{'label': name, 'value': name} for name in metricNames],
                            multi=True,
                            value=[],
                            className='dropdown'
                        ),
                        html.Div(id='weightSection'),
                        html.Div(id='weightSection2'),
                        html.Button(id='weightBtn', children='Score Constellations',
                            style={
                                'margin': '20px auto',
                                'background-color':'#03DAC5'
                        }),
                    ]),
                    html.Div([
                        html.Div(
                            children='Select Metrics that are better when smaller:'
                        ),
                        dcc.Checklist(
                            id='metric-checklist',
                            options=[{'label': name, 'value': name} for name in metricNames],
                            value=[],
                        ),
                    ]),
                    html.Div(id='scoreTableContainer'),
                ], 
                className='gridinput'),
            ], style={
                'padding':'2%',
                'width':'80%',
                'background-color':'#1F1B24',
                'margin':'20px auto',
            }),
               

            #container for plots
            html.Div([
                html.Div(id='metricGraphContainer', className='gridplot')
            ],
            style={'padding':'2%',
                'width':'80%',
                'background-color':'#1F1B24',
                'margin':'20px auto',
            }),
        ],)
    else:
        return html.Div([
            html.H3("Load in Data first. Go back to home page"),
        ])

# callback for making weight input boxes based on selected metrics from dropdown
@app.callback(
    Output('weightSection', 'children'),
    [Input('metric-dropdown', 'value'),
    Input('output_data', 'children')])
def createWeightBoxes(selectedMetrics, output):
    results = []

    for metric in selectedMetrics:
        results.append(
            html.Div(id='label ' + metric, children=[html.Label(metric)],
                    style={'width': '30%', 'margin-right': '1%'}))

        results.append(
            html.Div(id='metric ' + metric, children=[
                dcc.Input(
                    id='pm ' + metric,
                    placeholder='0.0 <= Weight <= 1.0',
                    value='',
                    type='number',
                    style={'height': '10%',
                        'font-size': '100%'}
                ),
            ], style={'width': '30%'}))

    return results

# callback for displaying weights on the page for metrics that weren't selected.
@app.callback(
    Output('weightSection2', 'children'),
    [Input('otherWts', 'children'), Input('output_data', 'children')],
    [State('metric-dropdown', 'value')])
def createOtherWeightBoxes(otherWts, output, selectedMetrics):
    # convert output (performance metrics) data from json to dataframe
    output = pd.read_json(output[0], orient='split').select_dtypes(exclude=[object])
    
    # determine the metrics that were not selected for weights
    otherMetrics = set(output.columns) - set(selectedMetrics)
    
    otherWts = str(round(otherWts,3))
    results = []

    # for the non-selected metrics, display their calculated weight
    for metric in otherMetrics:
        results.append(
            html.Div(id='label ' + metric, 
                    children=[html.Label(metric + ': ' + otherWts)],
                    style={'width': '30%', 'margin-right': '1%'}))

    return results

# callback to make score table based on all weights
@app.callback(
    Output('scoreTableContainer', 'children'),
    [Input('weightBtn', 'n_clicks'), 
    Input('output_data', 'children'),
    Input('metric-checklist', 'value')],
    [State('weightSection', 'children')]
)
def createScores(n_clicks, output, metrics, components):
    output = pd.read_json(output[0], orient='split').select_dtypes(exclude=[object])
    if metrics:
        for metric in metrics:
            output.loc[:,metric] = 1/output.loc[:,metric]
    metricNames = output.columns
    numMetrics = len(metricNames)
    allWts = pd.DataFrame(columns=metricNames)
    chosenWts = []
    chosenMetrics = []    

    # these are the div components within the weightSection div
    if components:
        for component in components:

            # reduce down to the leaf component
            component = component['props']['children'][0]['props']
            if 'children' in component: # it is a metric label
                chosenMetrics.append(component['children'])
            if 'value' in component: # it is an Input box
                # display error if weight was not entered ###################################
                if (component['value'] is None) | (component['value']==''):
                    return html.Div('Error: One or more weight boxes are empty.')

                wt = float(component['value'])
                # check if weights are between 0.0 and 1.0
                if (wt<0) | (wt>1.0):
                    return html.Div('Error: All weights must be in the range ' +
                        '0.0 to 1.0')
                chosenWts.append(float(component['value']))

        # ensure user's weights don't add to more than 1.0 
        if sum(chosenWts) > 1.0:
            return html.Div('Error: The sum of your weights cannot be more ' +
                'than 1.0')

    # put user selected weights into dataframe
    allWts.loc[0, chosenMetrics] = chosenWts

    # put rest of calculated weights into dataframe
    numOtherWts = numMetrics - len(chosenWts)
    if numOtherWts != 0: # avoid dividing by zero
        otherWts = (1-sum(chosenWts)) / numOtherWts
        allWts.fillna(otherWts, inplace=True)
    elif sum(chosenWts) < 1.0:
        return html.Div('Error: You selected all performance metrics, ' +
            ' but the weights do not add to 1.0')

    # make it a list to multiply by normalized output dataframe
    allWts = allWts.iloc[0].tolist()

    # calculate scores
    metricsNorm = (output-output.mean()) / output.std()

    # metricsNorm[metricsNorm.loc[:,:] > 3] = 3
    scoresIndiv = metricsNorm*allWts
    scores = (scoresIndiv).sum(axis=1)
    scores = scores.sort_values(ascending=False)
    scores = scores.round(3)

    # scores has to be a dataframe to make a dash data table
    scores = pd.DataFrame({'id':scores.index, 'Score':scores})

    if n_clicks:
        return [createScoreTable(scores),
            html.Div(otherWts, id='otherWts', style={'display': 'none'})]
def createScoreTable(scores):
    table = dash_table.DataTable(
        id='scoreTable',
        columns=[
            {"name": i if i!='id' else 'Constellation', "id": i, 
            "deletable": True, "selectable": True} for i in scores.columns
        ],
        data = scores.to_dict('records'),
        editable=True,
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        row_selectable="multi",
        row_deletable=True,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        page_current= 0,
        page_size= 7,
        style_header={'backgroundColor': 'rgb(30, 30, 30)'},
        style_cell={
        'backgroundColor': 'rgb(50, 50, 50)',
        'color': 'white'
        },
    )

    return table

# Callback for highlighting table rows when selected
@app.callback(
    Output('scoreTable', 'style_data_conditional'),
    [Input('scoreTable', 'selected_rows'),
    Input('scoreTable', 'active_cell')]
)
def update_styles(selected_rows, active_cell):
    conditional_style =  [
        {
            'if': { 'row_index': i },
            'background_color': '#BB86FC'
        } 
        for i in selected_rows]
    
    #add active style
    if active_cell:
        conditional_style.append({
            'if': { 'row_index': active_cell['row']},
            'background_color': '#03DAC5'
        })
    return conditional_style

# Callback for creating graphs of performance metric scores for selected
# constellations
@app.callback(
    Output('metricGraphContainer', 'children'),
    [Input('scoreTable', 'derived_virtual_row_ids'),
     Input('scoreTable', 'selected_row_ids'),
     Input('scoreTable', 'active_cell'),
     Input('output_data', 'children')],
)
def createMetricGraphs(row_ids, selected_row_ids, active_cell, output):
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncracy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.

    output = pd.read_json(output[0], orient='split').select_dtypes(exclude=[object])

    if selected_row_ids is None:
        return

    # resort selected rows based on current order of all rows from table
    selected_row_ids = [id for id in row_ids[::-1] 
                            if id in selected_row_ids]
    df = output.loc[selected_row_ids]

    active_row_id = active_cell['row_id'] if active_cell else None

    colors = []
    for id in selected_row_ids:
        if id == active_row_id: colors.append('#03DAC5')
        else: colors.append('#BB86FC')

    graphs = [
        dcc.Graph(
            id='metricGraph ' + column,
            figure={
                'data': [
                    {
                        'x': df[column],
                        'y': df.index,
                        'type': 'bar',
                        'orientation': 'h',
                        'marker': {'color': colors},
                    }
                ],
                'layout': {
                    'xaxis': {'automargin': True,
                        'title': {'text': column}
                    },
                    'yaxis': {
                        'automargin': True,
                        'title': {'text': 'Constellation'}
                    },
                    'margin': {'t': 10, 'l': 10, 'r': 10}
                },
                
            },
        )
    for column in df]

    return graphs

# store orbital elements for constellations selected in score table
@app.callback(
    Output('orbitPlotInfo', 'children'),
    [Input('scoreTable', 'derived_virtual_row_ids'),
     Input('scoreTable', 'selected_row_ids'),
     Input('input_data', 'children')]
)
def storeOrbitInfo(row_ids, selected_row_ids, input):
    input = pd.read_json(input[0], orient='split').select_dtypes(exclude=[object])
    if selected_row_ids is None:
        return

    # resort selected rows based on current order of all rows from table
    selected_row_ids = [id for id in row_ids if id in selected_row_ids]
    df = input.loc[selected_row_ids]
    return df.to_json(orient = 'split')

##TODO #################### Plots Page #########################

plots_page = html.Div([
    html.H1('Plots for Selected Constellations'),
    # This button will pull the JSON data into a new callback
    html.Div([html.Div(id='orbitPlotsContainer', className='gridplot2')],
            style={'padding':'2%',
                'width':'80%',
                'background-color':'#1F1B24',
                'margin':'20px auto',
            }
    ),
    dcc.Link('Back', href='/inputs', className='nav-button'),
])

@app.callback(
    Output('orbitPlotsContainer', 'children'),
    [Input('orbitPlotInfo', 'children')]
)
def createOrbitPlots(orbitElements):
    if orbitElements is None:
        return
    orbitElements = pd.read_json(orbitElements, 
                        orient='split').select_dtypes(exclude=[object])

    # remove special chars, spaces, puncation from variables, and make lowercase
    colNames = [re.sub('[^A-Za-z0-9]+', '', name).lower() for name in orbitElements.columns]
    orbitElements.columns = colNames

    # determine number of satellites by using regex on orbital element variables
    results = [re.search(r'\d+', name) for name in orbitElements.columns]
    numSats = max([int(result.group(0)) for result in results if result])

    graphs = []

    # create surface object for earth in 3d plot (radius 6371km)
    theta = np.linspace(0,2*pi,100)
    phi = np.linspace(0,pi,100)
    x = 6371*np.outer(np.cos(theta),np.sin(phi))
    y = 6371*np.outer(np.sin(theta),np.sin(phi))
    z = 6371*np.outer(np.ones(100),np.cos(phi))
    earth = go.Surface(x=x, y=y, z=z, showscale=False, colorscale='Blues')

    # for each constellation
    for const, row in orbitElements.iterrows():
        data = []

         # for each satellite in current constellation
        for sat in range(1,numSats+1):
            if 'sv1semimajoraxis' in colNames: # if classical orbital elems
                a = row['sv'+str(sat)+'semimajoraxis']
                e = row['sv'+str(sat)+'eccentricity']
                i = radians(row['sv'+str(sat)+'inclination'])
                raan = radians(row['sv'+str(sat)+'raan'])
                argPe = radians(row['sv'+str(sat)+'argumentofperigee'])
                truAnom = np.arange(0,2*pi,0.01)

            elif 'svalta1' in colNames: # if classical orbital elems but with altitudes
                alta = row['sv'+'alta'+str(sat)]
                altb = row['sv'+'altb'+str(sat)]
                r_a = max(alta, altb) + 6371
                r_p = min(alta, altb) + 6371
                a = (r_a + r_p) / 2
                e = (r_a - r_p) / (r_a + r_p)
                i = radians(row['sv'+'inc'+str(sat)])
                raan = radians(row['sv'+'raan'+str(sat)])
                argPe = radians(row['sv'+'aop'+str(sat)])
                truAnom = np.arange(0,2*pi,0.01)
            
            else: return
            
            orbitPos = calcPositions(a, e, i, raan, argPe, truAnom)
            orbitPos = array([orbitPos[j][0] for j in range(3)])

            trace = go.Scatter3d(x=orbitPos[0,:], y=orbitPos[1,:], 
                        z=orbitPos[2,:], mode="lines",
                        line=dict(width=2),
                        text='Satellite ' + str(sat),
                        name='Satellite ' + str(sat))

            data.append(trace)
        data.append(earth)

        # Create figure for this constellation
        fig = dict(
            data=data,
            layout=go.Layout(
                xaxis=dict(autorange=False, zeroline=False),
                yaxis=dict(autorange=False, zeroline=False),
                title_text=const, hovermode="closest",
                )
        )

        # create graph component containing constellation figure
        graph = dcc.Graph(id='orbitPlot ' + const,
                    figure=fig)

        # add graph to list of current constellation graphs
        graphs.append(graph)

    return graphs
# all angles in radians
def calcPositions(a,e,i,raan,argPe,truAnom):
    p = a*(1-pow(e,2)) # parameter
    r = p / (1+e*cos(truAnom))
    
    # r vec in perifocal frame
    rPerifocal = array([[r*cos(truAnom)], [r*sin(truAnom)], [0]])

    # calc elements of rotation matrix
    R11 = cos(raan)*cos(argPe)-sin(raan)*sin(argPe)*cos(i)
    R12 = -cos(raan)*sin(argPe)-sin(raan)*cos(argPe)*cos(i)
    R13 = sin(raan)*sin(i)
    R21 = sin(raan)*cos(argPe)+cos(raan)*sin(argPe)*cos(i)
    R22 = -sin(raan)*sin(argPe)+cos(raan)*cos(argPe)*cos(i)
    R23 = -cos(raan)*sin(i)
    R31 = sin(argPe)*sin(i)
    R32 = cos(argPe)*sin(i)
    R33 = cos(i)
    R = array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])

    # r vec in geocentric frame
    rGeo = R@rPerifocal
    return rGeo

#################### Router for app ##################

#router for app
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return index_page
    elif pathname == '/inputs':
        return input_page
    elif pathname == '/plots':
        return plots_page
    else:
        return "Error 404: Page Not Found."

#Run the app
if __name__ == '__main__':
    app.run_server(debug=True)