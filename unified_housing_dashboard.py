
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Load dataset
df = pd.read_csv("train.csv")

# Clean up missing data
df = df.loc[:, df.isnull().mean() < 0.5]

# Identify columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Treat MSSubClass as categorical
if 'MSSubClass' in df.columns and 'MSSubClass' not in categorical_cols:
    categorical_cols.append('MSSubClass')
    if 'MSSubClass' in numerical_cols:
        numerical_cols.remove('MSSubClass')

# Prepare features for dropdowns
numerical_features_for_comparison = [col for col in numerical_cols if col != 'SalePrice']
all_features = categorical_cols + numerical_features_for_comparison

# Initialize Dash app

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)


# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Unified Housing Data Dashboard", className="text-center mb-4"), width=12)
    ]),

    dbc.Tabs([
        dbc.Tab(label="Categorical Comparison", tab_id="tab-categorical"),
        dbc.Tab(label="Numerical Comparison", tab_id="tab-numerical"),
    ], id="tabs", active_tab="tab-categorical"),

    html.Div(id="tab-content")

], fluid=True)


# Callback for rendering tab content
@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab_content(tab):
    if tab == "tab-categorical":
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Categorical Variable:"),
                    dcc.Dropdown(
                        id='categorical-dropdown',
                        options=[{'label': col, 'value': col} for col in categorical_cols],
                        value='Neighborhood',
                        clearable=False
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Compare with:"),
                    dcc.Dropdown(
                        id='comparison-dropdown',
                        options=[{'label': col, 'value': col} for col in all_features + ['SalePrice']],
                        value='SalePrice',
                        clearable=False
                    )
                ], width=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='comparison-plot'), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.H2("Distribution of Categorical Variable", className="text-center mt-4 mb-3"), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='categorical-count-plot'), width=12)
            ])
        ])
    elif tab == "tab-numerical":
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Label("Select X-axis Variable:"),
                    dcc.Dropdown(
                        id='x-axis-dropdown',
                        options=[{'label': col, 'value': col} for col in numerical_features_for_comparison],
                        value='GrLivArea',
                        clearable=False
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Select Y-axis Variable:"),
                    dcc.Dropdown(
                        id='y-axis-dropdown',
                        options=[{'label': col, 'value': col} for col in numerical_features_for_comparison + ['SalePrice']],
                        value='SalePrice',
                        clearable=False
                    )
                ], width=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='scatter-plot'), width=12)
            ]),
            dbc.Row([
                dbc.Col(html.H2("Distribution of Sale Price", className="text-center mt-4 mb-3"), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='sale-price-histogram'), width=12)
            ])
        ])


# Callback: Categorical comparison plot
@app.callback(
    Output('comparison-plot', 'figure'),
    [Input('categorical-dropdown', 'value'),
     Input('comparison-dropdown', 'value')]
)
def update_comparison_plot(categorical_col, comparison_col):
    if comparison_col in categorical_cols:
        grouped_counts = df.groupby([categorical_col, comparison_col]).size().reset_index(name='Count')
        fig = px.bar(grouped_counts, x=categorical_col, y='Count', color=comparison_col,
                     title=f'Count of {categorical_col} by {comparison_col}')
    else:
        fig = px.box(df, x=categorical_col, y=comparison_col,
                     title=f'{comparison_col} Distribution by {categorical_col}')
    fig.update_layout(xaxis_title=categorical_col, yaxis_title=comparison_col)
    return fig


# Callback: Categorical count plot
@app.callback(
    Output('categorical-count-plot', 'figure'),
    [Input('categorical-dropdown', 'value')]
)
def update_categorical_count_plot(categorical_col):
    counts = df[categorical_col].value_counts().reset_index()
    counts.columns = [categorical_col, 'Count']
    fig = px.bar(counts, x=categorical_col, y='Count',
                 title=f'Count of Properties by {categorical_col}')
    fig.update_layout(xaxis_title=categorical_col, yaxis_title='Count')
    return fig


# Callback: Scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)
def update_scatter_plot(x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col,
                     title=f'{y_col} vs {x_col}')
    return fig

# Callback: Sale price histogram
@app.callback(
    Output('sale-price-histogram', 'figure'),
    [Input('sale-price-histogram', 'id')]
)
def update_sale_price_histogram(_):
    fig = px.histogram(df, x='SalePrice', nbins=50,
                       title='Distribution of Sale Price',
                       marginal='box')
    fig.update_layout(xaxis_title='Sale Price', yaxis_title='Count')
    return fig


if __name__ == '__main__':
    app.run(debug=True)
