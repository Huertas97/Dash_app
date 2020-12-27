# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:23:13 2020

@author: alvar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:55:08 2020

@author: alvar
"""


# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import plotly.express as px
import pandas as pd
import numpy as np
import tweepy
import json
from requests_oauthlib import OAuth1Session
import pickle
import dash_bootstrap_components as dbc

import random
import matplotlib.pyplot as plt
# %matplotlib inline 
import plotly.express as px
import plotly.graph_objects as go

# external_stylesheets = ["https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/sandstone/bootstrap.min.css"] #["https://codepen.io/chriddyp/pen/bWLwgP.css"] # ['https://codepen.io/chriddyp/pen/bWLwgP.css'] "https://www.w3schools.com/w3css/4/w3.css"
# external_stylesheets=external_stylesheets
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                title ="Semantic Search TFM")


# ============================================================================
# ============================================================================
# ---------------- VARIABLES --------------------
# ============================================================================
# ============================================================================


# ------------- Twitter ------------------- 
api_key = "h50aoVmiuNFuHI8o3dZE7C15N" 
api_secret_key = "Dz6jeUBUdGp43uJugObOgIqnVdCbUrbrkwkcjAibmlDQwq6sdL" 
access_token = "1311739853307027457-HqUEzNSGtzdFqkFDFmdYg5UcMEjPv2" 
access_token_secret = "iabmz6wZ0gucIodSNJ9TfSfnrT17yJXjDAu13y4QF8hLI" 

twitter = OAuth1Session(api_key,
                        client_secret=api_secret_key,
                        resource_owner_key=access_token,
                        resource_owner_secret=access_token_secret)
# auth = tweepy.AppAuthHandler(api_key, api_secret_key)
auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True) # When we set wait_on_rate_limit to True, we will have our program wait automatically 15 minutes so that Twitter does not lock us out, whenever we exceed the rate limit and we automatically continue to get new data!

all_t = api.trends_available()
    
from tqdm import tqdm
names = []
woeid = []
for dic in tqdm(all_t, desc = "Twitter"):
    names.append(dic["name"])
    woeid.append(dic["woeid"])
    
trends_woeids = dict(zip(names, woeid))

# Options for Dropdown element in webpage
options = []
for k, v in trends_woeids.items():
    d = {"label": k, "value": v}
    options.append(d)
options = sorted(options, key = lambda d: list(d.values())[0])  # Sort dropdown elements

# country = "Spain" # input("Select country: ")
# id_country = trends_woeids[country]
# trends = api.trends_place(id_country)
# type(trends[0])
# t = json.dumps(trends[0]["trends"])
# df_twitter = pd.read_json(t)
# df_twitter = df_twitter.dropna(how = "all", axis = "columns").fillna("")

# ----------------- MARKDOWN TEXT
markdown_text = '''
* Using [Tweepy](http://docs.tweepy.org/en/latest/) we access to the trending topic
over the world. It's also possible to select a place among the WORLD IDs to show its
particular trending topic'
* The Semantic Search is accomplished by using the module [sentence-transformers](https://www.sbert.net/) 

**Autor**

Álvaro Huertas García
'''

# ------------------------ Style links --------------------------
style_links = {"color": "blue", "text-decoration": "underline",
       # "background-color": "#f7d297", "padding": "4px",
       "visited" : {"color":"purple"}, "cursor":"pointer",
       "float":"right", "margin-left":"30px"
                                                        }


# -------------------------------------- HEATMAP --------------------
def fancy_hover_text(textos):
  # Formateo los textos para que se visualicen adecuadamente en Hover
  textos_acortado = []
  for texto in tqdm(textos, desc = "Progress"):
    # tokenized_text = nltk.word_tokenize(texto)
    tokenized_text = texto.split()
    i = 5
    while i < len(tokenized_text):
      # for i in range(10,len(tokenized_text), 10):
      if i < 30:
        tokenized_text.insert(i, "<br>")

      # si la longitud de palabras es igual a 50 se cogeran las 50 primeras palabras
      else:
        texto = " ".join(tokenized_text[:i])
        texto = texto+"..."
        i = len(tokenized_text)
        continue
    
      # en caso de que i sea menor que 50 pero se cabe el texto hay que unirlo
      if i + 10 >= len(tokenized_text):
        texto = " ".join(tokenized_text[:])
        i = len(tokenized_text)
        continue

      i += 10
    textos_acortado.append(texto)
  return textos_acortado


import plotly.figure_factory as ff
def heatmap_cosine(cosine_dist, labels, textos, query,  
                   kwargs_for_annotated_heatmap={}, 
                   kwargs_for_update_layout={}):
  """
  Labels example --> ["query", "answer"]
  
  Textos example --> ["this is abstract 1", "this is abstract 2"]
  """
  # -------------- DATOS --------------
  textos_acortado = fancy_hover_text(textos)
  data_df = pd.DataFrame({"label": labels,  "acortado": textos_acortado, "score": cosine_dist.squeeze().numpy().tolist() })
  data_df = data_df.sort_values(by= "score")

  # -------------- REPRESENTACIÓN --------------
  z_round = np.around(np.array(data_df["score"].tolist()).reshape(-1, 1), decimals = 3) # Mostramos el valor del coseno redondeado en las anotaciones. Completo en Hover

  if "font_colors" not in kwargs_for_annotated_heatmap :
    kwargs_for_annotated_heatmap[ "font_colors"]  = ['black', 'white']# Colores de las anotaciones
  if "showscale" not in kwargs_for_annotated_heatmap:
    kwargs_for_annotated_heatmap["showscale"] = True # Mostrar color bar
  if "hovertemplate" not in kwargs_for_annotated_heatmap: # determina el estilo de la hover info
    kwargs_for_annotated_heatmap["hovertemplate"] = "<b>Query:</b> %{x}<br>" + "<br><b>Text:</b> %{y}</br><extra></extra>" + "<br><b>Similitud coseno:</b> %{z}"
  if "colorscale" not in kwargs_for_annotated_heatmap:
    kwargs_for_annotated_heatmap["colorscale"] = "Portland" #"Geyser" #"Agsunset", # "Viridis", # 'gnbu',

  # Heatmap anotado. 
  fig = ff.create_annotated_heatmap(z = np.array(data_df["score"].tolist()).reshape(-1, 1),
                                    
                                  annotation_text=z_round,
                                  x = query, # aparecerá en hover info Query es sentence 1 en formato lista
                                  y =  data_df["acortado"].tolist(), # aparecerá en hover info
                                  # font_colors = font_colors,
                                  # showscale = True, # Mostrar color bar
                                  # hovertemplate = "<b>Texto A:</b> %{x}<br>" +
                                  #                 "<br><b>Texto B:</b> %{y}</br><extra></extra>" +
                                  #                 "<br><b>Similitud coseno:</b> %{z}", 
                                  # colorscale="Geyser", 
                                  **kwargs_for_annotated_heatmap)
  
  # Titulo, posición y tamaño de la color bar
  fig.data[0].colorbar = dict(title='Similitud coseno', titleside = 'right', titlefont = {"size": 19})

  # Mínimo y máximo de la color bar
  fig.data[0].update(zmin=-1, zmax=1)

  # Tamaño de las anotaciones en el heatmap
  for i in range(len(fig.layout.annotations)):
    fig.layout.annotations[i].font.size = 16

  # Determinamos los nuevos ticks de los ejes con nuestro texto en una lista. Hay que indicar texto y posición (tickval)
  fig.update_yaxes(tickvals = data_df.index.tolist(), ticktext =  data_df["label"], tickfont= {"size": 12})
  fig.update_xaxes(tickvals = [0], ticktext =["Query"], tickfont= {"size": 12})



  # if "title" not in kwargs_for_update_layout:
  #   kwargs_for_update_layout["title"] = "Heatmap" 
  # if "title_font" not in kwargs_for_update_layout:
  #   kwargs_for_update_layout["title_font"] = {"size": 20} 
  # if "width" not in kwargs_for_update_layout:
  #   kwargs_for_update_layout["width"] = 1000 
  # if "height" not in kwargs_for_update_layout:
  #   kwargs_for_update_layout["height"] = 1000  
  
  fig.update_layout(
    autosize=True,
    margin=dict(l=90,
                r=90,
                b=100,
                t=120,
                pad=2),
    hoverlabel=dict(bgcolor="white", # Hover info formato
                    font_size=16,
                    font_family="Arial"
                    ),
    paper_bgcolor="white", # Color fondo imagen
    **kwargs_for_update_layout # hereda los parámetros dirigidos a esta funcion
    )
  return fig


# -------------------------- SENTENCE TRANSFORMERS --------------------------
from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# sentences1 = ['The cat sits outside'] # QUery

# sentences2 = ['The dog plays in the garden',
#               'A woman watches TV',
#               'The new movie is so great',
#               "How are you?"]

# #Compute embedding for both lists
# embeddings1 = model.encode(sentences1, convert_to_tensor=True)
# embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# #Compute cosine-similarits
# cosine_dist = util.pytorch_cos_sim(embeddings1, embeddings2)
# # print(cosine_dist.cpu().numpy())
# #Output the pairs with their score
# for i in range(len(sentences2)):
#     print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[0], sentences2[i], cosine_dist[0][i]))

# textos = sentences2
# labels = ["Texto_"+str(i) for i in range(len(textos))]

# 口罩不安全 // Las mascarillas no son seguras
#   الأقنعة ليست آمنة 
# Маски небезопасны  // Las mascarillas no son seguras

def corpus():
    sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great',
              "How are you?",
              "¿Cómo estás?",
              "La COVID-19 es una farsa, los políticos nos mienten",
              "Las mascarillas protegen frente a la propagación del virus SARS-CoV-2",
              "Heißes Wasser oder Dampf härten COVID-19 nicht aus",
              "@jairbolsonaro Ni #vacunachina ni #vacunarusa cada quien que elija si quiere alterar su organismo. Ya han fallecido mas de 36 personas en Corea del Sur por la vacuna de la gripe y no dejan saber la composición de la vacunas covid, en algunos países..estar atentos.",
              "Esta yendo a traer la #vacunaRusa para el pueblo boliviano. Quienes lo conocemos sabemos q no pierde el tiempo, y el trabajo para reconstruir #Bolivia ya comenzó.\n#BoliviaRecuperaSuDemocracia https://t.co/1VuVMKJQtP",
              "#IndustriaFarmacéutica | La vacuna #neumocócica de @MSDEspana obtiene resultados positivos en ensayos",
              "Bélgica extiende el toque de queda a las 22h después de registrar el viernes un nuevo récord de positivos: más de 15.400 casos en un día.\n\nLas reuniones en espacios públicos serán de máximo cuatro personas. \nhttps://t.co/R47rwH5aiX",
              "Pedro Sánchez prepara un Consejo de Ministros extraordinario este domingo, después de que diez comunidades hayan pedido el estado de alarma.",
              "Today is WorldPolioDay! 5 of the 6 WHO regions – representing more 90% of the population – are now free of the wild poliovirus! EndPolio",
              "Many countries don't have enough oxygen available to assist sick patients as they struggle to breathe. The oxygen project reflects WHO’s commitment to end-to-end solutions and innovation to do what we do better, cheaper and to reach more people. #COVID19 https://t.co/hSsfn3xKOC",
              "Qué sabemos de la mujer que \"perdió líquido cerebral después de realizarse una PCR\": tenía un defecto previo en la base del cráneo",
              "¿Te han llegado mensajes “antimascarillas”? Las mascarillas son seguras para nuestra salud y previenen la COVID-19. #HablanLosExpertos",
              "Ni infecciones, ni intoxicaciones ni afecciones pulmonares...Las #mascarillas protegen contra #COVID19 @EFEVerifica @EFEnoticias",
              "El 5G no ha causado ni propagado la COVID-19. Las ondas 5G no pueden interactuar con un virus.\n\n#DatosCoronavirus"
              
              
              ]
    return sentences2

def similarity (query):
    model = SentenceTransformer('distiluse-base-multilingual-cased')  # 'distilbert-base-nli-stsb-mean-tokens'
    sentences2 = corpus()
    #Compute embedding for both lists
    embeddings1 = model.encode(query, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    #Compute cosine-similarits
    cosine_dist = util.pytorch_cos_sim(embeddings1, embeddings2)
    textos = sentences2
    labels = ["Texto_"+str(i+1) for i in range(len(textos))]
    return (cosine_dist, textos, labels)

# =============================== PCA Embeddings Multilinguals ================
def PCA_plot(models, m_emb, pca_cum_var_ratio_list, title):
  fig = go.Figure()

  for i, model in enumerate(models):
    max_features = m_emb[i].shape[0]
    pca_cum_var_ratio = pca_cum_var_ratio_list[i]

    # Add cumulative variance ratio scatterplot
    fig.add_trace(go.Scatter(
      x=list(range(1, max_features)),
      y= pca_cum_var_ratio,
      #mode='lines+markers',
      name= model,
      hovertemplate = "<br>Cumulative variance: %{y:.4f} </br>Components: %{x}",
      line=dict(
          shape='spline',
          color=px.colors.qualitative.Set1[i]
          )
      ))
  # Add 
  fig.update_xaxes(showspikes=True, spikecolor="grey", spikethickness=1) #  spikemode="across")
  fig.update_yaxes(showspikes=True, spikecolor="grey", spikethickness=1, spikesnap="cursor")
  fig.update_layout(spikedistance=1000, hoverdistance=100)
  fig.update_layout(
      hoverlabel=dict(
          font_size=14,
          font_family="Arial",
          bgcolor = "white"),

      height=700, 
      width = 1000, 
      title=title) # "PCA varianza acumulada embeddings modelos multilinguales"

  fig.update_xaxes(
          tickangle = 0,
          tickfont = {"size": 15},
          title_text = "Number of Components",
          title_font = {"size": 20},
          title_standoff = 10)

  fig.update_yaxes(
          tickfont = {"size": 15},
          title_font = {"size": 20},
          title_text = "Cumulative variance (%)",
          title_standoff = 10)
  
  fig.update_layout(legend=dict(
    yanchor="bottom",
    y=0.02,
    xanchor="auto",
    x=0.98,
    bgcolor="white",
        bordercolor="Black",
        borderwidth=2)
    )


  return fig

# df_multi_emb = pd.read_pickle("./assets/df_multilinguals_STSb_Train_pca.pkl")
# models = df_multi_emb["modelo"].to_list()
# modelos_embeddings_train = df_multi_emb["matriz"].to_list()
# pca_cum_var_ratio_list = df_multi_emb["pca_var_acum"].to_list()
# title = "PCA cumulative variance plot - STSb Train - Multilingual Models"
# PCA_cum_var_fig = PCA_plot(models, modelos_embeddings_train, pca_cum_var_ratio_list, title)

# ============================================================================
# ============================================================================
# ---------------- APP --------------------
# ============================================================================
# ============================================================================

# We apply basic HTML formatting to the layout
app.layout = html.Div([dcc.Location(id='url', refresh=False),
                                  html.Div(id='page-content')])


# ============================================================================
# ============================================================================
# ---------------- INDEX PAGE --------------------
# ============================================================================
# ============================================================================

PLOTLY_LOGO =  "/assets/bioinformatics-polo-ggb.png" # "https://www.pologgb.com/wp-content/uploads/2018/04/bioinformatics-polo-ggb.png"#  "https://cdn1.iconfinder.com/data/icons/laboratory-15/48/08-dna_structure-electronics-biology-dna-education-science-monitor-computer-512.png"

search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Holz")),
        dbc.Col(
            dbc.Button("Search", color="primary", className="ml-2"),
            width="auto",
        ),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

NavBar = dbc.Navbar([html.A(dbc.Row([dbc.Col(html.Img(src=PLOTLY_LOGO, height="40px")),
                                     dbc.Col(dbc.NavbarBrand("Index", className="ml-2")),
                                     dbc.Col(dbc.NavLink("Home", href="/")),
                                     dbc.Col(dbc.NavLink("Semantic Search", href="/page-1"), width= "10px"),
                                     dbc.Col(dbc.NavLink("Description App", href="/page-2"), width = "10px"),
                                     ],
                                    align="center",
                                    justify = "center", 
                                    no_gutters=True,
                                    ),
                            href="#",
                            ),
                     dbc.NavbarToggler(id="navbar-toggler"),

                     dbc.Collapse(search_bar, id="navbar-collapse", navbar=True),
                     ],color="primary",
                                         dark=True,
                                         fixed = "top"
                                         )


index_page = dbc.Container(
    style = {"margin-top": "5%",},
                      children =[
                          html.Br(),
                          html.Br(),
        html.H1(children='Welcome to the TFM development Webpage',
            style = {
                "textAlign": "center"}
            ),
        
        html.H3(children="By: Álvaro Huertas García", 
                style = {"textAlign": "center"
            }
            ),
        NavBar, 
        
])

# # add callback for toggling the collapse on small screens
# @app.callback(
#     Output("navbar-collapse", "is_open"),
#     [Input("navbar-toggler", "n_clicks")],
#     [State("navbar-collapse", "is_open")],
# )
# def toggle_navbar_collapse2(n, is_open):
#     if n:
#         return not is_open
#     return is_open

# ============================================================================
# ============================================================================
# ---------------- PAGE 1 --------------------
# ============================================================================
# ============================================================================
popover = html.Div(
    [
        dbc.Button(
            "Display information", id="popover-target", color="info"
        ),
        dbc.Popover(
            [
                dbc.PopoverHeader("How it works",
                                  style = {"font-size": 18, "font-weight": "bold"}),
                dbc.PopoverBody(html.P(
                    """Introduce a query and top k fact-checked news will be searched. \n
                    The search is done using semantic similarity.
                    
                    The metric used for measuring the similarity is cosine similarity"""),
                                style = {"font-size": 16, 'whiteSpace': 'pre-line', "height": "200px"}),
            ],
            id="popover",
            is_open=False,
            target="popover-target",
            placement = "left",
        ),
    ]
)

similarity_heatmap = dcc.Graph(id='similarity-heatmap',
              style={"height":"500px", "width":"700px", "padding":"10px"}) 

# df = pd.DataFrame(
#     {
#         "First Name": ["Arthur", "Ford", "Zaphod", "Trillian"],
#         "Last Name": ["Dent", "Prefect", "Beeblebrox", "Astra"],
#     }
# )

# table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True,
#                                  dark = True, className="table-dark",
#                                  style = {"border": "5px solid black"})

# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')
# t =  dcc.Graph(figure= go.Figure(data=[go.Table(
#     header=dict(values=list(df.columns),
#                 fill_color='paleturquoise',
#                 align='left'),
#     cells=dict(values=[df["First Name"], df["Last Name"]],
#                fill_color='lavender',
#                align='left'))
# ]))


page_1_layout = dbc.Container(
    style = {"margin-top": "5%",},
                         children=[
                             # Title
                             html.H1(children='Semantic search',             
                                     style = {"textAlign": "center",
                                              # "color": "orange",
                                              # "text-shadow": "-1px 0 black, 0 1px black, 1px 0 black, 0 -1px black"
                                              }
                                     ),
                             
                             
                            NavBar,
                             

                            
                            # html.Div(id="Table-dbc"),
                            
                             # Links
                             # dcc.Link(
                             #     dbc.Button('Go back to home', color = "Primary", className="btn-primary"),
                             #                href='/', style = {"margin":"2px"}
                             #         ),
                             # dcc.Link(
                             #     dbc.Button('Go to Description App', color = "Primary", className="btn-primary"),
                             #                href='/page-2',
                             #         ), 
                             html.Br(),
                             html.Br(),
                             
                             # Text
                             html.H2(children="Trending topic search"),
                             html.H4(children='Web application for retrieving the current trending topics around the world.'),
                             
                             # # Dropdown, seleccionables
                             html.Div(id = "demo2-dropdown", 
                                      children = [
                                 dbc.InputGroup([
                                     dbc.InputGroupAddon(
                                         dbc.Button("Random place", id ="trendy-button"), 
                                         addon_type="prepend"
                                         ),
                                     dbc.Select(id = "trendy-dropdown", #"demo-dropdown",
                                         options=options,
                                         placeholder="Default: World"
                                         ),
                                     ]),
                                 ]),
                             html.Label(id="dropdown_selection", ), # style={"left": "40px"}
                             html.Div(id='intermediate-value', style={'display': 'none'}),
                            
                            # table,
                            dbc.Row([dbc.Col(id = "Tabla-plotly1", ),
                                    dbc.Col(id = "Tabla-plotly2")
                                    ]
                                    ),
                             
                             # Tables Trendic Topic
                             # html.Div(className="flex-container", 
                             #          style = {'display' : 'flex', 
                             #                   "flex-direction":"row",
                             #                   #"align-items":"flex-start",
                             #                   "justify-content": "space-around",
                             #                   "padding": "5px",
                             #                   }, 
                             #          children=[
                             #              html.Div(className= "four columns", 
                             #                       children =[
                             #                           html.Label(id= "table-title", ), #style= {"fontSize":12, "color":"blue"}
                             #                           html.Div(id='output-tabla'),
                             #                           ], 
                             #                       ),
                             #              html.Div(className= "four columns", 
                             #                       children =[
                             #                           html.Label(id= "table-title-2", ),
                             #                           html.Div(id='output-tabla-2'),
                             #                           ]
                             #                       )
                             #              ]
                             #          ),
                             html.Br(),
                             html.Br(),
                             
                             # Semantic Search text
                             html.H3(children = "Similarity search"),
                             popover, 
                             
                             # Semantic Search Input
                             dcc.Input(id="page-1-insert-text",
                                       type = "text",
                                       style={"position": "relative", "left": "40px", "top":"2px",
                                              "transition": {"duration" : 500, "easing":'cubic-in-out'} 
                                              },
                                       placeholder = "Insert text",
                                       debounce = True
                                       ),
                             # Input text intermidiate saved
                             html.Div(id="intermidiate-result-insert-text", 
                                      style={"display": "none"}
                                      ),
                             
                             # Heatmap Semantic Search
                             html.Div(id="heatmap-container" , 
                                      children = [
                                          dbc.Col(similarity_heatmap)

                                          ],
                                      )
                             ], 
                         )

# Recogemos la selección del dropdown y devolvemos el valor (WOEID) correspondiente a la etiqueta seleccionada en dropdown
@app.callback(
    Output(component_id = "dropdown_selection", component_property='children'),
    [Input(component_id = 'trendy-dropdown', component_property='value')])
def update_output(value):
    if value != None:
        return 'You have selected the Where On Earth IDentifier (WOEID): "{}"'.format(value)
    
# Recoge el resultado del dropdown, recuperamos la etiqueta (país) correspondiente al WOEID seleccionado
@app.callback([Output('table-title', 'children'), 
               Output('table-title-2', 'children')], 
              [Input(component_id = 'trendy-dropdown', component_property='value')])
def update_table_title(id_pais):
    if id_pais == None:
        id_pais = 1
    id_pais = int(id_pais)
    country = list(trends_woeids.keys())[list(trends_woeids.values()).index(id_pais)]
    output_1 = 'Column 1: 10 tendring topics from "{}"'.format(country)
    output_2 = 'Column 2: 10 tendring topics from "{}"'.format(country)
    return [output_1, output_2]

# Recoge el valor del dropdown y genera el dataframe con los Trending Topics. Lo outputeamos en una división oculta. Importante que este como JSON
@app.callback(
    Output(component_id = 'intermediate-value', component_property='children'),
    [Input(component_id = 'trendy-dropdown', component_property='value')])
def update_table(id_pais):
    # country = "Spain" # input("Select country: ")
    if id_pais == None:
        id_pais = 1
    id_country = id_pais # trends_woeids[country]
    trends = api.trends_place(id_country)
    if len(trends[0]["trends"]) == 0:
        df_twitter = pd.DataFrame(columns = ["name"])
        return df_twitter.to_json(date_format='iso', orient='split')
    else:
        t = json.dumps(trends[0]["trends"])
        df_twitter = pd.read_json(t)
        # df_twitter = df_twitter.drop(df_twitter.columns.difference(["name"]), axis = 1)
        df_twitter = df_twitter.dropna(how = "all", axis = "columns").fillna("")
        return df_twitter.to_json(date_format='iso', orient='split') #  generate_table(df_twitter)

    
@app.callback(
    Output(component_id = 'Table-dbc', component_property='children'),
    [Input(component_id = 'trendy-dropdown', component_property='value')])
def update_tabl_dbe(id_pais):
    # country = "Spain" # input("Select country: ")
    if id_pais == None:
        id_pais = 1
    id_country = id_pais # trends_woeids[country]
    trends = api.trends_place(id_country)
    if len(trends[0]["trends"]) == 0:
        df_twitter = pd.DataFrame(columns = ["name"])
        return dbc.Table.from_dataframe(df = df_twitter,
                                                     striped=True, bordered=True, hover=True,
                                 dark = True, className="table-info",
                                 style = {"border": "5px solid black", "overflow": "auto"}) #df_twitter # .to_json(date_format='iso', orient='split')
    else:
        t = json.dumps(trends[0]["trends"])
        df_twitter = pd.read_json(t)
        # df_twitter = df_twitter.drop(df_twitter.columns.difference(["name"]), axis = 1)
        df_twitter = df_twitter.dropna(how = "all", axis = "columns").fillna("")

        return dbc.Table.from_dataframe(df = df_twitter,
                                                     striped=True, bordered=True, hover=True,
                                 dark = False, className="table-primary",
                                 style = {"border": "5px solid black", "overflowY": "scroll"}) # .to_json(date_format='iso', orient='split') #  generate_table(df_twitter)
     
        
    
    
@app.callback(
    [Output(component_id = 'Tabla-plotly1', component_property='children'),
     Output(component_id = 'Tabla-plotly2', component_property='children'),],
    [Input(component_id = 'trendy-dropdown', component_property='value')])
def update_tabl_dbe(id_pais):
    # country = "Spain" # input("Select country: ")
    if id_pais == None:
        id_pais = 1
    id_country = id_pais # trends_woeids[country]
    trends = api.trends_place(id_country)
    if len(trends[0]["trends"]) == 0:
        df_twitter = pd.DataFrame(columns = ["name"])
        output = dcc.Graph(figure= go.Figure(data=[go.Table(
                         header=dict(values=list(["Trending Topics"]),
                                     fill_color='paleturquoise',
                                     align='left'),
                         cells=dict(values=[df_twitter["name"], 
                                            # df_twitter["url"],
                                            # df_twitter["query"],
                                            # df_twitter["tweet_volume"]
                                            ],
                                    fill_color='lavender',
                                    align='left'))
            ]))

        return [output, output]
    
    else:
        t = json.dumps(trends[0]["trends"])
        df_twitter = pd.read_json(t)
        # df_twitter = df_twitter.drop(df_twitter.columns.difference(["name"]), axis = 1)
        df_twitter = df_twitter.dropna(how = "all", axis = "columns").fillna("")
        output = dcc.Graph(figure= go.Figure(data=[go.Table(
                         header=dict(values=list(["Trending Topic"]),
                                     fill_color='paleturquoise',
                                     align='left'),
                         cells=dict(values=[df_twitter["name"], 
                                            # df_twitter["url"],
                                            # df_twitter["query"],
                                            # df_twitter["tweet_volume"]
                                            ],
                                    fill_color='lavender',
                                    align='left'))
            ]))

        return [output, output]



# Recuperamos el dataframe en formato JSON y generamos una tabla que es sacada por una nueva divison
@app.callback([Output('output-tabla', 'children'),
               Output('output-tabla-2', 'children')], 
              [Input('intermediate-value', 'children')])
def update_graph(jsonified_cleaned_data):
    # tweet_volume -->  this is the volume of tweets per trend for the last 24 hours.
    # more generally, this line would be
    # json.loads(jsonified_cleaned_data)
    df_twitter = pd.read_json(jsonified_cleaned_data, orient='split')
    # df_twitter = df_twitter.sort_values(by = "tweet_volume") # Hay valores vacíos NaN
    # return generate_table(df_twitter)
    output_1 = dt.DataTable(
                        # style table
            style_table={
                    'maxHeight': '50ex',
                    'overflowY': 'scroll',
                    'width': '100%',
                    'minWidth': '100%',
                    
                    },
            style_cell_conditional=[
                {
                    'if': {'column_id': c},
                    'textAlign': 'left'
                    } for c in ['Date', 'Region']
                ],
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': "#f5c382"
                    }
                ],
            style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                    },
            style_cell={
                    'textAlign': 'left', 'padding': '5px', 'backgroundColor': "#f2b25e", 
                    "fontFamily": "verdana"
                    },
            style_as_list_view=False,
            data=df_twitter.iloc[:, :].to_dict('rows'),
            columns=[{"name": i, "id": i,} for i in ([df_twitter["name"].name])],
            row_selectable=False,
            sort_action='native'       
            )
    output_2 = output_1
    return [output_1, output_2]

# Devolvemos el texto introducido por el usuario
@app.callback([Output('intermidiate-result-insert-text', 'children')], 
              [Input('page-1-insert-text', 'value')])
def text_user(text):
    return [text]

# Recogemos el texto introducido por el usuario y calculamos la similitud
@app.callback([Output('similarity-heatmap', 'figure')], 
              [Input('intermidiate-result-insert-text', 'children')])
def heatmap_similarity(query):
    if query is not None:
        cosine_dist, textos, labels = similarity([query])
        fig=heatmap_cosine(cosine_dist.cpu(), labels, textos, [query], {"colorscale":"blues"}, 
                          {"title":"Semantic search results"} ), 
        print(type(fig))
        return fig
    else:
        
        return [{
            "layout": {
                "xaxis": {
                    "visible": False
                    },
                "yaxis": {
                    "visible": False
                    },
                "annotations": [{
                    "text": "No matching data found",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 28
                        }
                    }]
                }
            }]

@app.callback(Output("heatmap-container", 'style'),
              [Input("page-1-insert-text",'value')])
def hide_graph(input):
    if input:
        return {'display' : 'flex', "align-items":"flex-start",
                      "justify-content": "center", "heigth": 1000}
    else:
        return {'display' : 'flex', "align-items":"flex-start",
                      "justify-content": "center", "heigth": 1000}
    
# Dar valores al dropdown de Trending topics
@app.callback(
    Output("trendy-dropdown", "value"),
    [Input("trendy-button", "n_clicks")],
)
def on_button_click(n_clicks):
    if n_clicks:
        # woeid
        # which = n_clicks % len(names)
        # print(which)
        # print(names[which])
        return random.choice(woeid)
    else:
        return None

# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("popover", "is_open"),
    [Input("popover-target", "n_clicks")],
    [State("popover", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open

# ============================================================================
# ============================================================================
# ---------------- PAGE 2 --------------------
# ============================================================================
# ============================================================================

page_2_layout = dbc.Container(id = "page_2", 
                         style = {"margin-top": "1%"},
                         children = [
                             
                             html.H1("Webpage Information", 
                                     style = {"textAlign": "Center"}),
                             
                             html.Div(id="link-1-page-2", 
                                      style = {"fontFamily": "verdana", "display":"inline"},
                                      children = [
                                          dcc.Link(
                                              dbc.Button('Go back to home', color = "Primary", className="btn-primary"),
                                              href='/', style = {"margin":"2px"}
                                              ),
                                          dcc.Link(
                                              dbc.Button('Go to Semantic Search', color = "Primary", className="btn-primary"),
                                              href='/page-1'),
                                          ]
                                      ),
                              NavBar,
                             html.Br(),
                             html.Br(),
                             
                             html.H2("Information" 
                                    ),
                             dcc.Markdown(children = markdown_text,
                                       highlight_config = {"theme" : "dark"},
                                       dangerously_allow_html = True,
                                       # style = {"backgroundColor": "#e6e6e6", "fontFamily": "verdana"}
                                       ),

                             html.Br(),
                             html.Br(), 
                             # dcc.Graph(id ="PCA_cum",
                             #           style={"display":"block",
                             #                  "margin":"auto",
                             #                  },
                             #           figure = PCA_cum_var_fig),

                             ], 
                         )
# Update the index. El truco está en que en el layout inicial tienes un Div que mostrará la 
# página que desees. Se inicia poniendo la Página de índice. Cuando clicas el enlace a otra página 
# en el layout inicial se muestra el contenido de esa página. De modo que no cambias de página, sino
# que en una division inicial cambias la salida en función de la página que quieras. 
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here

if __name__ == '__main__':
    app.run_server(debug=True)