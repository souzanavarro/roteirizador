import requests
import streamlit as st
import random
import networkx as nx
from itertools import permutations
from geopy.distance import geodesic
from sklearn.cluster import KMeans, DBSCAN
import folium
from config import endereco_partida, endereco_partida_coords
import math
import pandas as pd
import logging
import numpy as np

def obter_coordenadas_opencage(endereco):
    """
    Obtém as coordenadas de um endereço utilizando a API do OpenCage.
    """
    try:
        api_key = "6f522c67add14152926990afbe127384"  # Substitua pela sua chave de API
        url = f"https://api.opencagedata.com/geocode/v1/json?q={endereco}&key={api_key}"
        response = requests.get(url)
        data = response.json()
        if 'status' in data and data['status']['code'] == 200 and 'results' in data:
            location = data['results'][0]['geometry']
            return (location['lat'], location['lng'])
        else:
            st.error(f"Não foi possível obter as coordenadas para o endereço: {endereco}.")
            return None
    except Exception as e:
        st.error(f"Erro ao tentar obter as coordenadas: {e}")
        return None

def obter_coordenadas_com_fallback(endereco, coordenadas_salvas):
    """
    Retorna as coordenadas salvas para um endereço ou tenta obtê-las via OpenCage.
    Se não obtiver, utiliza um dicionário de coordenadas manuais pré-definido.
    """
    if endereco in coordenadas_salvas:
        return coordenadas_salvas[endereco]
    
    coords = obter_coordenadas_opencage(endereco)
    if coords is None:
        # Exemplo de coordenadas manuais para endereços específicos
        coordenadas_manuais = {
            "Rua Araújo Leite, 146, Centro, Piedade, São Paulo, Brasil": (-23.71241093449893, -47.41796911054548)
        }
        coords = coordenadas_manuais.get(endereco, (None, None))
    
    if coords:
        coordenadas_salvas[endereco] = coords
    return coords

def calcular_distancia(coords_1, coords_2):
    """
    Calcula a distância em metros entre duas coordenadas.
    """
    if coords_1 and coords_2:
        return geodesic(coords_1, coords_2).meters
    return None

def resolver_vrp_separado(pedidos_df, caminhoes_df):
    """
    Resolve o problema do VRP com separação por vez das cargas e permite seleção de placas.
    """
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        return "Erro: OR-Tools não está instalado. Instale com: pip install ortools"

    # Obtenha as coordenadas dos pedidos
    coords = list(zip(pedidos_df['Latitude'], pedidos_df['Longitude']))
    if not coords:
        return "Sem pedidos para roteirização."

    depot = 0  # Local de partida

    def calcular_dist(i, j):
        return int(math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2) * 1000)

    N = len(coords)
    distance_matrix = [[calcular_dist(i, j) for j in range(N)] for i in range(N)]

    # Número de veículos igual ao número de pedidos (separação individual)
    num_vehicles = len(pedidos_df)

    # Seleção de placas
    st.write("Selecione as placas dos veículos:")
    placas = caminhoes_df['Placa'].tolist()
    placas_selecionadas = st.multiselect("Escolha as placas dos veículos:", options=placas, default=placas)

    if not placas_selecionadas:
        return "Nenhuma placa selecionada para roteirização."

    # Criação do modelo de roteamento
    manager = pywrapcp.RoutingIndexManager(N, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Parâmetros de busca
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        routes = {}
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(pedidos_df.iloc[node]['Endereço Completo'])
                index = solution.Value(routing.NextVar(index))
            if vehicle_id < len(placas_selecionadas):
                routes[f"Veículo {placas_selecionadas[vehicle_id]}"] = route
            else:
                routes[f"Veículo {vehicle_id + 1}"] = route
        return routes
    else:
        return "Não foi encontrada solução para o problema VRP."
