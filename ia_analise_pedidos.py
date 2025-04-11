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

def criar_grafo_tsp(pedidos_df):
    """
    Cria um grafo (usando NetworkX) para o problema do caixeiro viajante (TSP).
    O nó de partida é definido em config e os demais nós são os endereços únicos da planilha.
    """
    G = nx.Graph()
    enderecos = pedidos_df['Endereço Completo'].unique()
    # Nó de partida
    G.add_node(endereco_partida, pos=endereco_partida_coords)
    for endereco in enderecos:
        coords = (
            pedidos_df.loc[pedidos_df['Endereço Completo'] == endereco, 'Latitude'].values[0],
            pedidos_df.loc[pedidos_df['Endereço Completo'] == endereco, 'Longitude'].values[0]
        )
        G.add_node(endereco, pos=coords)
    for (end1, end2) in permutations([endereco_partida] + list(enderecos), 2):
        distancia = calcular_distancia(G.nodes[end1]['pos'], G.nodes[end2]['pos'])
        if distancia is not None:
            G.add_edge(end1, end2, weight=distancia)
    return G

def resolver_tsp_genetico(G):
    """
    Resolve o TSP utilizando um algoritmo genético simples.
    Retorna a melhor rota encontrada e sua distância total.
    """
    def fitness(route):
        return sum(G.edges[route[i], route[i+1]]['weight'] for i in range(len(route) - 1)) + \
               G.edges[route[-1], route[0]]['weight']

    def mutate(route):
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
        return route

    def crossover(route1, route2):
        size = len(route1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = route1[start:end]
        pointer = 0
        for i in range(size):
            if route2[i] not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = route2[i]
        return child

    def genetic_algorithm(population, generations=1000, mutation_rate=0.01):
        for _ in range(generations):
            population = sorted(population, key=lambda route: fitness(route))
            next_generation = population[:2]
            for _ in range(len(population) // 2 - 1):
                parents = random.sample(population[:10], 2)
                child = crossover(parents[0], parents[1])
                if random.random() < mutation_rate:
                    child = mutate(child)
                next_generation.append(child)
            population = next_generation
        return population[0], fitness(population[0])

    nodes = list(G.nodes)
    population = [random.sample(nodes, len(nodes)) for _ in range(100)]
    best_route, best_distance = genetic_algorithm(population)
    return best_route, best_distance

def resolver_vrp(pedidos_df, caminhoes_df):
    """
    Resolve o problema do VRP utilizando OR-Tools.
    
    O algoritmo constrói uma matriz de distâncias com base nas coordenadas dos pedidos.
    O número de veículos é determinado pelo número de caminhões disponíveis.
    
    Retorna:
      dict: Rotas para cada veículo, ou
      str: Mensagem de erro se a solução não for encontrada ou se OR-Tools não estiver instalado.
    """
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        return "Erro: OR-Tools não está instalado. Instale com: pip3 install ortools"

    # Obtenha as coordenadas dos pedidos
    coords = list(zip(pedidos_df['Latitude'], pedidos_df['Longitude']))
    if not coords:
        return "Sem pedidos para roteirização."

    depot = 0  # Usando o primeiro pedido (ou defina um depot específico)

    def calcular_dist(i, j):
        return int(math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2) * 1000)

    N = len(coords)
    distance_matrix = [[calcular_dist(i, j) for j in range(N)] for i in range(N)]

    num_vehicles = len(caminhoes_df)
    if num_vehicles < 1:
        return "Nenhum caminhão disponível para a roteirização."

    # Cria o index manager e o modelo de roteamento
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
            routes[f"Veículo {vehicle_id + 1}"] = route
        return routes
    else:
        return "Não foi encontrada solução para o problema VRP."

def otimizar_aproveitamento_frota(pedidos_df, caminhoes_df, percentual_frota, max_pedidos, n_clusters):
    """
    Otimiza a alocação dos pedidos aos caminhões disponíveis, agrupando os pedidos em regiões,
    atribuindo números de carga e placas.
    """
    # Inicializa as colunas
    pedidos_df['Carga'] = 0
    pedidos_df['Placa'] = ""
    carga_numero = 1
    
    # Ajusta a capacidade dos caminhões conforme o percentual informado
    caminhoes_df['Capac. Kg'] *= (percentual_frota / 100)
    caminhoes_df['Capac. Cx'] *= (percentual_frota / 100)
    # Filtra somente caminhões com disponibilidade "Ativo"
    caminhoes_df = caminhoes_df[caminhoes_df['Disponível'] == 'Ativo']
    
    # Agrupa os pedidos em regiões
    pedidos_df = agrupar_por_regiao(pedidos_df, n_clusters=n_clusters)
    
    for regiao in pedidos_df['Regiao'].unique():
        pedidos_regiao = pedidos_df[pedidos_df['Regiao'] == regiao]
        for _, caminhao in caminhoes_df.iterrows():
            capacidade_peso = caminhao['Capac. Kg']
            capacidade_caixas = caminhao['Capac. Cx']
            pedidos_alocados = pedidos_regiao[
                (pedidos_regiao['Peso dos Itens'] <= capacidade_peso) &
                (pedidos_regiao['Qtde. dos Itens'] <= capacidade_caixas)
            ]
            pedidos_alocados = pedidos_alocados.sample(n=min(max_pedidos, len(pedidos_alocados)))
            if not pedidos_alocados.empty:
                pedidos_df.loc[pedidos_alocados.index, 'Carga'] = carga_numero
                pedidos_df.loc[pedidos_alocados.index, 'Placa'] = caminhao['Placa']
                capacidade_peso -= pedidos_alocados['Peso dos Itens'].sum()
                capacidade_caixas -= pedidos_alocados['Qtde. dos Itens'].sum()
                carga_numero += 1
    
    if pedidos_df['Placa'].isnull().any() or pedidos_df['Carga'].isnull().any():
        st.error("Não foi possível atribuir placas ou números de carga a alguns pedidos.")
    
    return pedidos_df

def agrupar_por_regiao(pedidos_df, metodo='kmeans', n_clusters=3, eps=0.01, min_samples=2):
    """
    Agrupa os pedidos em regiões utilizando o nome da cidade e, dentro de cada cidade,
    aplica K-Means ou DBSCAN com base em Latitude e Longitude.

    Args:
        pedidos_df (pd.DataFrame): DataFrame contendo os pedidos com colunas 'Cidade de Entrega', 'Latitude' e 'Longitude'.
        metodo (str): Método de agrupamento ('kmeans' ou 'dbscan').
        n_clusters (int): Número de clusters (apenas para K-Means).
        eps (float): Distância máxima entre pontos para formar um cluster (apenas para DBSCAN).
        min_samples (int): Número mínimo de pontos para formar um cluster (apenas para DBSCAN).

    Returns:
        pd.DataFrame: DataFrame com a coluna 'Regiao' indicando o cluster de cada pedido.
    """
    if pedidos_df.empty:
        raise ValueError("O DataFrame de pedidos está vazio.")

    required_columns = ['Cidade de Entrega', 'Latitude', 'Longitude']
    if not all(col in pedidos_df.columns for col in required_columns):
        raise ValueError(f"As colunas necessárias {required_columns} não foram encontradas no DataFrame.")

    pedidos_df['Regiao'] = -1  # Inicializa a coluna de regiões
    regiao_id = 0

    # Agrupa os pedidos por cidade
    for cidade, grupo_cidade in pedidos_df.groupby('Cidade de Entrega'):
        coords = grupo_cidade[['Latitude', 'Longitude']].values

        if metodo == 'kmeans':
            # Agrupamento com K-Means
            kmeans = KMeans(n_clusters=min(n_clusters, len(grupo_cidade)), random_state=42)
            clusters = kmeans.fit_predict(coords)
        elif metodo == 'dbscan':
            # Agrupamento com DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
            clusters = dbscan.fit_predict(np.radians(coords))
            clusters = np.where(clusters == -1, np.nan, clusters)  # Marcar como NaN pontos não agrupados
        else:
            raise ValueError("Método de agrupamento inválido. Escolha 'kmeans' ou 'dbscan'.")

        # Atualiza o DataFrame com os clusters
        pedidos_df.loc[grupo_cidade.index, 'Regiao'] = clusters + regiao_id
        regiao_id += len(np.unique(clusters))  # Incrementa o ID da região para evitar sobreposição

    return pedidos_df

def criar_mapa(pedidos_df):
    """
    Cria e retorna um mapa Folium com marcadores para cada pedido e para o endereço de partida.
    """
    mapa = folium.Map(location=endereco_partida_coords, zoom_start=12)
    for _, row in pedidos_df.iterrows():
        popup_text = f"<b>Placa: {row['Placa']}</b><br>Endereço: {row['Endereço Completo']}"
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup_text,
            icon=folium.Icon(color='blue')
        ).add_to(mapa)
    folium.Marker(
        location=endereco_partida_coords,
        popup="Endereço de Partida",
        icon=folium.Icon(color='red')
    ).add_to(mapa)
    return mapa