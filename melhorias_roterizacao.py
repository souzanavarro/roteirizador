import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.cluster import KMeans, DBSCAN
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO, filename="roterizacao.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

def calcular_distancia(coord1, coord2):
    """
    Calcula a distância em km entre duas coordenadas.
    """
    try:
        return geodesic(coord1, coord2).km
    except Exception as e:
        st.write(f"Erro calculando distância: {e}")
        return float('inf')

@st.cache_data
def gerar_matriz_distancias(pedidos_df):
    """
    Gera e armazena em cache a matriz de distâncias com base nas coordenadas dos pedidos.
    """
    coords = list(zip(pedidos_df['Latitude'], pedidos_df['Longitude']))
    n = len(coords)
    matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matriz[i][j] = calcular_distancia(coords[i], coords[j])
            else:
                matriz[i][j] = 0
    return matriz

def tsp_nearest_neighbor(pedidos_df):
    """
    Aplica a heurística do vizinho mais próximo para TSP e retorna a ordem dos índices.
    """
    matriz = gerar_matriz_distancias(pedidos_df)
    n = len(matriz)
    if n == 0:
        return []
    start = 0
    visited = [False] * n
    rota = [start]
    visited[start] = True
    ultimo = start
    # Para cada ponto, busca o mais próximo que ainda não foi visitado
    for _ in range(n - 1):
        proximos = [(matriz[ultimo][j], j) for j in range(n) if not visited[j]]
        if not proximos:
            break
        next_index = min(proximos, key=lambda x: x[0])[1]
        rota.append(next_index)
        visited[next_index] = True
        ultimo = next_index
    return rota

def route_distance(rota, matriz):
    """
    Calcula a distância total de uma rota utilizando a matriz de distâncias.
    """
    dist = 0
    for i in range(len(rota) - 1):
        dist += matriz[rota[i]][rota[i+1]]
    return dist

def otimizacao_2opt(rota, matriz):
    """
    Melhora a rota do TSP utilizando a heurística 2-opt.
    """
    best = rota
    improved = True
    while improved:
        improved = False
        for i in range(1, len(rota) - 2):
            for j in range(i + 1, len(rota)):
                if j - i == 1:  # segue vizinhos adjacentes sem alteração
                    continue
                new_route = best[:]
                new_route[i:j] = best[j - 1:i - 1:-1]
                if route_distance(new_route, matriz) < route_distance(best, matriz):
                    best = new_route
                    improved = True
        rota = best
    return best

def agrupar_por_regiao(pedidos_df, metodo='kmeans', n_clusters=3, eps=0.01, min_samples=2):
    """
    Agrupa os pedidos em regiões utilizando K-Means ou DBSCAN com base em Latitude e Longitude.
    """
    if pedidos_df.empty:
        pedidos_df['Regiao'] = []
        return pedidos_df

    coords = pedidos_df[['Latitude', 'Longitude']].values

    if metodo == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        pedidos_df['Regiao'] = kmeans.fit_predict(coords)
    elif metodo == 'dbscan':
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
        pedidos_df['Regiao'] = dbscan.fit_predict(np.radians(coords))
        pedidos_df['Regiao'] = pedidos_df['Regiao'].replace(-1, np.nan)  # Marcar como NaN pontos não agrupados
    else:
        raise ValueError("Método de agrupamento inválido. Escolha 'kmeans' ou 'dbscan'.")

    return pedidos_df

# Bloco de interface Streamlit para testes do TSP
try:
    pedidos_df = pd.read_excel("database/roterizacao_resultado.xlsx", engine="openpyxl")
except Exception as e:
    st.error("Planilha de Pedidos não encontrada. Envie a planilha de pedidos.")
    pedidos_df = pd.DataFrame()

# Validação de dados
required_columns = ['Latitude', 'Longitude', 'Peso dos Itens', 'Qtde. dos Itens']
if not all(col in pedidos_df.columns for col in required_columns):
    st.error(f"As colunas necessárias {required_columns} não foram encontradas no DataFrame.")
    st.stop()

if pedidos_df[required_columns].isnull().any().any():
    st.error("O DataFrame contém valores nulos. Verifique os dados e tente novamente.")
    st.stop()

if st.button("Roteirizar Pedidos"):
    st.write("Roteirização em execução...")
    st.write("Aguarde, estamos agrupando os pedidos por regiões...")
    
    # Escolha do método de agrupamento
    metodo = st.selectbox("Escolha o método de agrupamento:", ["kmeans", "dbscan"])
    if metodo == "kmeans":
        n_clusters = st.slider("Número de Clusters (K-Means):", min_value=2, max_value=10, value=3)
        pedidos_df = agrupar_por_regiao(pedidos_df, metodo=metodo, n_clusters=n_clusters)
    elif metodo == "dbscan":
        eps = st.slider("Distância Máxima (DBSCAN - eps):", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
        min_samples = st.slider("Mínimo de Pontos por Cluster (DBSCAN):", min_value=1, max_value=10, value=2)
        pedidos_df = agrupar_por_regiao(pedidos_df, metodo=metodo, eps=eps, min_samples=min_samples)

    st.write("Pedidos agrupados com sucesso!")
    
    # Relatório de clusters
    cluster_report = pedidos_df.groupby('Regiao').agg({
        'Peso dos Itens': 'sum',
        'Qtde. dos Itens': 'sum',
        'Regiao': 'count'
    }).rename(columns={'Regiao': 'Total de Pedidos'})

    st.write("Relatório de Clusters:")
    st.dataframe(cluster_report)
    
    # Seleciona os pedidos da região 0 para rodar o TSP
    pedidos_regiao = pedidos_df[pedidos_df['Regiao'] == 0].reset_index(drop=True)
    if not pedidos_regiao.empty:
        st.write("Calculando a rota otimizada...")
        
        # Ordena os pedidos por peso, quantidade de itens e coordenadas
        pedidos_regiao = pedidos_regiao.sort_values(
            by=['Peso dos Itens', 'Qtde. dos Itens', 'Latitude', 'Longitude'], 
            ascending=[False, False, True, True]
        )
        
        rota = tsp_nearest_neighbor(pedidos_regiao)
        matriz = gerar_matriz_distancias(pedidos_regiao)
        rota_otimizada = otimizacao_2opt(rota, matriz)
        distancia_total = route_distance(rota_otimizada, matriz)
        rota_enderecos = " → ".join(pedidos_regiao.loc[i, 'Endereço Completo'] for i in rota_otimizada)
        st.success(f"Rota Otimizada: {rota_enderecos}")
        st.info(f"Distância Total da Rota: {distancia_total:.2f} km")
    else:
        st.error("Não há pedidos na região selecionada para roteirização.")