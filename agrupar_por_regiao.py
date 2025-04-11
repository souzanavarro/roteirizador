import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

def agrupar_por_regiao(pedidos_df, metodo='kmeans', n_clusters=3, eps=0.01, min_samples=2):
    """
    Agrupa os pedidos em regiões utilizando K-Means ou DBSCAN com base em Latitude e Longitude.
    Adiciona a coluna 'Regiao' no dataframe.

    Args:
        pedidos_df (pd.DataFrame): DataFrame contendo os pedidos com colunas 'Latitude' e 'Longitude'.
        metodo (str): Método de agrupamento ('kmeans' ou 'dbscan').
        n_clusters (int): Número de clusters (apenas para K-Means).
        eps (float): Distância máxima entre pontos para formar um cluster (apenas para DBSCAN).
        min_samples (int): Número mínimo de pontos para formar um cluster (apenas para DBSCAN).

    Returns:
        pd.DataFrame: DataFrame com a coluna 'Regiao' indicando o cluster de cada pedido.
    """
    print("Aguarde, estamos separando os pedidos por regiões...")  # Mensagem de status

    required_columns = ['Latitude', 'Longitude']
    # Verifica se as colunas necessárias estão presentes
    if not all(col in pedidos_df.columns for col in required_columns):
        raise ValueError(f"As colunas necessárias {required_columns} não foram encontradas no DataFrame.")
    
    if pedidos_df.empty:
        pedidos_df['Regiao'] = []
        return pedidos_df

    coords = pedidos_df[required_columns].values

    if metodo == 'kmeans':
        # Agrupamento com K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        pedidos_df['Regiao'] = kmeans.fit_predict(coords)
    elif metodo == 'dbscan':
        # Agrupamento com DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
        pedidos_df['Regiao'] = dbscan.fit_predict(np.radians(coords))
    else:
        raise ValueError("Método de agrupamento inválido. Escolha 'kmeans' ou 'dbscan'.")

    # Verifica se algum ponto ficou sem cluster (apenas para DBSCAN)
    if metodo == 'dbscan' and (pedidos_df['Regiao'] == -1).any():
        pedidos_df['Regiao'] = pedidos_df['Regiao'].replace(-1, np.nan)  # Marcar como NaN para pontos não agrupados

    return pedidos_df