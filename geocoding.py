"""
Módulo de geocodificação

Contém funções que convertem endereços em coordenadas.
Utiliza caching em memória com functools.lru_cache para reduzir chamadas repetitivas.
"""

import os
import pandas as pd
import numpy as np
import logging
from functools import lru_cache
from geopy.geocoders import Nominatim
from config import DATABASE_FOLDER, GEOCODER_USER_AGENT, OPENCAGE_API_KEY

logging.basicConfig(level=logging.INFO, filename="geocoding.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

geolocator = Nominatim(user_agent=GEOCODER_USER_AGENT)

@lru_cache(maxsize=128)
def geocode_endereco(endereco):
    """
    Converte um endereço em (latitude, longitude).
    
    Retorna:
      tuple: (latitude, longitude) ou None se não conseguir geocodificar.
    """
    try:
        local = geolocator.geocode(endereco)
        if local:
            return (local.latitude, local.longitude)
    except Exception as e:
        logging.error(f"Erro na geocodificação do endereço '{endereco}': {e}")
    return None

def converter_enderecos(df, endereco_coluna="Endereço Completo", cache_filename="coordenadas_cache.xlsx"):
    """
    Atualiza o DataFrame com as colunas 'Latitude' e 'Longitude' para cada endereço.
    
    Utiliza um arquivo de cache para evitar geocodificações repetitivas.
    Atualiza o cache no disco somente se houver novas entradas.
    
    Parâmetros:
      df (DataFrame): DataFrame com os endereços.
      endereco_coluna (str): Nome da coluna de endereços.
      cache_filename (str): Nome do arquivo de cache.
    
    Retorna:
      DataFrame: com colunas 'Latitude' e 'Longitude' populadas.
    """
    cache_file = os.path.join(DATABASE_FOLDER, cache_filename)
    
    # Tenta carregar cache a partir do disco
    try:
        cache_df = pd.read_excel(cache_file, engine="openpyxl")
        cache = dict(zip(cache_df['Endereço'], zip(cache_df['Latitude'], cache_df['Longitude'])))
    except Exception as e:
        logging.info(f"Cache não encontrado ou erro na leitura: {e}")
        cache = {}
    
    latitudes = []
    longitudes = []
    for endereco in df[endereco_coluna]:
        if endereco in cache:
            lat, lon = cache[endereco]
        else:
            latlon = geocode_endereco(endereco)
            if latlon is None:
                lat, lon = (np.nan, np.nan)
            else:
                lat, lon = latlon
            cache[endereco] = (lat, lon)
        latitudes.append(lat)
        longitudes.append(lon)

    df['Latitude'] = latitudes
    df['Longitude'] = longitudes

    try:
        cache_df = pd.DataFrame(list(cache.items()), columns=['Endereço', 'Coordenadas'])
        cache_df[['Latitude', 'Longitude']] = pd.DataFrame(cache_df['Coordenadas'].tolist(), index=cache_df.index)
        cache_df.drop(columns=['Coordenadas'], inplace=True)
        cache_df.to_excel(cache_file, index=False)
    except Exception as e:
        logging.error(f"Erro ao atualizar o cache: {e}")
    
    return df