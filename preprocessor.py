"""
Módulo de pré-processamento

Realiza a validação, limpeza e normalização dos dados recebidos nas planilhas.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, filename="preprocessor.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

def preprocessar_dados(df):
    """
    Pré-processa os dados:
      - Preenche valores faltantes.
      - Converte colunas para numérico.
      - Normaliza colunas, se aplicável.
    
    Parâmetros:
      df (DataFrame): Dados a serem processados.
    
    Retorna:
      DataFrame: Dados pré-processados.
    """
    df.fillna(0, inplace=True)
    for coluna in ['Peso dos Itens', 'Volume', 'Distância']:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
            max_val = df[coluna].max()
            if max_val and max_val > 0:
                df[coluna] = df[coluna] / max_val
    return df