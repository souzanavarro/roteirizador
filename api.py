from flask import Flask, request, jsonify, send_file
import os
import io
import folium
import pandas as pd
import numpy as np
import random
from datetime import datetime
import logging

from geocoding import converter_enderecos
from preprocessor import preprocessar_dados
from optimization import run_genetic_algorithm
from config import DATABASE_FOLDER

# Configuração de logging para a API
logging.basicConfig(level=logging.INFO, filename="api.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

if not os.path.exists(DATABASE_FOLDER):
    os.makedirs(DATABASE_FOLDER)

def ler_planilha(nome_arquivo, colunas_obrigatorias):
    """
    Lê um arquivo .xlsx a partir da pasta de dados e valida as colunas obrigatórias.
    """
    caminho = os.path.join(DATABASE_FOLDER, nome_arquivo)
    df = pd.read_excel(caminho, engine="openpyxl")
    for coluna in colunas_obrigatorias:
        if coluna not in df.columns:
            logging.error(f"Coluna obrigatória '{coluna}' não encontrada em {nome_arquivo}.")
            raise ValueError(f"Coluna obrigatória '{coluna}' não encontrada em {nome_arquivo}.")
    return df

def gerar_mapa(pedidos_df):
    """
    Gera um mapa interativo com Folium exibindo os pedidos.
    """
    if pedidos_df.empty:
        return folium.Map(location=[0, 0], zoom_start=2)
    centro = [pedidos_df.iloc[0]['Latitude'], pedidos_df.iloc[0]['Longitude']]
    mapa = folium.Map(location=centro, zoom_start=12)
    for _, row in pedidos_df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=row.get("Endereço Completo", "Sem endereço"),
            icon=folium.Icon(color="blue")
        ).add_to(mapa)
    return mapa

# ---------- Endpoints da API REST ----------

@app.route('/upload', methods=['POST'])
def upload_files():
    """
    POST /upload: Recebe os arquivos Pedidos.xlsx, Caminhoes.xlsx, IA.xlsx e os salva na pasta DATABASE_FOLDER.
    """
    result = {}
    for nome in ["Pedidos.xlsx", "Caminhoes.xlsx", "IA.xlsx"]:
        if nome in request.files:
            file = request.files[nome]
            caminho = os.path.join(DATABASE_FOLDER, nome)
            file.save(caminho)
            result[nome] = "Arquivo enviado com sucesso"
        else:
            result[nome] = "Arquivo não enviado"
    return jsonify(result)

@app.route('/resultado', methods=['GET'])
def get_resultado():
    """
    GET /resultado: Lê os arquivos de Pedidos e Caminhões, pré-processa e executa o algoritmo genético.
    Retorna a melhor solução encontrada.
    """
    try:
        pedidos_df = ler_planilha("Pedidos.xlsx", ["Endereço de Entrega", "Bairro de Entrega", "Cidade de Entrega", "Peso dos Itens"])
        caminhoes_df = ler_planilha("Caminhoes.xlsx", ["Placa", "Capac. Kg", "Capac. Cx", "Disponível"])
    except Exception as e:
        logging.error(f"Erro na leitura dos arquivos: {e}")
        return jsonify({"error": f"Erro na leitura dos arquivos: {str(e)}"}), 400

    pedidos_df["Endereço Completo"] = pedidos_df["Endereço de Entrega"] + ", " + pedidos_df["Bairro de Entrega"] + ", " + pedidos_df["Cidade de Entrega"]
    pedidos_df = converter_enderecos(pedidos_df)
    pedidos_df = preprocessar_dados(pedidos_df)

    solucao = run_genetic_algorithm(pedidos_df, caminhoes_df)
    return jsonify(solucao)

@app.route('/mapa', methods=['GET'])
def get_mapa():
    """
    GET /mapa: Gera e retorna uma página HTML com o mapa interativo dos pedidos.
    """
    try:
        pedidos_df = ler_planilha("Pedidos.xlsx", ["Endereço de Entrega", "Bairro de Entrega", "Cidade de Entrega"])
        pedidos_df["Endereço Completo"] = pedidos_df["Endereço de Entrega"] + ", " + pedidos_df["Bairro de Entrega"] + ", " + pedidos_df["Cidade de Entrega"]
        pedidos_df = converter_enderecos(pedidos_df)
    except Exception as e:
        logging.error(f"Erro ao ler ou processar os pedidos: {e}")
        return jsonify({"error": f"Erro ao ler ou processar os pedidos: {str(e)}"}), 400

    mapa = gerar_mapa(pedidos_df)
    return mapa._repr_html_()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)