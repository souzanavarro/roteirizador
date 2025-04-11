import streamlit as st
import pandas as pd
from io import BytesIO

REQUIRED_COLUMNS = ["Endereço de Entrega", "Bairro de Entrega", "Cidade de Entrega"]

def processar_pedidos():
    uploaded_pedidos = st.file_uploader("Escolha o arquivo Excel de Pedidos", type=["xlsx", "xlsm"])
    if uploaded_pedidos is None:
        st.info("Envie a planilha de pedidos para continuação.")
        return None

    try:
        pedidos_df = pd.read_excel(uploaded_pedidos, engine='openpyxl')
    except Exception as e:
        st.error("Erro ao ler a planilha: " + str(e))
        return None

    # Verifica se as colunas necessárias estão presentes
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in pedidos_df.columns]
    if missing_cols:
        st.error(f"As seguintes colunas necessárias não foram encontradas: {', '.join(missing_cols)}")
        return None

    # Cria a coluna 'Endereço Completo'
    pedidos_df['Endereço Completo'] = (
        pedidos_df['Endereço de Entrega'].astype(str) + ', ' +
        pedidos_df['Bairro de Entrega'].astype(str) + ', ' +
        pedidos_df['Cidade de Entrega'].astype(str)
    )
    
    # Carrega coordenadas salvas (se houver)
    try:
        coordenadas_salvas_df = pd.read_excel("database/coordenadas_salvas.xlsx", engine='openpyxl')
        coordenadas_salvas = dict(zip(
            coordenadas_salvas_df['Endereço'],
            zip(coordenadas_salvas_df['Latitude'], coordenadas_salvas_df['Longitude'])
        ))
    except FileNotFoundError:
        coordenadas_salvas = {}
    
    return pedidos_df, coordenadas_salvas

def salvar_coordenadas(coordenadas_salvas):
    # Salva as coordenadas atualizadas num arquivo Excel
    coordenadas_salvas_df = pd.DataFrame(coordenadas_salvas.items(), columns=['Endereço', 'Coordenadas'])
    coordenadas_salvas_df[['Latitude','Longitude']] = pd.DataFrame(
        coordenadas_salvas_df['Coordenadas'].tolist(),
        index=coordenadas_salvas_df.index
    )
    coordenadas_salvas_df.drop(columns=['Coordenadas'], inplace=True)
    coordenadas_salvas_df.to_excel("database/coordenadas_salvas.xlsx", index=False)