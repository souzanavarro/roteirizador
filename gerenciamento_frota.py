import streamlit as st
import pandas as pd

def cadastrar_caminhoes():
    st.title("Cadastro de Caminhões da Frota")
    
    # Tenta carregar a frota existente ou cria uma nova
    try:
        caminhoes_df = pd.read_excel("database/caminhoes_frota.xlsx", engine='openpyxl')
    except FileNotFoundError:
        caminhoes_df = pd.DataFrame(columns=[
            'Placa', 'Transportador', 'Descrição Veículo', 'Capac. Cx', 'Capac. Kg', 'Disponível'
        ])
    
    # Upload de nova planilha de caminhões
    uploaded_caminhoes = st.file_uploader("Escolha o arquivo Excel de Caminhões", type=["xlsx", "xlsm"])
    
    if uploaded_caminhoes is not None:
        novo_caminhoes_df = pd.read_excel(uploaded_caminhoes, engine='openpyxl')
        colunas_caminhoes = ['Placa', 'Transportador', 'Descrição Veículo', 'Capac. Cx', 'Capac. Kg', 'Disponível']
        if not all(col in novo_caminhoes_df.columns for col in colunas_caminhoes):
            st.error("As colunas necessárias não foram encontradas na planilha de caminhões.")
            return
        
        # Exclui placas indesejadas
        placas_excluir = ["FLB1111", "FLB2222", "FLB3333", "FLB4444",
                           "FLB5555", "FLB6666", "FLB7777", "FLB8888", "FLB9999"]
        novo_caminhoes_df = novo_caminhoes_df[~novo_caminhoes_df['Placa'].isin(placas_excluir)]
        
        if st.button("Carregar Frota"):
            caminhoes_df = pd.concat([caminhoes_df, novo_caminhoes_df], ignore_index=True)
            caminhoes_df.to_excel("database/caminhoes_frota.xlsx", index=False)
            st.success("Frota carregada com sucesso!")
    
    if st.button("Limpar Frota"):
        caminhoes_df = pd.DataFrame(columns=[
            'Placa', 'Transportador', 'Descrição Veículo', 'Capac. Cx', 'Capac. Kg', 'Disponível'
        ])
        caminhoes_df.to_excel("database/caminhoes_frota.xlsx", index=False)
        st.success("Frota limpa com sucesso!")
    
    st.subheader("Caminhões Cadastrados")
    edited_caminhoes_df = st.data_editor(caminhoes_df, num_rows="dynamic")
    
    if st.button("Salvar Alterações"):
        edited_caminhoes_df.to_excel("database/caminhoes_frota.xlsx", index=False)
        st.success("Alterações salvas com sucesso!")