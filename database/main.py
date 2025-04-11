import streamlit as st
import pandas as pd
from streamlit_folium import folium_static
from db.database import Database

from gerenciamento_frota import cadastrar_caminhoes
from subir_pedidos import processar_pedidos, salvar_coordenadas
import ia_analise_pedidos as ia

def main():
    st.title("Roteirizador de Pedidos")

    # Inicializa o banco de dados
    db = Database()
    db.create_tables()

    # Subir e processar a planilha de pedidos
    pedidos_result = processar_pedidos()
    if pedidos_result is None:
        return
    pedidos_df, coordenadas_salvas = pedidos_result

    # Obter coordenadas para cada pedido
    with st.spinner("Obtendo coordenadas..."):
        pedidos_df['Latitude'] = pedidos_df['Endereço Completo'].apply(lambda x: ia.obter_coordenadas_com_fallback(x, coordenadas_salvas)[0])
        pedidos_df['Longitude'] = pedidos_df['Endereço Completo'].apply(lambda x: ia.obter_coordenadas_com_fallback(x, coordenadas_salvas)[1])
    salvar_coordenadas(coordenadas_salvas)
    
    if pedidos_df['Latitude'].isnull().any() or pedidos_df['Longitude'].isnull().any():
        st.error("Alguns endereços não obtiveram coordenadas. Verifique os dados.")
        return

    # Carrega a frota cadastrada
    try:
        caminhoes_df = pd.read_excel("caminhoes_frota.xlsx", engine="openpyxl")
    except FileNotFoundError:
        st.error("Nenhum caminhão cadastrado. Cadastre a frota na aba de gerenciamento.")
        return
    
    # Opções de configuração
    n_clusters = st.slider("Número de regiões para agrupar", min_value=1, max_value=10, value=5)
    percentual_frota = st.slider("Capacidade da frota a ser usada (%)", min_value=0, max_value=100, value=100)
    max_pedidos = st.slider("Número máximo de pedidos por veículo", min_value=1, max_value=30, value=12)
    aplicar_tsp = st.checkbox("Aplicar TSP")
    aplicar_vrp = st.checkbox("Aplicar VRP")
    
    if st.button("Roteirizar"):
        pedidos_df = pedidos_df[pedidos_df['Peso dos Itens'] > 0]
        pedidos_df = ia.agrupar_por_regiao(pedidos_df, n_clusters)
        pedidos_df = ia.otimizar_aproveitamento_frota(pedidos_df, caminhoes_df, percentual_frota, max_pedidos, n_clusters)
        
        if aplicar_tsp:
            G = ia.criar_grafo_tsp(pedidos_df)
            melhor_rota, menor_distancia = ia.resolver_tsp_genetico(G)
            st.write("Melhor rota TSP:")
            st.write("\n".join(melhor_rota))
            st.write(f"Menor distância TSP: {menor_distancia}")
            pedidos_df['Ordem de Entrega TSP'] = pedidos_df['Endereço Completo'].apply(lambda x: melhor_rota.index(x) + 1)
        
        if aplicar_vrp:
            rota_vrp = ia.resolver_vrp(pedidos_df, caminhoes_df)
            st.write(f"Melhor rota VRP: {rota_vrp}")
        
        st.write("Dados dos Pedidos:")
        st.dataframe(pedidos_df)
        
        mapa = ia.criar_mapa(pedidos_df)
        folium_static(mapa)
        
        output_file_path = "roterizacao_resultado.xlsx"
        pedidos_df.to_excel(output_file_path, index=False)
        st.write(f"Arquivo salvo: {output_file_path}")
        with open(output_file_path, "rb") as file:
            st.download_button("Baixar planilha", data=file, file_name="roterizacao_resultado.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    # Abas para Gerenciamento da Frota e Upload de Roteirizações
    if st.checkbox("Cadastrar Caminhões"):
        cadastrar_caminhoes()
    
    if st.checkbox("Subir Planilhas de Roteirizações"):
        # Aqui você pode incluir funcionalidades extras para roteirizações
        st.info("Funcionalidade de upload de roteirizações a ser implementada.")

if __name__ == "__main__":
    main()