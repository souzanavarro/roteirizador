import streamlit as st
import pandas as pd
from streamlit_folium import folium_static
import requests
import time

from gerenciamento_frota import cadastrar_caminhoes
from subir_pedidos import processar_pedidos, salvar_coordenadas
import ia_analise_pedidos as ia

# Exemplo de função para definir a ordem de entrega por carga
def definir_ordem_por_carga(pedidos_df, ordem_tsp):
    """
    Define a coluna 'Ordem de Entrega TSP' com base na ordem definida pelo TSP,
    agrupando os pedidos por 'Carga' e atribuindo uma sequência para cada entrega.
    """
    rota_indices = {endereco: idx for idx, endereco in enumerate(ordem_tsp)}
    pedidos_df['Ordem de Entrega TSP'] = ""
    for carga in pedidos_df['Carga'].unique():
        mask = pedidos_df['Carga'] == carga
        df_carga = pedidos_df.loc[mask].copy()
        df_carga = df_carga.sort_values(
            by='Endereço Completo', 
            key=lambda col: col.map(lambda x: rota_indices.get(x, float('inf')))
        )
        for seq, idx in enumerate(df_carga.index, start=1):
            pedidos_df.at[idx, 'Ordem de Entrega TSP'] = f"{carga}-{seq}"
    return pedidos_df

def main():
    st.title("Roteirizador de Pedidos")
    
    st.markdown(
        """
        <style>
        div[data-baseweb="radio"] ul {
            list-style: none;
            padding-left: 0;
        }
        div[data-baseweb="radio"] li {
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Menu lateral
    menu_opcao = st.sidebar.radio("Menu", options=[
        "Dashboard", 
        "Cadastro da Frota", 
        "IA Analise",
        "API REST"
    ])
    
    if menu_opcao == "Dashboard":
        st.header("Dashboard - Envio de Pedidos")
        st.write("Bem-vindo! Envie a planilha de pedidos para iniciar:")
        pedidos_result = processar_pedidos()
        if pedidos_result is None:
            st.info("Aguardando envio da planilha de pedidos.")
        else:
            pedidos_df, coordenadas_salvas = pedidos_result
            
            with st.spinner("Obtendo coordenadas..."):
                pedidos_df['Latitude'] = pedidos_df['Endereço Completo'].apply(
                    lambda x: ia.obter_coordenadas_com_fallback(x, coordenadas_salvas)[0]
                )
                pedidos_df['Longitude'] = pedidos_df['Endereço Completo'].apply(
                    lambda x: ia.obter_coordenadas_com_fallback(x, coordenadas_salvas)[1]
                )
            pedidos_df['Latitude'] = pedidos_df['Latitude'].fillna(0)
            pedidos_df['Longitude'] = pedidos_df['Longitude'].fillna(0)
            salvar_coordenadas(coordenadas_salvas)
            
            st.dataframe(pedidos_df)
            st.write("Cabeçalho da planilha:", list(pedidos_df.columns))
            
            st.markdown("### Configurações para Roteirização")
            n_clusters = st.slider("Número de regiões para agrupar", min_value=1, max_value=10, value=1)
            percentual_frota = st.slider("Capacidade da frota a ser usada (%)", min_value=0, max_value=100, value=100)
            max_pedidos = st.slider("Número máximo de pedidos por veículo", min_value=1, max_value=30, value=12)
            aplicar_tsp = st.checkbox("Aplicar TSP")
            
            st.markdown("""
            **Aplicar TSP:**  
            Utiliza um algoritmo genético para encontrar a rota que minimiza a distância total entre todos os pontos de entrega.
            """)

            aplicar_vrp = st.checkbox("Aplicar VRP")

            st.markdown("""
            **Aplicar VRP:**  
            Distribui os pedidos entre os veículos disponíveis, respeitando as restrições de capacidade e minimizando a distância percorrida.
            """)
            
            if st.button("Roteirizar Pedidos"):
                st.write("Roteirização em execução...")
                progress_bar = st.empty()
                for progresso in range(100):
                    time.sleep(0.05)
                    progress_bar.progress(progresso + 1)
                
                pedidos_df = pedidos_df[pedidos_df['Peso dos Itens'] > 0]
                
                try:
                    caminhoes_df = pd.read_excel("database/caminhoes_frota.xlsx", engine="openpyxl")
                except FileNotFoundError:
                    st.error("Nenhum caminhão cadastrado. Cadastre a frota na opção 'Cadastro da Frota'.")
                    return
                
                # Verifica se o DataFrame contém as colunas necessárias
                required_columns = ['Latitude', 'Longitude', 'Cidade de Entrega']
                if not all(col in pedidos_df.columns for col in required_columns):
                    st.error(f"As colunas necessárias {required_columns} não foram encontradas no DataFrame.")
                    st.stop()

                # Verifica se há valores nulos nas colunas de coordenadas
                if pedidos_df[required_columns].isnull().any().any():
                    st.error("O DataFrame contém valores nulos nas colunas de coordenadas ou cidade. Verifique os dados e tente novamente.")
                    st.stop()

                # Agrupamento por cidade e coordenadas
                try:
                    pedidos_df = ia.agrupar_por_regiao(pedidos_df, metodo='kmeans', n_clusters=n_clusters)
                    st.write("Pedidos agrupados por região:")
                    st.dataframe(pedidos_df[['Cidade de Entrega', 'Latitude', 'Longitude', 'Regiao']])
                except Exception as e:
                    st.error(f"Erro ao agrupar pedidos por região: {e}")
                    st.stop()
                
                # Otimização da frota com base nas coordenadas
                pedidos_df = ia.otimizar_aproveitamento_frota(pedidos_df, caminhoes_df, percentual_frota, max_pedidos, n_clusters)
                
                # Relatório de alocação por região
                alocacao_report = pedidos_df.groupby(['Regiao', 'Placa']).agg({
                    'Peso dos Itens': 'sum',
                    'Qtde. dos Itens': 'sum',
                    'Regiao': 'count'
                }).rename(columns={'Regiao': 'Total de Pedidos'})
                
                # Relatório de alocação por região
                st.write("Relatório de Alocação por Região e Veículo:")
                st.dataframe(alocacao_report)

                # Ajusta o estilo do mapa para ocupar 100% da largura
                st.markdown(
                    """
                    <style>
                    .folium-map {
                        width: 100% !important;
                        height: 700px !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Exibe o mapa
                mapa = ia.criar_mapa(pedidos_df)
                folium_static(mapa)
                
                if aplicar_tsp:
                    for regiao in pedidos_df['Regiao'].unique():
                        pedidos_regiao = pedidos_df[pedidos_df['Regiao'] == regiao]
                        if not pedidos_regiao.empty:
                            G = ia.criar_grafo_tsp(pedidos_regiao)
                            melhor_rota, menor_distancia = ia.resolver_tsp_genetico(G)
                            st.write(f"Melhor rota TSP para a região {regiao}:")
                            st.write("\n".join(melhor_rota))
                            st.write(f"Menor distância TSP para a região {regiao}: {menor_distancia}")
                            
                            # Define a ordem de entrega baseada no campo 'Carga'
                            pedidos_df = definir_ordem_por_carga(pedidos_df, melhor_rota)
                
                st.write("Dados dos Pedidos:")
                st.dataframe(pedidos_df)
                                            
                output_file_path = "database/roterizacao_resultado.xlsx"
                pedidos_df.to_excel(output_file_path, index=False)
                st.write(f"Arquivo salvo: {output_file_path}")
                with open(output_file_path, "rb") as file:
                    st.download_button(
                        "Baixar planilha",
                        data=file,
                        file_name="roterizacao_resultado.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
            st.markdown("**Edite a planilha de Pedidos, se necessário:**")
            dados_editados = st.data_editor(pedidos_df, num_rows="dynamic")
            if st.button("Salvar alterações na planilha"):
                dados_editados.to_excel("database/Pedidos.xlsx", index=False)
                st.success("Planilha editada e salva com sucesso!")
    
    elif menu_opcao == "Cadastro da Frota":
        st.header("Cadastro da Frota")
        from gerenciamento_frota import cadastrar_caminhoes
        if st.checkbox("Cadastrar Caminhões"):
            cadastrar_caminhoes()
    
    elif menu_opcao == "IA Analise":
        st.header("IA Analise")
        st.write("Envie a planilha de pedidos para análise e edite os dados, se necessário:")
        pedidos_result = processar_pedidos()
        if pedidos_result is None:
            st.info("Aguardando envio da planilha de pedidos.")
        else:
            pedidos_df, coordenadas_salvas = pedidos_result
            with st.spinner("Obtendo coordenadas..."):
                pedidos_df['Latitude'] = pedidos_df['Endereço Completo'].apply(
                    lambda x: ia.obter_coordenadas_com_fallback(x, coordenadas_salvas)[0]
                )
                pedidos_df['Longitude'] = pedidos_df['Endereço Completo'].apply(
                    lambda x: ia.obter_coordenadas_com_fallback(x, coordenadas_salvas)[1]
                )
            pedidos_df['Latitude'] = pedidos_df['Latitude'].fillna(0)
            pedidos_df['Longitude'] = pedidos_df['Longitude'].fillna(0)
            salvar_coordenadas(coordenadas_salvas)
            
            st.dataframe(pedidos_df)
            if st.button("Salvar alterações na planilha"):
                pedidos_df.to_excel("database/Pedidos.xlsx", index=False)
                st.success("Planilha editada e salva com sucesso!")
            with open("database/Pedidos.xlsx", "rb") as file:
                st.download_button(
                    "Baixar planilha de Pedidos",
                    data=file,
                    file_name="Pedidos.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
    elif menu_opcao == "API REST":
        st.header("Interação com API REST")
        st.write("Teste os endpoints:")
        st.markdown("""
        - **POST /upload**: Faz upload dos arquivos (Pedidos.xlsx, Caminhoes.xlsx, IA.xlsx).
        - **GET /resultado**: Retorna a solução do algoritmo genético.
        - **GET /mapa**: Exibe o mapa interativo.
        """)
        if st.button("Testar /resultado"):
            try:
                resposta = requests.get("http://localhost:5000/resultado")
                st.json(resposta.json())
            except Exception as e:
                st.error(f"Erro na requisição: {e}")

if __name__ == "__main__":
    main()