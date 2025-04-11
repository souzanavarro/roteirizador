import streamlit as st

def otimizar_aproveitamento_frota(pedidos_df, caminhoes_df, percentual_frota, max_pedidos, n_clusters=3):
    # Inicializa as colunas de alocação
    pedidos_df['Carga'] = 0
    pedidos_df['Placa'] = ""
    carga_numero = 1
    
    # Ajusta a capacidade da frota com base no percentual informado
    caminhoes_df['Capac. Kg'] *= (percentual_frota / 100)
    caminhoes_df['Capac. Cx'] *= (percentual_frota / 100)
    # Filtra somente os caminhões disponíveis ("Sim")
    caminhoes_df = caminhoes_df[caminhoes_df['Disponível'] == 'Sim']
    
    # Agrupa os pedidos por região utilizando o valor informado em n_clusters (default=3)
    from agrupar_por_regiao import agrupar_por_regiao
    pedidos_df = agrupar_por_regiao(pedidos_df, n_clusters)
    
    # Para cada região, aloca os pedidos aos caminhões disponíveis
    for regiao in pedidos_df['Regiao'].unique():
        pedidos_regiao = pedidos_df[pedidos_df['Regiao'] == regiao]
        for _, caminhao in caminhoes_df.iterrows():
            capacidade_peso = caminhao['Capac. Kg']
            capacidade_caixas = caminhao['Capac. Cx']
            
            # Seleciona pedidos que cabem nas capacidades do caminhão
            pedidos_alocados = pedidos_regiao[
                (pedidos_regiao['Peso dos Itens'] <= capacidade_peso) & 
                (pedidos_regiao['Qtde. dos Itens'] <= capacidade_caixas)
            ]
            # Limita o número máximo de pedidos a serem alocados
            pedidos_alocados = pedidos_alocados.sample(n=min(max_pedidos, len(pedidos_alocados)))
            
            if not pedidos_alocados.empty:
                pedidos_df.loc[pedidos_alocados.index, 'Carga'] = carga_numero
                pedidos_df.loc[pedidos_alocados.index, 'Placa'] = caminhao['Placa']
                
                # Atualiza as capacidades do caminhão
                capacidade_peso -= pedidos_alocados['Peso dos Itens'].sum()
                capacidade_caixas -= pedidos_alocados['Qtde. dos Itens'].sum()
                carga_numero += 1

    # Verifica se houve erro na alocação
    if pedidos_df['Placa'].isnull().any() or pedidos_df['Carga'].isnull().any():
        st.error("Não foi possível atribuir placas ou números de carga a alguns pedidos. Verifique os dados e tente novamente.")
    
    return pedidos_df