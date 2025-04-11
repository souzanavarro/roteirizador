import streamlit as st
from agrupar_por_regiao import agrupar_por_regiao

def otimizar_aproveitamento_frota(pedidos_df, caminhoes_df, percentual_frota, max_pedidos, n_clusters=3, metodo='kmeans'):
    # Inicializa as colunas de alocação
    pedidos_df['Carga'] = 0
    pedidos_df['Placa'] = ""
    carga_numero = 1
    
    # Ajusta a capacidade da frota com base no percentual informado
    caminhoes_df['Capac. Kg'] *= (percentual_frota / 100)
    caminhoes_df['Capac. Cx'] *= (percentual_frota / 100)
    # Filtra somente os caminhões disponíveis ("Sim")
    caminhoes_df = caminhoes_df[caminhoes_df['Disponível'] == 'Sim']
    
    # Agrupa os pedidos por região utilizando o método e os clusters informados
    pedidos_df = agrupar_por_regiao(pedidos_df, metodo=metodo, n_clusters=n_clusters)
    
    # Para cada região, aloca os pedidos aos caminhões disponíveis
    for regiao in pedidos_df['Regiao'].unique():
        st.write(f"Alocando pedidos para a região {regiao}...")
        pedidos_regiao = pedidos_df[pedidos_df['Regiao'] == regiao]
        
        # Ordena os pedidos por peso e quantidade de itens (priorizando os maiores)
        pedidos_regiao = pedidos_regiao.sort_values(by=['Peso dos Itens', 'Qtde. dos Itens'], ascending=False)
        
        for _, caminhao in caminhoes_df.iterrows():
            capacidade_peso = caminhao['Capac. Kg']
            capacidade_caixas = caminhao['Capac. Cx']
            
            # Seleciona pedidos que cabem nas capacidades do caminhão
            pedidos_alocados = pedidos_regiao[
                (pedidos_regiao['Peso dos Itens'] <= capacidade_peso) & 
                (pedidos_regiao['Qtde. dos Itens'] <= capacidade_caixas)
            ]
            
            # Ordena os pedidos para maximizar o aproveitamento do caminhão
            pedidos_alocados = pedidos_alocados.sort_values(by=['Peso dos Itens', 'Qtde. dos Itens'], ascending=False)
            pedidos_alocados = pedidos_alocados.head(max_pedidos)
            
            if not pedidos_alocados.empty:
                pedidos_df.loc[pedidos_alocados.index, 'Carga'] = carga_numero
                pedidos_df.loc[pedidos_alocados.index, 'Placa'] = caminhao['Placa']
                
                # Atualiza as capacidades do caminhão
                capacidade_peso -= pedidos_alocados['Peso dos Itens'].sum()
                capacidade_caixas -= pedidos_alocados['Qtde. dos Itens'].sum()
                carga_numero += 1
                
                # Verifica se o caminhão foi sobrecarregado
                if capacidade_peso < 0 or capacidade_caixas < 0:
                    st.error(f"O caminhão {caminhao['Placa']} foi sobrecarregado. Verifique os dados.")
            else:
                st.warning(f"Nenhum pedido foi alocado para o caminhão {caminhao['Placa']} na região {regiao}.")
    
    # Verifica se houve erro na alocação
    if pedidos_df['Placa'].isnull().any() or pedidos_df['Carga'].isnull().any():
        st.error("Não foi possível atribuir placas ou números de carga a alguns pedidos. Verifique os dados e tente novamente.")
    
    # Relatório final
    total_pedidos = len(pedidos_df)
    pedidos_alocados = len(pedidos_df[pedidos_df['Placa'] != ""])
    st.success(f"Pedidos alocados: {pedidos_alocados}/{total_pedidos} ({(pedidos_alocados / total_pedidos) * 100:.2f}%)")
    
    return pedidos_df