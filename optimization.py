"""
Módulo de otimização

Contém funções do algoritmo genético para otimização de cargas.
"""

import random
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, filename="optimization.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

def populacao_inicial(pedidos_df, caminhoes_df, tamanho=50):
    """
    Cria uma população inicial aleatória de soluções.
    
    Cada solução é um dicionário mapeando IDs de pedidos a IDs de caminhões.
    """
    population = []
    pedidos_ids = pedidos_df.index.tolist()
    caminhoes_ids = caminhoes_df.index.tolist()
    for _ in range(tamanho):
        sol = {pedido: random.choice(caminhoes_ids) for pedido in pedidos_ids}
        population.append(sol)
    logging.info(f"População inicial criada com {tamanho} soluções.")
    return population

def avaliacao_fitness(solucao, pedidos_df, caminhoes_df):
    """
    Calcula o fitness de uma solução considerando peso, volume e capacidade.
    
    Retorna:
      float: Valor de fitness.
    """
    fitness = 0
    capacidade_excedida = False
    for caminhao_id in caminhoes_df.index:
        pedidos_caminhao = [pedido for pedido, caminhao in solucao.items() if caminhao == caminhao_id]
        peso_total = pedidos_df.loc[pedidos_caminhao, "Peso dos Itens"].sum()
        volume_total = pedidos_df.loc[pedidos_caminhao, "Qtde. dos Itens"].sum()
        capacidade_peso = caminhoes_df.loc[caminhao_id, "Capac. Kg"]
        capacidade_volume = caminhoes_df.loc[caminhao_id, "Capac. Cx"]

        # Penaliza soluções que excedem a capacidade do caminhão
        if peso_total > capacidade_peso or volume_total > capacidade_volume:
            capacidade_excedida = True
            fitness -= 1000  # Penalidade alta para soluções inválidas
        else:
            fitness += peso_total + volume_total  # Maximiza o uso da capacidade

    if capacidade_excedida:
        logging.warning("Solução com capacidade excedida encontrada.")
    return fitness

def validar_solucao(solucao, pedidos_df, caminhoes_df):
    """
    Valida se a solução respeita as restrições de capacidade dos caminhões.
    
    Retorna:
      bool: True se a solução for válida, False caso contrário.
    """
    for caminhao_id in caminhoes_df.index:
        pedidos_caminhao = [pedido for pedido, caminhao in solucao.items() if caminhao == caminhao_id]
        peso_total = pedidos_df.loc[pedidos_caminhao, "Peso dos Itens"].sum()
        volume_total = pedidos_df.loc[pedidos_caminhao, "Qtde. dos Itens"].sum()
        capacidade_peso = caminhoes_df.loc[caminhao_id, "Capac. Kg"]
        capacidade_volume = caminhoes_df.loc[caminhao_id, "Capac. Cx"]

        if peso_total > capacidade_peso or volume_total > capacidade_volume:
            return False
    return True

def selecionar(population, fitnesses, num=10):
    """
    Seleciona as melhores soluções com base em sua fitness.
    
    Retorna:
      list: Subconjunto da população.
    """
    sorted_population = [sol for _, sol in sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)]
    logging.info(f"Selecionadas as {num} melhores soluções.")
    return sorted_population[:num]

def cruzar(sol1, sol2):
    """
    Realiza crossover entre duas soluções.
    """
    filho = {}
    for key in sol1.keys():
        filho[key] = sol1[key] if random.random() < 0.5 else sol2[key]
    logging.debug("Crossover realizado entre duas soluções.")
    return filho

def mutacao(solucao, caminhoes_ids, taxa=0.1):
    """
    Aplica mutação à solução, alterando mapeamentos aleatórios.
    """
    for pedido in solucao.keys():
        if random.random() < taxa:
            solucao[pedido] = random.choice(caminhoes_ids)
    logging.debug("Mutação aplicada a uma solução.")
    return solucao

def run_genetic_algorithm(pedidos_df, caminhoes_df, geracoes=100, tamanho_pop=50):
    """
    Executa o algoritmo genético e retorna a melhor solução encontrada.
    
    Retorna:
      dict: Contendo a solução e o fitness.
    """
    population = populacao_inicial(pedidos_df, caminhoes_df, tamanho=tamanho_pop)
    melhor_solucao = None
    melhor_fitness = -np.inf

    for geracao in range(geracoes):
        fitnesses = [avaliacao_fitness(sol, pedidos_df, caminhoes_df) for sol in population]
        melhores = selecionar(population, fitnesses, num=10)
        nova_pop = []
        for _ in range(tamanho_pop):
            sol1, sol2 = random.sample(melhores, 2)
            filho = cruzar(sol1, sol2)
            filho = mutacao(filho, caminhoes_df.index.tolist())
            if validar_solucao(filho, pedidos_df, caminhoes_df):
                nova_pop.append(filho)
        population = nova_pop

        melhor_iter = max(fitnesses)
        if melhor_iter > melhor_fitness:
            melhor_fitness = melhor_iter
            melhor_solucao = population[fitnesses.index(melhor_iter)]

        logging.info(f"Geração {geracao + 1}/{geracoes}: Melhor fitness = {melhor_fitness:.2f}")

    logging.info("Algoritmo genético concluído.")
    return {"solucao": melhor_solucao, "fitness": melhor_fitness}