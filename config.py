"""
Configuração do sistema de roteirização

- Define parâmetros gerais, como pasta de dados.
- Utiliza variáveis de ambiente para informações sensíveis (por exemplo, chave da API).
"""

import os

DATABASE_FOLDER = "database"
if not os.path.exists(DATABASE_FOLDER):
    os.makedirs(DATABASE_FOLDER)

# Parâmetros de geocodificação
GEOCODER_USER_AGENT = os.environ.get("GEOCODER_USER_AGENT", "logistica_app")
OPENCAGE_API_KEY = os.environ.get("OPENCAGE_API_KEY", "6f522c67add14152926990afbe127384")

# Parâmetros de rota de partida
endereco_partida = "Avenida Antonio Ortega, 3604 - Pinhal, Cabreúva - SP, São Paulo, Brasil"
endereco_partida_coords = (-23.0838, -47.1336)