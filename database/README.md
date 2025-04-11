# My Python App

Este projeto é uma aplicação em Python que utiliza Streamlit para criar uma interface de usuário interativa para o gerenciamento de pedidos e frota. A aplicação permite processar pedidos, otimizar rotas e gerenciar informações sobre caminhões.

## Estrutura do Projeto

```
my-python-app
├── main.py                  # Ponto de entrada da aplicação
├── db                       # Pacote para interação com o banco de dados
│   ├── __init__.py         # Inicialização do pacote db
│   └── database.py         # Lógica para interagir com o banco de dados local
├── gerenciamento_frota.py   # Funções para gerenciar a frota de caminhões
├── subir_pedidos.py        # Funções para processar pedidos
├── ia_analise_pedidos.py   # Funções de análise de pedidos usando IA
├── requirements.txt         # Dependências do projeto
└── README.md                # Documentação do projeto
```

## Instalação

1. Clone o repositório:
   ```
   git clone <URL_DO_REPOSITORIO>
   cd my-python-app
   ```

2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

## Uso

Para iniciar a aplicação, execute o seguinte comando:
```
streamlit run main.py
```

A aplicação abrirá em seu navegador padrão, onde você poderá interagir com a interface para gerenciar pedidos e a frota.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para mais detalhes.