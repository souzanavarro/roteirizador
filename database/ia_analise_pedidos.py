import sqlite3

def conectar_banco():
    conn = sqlite3.connect('banco_de_dados.db')
    return conn

def criar_tabelas():
    conn = conectar_banco()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pedidos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        endereco TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        peso_itens REAL,
        ordem_entrega INTEGER
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS frota (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        modelo TEXT NOT NULL,
        capacidade REAL NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

def inserir_pedido(endereco, latitude, longitude, peso_itens, ordem_entrega):
    conn = conectar_banco()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO pedidos (endereco, latitude, longitude, peso_itens, ordem_entrega)
    VALUES (?, ?, ?, ?, ?)
    ''', (endereco, latitude, longitude, peso_itens, ordem_entrega))
    
    conn.commit()
    conn.close()

def inserir_caminhao(modelo, capacidade):
    conn = conectar_banco()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO frota (modelo, capacidade)
    VALUES (?, ?)
    ''', (modelo, capacidade))
    
    conn.commit()
    conn.close()

def consultar_pedidos():
    conn = conectar_banco()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM pedidos')
    resultados = cursor.fetchall()
    
    conn.close()
    return resultados

def consultar_frota():
    conn = conectar_banco()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM frota')
    resultados = cursor.fetchall()
    
    conn.close()
    return resultados

criar_tabelas()