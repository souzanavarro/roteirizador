import sqlite3

def conectar_db():
    conn = sqlite3.connect('frota_ia.db')
    return conn

def criar_tabelas():
    conn = conectar_db()
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS frota (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        modelo TEXT NOT NULL,
        capacidade INTEGER NOT NULL,
        placa TEXT NOT NULL UNIQUE
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pedidos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        endereco TEXT NOT NULL,
        peso INTEGER NOT NULL,
        latitude REAL,
        longitude REAL
    )
    ''')
    
    conn.commit()
    conn.close()

def cadastrar_caminhao(modelo, capacidade, placa):
    conn = conectar_db()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO frota (modelo, capacidade, placa) VALUES (?, ?, ?)
    ''', (modelo, capacidade, placa))
    
    conn.commit()
    conn.close()

def consultar_frota():
    conn = conectar_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM frota')
    resultados = cursor.fetchall()
    
    conn.close()
    return resultados

def atualizar_caminhao(id, modelo, capacidade, placa):
    conn = conectar_db()
    cursor = conn.cursor()
    
    cursor.execute('''
    UPDATE frota SET modelo = ?, capacidade = ?, placa = ? WHERE id = ?
    ''', (modelo, capacidade, placa, id))
    
    conn.commit()
    conn.close()