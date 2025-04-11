import sqlite3

def connect_db(db_name="database.db"):
    conn = sqlite3.connect(db_name)
    return conn

def create_tables(conn):
    cursor = conn.cursor()
    
    # Tabela para armazenar as planilhas de IA
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ia_planilhas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            dados BLOB NOT NULL
        )
    ''')
    
    # Tabela para armazenar informações da frota
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS frota (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            modelo TEXT NOT NULL,
            capacidade INTEGER NOT NULL,
            placa TEXT NOT NULL UNIQUE
        )
    ''')
    
    conn.commit()

def insert_ia_planilha(conn, nome, dados):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO ia_planilhas (nome, dados) VALUES (?, ?)
    ''', (nome, dados))
    conn.commit()

def insert_frota(conn, modelo, capacidade, placa):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO frota (modelo, capacidade, placa) VALUES (?, ?, ?)
    ''', (modelo, capacidade, placa))
    conn.commit()

def query_ia_planilhas(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM ia_planilhas')
    return cursor.fetchall()

def query_frota(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM frota')
    return cursor.fetchall()