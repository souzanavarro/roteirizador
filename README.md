# 🚀 Sistema de Roteirização com IA

Este projeto é uma plataforma completa de roteirização de entregas utilizando inteligência artificial. Ele combina um **frontend em React + Vite** com um **backend em FastAPI**, tudo pronto para deploy local ou em produção.

---

## 📦 Tecnologias Utilizadas

- 🧠 Backend: [FastAPI](https://fastapi.tiangolo.com/)
- ⚛️ Frontend: [React](https://reactjs.org/) + [Vite](https://vitejs.dev/)
- 📊 Visualização: Matplotlib, Folium, Gráficos
- 📁 Exportação: Excel, PDF
- 🔒 Autenticação: Sessão com usuário fixo (Orlando / Picole2024@)
- 🧭 Algoritmos de otimização de rotas (NN, agrupamento por região, etc.)

---

## 🚀 Como rodar localmente

### 🔧 Pré-requisitos

- Python 3.10+
- Node.js 18+

### 1️⃣ Rodando o Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn main:app --reload

🧠 Autor
Desenvolvido com ❤️ por Orlando & IA
Contribuições e sugestões são bem-vindas!
