#Bibliotecas
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.retrievers import NLSQLRetriever
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core import SQLDatabase
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI
from IPython.display import Markdown, display
from sqlalchemy import create_engine
import plotly.graph_objects as go
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import openai
import time
import ast
import os

##Titulo
#st.title('Modelo LLM  para boletins de ocorrência.')

load_dotenv()

#Variaveis de ambiente
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GROQ_API = os.environ.get("GROQ_API")
DB_CONNECTION_STRING = os.environ.get("DB_CONNECTION_STRING")

table_name = "boletins_ocorrencia"
model = "gpt-3.5-turbo"

#Funcões
def nlsql_table_query_engine(query_str, db_connection_str, table_name, model):
  db_engine = create_engine(db_connection_str)
  sql_database = SQLDatabase(db_engine, include_tables=[table_name])
  llm = OpenAI(temperature=0.1, model=model)

  query_engine = NLSQLTableQueryEngine(
      sql_database=sql_database,
      tables=[table_name],
      llm=llm)
  return query_engine.query(query_str)

def nlsql_retriever(query_str, db_connection_str, table_name):
  db_engine = create_engine(db_connection_str)
  sql_database = SQLDatabase(db_engine, include_tables=[table_name])
  nl_sql_retriever = NLSQLRetriever(
      sql_database,
      tables=[table_name],
      return_raw=True)
  return nl_sql_retriever.retrieve(query_str)

def bar_chart(df):
  return st.plotly_chart(go.Figure(go.Bar(x=df[0], y=df[1])))

def pie_chat(df):
  return st.plotly_chart(go.Figure(go.Pie(labels=df[0], values=df[1])))

def line_chart(df):
  return st.plotly_chart(go.Figure(go.Scatter(x=df[0], y=df[1], mode='lines+markers')))

def card_chart(df):
  return st.plotly_chart(go.Figure(go.Indicator(mode="number", value=df[0][0], number = {'valueformat':'f'})))

def principal(df1, df2):
  if((len(df1) < 1) | df2.empty):
    return 'Não foi possivel construir o gráfico!'
  elif df2.shape[0] == 1 and df2.shape[1] == 1:
    return card_chart(df2)
  elif df2.shape[0] == 2 and df2.shape[1] <= 2:
    return pie_chat(df2)
  elif df2.shape[0] > 2 and df2.shape[0] <= 10 and df2.shape[1] == 2:
    return bar_chart(df2)
  elif df2.shape[0] > 10 and df2.shape[1] == 2:
    df2 = df2.sort_values(by=0, ascending=False)
    return line_chart(df2)
  else:
    return f'Não foi possivel construir o gráfico! erro shape: {df2.shape}'

st.sidebar.image("https://www.ufpb.br/educacaofinanceira/contents/imagens/brasoes-universidades/ufpa.png/@@images/image.png", width=120)
st.sidebar.markdown("**Universidade Federal do Pará - UFPA**")
st.sidebar.markdown("**Programa de Pós-Graduação em Computação Aplicada - PPCA**")
st.sidebar.markdown("**Discente:** Anselmo Mendes Oliveira")
st.sidebar.markdown("**Orientador:** Adam Dreyton Ferreira dos Santos")

head = st.empty()

st.markdown("# Sistema Inteligente de Análise de Ocorrências do Pará (SIAOP)")



st.markdown("Este sistema foi desenvolvido com o objetivo de facilitar a análise de dados de boletins de ocorrência do Pará. \
            O sistema utiliza um modelo de linguagem natural para responder perguntas sobre os dados de boletins de ocorrência. \
            Para fazer uma pergunta, basta digitar a pergunta no campo abaixo e clicar no botão 'Pesquisar'. \
            O sistema irá retornar um gráfico com a resposta para a sua pergunta.")

#query_str = input.text_input("Faça sua pergunta sobre os dados de boletins de ocorrência", "Ex: Qual a quantidade de boletins de ocorrência por mês em 2024?")
query_str = st.text_input("Faça sua pergunta sobre os dados de boletins de ocorrência",
                          placeholder="Ex: Qual a quantidade total mensal de registros na tabela de boletins?")


button = st.button("Pesquisar")

placeholder = st.empty()

if button:
    
    placeholder.progress(0, "Wait for it...")
    time.sleep(1)

    placeholder.progress(25, "Wait for it...")
    time.sleep(1)

    df = nlsql_table_query_engine(query_str, DB_CONNECTION_STRING, table_name, model)

    placeholder.progress(50, "Wait for it...")
    time.sleep(1)

    placeholder.progress(75, "Wait for it...")
    time.sleep(1)

    data = nlsql_retriever(query_str, DB_CONNECTION_STRING, table_name)
    data = data[0].get_text()

    data_list = ast.literal_eval(data)

    df2 = pd.DataFrame(data_list)
    
    placeholder.progress(98, "Wait for it...")
    time.sleep(1)

    principal(df.response, df2)
    st.write(df.response)

    placeholder.empty()