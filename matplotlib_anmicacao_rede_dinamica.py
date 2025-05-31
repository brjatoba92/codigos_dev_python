import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from datetime import datetime, timedelta

# ======================================
# Visualização temporal de mudanças na rede de fornecedores
# ======================================

def animacao_rede_dinamica():
    """
    Animação sofisticada mostrando evolução temporal de uma rede de fornecedores
    com métricas de centralidade, fluxos de materiais e detecção de comunidades
    """
     
    # criar rede base de fornecedores
    G = nx.barabasi_albert_graph(50, 2, seed=42)
    # adicionar atributos realistas aos nós
    empresas_tipos = ['Fornecedor', 'Distribuidor', 'Loja', 'Construtora']
    for node in G.nodes():
        G.nodes[node]['tipo'] = np.random.choice(empresas_tipos)
        G.nodes[node]['capacidade'] = np.random.exponential(100)
        G.nodes[node]['localizacao'] = (np.random.uniform(-50, 50), np.random.uniform(-50, 50))
    #adicionar pesos às arestas (volume de negócios)
    for edge in G.edges():
        G.edges[edge]['peso'] = np.random.exponential(500)
        G.edges[edge]['material'] = np.random.choice(['Aço', 'Madeira', 'Cimento', 'Cerâmica'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ANÁLISE DINÂMICA DE REDE DE FORNECEDORES\nEvolução Temporal e Métricas de Centralidade', fontsize=14, fontweight='bold')