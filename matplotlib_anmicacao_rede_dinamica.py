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

    def atualizar_frame(frame):
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        # simular mudanças temporais na rede
        tempo = frame / 10

        # 1 - Rede principal com layout spring
        pos = nx.spring_layout(G, k=2, iterations=50)

        # calcular metricas de centralidade
        centralidade_betweenness = nx.betweenness_centrality(G, normalized=True)
        centralidade_closeness = nx.closeness_centrality(G, normalized=True)
        centralidade_degree = nx.degree_centrality(G)

        # cores por tipo de empresa
        cores_tipos = {
            'Fornecedor': '#2E8B57', 
            'Distribuidor': '#4682B4', 
            'Loja': '#CD853F', 
            'Construtora': '#DC143C'
        }
        node_colors = [cores_tipos[G.nodes[node]['tipo']] for node in G.nodes()]

        # Tamanhos proporcionais à centralidade
        node_sizes = [2000 * centralidade_betweenness[node] + 300 for node in G.nodes()]

        # Desenhar rede no ax1
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax1, alpha=0.8)

        # desenhar arestas com espessura proporcional ao peso
        edge_weights = [G.edges[edge]['peso'] for edge in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax1, width=np.array(edge_weights) / 200, alpha=0.6, edge_color='gray')

        # labels dos nós
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8, font_weight='bold')
        ax1.set_title('Rede de Fornecedores (t={tempo:.1f})\nTamanho = Centralidade Betweenness')
        ax1.axis('off')

        # 2 - Analise da cetralidade por tipo
        tipos_data = {}
        for node in G.nodes():
            tipo = G.nodes[node]['tipo']
            if tipo not in tipos_data:
                tipos_data[tipo] = {'betweenness': [], 'closeness': [], 'degree': []}
            tipos_data[tipo]['betweenness'].append(centralidade_betweenness[node])
            tipos_data[tipo]['closeness'].append(centralidade_closeness[node])
            tipos_data[tipo]['degree'].append(centralidade_degree[node])
        # boxplot de centralidade por tipo
        tipos = list(tipos_data.keys())
        betweenness_data = [tipos_data[tipo]['betweenness'] for tipo in tipos]

        bp = ax2.boxplot(betweenness_data, labels=tipos, patch_artist=True)
        for patch, tipo in zip(bp['boxes'], tipos):
            patch.set_facecolor(cores_tipos[tipo])
            patch.set_alpha(0.7)
        ax2.set_title('Centralidade Betweenness \npor Tipo de Empresa')
        ax2.set_ylabel('Centralidade Betweenness')
        ax2.tick_params(axis='x', rotation=45)

        # 3 - Fluxo de materiais (chord diagram simplificado)
        materiais = ['Cimento', 'Aço', 'Madeira', 'Cerâmica']
        fluxo_matrix = np.zeros((len(materiais), len(materiais)))

        # simular fluxos baseados nas conexões
        for edge in G.edges():
            material = G.edges[edge]['material']
            peso = G.edges[edge]['peso']

            # distribuir fluxos entre materiais
            i = materiais.index(material)
            j = (i + np.random.randint(1, len(materiais))) % len(materiais)
            fluxo_matrix[i, j] += peso * np.sin(tempo + i) ** 2
        im3 = ax3.imshow(fluxo_matrix, cmap='Blues', interpolation='nearest')
        ax3.set_title('Matriz de Fluxo entre Materiais\n(Intensidade Temporal)')
        ax3.set_xticks(np.arange(len(materiais)))
        ax3.set_xticklabels(materiais, rotation=45)
        ax3.set_yticks(np.arange(len(materiais)))
        ax3.set_yticklabels(materiais)

        # adicionar valores na matriz
        for i in range(len(materiais)):
            for j in range(len(materiais)):
                ax3.text(j, i, f'{fluxo_matrix[i, j]:.0f}', ha='center', va='center', color='red', fontweight='bold')