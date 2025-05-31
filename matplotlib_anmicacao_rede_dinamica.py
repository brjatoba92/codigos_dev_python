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

def animacao_rede_dinamica():
    """
    Animação sofisticada mostrando evolução temporal de uma rede de fornecedores
    com métricas de centralidade, fluxos de materiais e detecção de comunidades
    """
    
    # Criar rede base de fornecedores
    G = nx.barabasi_albert_graph(20, 3, seed=42)
    
    # Adicionar atributos realistas aos nós
    empresas_tipos = ['Fornecedor', 'Distribuidor', 'Loja', 'Construtora']
    for node in G.nodes():
        G.nodes[node]['tipo'] = np.random.choice(empresas_tipos)
        G.nodes[node]['capacidade'] = np.random.exponential(1000)
        G.nodes[node]['localizacao'] = (np.random.uniform(-50, 50), 
                                       np.random.uniform(-50, 50))
    
    # Adicionar pesos às arestas (volume de negócios)
    for edge in G.edges():
        G.edges[edge]['peso'] = np.random.exponential(500)
        G.edges[edge]['material'] = np.random.choice(['Cimento', 'Aço', 'Madeira', 'Cerâmica'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('ANÁLISE DINÂMICA DE REDE DE FORNECEDORES\nEvolução Temporal e Métricas de Centralidade', 
                fontsize=14, fontweight='bold')
    
    def atualizar_frame(frame):
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        # Simular mudanças temporais na rede
        tempo = frame / 10.0
        
        # 1. Rede principal com layout spring
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Calcular métricas de centralidade
        centralidade_betweenness = nx.betweenness_centrality(G)
        centralidade_closeness = nx.closeness_centrality(G)
        centralidade_degree = nx.degree_centrality(G)
        
        # Cores por tipo de empresa
        cores_tipos = {'Fornecedor': '#FF6B6B', 'Distribuidor': '#4ECDC4', 
                      'Loja': '#45B7D1', 'Construtora': '#96CEB4'}
        node_colors = [cores_tipos[G.nodes[node]['tipo']] for node in G.nodes()]
        
        # Tamanhos proporcionais à centralidade
        node_sizes = [centralidade_betweenness[node] * 2000 + 300 for node in G.nodes()]
        
        # Desenhar rede no ax1
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8)
        
        # Desenhar arestas com espessura proporcional ao peso
        edge_weights = [G.edges[edge]['peso'] for edge in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax1, width=np.array(edge_weights)/200, 
                              alpha=0.6, edge_color='gray')
        
        # Labels dos nós
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8, font_weight='bold')
        
        ax1.set_title(f'Rede de Fornecedores (t={tempo:.1f})\nTamanho = Centralidade Betweenness')
        ax1.axis('off')
        
        # 2. Análise de centralidade por tipo
        tipos_data = {}
        for node in G.nodes():
            tipo = G.nodes[node]['tipo']
            if tipo not in tipos_data:
                tipos_data[tipo] = {'betweenness': [], 'closeness': [], 'degree': []}
            tipos_data[tipo]['betweenness'].append(centralidade_betweenness[node])
            tipos_data[tipo]['closeness'].append(centralidade_closeness[node])
            tipos_data[tipo]['degree'].append(centralidade_degree[node])
        
        # Boxplot de centralidade por tipo
        tipos = list(tipos_data.keys())
        betweenness_data = [tipos_data[tipo]['betweenness'] for tipo in tipos]
        
        bp = ax2.boxplot(betweenness_data, labels=tipos, patch_artist=True)
        for patch, tipo in zip(bp['boxes'], tipos):
            patch.set_facecolor(cores_tipos[tipo])
            patch.set_alpha(0.7)
        
        ax2.set_title('Centralidade Betweenness\npor Tipo de Empresa')
        ax2.set_ylabel('Centralidade')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Fluxo de materiais (chord diagram simplificado)
        materiais = ['Cimento', 'Aço', 'Madeira', 'Cerâmica']
        fluxo_matrix = np.zeros((len(materiais), len(materiais)))
        
        # Simular fluxos baseados nas conexões
        for edge in G.edges():
            material = G.edges[edge]['material']
            peso = G.edges[edge]['peso']
            
            # Distribuir fluxo entre materiais
            i = materiais.index(material)
            j = (i + np.random.randint(1, len(materiais))) % len(materiais)
            fluxo_matrix[i, j] += peso * np.sin(tempo + i) ** 2
        
        im3 = ax3.imshow(fluxo_matrix, cmap='Blues', interpolation='nearest')
        ax3.set_xticks(range(len(materiais)))
        ax3.set_yticks(range(len(materiais)))
        ax3.set_xticklabels(materiais, rotation=45)
        ax3.set_yticklabels(materiais)
        ax3.set_title('Matriz de Fluxo entre Materiais\n(Intensidade Temporal)')
        
        # Adicionar valores na matriz
        for i in range(len(materiais)):
            for j in range(len(materiais)):
                ax3.text(j, i, f'{fluxo_matrix[i, j]:.0f}', 
                        ha='center', va='center', color='red', fontweight='bold')
        
        # 4. Métricas temporais da rede
        if not hasattr(atualizar_frame, 'historico'):
            atualizar_frame.historico = {
                'tempo': [],
                'densidade': [],
                'clustering': [],
                'centralidade_media': [],
                'num_componentes': []
            }
        
        # Calcular métricas da rede
        densidade = nx.density(G)
        clustering = nx.average_clustering(G)
        centralidade_media = np.mean(list(centralidade_betweenness.values()))
        num_componentes = nx.number_connected_components(G)
        
        # Adicionar ao histórico
        atualizar_frame.historico['tempo'].append(tempo)
        atualizar_frame.historico['densidade'].append(densidade)
        atualizar_frame.historico['clustering'].append(clustering)
        atualizar_frame.historico['centralidade_media'].append(centralidade_media)
        atualizar_frame.historico['num_componentes'].append(num_componentes)
        
        # Plotar séries temporais
        ax4.plot(atualizar_frame.historico['tempo'], 
                atualizar_frame.historico['densidade'], 
                'o-', label='Densidade', color='red')
        ax4.plot(atualizar_frame.historico['tempo'], 
                atualizar_frame.historico['clustering'], 
                's-', label='Clustering', color='blue')
        ax4.plot(atualizar_frame.historico['tempo'], 
                [x*10 for x in atualizar_frame.historico['centralidade_media']], 
                '^-', label='Centralidade Média (×10)', color='green')
        
        ax4.set_title('Evolução das Métricas de Rede')
        ax4.set_xlabel('Tempo')
        ax4.set_ylabel('Valor da Métrica')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Adicionar informações textuais
        info_text = f"""
        Tempo: {tempo:.1f}
        Nós: {G.number_of_nodes()}
        Arestas: {G.number_of_edges()}
        Densidade: {densidade:.3f}
        Clustering: {clustering:.3f}
        """
        ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Criar animação
    print("Iniciando animação de rede dinâmica...")
    print("Pressione Ctrl+C para interromper a animação")
    
    try:
        ani = FuncAnimation(fig, atualizar_frame, frames=50, interval=500, repeat=True)
        plt.tight_layout(pad=4)
        plt.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.97,
                    hspace=0.5, wspace=0.3)
        plt.show()
    except KeyboardInterrupt:
        print("Animação interrompida pelo usuário")
    
    return G

animacao_rede_dinamica()