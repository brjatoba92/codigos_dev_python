import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from mpl.toolkits.mplot3d import Axes3D
import seaborn as sns
from datatime import datetime, timedelta

#Configuração global para estilo corporativo
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E8B57', '#4682B4', '#CD853F', '#DC143C', '#9932CC']

# Analise complexa de multiplas métricas simultaneamente
def exemplo_1_dashboard_performance():
    """
    Dashboard avançado mostrando performance de rede empresarial com 6 subplots
    interconectados, incluindo análise de correlação e métricas de centralidade
    """

    # dados simulados de uma rede de 15 lojas de materiais de construção
    np.random.seed(42)
    lojas = [f'Loja {i+1}' for i in range(15)]

    # metricas
    vendas_mensais = np.random.exponential(10000, (15, 12)) + np.random.normal(50000, 1000, (15, 12))
    margem_lucro = np.random.beta(2,5,15)*100 # beta distribuição para margem mais realista
    tempo_entrega = np.random.gamma(2,3,15) # gamma distribuição para tempos de entrega
    satisfacao_cliente = np.random.normal(8.5, 1.2, 15)
    market_share = np.random.dirichlet(np.ones(15)) * 100  # Soma = 100%
    produtos_vendidos = np.random.poisson(50, 15)

    # Criar correlações realistas
    vendas_totais = vendas_mensais.sum(axis=1)
    margem_lucro += (vendas_totais - vendas_totais.mean()) / vendas_totais.std() * 5
    satisfacao_cliente += (margem_lucro - margem_lucro.mean()) / margem_lucro.std() * 0.5

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('DASHBOARD ANALÍTICO - REDE DE MATERIAIS DE CONSTRUÇÃO\nAnálise Multi-dimensional de Performance', fontsize=16, fontweight='bold', y=0.95)

    # 1 - Heatmap de vendas mensais com padrões sazonais
    ax1 = fig.add_subplot(3, 3, 1)
    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

    # adicionanod sazonalidade realista (picos no verão)
    sazonalidade = np.array([0.8, 0.7, 0.9, 1.1, 1.3, 1.4, 1.5, 1.4, 1.2, 1.0, 0.8, 0.7])
    vendas_sazonais = vendas_mensais * sazonalidade

    im1 = ax1.imshow(vendas_sazonais, cmap="RdYiGn", aspect='auto')
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(meses, rotation=45)
    ax1.set_yticks(range(15))
    ax1.set_yticklabels([f'E{i+1}' for i in range(15)])
    ax1.set_title('Vendas Mensais (R$ mil)\ncom Padrão Sazonal', fontweight='bold')
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 2 - Scatter plot 3D, Vendas vs Margem de Satisfação
    ax2 = plt.subplot(3, 3, 2, projection='3d')
    scatter = ax2.scatter(vendas_totais/1000, margem_lucro,satisfacao_cliente, c=market_share, s=produtos_vendidos*3, alpha=0.7, cmap='viridis')
    ax2.set_xlabel('Vendas Totais (R$ mil)')
    ax2.set_ylabel('Margem de Lucro(%)')
    ax2.set_zlabel('Satisfação do Cliente')
    ax2.set_title('Analise 3D: Performance\nIntegrada', fontweight='bold')

    # 3 - BoxPlot comparativo de tempo de entrega
    ax3 = plt.subplot(3,3,3)

    # Criar categorias baseadas em performance
    performance_cat = pd.cut(vendas_totais, bins=3, labels=['Baixa', 'Media', 'Alta'])
    df_box = pd.DataFrame({
        'Performance': performance_cat,
        'Tempo_Entrega': tempo_entrega
    })

    box_data = [df_box[df_box['Performance'] == cat]['Tempo_Entrega'].values for cat in ['Baixa', 'Media', 'Alta']]

    bp = ax3.boxplot(box_data, labels=['Baixa', 'Média', 'Alta'], patch_artist=True)
    colors_box = ['#ff9999', '#ffff99', '#99ff99']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)

    ax3.set_title('Comparação de Tempo de Entrega\npor Categoria Performance', fontweight='bold')
    ax3.set_ylabel('Tempo de Entrega (dias)')

    # 4 - Grafico de Radar para ttop 5 lojas
    ax4 = plt.subplot(3,3,4, projection='polar')

    # Selecionar top 5 por vendas
    top5_idx = np.argsort(vendas_totais)[-5:]

    # Normatizar métricas para o radar (0-10)
    metricas_norm = np.array([
        vendas_totais[top5_idx] / vendas_totais.max() * 10,
        margem_lucro[top5_idx] / margem_lucro.max() * 10,
        satisfacao_cliente [top5_idx],
        (1/tempo_entrega[top5_idx]) * 10, # inverter o tempo (menos é melhor)
        market_share[top5_idx] / market_share.max() * 10
    ]).T

    angulos = np.linspace(0,2*np.pi,5,endpoint=False).tolist()
    angulos += angulos[:1] # fechar o poligono

    for i, idx in enumerate(top5_idx):
        valores = metricas_norm[i].tolist()
        valores += valores[:1]
        ax4.plot(angulos, valores, 'o-', label=f'Loja {idx+1}')
        ax4.fill(angulos, valores, alpha=0.1)

    ax4.set_xticks(angulos[:-1])
    ax4.set_xticklabels(['Vendas', 'Margem', 'Satisfação', 'Agilidade', 'Market Share'])
    ax4.set_title('Radar Top 5 Lojas \n Por Performance', fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # 5 - Analise de correlação avançada
    ax5 = plt.subplot(3,3,5)
    
    # Matriz correlação
    dados_corr = np.column_stack([
        vendas_totais, margem_lucro, satisfacao_cliente, tempo_entrega, market_share, produtos_vendidos
    ])

    labels_corr = ['Vendas', 'Margem', 'Satisfação', 'T.Entrega', 'M.Share', 'N.Produtos']
    corr_matrix = np.corrcoef(dados_corr.T)

    # Criar mascara para triangulo superior
    mask = np.triu(np.ones_like(corr_matrix), dtype=bool)

    im5 = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax5.set_xticks(np.arange(len(labels_corr)))
    ax5.set_yticks(np.arange(len(labels_corr)))
    ax5.set_xticklabels(labels_corr, rotation=45)
    ax5.set_yticklabels(labels_corr)

    #adcionar valores de correlação
    for i in range(len(labels_corr)):
        for j in range(len(labels_corr)):
            if not mask[i,j]:
                text = ax5.text(j,i, f'{corr_matrix[i, j]:.2f}', ha="center", va="center", color="black", fontweight="bold")
    ax5.set_title('Matriz de Correlação \nMetricas Empresariais', fontweight='bold')    
    plt.colorbar(im5, ax=ax5, shrink=0.8)

    # 6 - Serie Temporal com Tendencias