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

# Configuração global para estilo corporativo
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E8B57', '#4682B4', '#CD853F', '#DC143C', '#9932CC']

# =============================================================================
# EXEMPLO 1: DASHBOARD MULTI-DIMENSIONAL DE PERFORMANCE DE REDE
# Análise complexa de múltiplas métricas simultaneamente
# =============================================================================

def dashboard_performance():
    """
    Dashboard avançado mostrando performance de rede empresarial com 6 subplots
    interconectados, incluindo análise de correlação e métricas de centralidade
    """
    # Dados simulados de uma rede de 15 empresas de materiais de construção
    np.random.seed(42)
    empresas = [f'Empresa_{i+1}' for i in range(15)]
    
    # Métricas complexas
    vendas_mensais = np.random.exponential(2000, (15, 12)) + np.random.normal(5000, 1000, (15, 12))
    margem_lucro = np.random.beta(2, 5, 15) * 100  # Beta distribution para margem mais realista
    tempo_entrega = np.random.gamma(2, 3, 15)  # Gamma para tempos de entrega
    satisfacao_cliente = np.random.normal(8.5, 1.2, 15)
    market_share = np.random.dirichlet(np.ones(15)) * 100  # Soma = 100%
    produtos_vendidos = np.random.poisson(50, 15)
    
    # Criar correlações realistas
    vendas_totais = vendas_mensais.sum(axis=1)
    margem_lucro += (vendas_totais - vendas_totais.mean()) / vendas_totais.std() * 5
    satisfacao_cliente += (margem_lucro - margem_lucro.mean()) / margem_lucro.std() * 0.5
    
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('DASHBOARD ANALÍTICO - REDE DE MATERIAIS DE CONSTRUÇÃO\nAnálise Multi-dimensional de Performance', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # 1. Heatmap de Vendas Mensais com padrões sazonais
    ax1 = plt.subplot(3, 3, 1)
    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
             'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    
    # Adicionar sazonalidade realista (picos no verão)
    sazonalidade = np.array([0.8, 0.7, 0.9, 1.1, 1.3, 1.4, 1.5, 1.4, 1.2, 1.0, 0.8, 0.7])
    vendas_sazonais = vendas_mensais * sazonalidade
    
    im1 = ax1.imshow(vendas_sazonais, cmap='RdYlGn', aspect='auto')
    ax1.set_xticks(range(12))
    ax1.set_xticklabels(meses, rotation=45)
    ax1.set_yticks(range(15))
    ax1.set_yticklabels([f'E{i+1}' for i in range(15)])
    ax1.set_title('Vendas Mensais (R$ mil)\ncom Padrão Sazonal', fontweight='bold')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 2. Scatter plot 3D - Vendas vs Margem vs Satisfação
    ax2 = plt.subplot(3, 3, 2, projection='3d')
    scatter = ax2.scatter(vendas_totais/1000, margem_lucro, satisfacao_cliente, 
                         c=market_share, s=produtos_vendidos*3, alpha=0.7, cmap='viridis')
    ax2.set_xlabel('Vendas Totais (R$ M)')
    ax2.set_ylabel('Margem Lucro (%)')
    ax2.set_zlabel('Satisfação Cliente')
    ax2.set_title('Análise 3D: Performance\nIntegrada', fontweight='bold')
    
    # 3. Boxplot comparativo de tempo de entrega
    ax3 = plt.subplot(3, 3, 3)
    
    # Criar categorias baseadas em performance
    performance_cat = pd.cut(vendas_totais, bins=3, labels=['Baixa', 'Média', 'Alta'])
    df_box = pd.DataFrame({
        'Performance': performance_cat,
        'Tempo_Entrega': tempo_entrega
    })
    
    box_data = [df_box[df_box['Performance'] == cat]['Tempo_Entrega'].values 
                for cat in ['Baixa', 'Média', 'Alta']]
    
    bp = ax3.boxplot(box_data, labels=['Baixa', 'Média', 'Alta'], patch_artist=True)
    colors_box = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_title('Tempo Entrega por\nCategoria Performance', fontweight='bold')
    ax3.set_ylabel('Dias')
    
    # 4. Gráfico de radar para top 5 empresas
    ax4 = plt.subplot(3, 3, 4, projection='polar')
    
    # Selecionar top 5 por vendas
    top5_idx = np.argsort(vendas_totais)[-5:]
    
    # Normalizar métricas para radar (0-10)
    metricas_norm = np.array([
        vendas_totais[top5_idx] / vendas_totais.max() * 10,
        margem_lucro[top5_idx] / margem_lucro.max() * 10,
        satisfacao_cliente[top5_idx],
        (1 / tempo_entrega[top5_idx]) * 10,  # Inverter tempo (menor é melhor)
        market_share[top5_idx] / market_share.max() * 10
    ]).T
    
    angulos = np.linspace(0, 2*np.pi, 5, endpoint=False).tolist()
    angulos += angulos[:1]  # Fechar o polígono
    
    for i, idx in enumerate(top5_idx):
        valores = metricas_norm[i].tolist()
        valores += valores[:1]
        ax4.plot(angulos, valores, 'o-', linewidth=2, label=f'Empresa_{idx+1}')
        ax4.fill(angulos, valores, alpha=0.1)
    
    ax4.set_xticks(angulos[:-1])
    ax4.set_xticklabels(['Vendas', 'Margem', 'Satisfação', 'Agilidade', 'Market Share'])
    ax4.set_title('Radar TOP 5 Empresas\nPor Performance', fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 5. Análise de correlação avançada
    ax5 = plt.subplot(3, 3, 5)
    
    # Matriz de correlação
    dados_corr = np.column_stack([
        vendas_totais, margem_lucro, satisfacao_cliente, 
        tempo_entrega, market_share, produtos_vendidos
    ])
    
    labels_corr = ['Vendas', 'Margem', 'Satisfação', 'T.Entrega', 'M.Share', 'N.Produtos']
    corr_matrix = np.corrcoef(dados_corr.T)
    
    # Criar máscara para triângulo superior
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    im5 = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax5.set_xticks(range(len(labels_corr)))
    ax5.set_yticks(range(len(labels_corr)))
    ax5.set_xticklabels(labels_corr, rotation=45)
    ax5.set_yticklabels(labels_corr)
    
    # Adicionar valores de correlação
    for i in range(len(labels_corr)):
        for j in range(len(labels_corr)):
            if not mask[i, j]:
                text = ax5.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')
    
    ax5.set_title('Matriz Correlação\nMétricas Empresariais', fontweight='bold')
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    
    # 6. Série temporal com tendências
    ax6 = plt.subplot(3, 3, (6, 9))  # Ocupar espaço de 2 subplots
    
    # Gerar série temporal realista
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    
    # Múltiplas séries com tendências diferentes
    base_demand = 1000 + 200 * np.sin(2 * np.pi * np.arange(365) / 365)  # Sazonalidade anual
    weekly_pattern = 50 * np.sin(2 * np.pi * np.arange(365) / 7)  # Padrão semanal
    trend = np.linspace(0, 300, 365)  # Tendência crescimento
    noise = np.random.normal(0, 30, 365)
    
    # Diferentes categorias de produtos
    cimento = base_demand + trend + noise + 200
    aco = base_demand * 0.8 + trend * 1.2 + noise + weekly_pattern + 150
    madeira = base_demand * 0.6 + trend * 0.8 + noise * 1.5 + 100
    
    ax6.plot(dates, cimento, label='Cimento', linewidth=2, color='#8B4513')
    ax6.plot(dates, aco, label='Aço', linewidth=2, color='#2F4F4F')
    ax6.plot(dates, madeira, label='Madeira', linewidth=2, color='#228B22')
    
    # Adicionar médias móveis
    window = 30
    ax6.plot(dates[window-1:], np.convolve(cimento, np.ones(window)/window, mode='valid'), 
             '--', alpha=0.7, color='#8B4513', label='Média Móvel Cimento')
    ax6.plot(dates[window-1:], np.convolve(aco, np.ones(window)/window, mode='valid'), 
             '--', alpha=0.7, color='#2F4F4F', label='Média Móvel Aço')
    ax6.plot(dates[window-1:], np.convolve(madeira, np.ones(window)/window, mode='valid'), 
             '--', alpha=0.7, color='#228B22', label='Média Móvel Madeira')
    
    ax6.set_title('Demanda por Categoria de Produto\ncom Tendências e Sazonalidade', fontweight='bold')
    ax6.set_xlabel('Período')
    ax6.set_ylabel('Demanda (toneladas)')
    ax6.legend(ncol=2)
    ax6.grid(True, alpha=0.3)
    
    # Formatação de datas
    import matplotlib.dates as mdates
    ax6.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # 7. Análise de eficiência operacional
    ax7 = plt.subplot(3, 3, 7)
    
    # Calcular eficiência como função de múltiplas variáveis
    eficiencia = (satisfacao_cliente * margem_lucro) / (tempo_entrega + 1)
    
    # Scatter com tamanho proporcional às vendas
    scatter7 = ax7.scatter(market_share, eficiencia, 
                          s=vendas_totais/100, 
                          c=produtos_vendidos, 
                          alpha=0.7, cmap='plasma', edgecolors='black')
    
    # Adicionar linha de tendência
    z = np.polyfit(market_share, eficiencia, 1)
    p = np.poly1d(z)
    ax7.plot(market_share, p(market_share), "r--", alpha=0.8, linewidth=2)
    
    ax7.set_xlabel('Market Share (%)')
    ax7.set_ylabel('Índice Eficiência')
    ax7.set_title('Eficiência vs Market Share\n(Tamanho = Vendas)', fontweight='bold')
    
    # Anotar empresas top
    top3_eff = np.argsort(eficiencia)[-3:]
    for idx in top3_eff:
        ax7.annotate(f'E{idx+1}', (market_share[idx], eficiencia[idx]),
                    xytext=(5, 5), textcoords='offset points',
                    fontweight='bold', color='red')
    
    plt.colorbar(scatter7, ax=ax7, label='N° Produtos')
    
    # 8. Distribuição geográfica simulada
    ax8 = plt.subplot(3, 3, 8)
    
    # Coordenadas simuladas de filiais
    np.random.seed(123)
    lat = np.random.normal(-23.5505, 2, 15)  # São Paulo como centro
    lon = np.random.normal(-46.6333, 2, 15)
    
    # Criar mapa de calor de vendas por região
    scatter8 = ax8.scatter(lon, lat, s=vendas_totais/50, c=margem_lucro, 
                          cmap='RdYlBu_r', alpha=0.7, edgecolors='black')
    
    # Adicionar círculos para indicar zonas de influência
    for i in range(len(lat)):
        circle = Circle((lon[i], lat[i]), 0.3, fill=False, 
                       linestyle='--', alpha=0.5, color='gray')
        ax8.add_patch(circle)
    
    ax8.set_xlabel('Longitude')
    ax8.set_ylabel('Latitude')
    ax8.set_title('Distribuição Geográfica\n(Tamanho=Vendas, Cor=Margem)', fontweight='bold')
    plt.colorbar(scatter8, ax=ax8, label='Margem Lucro (%)')
    
    plt.tight_layout()
    plt.show()
    
    # Retornar dados para análises posteriores
    return {
        'empresas': empresas,
        'vendas_totais': vendas_totais,
        'margem_lucro': margem_lucro,
        'satisfacao_cliente': satisfacao_cliente,
        'eficiencia': eficiencia
    }

# Executar o dashboard
dashboard_performance()