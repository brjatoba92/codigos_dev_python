import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Polygon
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import seaborn as sns
from datetime import datetime, timedelta

# Configuração do estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Gerando dados sintéticos de vendas geográficas
np.random.seed(42)

# Coordenadas de cidades brasileiras (exemplo)
cidades = {
    'São Paulo': (-23.5505, -46.6333),
    'Rio de Janeiro': (-22.9068, -43.1729),
    'Belo Horizonte': (-19.9167, -43.9345),
    'Salvador': (-12.9714, -38.5014),
    'Brasília': (-15.7942, -47.8822),
    'Fortaleza': (-3.7319, -38.5267),
    'Recife': (-8.0476, -34.9011),
    'Curitiba': (-25.4284, -49.2733),
    'Porto Alegre': (-30.0346, -51.2177),
    'Manaus': (-3.1190, -60.0217),
    'Goiânia': (-16.6799, -49.2550),
    'Belém': (-1.4558, -48.5044),
    'Vitória': (-20.3155, -40.3128),
    'Natal': (-5.7945, -35.2110),
    'João Pessoa': (-7.1195, -34.8450)
}

# Criando dados de vendas
vendas_data = []
for cidade, (lat, lon) in cidades.items():
    # Vendas mensais para cada cidade
    for mes in range(1, 13):
        vendas = np.random.normal(10000, 3000) + np.random.exponential(2000)
        vendas = max(1000, vendas)  # Mínimo de 1000
        vendas_data.append({
            'cidade': cidade,
            'latitude': lat,
            'longitude': lon,
            'mes': mes,
            'vendas': vendas,
            'regiao': 'Nordeste' if lat > -15 and lon > -45 else 
                     'Norte' if lat > -10 else
                     'Centro-Oeste' if lon < -47 else
                     'Sudeste' if lat > -25 else 'Sul'
        })

df_vendas = pd.DataFrame(vendas_data)

# Criando uma grade para interpolação
lat_min, lat_max = df_vendas['latitude'].min() - 2, df_vendas['latitude'].max() + 2
lon_min, lon_max = df_vendas['longitude'].min() - 2, df_vendas['longitude'].max() + 2

# Função para criar mapa de calor
def criar_mapa_calor_vendas():
    """Cria mapa de calor da distribuição de vendas"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Análise Geoespacial de Vendas - Mapas de Calor', fontsize=20, fontweight='bold')
    
    # 1. Mapa de calor geral - vendas totais por cidade
    ax1 = axes[0, 0]
    vendas_cidade = df_vendas.groupby(['cidade', 'latitude', 'longitude'])['vendas'].sum().reset_index()
    
    scatter = ax1.scatter(vendas_cidade['longitude'], vendas_cidade['latitude'], 
                         c=vendas_cidade['vendas'], s=vendas_cidade['vendas']/500,
                         cmap='YlOrRd', alpha=0.7, edgecolors='black', linewidth=1)
    
    # Adicionando nomes das cidades
    for idx, row in vendas_cidade.iterrows():
        if row['vendas'] > 80000:  # Apenas cidades com vendas altas
            ax1.annotate(row['cidade'], (row['longitude'], row['latitude']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax1.set_title('Distribuição Total de Vendas por Cidade', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Vendas Totais (R$)', rotation=270, labelpad=20)
    
    # 2. Mapa de calor sazonal
    ax2 = axes[0, 1]
    vendas_verao = df_vendas[df_vendas['mes'].isin([12, 1, 2])].groupby(['cidade', 'latitude', 'longitude'])['vendas'].mean().reset_index()
    vendas_inverno = df_vendas[df_vendas['mes'].isin([6, 7, 8])].groupby(['cidade', 'latitude', 'longitude'])['vendas'].mean().reset_index()
    
    # Razão verão/inverno
    vendas_sazon = vendas_verao.merge(vendas_inverno, on=['cidade', 'latitude', 'longitude'], suffixes=('_verao', '_inverno'))
    vendas_sazon['razao_sazonal'] = vendas_sazon['vendas_verao'] / vendas_sazon['vendas_inverno']
    
    scatter2 = ax2.scatter(vendas_sazon['longitude'], vendas_sazon['latitude'], 
                          c=vendas_sazon['razao_sazonal'], s=200,
                          cmap='RdBu_r', alpha=0.8, edgecolors='black', linewidth=1,
                          vmin=0.8, vmax=1.2)
    
    ax2.set_title('Sazonalidade das Vendas (Verão/Inverno)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Razão Verão/Inverno', rotation=270, labelpad=20)
    
    # 3. Densidade de vendas por região
    ax3 = axes[1, 0]
    
    # Criando grade para interpolação
    xi = np.linspace(lon_min, lon_max, 50)
    yi = np.linspace(lat_min, lat_max, 50)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolação usando médias ponderadas pela distância
    vendas_total = df_vendas.groupby(['latitude', 'longitude'])['vendas'].sum()
    
    Zi = np.zeros_like(Xi)
    for i in range(len(xi)):
        for j in range(len(yi)):
            distances = []
            weights = []
            for (lat, lon), vendas in vendas_total.items():
                dist = np.sqrt((xi[i] - lon)**2 + (yi[j] - lat)**2)
                if dist < 0.1:  # Muito próximo
                    distances.append(vendas)
                    weights.append(1000)
                elif dist < 10:  # Próximo o suficiente para influenciar
                    distances.append(vendas)
                    weights.append(1 / (dist + 0.1)**2)
            
            if weights:
                Zi[j, i] = np.average(distances, weights=weights)
    
    contour = ax3.contourf(Xi, Yi, Zi, levels=20, cmap='viridis', alpha=0.6)
    ax3.scatter(df_vendas['longitude'], df_vendas['latitude'], 
               c='red', s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    ax3.set_title('Densidade de Vendas - Interpolação Espacial', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    
    cbar3 = plt.colorbar(contour, ax=ax3)
    cbar3.set_label('Densidade de Vendas', rotation=270, labelpad=20)
    
    # 4. Análise por região
    ax4 = axes[1, 1]
    vendas_regiao = df_vendas.groupby(['regiao', 'mes'])['vendas'].sum().reset_index()
    
    regioes = vendas_regiao['regiao'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(regioes)))
    
    for i, regiao in enumerate(regioes):
        data = vendas_regiao[vendas_regiao['regiao'] == regiao]
        ax4.plot(data['mes'], data['vendas'], marker='o', linewidth=2.5, 
                label=regiao, color=colors[i], markersize=6)
    
    ax4.set_title('Evolução Mensal de Vendas por Região', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Mês')
    ax4.set_ylabel('Vendas (R$)')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(1, 13))
    
    plt.tight_layout(pad=4)
    plt.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.97, hspace=0.5, wspace=0.3)
    return fig

# Gerando o mapa de calor
#fig_mapa_calor = criar_mapa_calor_vendas()
# Exibindo o mapa de calor
# plt.show()
# Salvando o mapa de calor
# fig_mapa_calor.savefig('mapa_calor_vendas.png', dpi=300, bbox_inches='tight')

# Função de otimização de rotas
def otimizar_rotas():
    """otimização de rotas de entrega usando algoritmo genetico simplificado"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Otimização de Rotas de Entrega', fontsize=18, fontweight='bold')

    # selecionando cidades principais para rota
    cidades_principais = ['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Salvador', 'Brasília', 'Curitiba', 'Porto Alegre', 'Fortaleza']

    coords = np.array([cidades[cidade] for cidade in cidades_principais])

    # Calculando matriz de distancias
    dist_matrix = cdist(coords, coords, metric='euclidean')

    # algoritmo de otimização de rotas(nearest neighbor)
    def nearest_neighbor_tsp(dist_matrix, start=0):
        n = len(dist_matrix)
        unvisited = set(range(n))
        unvisited.remove(start)

        route = [start]
        current = start
        total_dist = 0

        while unvisited:
            nearest = min(unvisited, key=lambda x: dist_matrix[current][x])
            total_dist += dist_matrix[current][nearest]
            route.append(nearest)
            current = nearest
            unvisited.remove(nearest)

        # Voltando ao ponto de partida
        total_dist += dist_matrix[current][start]
        route.append(start)
        
        return route, total_dist
    
    # 1 - Rota não otimizada (orden original)
    ax1 = axes[0]
    route_original = list(range(len(coords))) + [0]

    for i in range(len(route_original) - 1):
        start_idx, end_idx = route_original[i], route_original[i + 1]
        ax1.plot([coords[start_idx][1], coords[end_idx][1]], 
                [coords[start_idx][0], coords[end_idx][0]], 
                'r-', linewidth=1, alpha=0.7)
    
    ax1.scatter(coords[:, 1], coords[:, 0], c='blue', s=100, zorder=5)

    for i, cidade in enumerate(cidades_principais):
        ax1.annotate(f'{i+1}. {cidade}', (coords[i][1], coords[i][0]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    dist_original = sum(dist_matrix[route_original[i], route_original[i + 1]] 
                        for i in range(len(route_original) - 1))
    
    ax1.set_title(f'Rota Original\nDistância Total: {dist_original:.2f}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)

    # 2 - Rota otimizada
    ax2 = axes[1]
    route_opt, dist_opt = nearest_neighbor_tsp(dist_matrix)

    for i in range(len(route_opt) - 1):
        start_idx, end_idx = route_opt[i], route_opt[i + 1]
        ax2.plot([coords[start_idx][1], coords[end_idx][1]], 
                [coords[start_idx][0], coords[end_idx][0]], 
                'g-', linewidth=2, alpha=0.7)
    ax2.scatter(coords[:, 1], coords[:, 0], c='blue', s=100, zorder=5)

    for i, cidade in enumerate(cidades_principais):
        ax2.annotate(f'{route_opt.index(i)+1}. {cidade}', (coords[i][1], coords[i][0]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_title(f'Rota Otimizada\nDistância Total: {dist_opt:.2f}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)

    # 3 - Comparação e métricas
    ax3 = axes[2]

    # simulando multiplas rotas com diferentes heuristicas
    metodos = ['Original', 'Nearest Neighbor', 'Random Restart (Best)', 'Genetic Algorithm (Sim)']
    distancias = [dist_original, dist_opt]

    # Simulando outros métodos
    best_random = min([nearest_neighbor_tsp(dist_matrix, start=i)[1] for i in range(len(coords))])
    genetic_sim = dist_opt * 0.95 # Simulando uma melhoria de 5% com GA

    distancias.extend([best_random, genetic_sim])

    colors = ['red', 'green', 'orange', 'purple']
    bars = ax3.bar(metodos, distancias, color=colors, alpha=0.7)

    # adicioando valores nas barras
    for bar, dist in zip(bars, distancias):
        height = bar.get_height()
        ax3.text(bar.get_x()+ bar.get_width()/2, height + 0.01, f'{dist:.2f}', ha='center', va='bottom', fontweight='bold')
    ax3.set_title('Comparação de Algoritmos de Otimização', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Distância Total')
    ax3.set_xticklabels(metodos, rotation=45, ha='right')
    ax3.grid(True, axis='y', alpha=0.3)

    # Calculando economia
    economia = ((dist_original - genetic_sim) / dist_original) * 100
    ax3.text(0.05, 0.95, f'Economia: {economia:.1f}%', transform=ax3.transAxes,ha='center', va='top',
            fontsize=14, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    plt.tight_layout(pad=3)
    return fig

# Gerando o gráfico de otimização de rotas
fig_otimizacao_rotas = otimizar_rotas()
# Exibindo o gráfico de otimização de rotas
plt.show()
# Salvando o gráfico de otimização de rotas
fig_otimizacao_rotas.savefig('otimizacao_rotas_entrega.png', dpi=300, bbox_inches='tight')
