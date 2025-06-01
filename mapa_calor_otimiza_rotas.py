import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Curcle, Polygon
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import seaborn as sns
from datetime import datetime, timedelta

# Configuração estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')

# Gerando dados sinteticos de vendas geograficas
np.random.seed(42)

# Coordenadas de cidades brasileiras (exemplo)
cidades = {
    'São Paulo': (-23.5505, -46.6333),
    'Rio de Janeiro': (-22.9068, -43.1729),
    'Belo Horizonte': (-19.9167, -43.9345),
    'Curitiba': (-25.4284, -49.2733),
    'Porto Alegre': (-30.0346, -51.2177),
    'Salvador': (-12.9714, -38.5014),
    'Fortaleza': (-3.7172, -38.5433),
    'Brasília': (-15.7801, -47.9292),
    'Manaus': (-3.1190, -60.0217),
    'Recife': (-8.0476, -34.8770),
    'Goiânia': (-16.6869, -49.2643),
    'Belém': (-1.4558, -48.4902),
    'Natal': (-5.7945, -35.2110),
    'Campo Grande': (-20.4697, -54.6201),
    'Florianópolis': (-27.5954, -48.5480),
    'Maceió': (-9.6659, -35.7350),
    'João Pessoa': (-7.1150, -34.8631),
    'Aracaju': (-10.9472, -37.0731),
    'Teresina': (-5.0892, -42.8013),
    'Cuiabá': (-15.6010, -56.0979)
}

# criando dados de vendas
vendas_data = []
for cidade, (lat, lon) in cidades.items():
    # vendas mensais para cada cidade
    for mes in range(1, 13):
        vendas = np.random.normal(1000, 3000) + np.random.exponential(2000)
        vendas = max(1000, vendas) #minimo de 1000
        vendas_data.append({
            'cidade': cidade,
            'mes': mes,
            'vendas': vendas,
            'latitude': lat,
            'longitude': lon,
            'regiao': 'Nordeste' if lat > -15 and lon > -45 else 
                    'Norte' if lat > -10 else 
                    'Centro-Oeste' if lat > -20 else 
                    'Sudeste' if lat > -25 else 'Sul'
        })
df_vendas = pd.DataFrame(vendas_data)

# Criando uma grade para interpolação
lat_min, lat_max =  df_vendas['latitude'].min(), df_vendas['latitude'].max() + 2
lon_min, lon_max =  df_vendas['longitude'].min(), df_vendas['longitude'].max() + 2
