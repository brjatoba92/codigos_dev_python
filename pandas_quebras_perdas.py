import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import itertools
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Simulando layout de loja com coordenadas e características
setores_loja = {
    'Entrada': {'x_range': (0, 10), 'y_range': (0, 5), 'trafego': 100, 'conversao': 0.15},
    'Cimentos_Cal': {'x_range': (0, 15), 'y_range': (5, 15), 'trafego': 60, 'conversao': 0.25},
    'Ferragens': {'x_range': (15, 25), 'y_range': (0, 10), 'trafego': 45, 'conversao': 0.30},
    'Tintas': {'x_range': (15, 25), 'y_range': (10, 20), 'trafego': 40, 'conversao': 0.20},
    'Ceramicas': {'x_range': (25, 40), 'y_range': (0, 15), 'trafego': 35, 'conversao': 0.18},
    'Madeiras': {'x_range': (25, 40), 'y_range': (15, 25), 'trafego': 30, 'conversao': 0.22},
    'Fundo_Loja': {'x_range': (40, 50), 'y_range': (0, 25), 'trafego': 15, 'conversao': 0.35}
}

#Tipos perdas especificas do setor
tipos_perdas = {
    'Quebra Manuseio': {'freq': 0.25, 'valor_medio': 150, 'sazonalidade': [1.2, 1.0, 0.8]}, # verao, meio, inverno
    'Vencimento': {'freq': 0.15, 'valor_medio': 80, 'sazonalidade': [0.8, 1.0, 1.3]},
    'Furto_Interno': {'freq': 0.12, 'valor_medio': 50, 'sazonalidade': [1.0, 1.1, 1.2]},
    'Furto_Externo': {'freq': 0.18, 'valor_medio': 50, 'sazonalidade': [1.1, 1.0, 1.3]},
    'Dano_Transporte': {'freq': 0.10, 'valor_medio': 50, 'sazonalidade': [1.0, 1.0, 1.0]},
    'Erro_Inventario': {'freq': 0.08, 'valor_medio': 30, 'sazonalidade': [0.9, 1.0, 1.1]},
    'Deteriorização': {'freq': 0.12, 'valor_medio': 30, 'sazonalidade': [1.4, 1.0, 0.7]},
}

# Gerando dados de perdas
perdas_data = []
data_inicio = pd.Timestamp('2024-01-01')

for dia in range(365):
    data_atual = data_inicio + timedelta(days=dia)

    # determinando a estação do ano (simplificada)
    if data_atual.month in [12, 1, 2]:
        estacao_idx = 0 # Verão
    elif data_atual.month in [6, 7, 8]:
        estacao_idx = 2 # Inverno
    else:
        estacao_idx = 1 # Meio - Outono e Primavera

    # Fator dia da semana (mais perdas no final de semana, maior movimento)
    fator_dia_semana = 1.3 if data_atual.weekday() >= 5 else 1.0

    for tipo_perda, config in tipos_perdas.items():
        # probabilidade de ocorrencia ajustada por sazonalidade
        prob_ocorrencia = config['freq'] * config['sazonalidade'][estacao_idx] * fator_dia_semana
        if np.random.random() < prob_ocorrencia:
            # valor da perda com variação
            valor_perda = max(10, np.random.normal(config['valor_medio'], config['valor_medio'] * 0.4))
            # Atribuindo a produtos/setores especificos
            setor_afetado = np.random.choice(list(setores_loja.keys()))

            # simulando detalhes adicionais
            turno = np.random.choice(['Manha', 'Tarde', 'Noite'], p=[0.5, 0.3, 0.2])
            funcionario_id = f'FUNC{np.random.randint(1, 21)}:03d' # 20 funcionarios

            perdas_data.append({
                'data_ocorrencia': data_atual,
                'tipo_perda': tipo_perda,
                'setor_afetado': setor_afetado,
                'valor_perda': round(valor_perda, 2),
                'turno': turno,
                'funcionario_responsavel': funcionario_id,
                'dia_semana': data_atual.day_name(),
                'mes': data_atual.month,
                'estacao': ['Verao', 'Meio-Ano', 'Inverno'][estacao_idx]
            })
print("Analise Detalhada de Perdas Operacionais:")
print("-" * 50)
