import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def previsao_demanda():
# Dados históricos de vendas (últimos 12 meses)
    np.random.seed(42)
    datas = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    
    produtos = ['Cimento 50kg', 'Tijolo Cerâmico', 'Tinta Látex']
    vendas_historicas = []
    
    for produto in produtos:
        for data in datas:
            # Simulando padrão semanal e sazonal
            base_vendas = 50 if produto == 'Cimento 50kg' else (200 if produto == 'Tijolo Cerâmico' else 20)
            
            # Padrão semanal (menos vendas no fim de semana)
            multiplicador_semanal = 0.3 if data.weekday() >= 5 else 1.0
            
            # Sazonalidade (mais vendas no verão)
            multiplicador_sazonal = 1.5 if data.month in [11, 12, 1, 2] else 1.0
            
            vendas = int(base_vendas * multiplicador_semanal * multiplicador_sazonal * np.random.uniform(0.7, 1.3))
            
            vendas_historicas.append({
                'data': data,
                'produto': produto,
                'vendas': max(0, vendas)
            })
    
    df_historico = pd.DataFrame(vendas_historicas)
    
    # Previsão por produto
    previsoes = []
    
    for produto in produtos:
        df_produto = df_historico[df_historico['produto'] == produto].copy()
        df_produto = df_produto.sort_values('data')
        
        # Média móvel de 7 e 30 dias
        df_produto['ma_7'] = df_produto['vendas'].rolling(window=7).mean()
        df_produto['ma_30'] = df_produto['vendas'].rolling(window=30).mean()
        
        # Tendência (regressão linear simples)
        df_produto['dias'] = (df_produto['data'] - df_produto['data'].min()).dt.days
        coef = np.polyfit(df_produto['dias'], df_produto['vendas'], 1)
        
        # Previsão para próximos 30 dias
        ultimo_dia = df_produto['dias'].max()
        proximos_30_dias = range(ultimo_dia + 1, ultimo_dia + 31)
        
        previsao_30_dias = np.polyval(coef, proximos_30_dias)
        media_recente = df_produto['ma_7'].iloc[-7:].mean()
        
        # Ajuste da previsão com média móvel
        previsao_ajustada = (previsao_30_dias + media_recente) / 2
        
        previsoes.append({
            'produto': produto,
            'media_vendas_mes': df_produto['vendas'].tail(30).mean().round(1),
            'tendencia': 'Crescente' if coef[0] > 0 else 'Decrescente',
            'previsao_proximo_mes': previsao_ajustada.mean().round(1),
            'estoque_sugerido': int(previsao_ajustada.sum() * 1.2)  # 20% margem segurança
        })
    
    df_previsoes = pd.DataFrame(previsoes)
    
    print("\n=== PREVISÃO DE DEMANDA ===")
    print(df_previsoes)

    return df_previsoes

previsao_demanda()