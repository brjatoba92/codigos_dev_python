import pandas as pd
import numpy as np

def gestao_pedidos_logistica():
    np.random.seed(42)
    
    # Dados de pedidos
    pedidos_data = []
    for i in range(200):
        data_pedido = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        
        pedidos_data.append({
            'pedido_id': f'PED{i+1:04d}',
            'data_pedido': data_pedido,
            'cliente_tipo': np.random.choice(['Pessoa Física', 'Construtor', 'Revenda'], p=[0.4, 0.4, 0.2]),
            'regiao_entrega': np.random.choice(['Centro', 'Norte', 'Sul', 'Leste', 'Oeste']),
            'valor_pedido': np.random.uniform(500, 15000),
            'peso_kg': np.random.uniform(100, 5000),
            'prazo_prometido': np.random.choice([1, 2, 3, 5, 7]),  # dias
            'prazo_real': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),  # dias
            'transportadora': np.random.choice(['Express', 'RodoLog', 'Própria', 'CargoFast']),
            'status': np.random.choice(['Entregue', 'Em Trânsito', 'Atrasado'], p=[0.8, 0.1, 0.1])
        })
    
    df_pedidos = pd.DataFrame(pedidos_data)
    
    # Análises logísticas
    df_pedidos['atraso_dias'] = df_pedidos['prazo_real'] - df_pedidos['prazo_prometido']
    df_pedidos['no_prazo'] = df_pedidos['atraso_dias'] <= 0
    df_pedidos['custo_entrega_kg'] = df_pedidos['valor_pedido'] * 0.05 / df_pedidos['peso_kg']  # Estimativa

    # Performance por transportadora
    perf_transportadora = df_pedidos.groupby('transportadora').agg({
        'no_prazo': 'mean',
        'atraso_dias': 'mean',
        'custo_entrega_kg': 'mean',
        'pedido_id': 'count'
    }).round(3)
    perf_transportadora.columns = ['taxa_pontualidade', 'atraso_medio_dias', 'custo_medio_kg', 'total_pedidos']
    
    # Performance por região
    perf_regiao = df_pedidos.groupby('regiao_entrega').agg({
        'no_prazo': 'mean',
        'valor_pedido': 'mean',
        'atraso_dias': 'mean'
    }).round(2)
    perf_regiao.columns = ['taxa_pontualidade', 'ticket_medio', 'atraso_medio']
    
    # Análise de sazonalidade de pedidos
    df_pedidos['mes'] = df_pedidos['data_pedido'].dt.month
    pedidos_mes = df_pedidos.groupby('mes').agg({
        'pedido_id': 'count',
        'valor_pedido': 'sum',
    }).round(0)
    pedidos_mes.columns = ['quantidade_pedidos', 'faturamento']
    
    print("\n=== PERFORMANCE LOGÍSTICA POR TRANSPORTADORA ===")
    print(perf_transportadora.sort_values('taxa_pontualidade', ascending=False))
    
    print("\n=== PERFORMANCE POR REGIÃO DE ENTREGA ===")
    print(perf_regiao.sort_values('taxa_pontualidade', ascending=False))

    return df_pedidos

gestao_pedidos_logistica()