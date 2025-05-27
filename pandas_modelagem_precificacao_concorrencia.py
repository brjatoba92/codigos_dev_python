import pandas as pd
import numpy as np

# Simulando com diferentes dp mercado e concorrencia
np.random.seed(789)

# Produtos com diferentes elasticidades de preço
produtos_mercado = [
    {'nome': 'Cimento CP2 50 Kg', 'elasticidade': -0.8, 'custo': 20, 'preco_base': 30},
    {'nome': 'Tijolo Ceramico', 'elasticidade': -1.2, 'custo': 20, 'preco_base': 0.75},
    {'nome': 'Tinta Latex 18L', 'elasticidade': -0.6, 'custo': 20, 'preco_base': 185},
    {'nome': 'Piso Ceramico m2', 'elasticidade': -1.8, 'custo': 15, 'preco_base': 28},
    {'nome': 'Ferro 10mm barra', 'elasticidade': -0.4, 'custo': 28, 'preco_base': 42}
]

concorrentes = ['Concorrente A', 'Concorrente B', 'Concorrente C', 'Concorrente D']

# Gerando historico de preços e vendas
precificacao_data = []
datas_analise = pd.date_range('2024-01-01', '2024-12-31', freq='W') #semanal

for data in datas_analise:
    for produto in produtos_mercado:
        # preços dps concorrentes com variação semanal
        precos_concorrentes = {}
        for conc in concorrentes:
            variacao_semanal = np.random.normal(1, 0.05) # 5% de variacao
            preco_conc = produto['preco_base'] * variacao_semanal
            precos_concorrentes[conc] = round(preco_conc, 2)
        # Calculando posição mercado
        todos_precos = list(precos_concorrentes.values())
        preco_medio_mercado = np.mean(todos_precos)
        preco_min_mercado = min(todos_precos)
        preco_max_mercado = max(todos_precos)

        #nosso preço inicial (base)
        nosso_preco = produto['preco_base']

        # calculando demanda baseada na elasticidade preço
        # Q = Q0 * (P/P0)^E
        razao_preco = nosso_preco / preco_medio_mercado
        fator_elasticidade = razao_preco ** produto['elasticidade']

        #Demanda base ajustada por sazonalidade
        demanda_base = 100 #unidades base
        if data.month in [4,5,6,7,8,9]: #alta temporada construção:
            demanda_base *= 1.4
        
        demanda_estimada = demanda_base * fator_elasticidade * np.random.normal(1, 0.15)
        demanda_estimada = max(5, demanda_estimada)

        # calculando metricas financeiras
        receita = nosso_preco * demanda_estimada
        custo_total = produto['custo'] * demanda_estimada
        lucro = receita - custo_total
        margem_percentual = ((nosso_preco - produto['custo']) / nosso_preco) * 100

        # Market Share aproximado na posição do de preço
        if nosso_preco <= preco_min_mercado:
            market_share = 0.35 # lider em preço
        elif nosso_preco <= preco_medio_mercado:
            market_share = 0.25 # competitivo
        elif nosso_preco <= preco_max_mercado:
            market_share = 0.15 # premium
        else:
            market_share = 0.08 # muito caro
        
        market_share *= np.random.normal(1, 0.2) #add variação
        market_share = max(0.02, min (0.5, market_share)) #limitando

        precificacao_data.append({
            'data': data,
            'produto': produto['nome'],
            'nosso_preco': round(nosso_preco, 2),
            'preco_medio_mercado': round(preco_medio_mercado, 2),
            'preco_min_mercado': round(preco_min_mercado, 2),
            'preco_max_mercado': round(preco_max_mercado, 2),
            'demanda_estimada': round(demanda_estimada, 1),
            'receita': round(receita, 2),
            'custo_total': round(custo_total, 2),
            'lucro': round(lucro, 2),
            'margem_percentual': round(margem_percentual, 2),
            'market_share': round(market_share, 3),
            'elasticidade_preco': produto['elasticidade'],
            **{f'preco_{conc.lower().replace(" ", "_")}': precos_concorrentes[conc] for conc in concorrentes}
        })

df_precificacao = pd.DataFrame(precificacao_data)

print("Análise de Precificacão Dinamica:")
print("-" * 50)

