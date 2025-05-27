import pandas as pd
import numpy as np

# Simulando dados de mercado e concorrência
np.random.seed(789)

# Produtos com diferentes elasticidades de preço
produtos_mercado = [
    {'nome': 'Cimento CP2 50kg', 'elasticidade': -0.8, 'custo': 20.00, 'preco_base': 28.00},
    {'nome': 'Tijolo Ceramico', 'elasticidade': -1.2, 'custo': 0.45, 'preco_base': 0.75},
    {'nome': 'Tinta Latex 18L', 'elasticidade': -0.6, 'custo': 120.00, 'preco_base': 185.00},
    {'nome': 'Piso Ceramico m2', 'elasticidade': -1.5, 'custo': 15.00, 'preco_base': 28.00},
    {'nome': 'Ferro 12mm Barra', 'elasticidade': -0.4, 'custo': 28.00, 'preco_base': 42.00}
]

concorrentes = ['Concorrente A', 'Concorrente B', 'Concorrente C', 'Concorrente D']

# Gerando histórico de preços e vendas
precificacao_data = []
datas_analise = pd.date_range('2024-01-01', '2024-12-31', freq='W')  # Semanal

for data in datas_analise:
    for produto in produtos_mercado:
        # Preços dos concorrentes com variação semanal
        precos_concorrentes = {}
        for conc in concorrentes:
            variacao_semanal = np.random.normal(1, 0.05)  # 5% de variação
            preco_conc = produto['preco_base'] * variacao_semanal
            precos_concorrentes[conc] = round(preco_conc, 2)
        
        # Calculando posição no mercado
        todos_precos = list(precos_concorrentes.values())
        preco_medio_mercado = np.mean(todos_precos)
        preco_min_mercado = min(todos_precos)
        preco_max_mercado = max(todos_precos)
        
        # Nosso preço inicial (base)
        nosso_preco = produto['preco_base']
        
        # Calculando demanda baseada na elasticidade-preço
        # Q = Q0 * (P/P0)^elasticidade
        razao_preco = nosso_preco / preco_medio_mercado
        fator_elasticidade = razao_preco ** produto['elasticidade']
        
        # Demanda base ajustada por sazonalidade
        demanda_base = 100  # unidades base
        if data.month in [4, 5, 6, 7, 8, 9]:  # Alta temporada construção
            demanda_base *= 1.4
        
        demanda_estimada = demanda_base * fator_elasticidade * np.random.normal(1, 0.15)
        demanda_estimada = max(5, demanda_estimada)
        
        # Calculando métricas financeiras
        receita = nosso_preco * demanda_estimada
        custo_total = produto['custo'] * demanda_estimada
        lucro = receita - custo_total
        margem_percentual = ((nosso_preco - produto['custo']) / nosso_preco) * 100
        
        # Market share aproximado baseado na posição de preço
        if nosso_preco <= preco_min_mercado:
            market_share = 0.35  # Líder em preço
        elif nosso_preco <= preco_medio_mercado:
            market_share = 0.25  # Competitivo
        elif nosso_preco <= preco_max_mercado:
            market_share = 0.15  # Premium
        else:
            market_share = 0.08  # Muito caro
        
        market_share *= np.random.normal(1, 0.2)  # Adicionando variação
        market_share = max(0.02, min(0.50, market_share))
        
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
            'margem_percentual': round(margem_percentual, 1),
            'market_share': round(market_share, 3),
            'elasticidade': produto['elasticidade'],
            **{f'preco_{conc.lower().replace(" ", "_")}': precos_concorrentes[conc] 
               for conc in concorrentes}
        })

df_precificacao = pd.DataFrame(precificacao_data)

print("Análise de Precificação Dinâmica:")
print("-" * 40)

# Otimização de preços usando análise de elasticidade
def otimizar_precos_elasticidade(df):
    """Calcula preços ótimos baseado em elasticidade e margem desejada"""
    
    resultados_otimizacao = []
    
    for produto in df['produto'].unique():
        dados_produto = df[df['produto'] == produto].copy()
        
        # Calculando correlação entre preço relativo e demanda
        dados_produto['preco_relativo'] = (dados_produto['nosso_preco'] / 
                                          dados_produto['preco_medio_mercado'])
        
        # Regressão simples para estimar elasticidade real
        if len(dados_produto) > 10:
            correlacao = dados_produto[['preco_relativo', 'demanda_estimada']].corr().iloc[0,1]
            elasticidade_real = correlacao * -1  # Convertendo para elasticidade
        else:
            elasticidade_real = dados_produto['elasticidade'].iloc[0]
        
        # Dados mais recentes (última semana)
        dados_recentes = dados_produto.iloc[-1]
        
        # Testando diferentes cenários de preço
        cenarios_preco = []
        preco_atual = dados_recentes['nosso_preco']
        custo_unit = dados_recentes['custo_total'] / dados_recentes['demanda_estimada']
        
        for ajuste in [-0.10, -0.05, 0, 0.05, 0.10, 0.15]:  # -10% a +15%
            novo_preco = preco_atual * (1 + ajuste)
            
            # Não vender abaixo do custo + margem mínima
            if novo_preco < custo_unit * 1.10:
                continue
                
            # Estimando nova demanda
            fator_preco = (novo_preco / dados_recentes['preco_medio_mercado'])
            nova_demanda = 100 * (fator_preco ** elasticidade_real)
            
            nova_receita = novo_preco * nova_demanda
            novo_custo = custo_unit * nova_demanda
            novo_lucro = nova_receita - novo_custo
            nova_margem = ((novo_preco - custo_unit) / novo_preco) * 100
            
            cenarios_preco.append({
                'produto': produto,
                'ajuste_percentual': ajuste * 100,
                'preco_proposto': round(novo_preco, 2),
                'demanda_estimada': round(nova_demanda, 1),
                'receita_estimada': round(nova_receita, 2),
                'lucro_estimado': round(novo_lucro, 2),
                'margem_estimada': round(nova_margem, 1),
                'preco_atual': preco_atual,
                'lucro_atual': dados_recentes['lucro']
            })
        
        # Encontrando o cenário que maximiza lucro
        if cenarios_preco:
            melhor_cenario = max(cenarios_preco, key=lambda x: x['lucro_estimado'])
            melhor_cenario['melhoria_lucro'] = ((melhor_cenario['lucro_estimado'] - 
                                               melhor_cenario['lucro_atual']) / 
                                              melhor_cenario['lucro_atual'] * 100)
            resultados_otimizacao.append(melhor_cenario)
    
    return pd.DataFrame(resultados_otimizacao)

otimizacao_precos = otimizar_precos_elasticidade(df_precificacao)

print("Recomendações de Otimização de Preços:")
for _, produto in otimizacao_precos.iterrows():
    status = "↑" if produto['ajuste_percentual'] > 0 else "↓" if produto['ajuste_percentual'] < 0 else "→"
    print(f"{produto['produto'][:20]:20} | {status} {produto['ajuste_percentual']:+5.1f}% | "
          f"R${produto['preco_atual']:6.2f} → R${produto['preco_proposto']:6.2f} | "
          f"Lucro: {produto['melhoria_lucro']:+5.1f}%")

# Análise de posicionamento competitivo
print("\nAnálise de Posicionamento vs Concorrência:")
posicionamento = df_precificacao.groupby('produto').agg({
    'nosso_preco': 'mean',
    'preco_medio_mercado': 'mean',
    'market_share': 'mean',
    'margem_percentual': 'mean'
}).round(2)

posicionamento['posicao_relativa'] = ((posicionamento['nosso_preco'] / 
                                      posicionamento['preco_medio_mercado']) - 1) * 100

for produto in posicionamento.index:
    dados = posicionamento.loc[produto]
    posicao = "PREMIUM" if dados['posicao_relativa'] > 5 else "COMPETITIVO" if dados['posicao_relativa'] > -5 else "AGRESSIVO"
    print(f"{produto[:20]:20} | {posicao:11} | Share: {dados['market_share']:5.1%} | "
          f"Margem: {dados['margem_percentual']:4.1f}% | "
          f"Dif. Mercado: {dados['posicao_relativa']:+5.1f}%")

print("\n" + "=" * 70)
print("NOVOS INSIGHTS ESPECÍFICOS GERADOS:")
print("=" * 70)
print("1. LAYOUT: Análise espacial de rentabilidade por m² com recomendações de realocação")
print("2. PERDAS: Sistema multidimensional de tracking com identificação de padrões anômalos")  
print("3. PREÇOS: Otimização dinâmica baseada em elasticidade real e posicionamento competitivo")
print("=" * 70)