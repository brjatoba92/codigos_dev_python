import pandas as pd
import numpy as np


def analisar_orcamentos_projetos():
    np.random.seed(42)
    # Dados de orçamentos e projetos
    projetos_data = []

    tipos_obra = ['Comercial', 'Reforma', 'Residencial', 'Industrial', 'Outros']
    status_projeto = ['Em orçamento', 'Aprovado', 'Em Execução', 'Concluido', 'Cancelado']

    for i in range(80):
        data_inicio = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        projetos_data.append({
            'projeto_id': f'PROJ{i+1:04d}',
            'cliente': f'Cliente Projeto {i+1}',
            'tipo_obra': np.random.choice(tipos_obra),
            'area_m2': np.random.uniform(100, 5000),
            'data_orcamento': data_inicio,
            'prazo_estimado_dias': np.random.randint(30, 180),
            'orcamento_inicial' : np.random.uniform(10000, 500000),
            'orcamento_aprovado' : np.random.uniform(10000, 500000),
            'gasto_atual': np.random.uniform(5000, 120000),
            'status_projeto': np.random.choice(status_projeto, p=[0.6, 0.2, 0.1, 0.075, 0.025]),
            'margem_esperada': np.random.uniform(0.1, 0.5)
        })
    df_projetos = pd.DataFrame(projetos_data)
    print(df_projetos.head(25))

    # Analises de orçamentos
    df_projetos['valor_m2'] = df_projetos['orcamento_aprovado'] / df_projetos['area_m2']
    df_projetos['variacao_orcamento'] = (
        (df_projetos['orcamento_aprovado'] - df_projetos['orcamento_inicial']) / df_projetos['orcamento_inicial'] *100
    ).round(2)
    df_projetos['perc_gasto'] = (df_projetos['gasto_atual'] / df_projetos['orcamento_aprovado']*100).round(1)
    df_projetos['margem_realizada'] = (
        (df_projetos['orcamento_aprovado'] - df_projetos['gasto_atual']) / df_projetos['orcamento_aprovado']
    ).round(3)

    # Analise por tipo de obra
    analise_tipo_obra = df_projetos.groupby('tipo_obra').agg({
        'projeto_id' : 'count',
        'orcamento_aprovado': 'mean',
        'valor_m2': 'mean',
        'margem_realizada': 'mean',
        'prazo_estimado_dias': 'mean'
    }).round(2)
    analise_tipo_obra.columns = ['qtd_projetos', 'orcamento_medio', 'valor_m2_medio', 'margem_media', 'prazo_medio']

    # Projetos com alerta (acima de 80% do orçamento)

    projetos_alerta = df_projetos[
        (df_projetos['perc_gasto'] > 80) & (df_projetos['status_projeto'].isin(['Aprovado', 'Em Execução']))
    ][['projeto_id', 'tipo_obra', 'orcamento_aprovado', 'gasto_atual', 'perc_gasto', 'status_projeto']]

    # Taxa de conversão de orçamentos

    conversao = df_projetos['status_projeto'].value_counts()
    taxa_aprovacao = (conversao.get('Aprovado', 0) +  conversao.get('Em Execução', 0) + 
                      conversao.get('Concluido', 0)) / len(df_projetos) * 100
    print("\n === ANALISE DE ORÇAMENTOS E PROJETOS ===")
    print(f"Taxa de aprovação de orçamento: {taxa_aprovacao:.1f}%")

    print("\nAnalise por tipo de obra")
    print(analise_tipo_obra)

    if len(projetos_alerta)>0:
        print(f"\n ⚠️  PROJETOS EM ALERTA: ({len(projetos_alerta)} projetos)")
        print(projetos_alerta)
    else:
        print("\n ✅ Nenhum projeto em situação crítica de orçamento")
    return df_projetos, analise_tipo_obra


analisar_orcamentos_projetos()