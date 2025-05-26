"""
1. Como você ajudaria a Carajás a prever a demanda de cimento ou piso com Python?
Resposta sugerida:
"Em uma loja como a Carajás, prever a demanda de cimento, pisos ou tintas é essencial para manter o equilíbrio entre excesso e falta de estoque. 
Eu usaria pandas para processar o histórico de vendas, agregando por mês e por loja. Depois aplicaria modelos como regressão linear ou Prophet para prever a demanda futura. 
Isso ajudaria a comprar melhor e evitar rupturas."
"""

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("vendas_cimento.csv")

modelo = Prophet()
modelo.fit(df)

futuro = modelo.make_future_dataframe(periods=3, freq='M')
previsao = modelo.predict(futuro)

print(previsao[['ds', 'yhat']].tail(3))

modelo.plot(previsao)
plt.title("Previsao de Vendas de Cimento (3 meses)")

plt.xlabel("Data")
plt.ylabel("Unidades Vendidas")
plt.tight_layout()
plt.show()