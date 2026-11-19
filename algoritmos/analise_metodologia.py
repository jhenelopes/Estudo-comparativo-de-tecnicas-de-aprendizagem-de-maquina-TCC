# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import os
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv("/content/drive/MyDrive/resultados_rede_neural_epocas/resultados_rede_neural.csv")

# Calcula o score de cada treinamento
def score_medio():
    df["Score"] = (
        df["Acurácia"] +
        df["Precisão"] +
        df["Recall (Sensibilidade)"] +
        df["F1-Score"]
    ) / 4

    # Media do Score dos três modelos
    score_epoca = (
        df.groupby("Quant_Epocas")["Score"]
          .mean()
          .reset_index()
    )

    melhor = score_epoca.loc[score_epoca["Score"].idxmax()]
    return melhor, score_epoca

#plota a figura score medio dos modelos por época
def plotar_score_medio_epoca(score_epoca, melhor):
    plt.figure(figsize=(8,5))

    plt.plot(
        score_epoca["Quant_Epocas"],
        score_epoca["Score"],
        marker='o',
        linewidth=2,
        label="Score Médio"
    )

    # marca a melhor época
    plt.axvline(
        melhor["Quant_Epocas"],
        linestyle='--',
        linewidth=2,
        label=f'Melhor época ({int(melhor["Quant_Epocas"])})'
    )

    plt.title("Score médio dos modelos por época")
    plt.xlabel("Número de épocas")
    plt.ylabel("Score")

    plt.xticks(score_epoca["Quant_Epocas"])   # mostra todas as épocas existentes
    plt.ylim(0.70,1.00)

    plt.grid(True)
    plt.legend()
    plt.show()

#plotar gráfico - score por modelo e evolução por época
def score_modelo_epoca():
    score_modelo_epoca = (
      df.groupby(["Quant_Epocas", "Modelo"])["Score"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10,6))

    for modelo in score_modelo_epoca["Modelo"].unique():
        dados = score_modelo_epoca[
            score_modelo_epoca["Modelo"] == modelo
        ].sort_values("Quant_Epocas")
        plt.plot(
            dados["Quant_Epocas"],
            dados["Score"],
            marker="o",
            linewidth=2,
            label=modelo
        )

    plt.title("Score médio por época e por modelo")
    plt.xlabel("Número de épocas")
    plt.ylabel("Score médio")

    plt.xticks(sorted(df["Quant_Epocas"].unique()))
    plt.ylim(0.55,1.00)
    plt.grid(True)
    plt.legend()
    plt.show()


#Chamada Funções
melhor, score_epoca = score_medio()
plotar_score_medio_epoca(score_epoca, melhor)
score_modelo_epoca()