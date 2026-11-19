# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import os
from google.colab import drive

drive.mount('/content/drive')
df = pd.read_csv("/content/drive/MyDrive/resultados_rede_neural_amostragem/resultados_rede_neural_amostragem.csv")

#plotar graficos das metricas de cada modelo
def plotar_metricas_amostragem():
    modelos = df["Modelo"].unique()
    for modelo in modelos:
        dados = df[df["Modelo"] == modelo]
        dados = dados.sort_values("Amostragem")
        plt.figure(figsize=(9,5))
        plt.plot(dados["Amostragem"], dados["Acurácia"],
                marker='o', linewidth=2, label="Acurácia")
        plt.plot(dados["Amostragem"], dados["Precisão"],
                marker='o', linewidth=2, label="Precisão")
        plt.plot(dados["Amostragem"], dados["Recall (Sensibilidade)"],
                marker='o', linewidth=2, label="Sensibilidade")
        plt.plot(dados["Amostragem"], dados["F1-Score"],
                marker='o', linewidth=2, label="F1-Score")
        plt.plot(dados["Amostragem"], dados["AUC"],
                marker='o', linewidth=2, label="AUC")

        plt.xlabel("Número de Amostragens")
        plt.ylabel("Valor")
        plt.ylim(0.40, 1.00)
        plt.grid(True)
        plt.legend()
        plt.show()

# Plotar gráficos de barras: média das métricas de cada modelo
def plotar_medias_metrica_por_modelo():

    media_metricas = (
        df.groupby("Modelo")[
            ["Acurácia", "Precisão", "Recall (Sensibilidade)", "F1-Score", "AUC"]
        ]
        .mean()
    )
    cores = ["royalblue", "darkorange", "forestgreen", "crimson", "purple"]
    nomes_metricas = [
        "Acurácia",
        "Precisão",
        "Sensibilidade",
        "F1-Score",
        "AUC"
    ]
    for modelo in media_metricas.index:
        metricas = media_metricas.loc[modelo]
        fig, ax = plt.subplots(figsize=(8, 5))
        barras = ax.bar(
            nomes_metricas,
            metricas.values,
            color=cores,
            width=0.65
        )

        ax.set_ylabel("Valor médio")
        ax.set_ylim(0.40, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        for barra in barras:
          altura = barra.get_height()

          ax.text(
              barra.get_x() + barra.get_width() / 2,
              0.5,
              f"{altura:.3f}",
              ha="center",
              va="bottom",
              color="white",
              fontweight="bold",
              fontsize=9
        )

        plt.tight_layout()
        plt.show()

# Plotar gráfico (barras): tempo de execução por modelo

def plotar_tempo_execucao_por_modelo():

    df["Tempo_Min"] = df["Tempo de Execução"] / 60

    tempo_medio = (
        df.groupby("Modelo")["Tempo_Min"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    barras = ax.bar(
        tempo_medio["Modelo"],
        tempo_medio["Tempo_Min"],
        color=["royalblue", "darkorange", "forestgreen"],
        width=0.65
    )

    ax.set_xlabel("Modelo")
    ax.set_ylabel("Tempo médio (min)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for barra in barras:
        altura = barra.get_height()

        ax.text(
            barra.get_x() + barra.get_width() / 2,
            10,
            f"{altura:.3f}",
            ha="center",
            va="bottom",
            color="white",
            fontweight="bold",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

# Plotar o Score médio dos modelos (comparativo)
# Gráfico de barras

def plotar_score_medio_por_modelo():

    df["Score"] = (
        df["Acurácia"] +
        df["Precisão"] +
        df["Recall (Sensibilidade)"] +
        df["F1-Score"]
    ) / 4

    # Média do Score por modelo
    media_modelo = (
        df.groupby("Modelo")["Score"]
          .mean()
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    barras = ax.bar(
        media_modelo["Modelo"],
        media_modelo["Score"],
        color=["Maroon", "Darkblue", "Olive"],
        width=0.65
    )
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Score médio")
    ax.set_ylim(0.70, 1.00)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


    for i, valor in enumerate(media_modelo["Score"]):
        ax.text(
            i,
            0.75,
            f"{valor:.3f}",
            ha="center",
            va="bottom",
            color="white",
            fontweight="bold",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()




# plotar o score médio (grafico de barras) por amostragem
def plotar_score_medio_por_amostragem():
    score_modelo_amost = (
        df.groupby(["Amostragem", "Modelo"])["Score"]
          .mean()
          .reset_index()
    )
    plt.figure(figsize=(10,6))
    for modelo in score_modelo_amost["Modelo"].unique():
        dados = score_modelo_amost[
            score_modelo_amost["Modelo"] == modelo
        ].sort_values("Amostragem")

        plt.plot(
            dados["Amostragem"],
            dados["Score"],
            marker="o",
            linewidth=2,
            label=modelo
        )

    plt.xlabel("Número de Amostragem")
    plt.ylabel("Score médio")
    plt.xticks(sorted(df["Amostragem"].unique()))
    plt.ylim(0.70,1.00)
    plt.grid(True)
    plt.legend()
    plt.show()

"""Chamada de Funções - Análise Por modelo"""

plotar_metricas_amostragem()
plotar_medias_metrica_por_modelo()

"""Chamada de Funções - Análise Comparativa"""

plotar_tempo_execucao_por_modelo()
plotar_score_medio_por_modelo()
plotar_score_medio_por_amostragem()