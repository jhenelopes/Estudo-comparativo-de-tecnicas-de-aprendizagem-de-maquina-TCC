# -*- coding: utf-8 -*-

import tensorflow as tf
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

from tensorflow import keras
from keras import layers, models
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from datetime import datetime
from google.colab import drive
drive.mount('/content/drive')

#Aquisição das Imagens | Importação do Dataset

path = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
DATA_PATH = Path(path) / "TB_Chest_Radiography_Database"

#Parâmetros Globais

IMG_SIZE = 128
BATCH_SIZE = 10
EPOCHS = 20
SEED = 123

#Aqui os resultados (métricas e imagens) foram salvos no google drive 
#por ter sido utilizado o colab originalmente

#caminho_pasta = "/content/drive/MyDrive/resultados_rede_neural_epocas" #materiais e métodos
caminho_pasta = "/content/drive/MyDrive/resultados_rede_neural_amostragem" #experimentos e resultados
pasta_imagem = os.path.join(caminho_pasta, "imagens")
arquivo_csv = os.path.join(caminho_pasta, "resultados_rede_neural_amostragem.csv")

os.makedirs(caminho_pasta, exist_ok=True)
os.makedirs(pasta_imagem, exist_ok=True)

#Dividir conjuntos para Treinamento e Validação

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

#Pré Processamento

data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.1),
])

normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (data_augmentation(normalization_layer(x)), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

#Otimização dos modelos | Cache

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Construção do Modelo - Transfer Learning

def get_model(model_name):
    #configuração de pesos e parametros
    if model_name == "VGG16":
        base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif model_name == "ResNet50":
        base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif model_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    else:
        raise ValueError("Modelo não suportado")

    # Congela pesos do modelo base
    for layer in base_model.layers:
        layer.trainable = False

    #Constrói a própria camada de saída
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)

    return model

#Compilação com Otimizador Adam

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

#Treinamento

def train_model(model, epochs):
    tempo_exe = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    tempo_exe = time.perf_counter() - tempo_exe
    return history, tempo_exe

#Execução Mobile Net
modelMobileNet = get_model("MobileNetV2")
modelMobileNet = compile_model(modelMobileNet)
historyMobileNet, tempo_MobileNet = train_model(modelMobileNet,  EPOCHS)

#Execução ResNet50
modelResNet50 = get_model("ResNet50")
modelResNet50 = compile_model(modelResNet50)
historyResNet50, tempo_ResNet = train_model(modelResNet50,  EPOCHS)

#Execução VGG16
modelVGG16 = get_model("VGG16")
modelVGG16 = compile_model(modelVGG16)
historyVGG16, tempo_VGG = train_model(modelVGG16,  EPOCHS)

#Etapa de Validação
lossMobile, accMobile = modelMobileNet.evaluate(val_ds)
lossRes, accRes = modelResNet50.evaluate(val_ds)
lossVGG, accVGG = modelVGG16.evaluate(val_ds)

#Seção de Avaliação dos Modelos

#Função para obter Predições
def obter_predicoes(model, dataset):
    y_true = []
    y_pred = []
    y_score = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)

        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        y_score.extend(preds[:, 1])

    return np.array(y_true), np.array(y_pred), np.array(y_score)

#Função para calcular as Métricas Básicas dos Modelos

def calcular_metricas(y_true, y_pred, y_score, nome_modelo, temp_exec):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "Modelo": nome_modelo,
        "Acurácia": accuracy_score(y_true, y_pred),
        "Precisão": precision_score(y_true, y_pred),
        "Recall (Sensibilidade)": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "y_true":y_true,
        "y_pred":y_pred,
        "Tempo de Execução": temp_exec

    }

#Função - Salvar Resultados

def salvar_resultados(resultados):
    df = pd.DataFrame(resultados)

    data_execucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(arquivo_csv):
        df_existente = pd.read_csv(arquivo_csv)
        amostragem = (len(df_existente) // len(df)) + 1
    else:
        amostragem = 1

    df.insert(0, "Amostragem", amostragem)
    df.insert(1, "Data_Execucao", data_execucao)
    df.insert(2, "Quant_Epocas", EPOCHS)

    if os.path.exists(arquivo_csv):
        df.to_csv(
            arquivo_csv,
            mode='a',
            header=False,
            index=False
        )
    else:
        df.to_csv(
            arquivo_csv,
            mode='w',
            header=True,
            index=False
        )

    # Cria uma pasta específica por amostragem e quantidade de épocas
    pasta_amostragem = os.path.join(
        pasta_imagem,
        f"Amostragem_{amostragem:03d}_Epocas_{EPOCHS:03d}"
    )

    os.makedirs(pasta_amostragem, exist_ok=True)

    return df, amostragem, pasta_amostragem

#Função - Matriz de Confusão

def plotar_matrizes_confusao(modelos, nomes, dataset, pasta_imagens):

    for model, nome in zip(modelos, nomes):

        y_true, y_pred, _ = obter_predicoes(model, dataset)

        cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm
        )

        disp.plot()

        plt.title(f"Matriz de Confusão - {nome}")

        caminho = os.path.join(
            pasta_imagens,
            f"matriz_de_confusao_{nome}.png"
        )

        plt.savefig(
            caminho,
            dpi=300,
            bbox_inches='tight'
        )

        plt.show()
        plt.close()

#Função - Análise Comparativa

def plotar_roc_comparativo(modelos, nomes, dataset, pasta_imagens):
    plt.figure()

    for model, nome in zip(modelos, nomes):
        y_true, _, y_score = obter_predicoes(model, dataset)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{nome} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("Falso Positivo")
    plt.ylabel("Verdadeiro Positivo")
    plt.title("Curva ROC - Comparação entre Modelos")
    plt.legend(title="Modelos", loc="lower right")
    plt.grid(alpha=0.3)

    caminho = os.path.join(
        pasta_imagens,
        "ROC_Comparativo.png"
    )

    plt.savefig(
        caminho,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close()

#Analise Roc - Individual

def plotar_roc_individual(modelos, nomes, dataset, pasta_imagens):

    for model, nome in zip(modelos, nomes):
        y_true, _, y_score = obter_predicoes(model, dataset)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.4f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle='--'
        )

        plt.xlabel("Falso Positivo")
        plt.ylabel("Verdadeiro Positivo")
        plt.title(f"Curva ROC - {nome}")

        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        caminho = os.path.join(
            pasta_imagens,
            f"ROC_{nome}.png"
        )

        plt.savefig(
            caminho,
            dpi=300,
            bbox_inches='tight'
        )

        plt.show()
        plt.close()

#Função - Comparação Acurácia por Épocas


def plotar_acuracia_por_epoca():
    plt.figure(figsize=(8,5))

    plt.plot(historyMobileNet.history['accuracy'], marker='o', label='MobileNetV2')
    plt.plot(historyResNet50.history['accuracy'], marker='o', label='ResNet50')
    plt.plot(historyVGG16.history['accuracy'], marker='o', label='VGG16')

    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.title('Comparação da Acurácia de Treinamento')
    plt.legend()
    plt.grid(True)

    caminho = os.path.join(
        pasta_imagens,
        "acuracia_por_epoca.png"
    )

    plt.savefig(
        caminho,
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()
    plt.close()

#Executa as funções | Gera os Gráficos e Tabelas

resultados = []

y_true, y_pred, y_score = obter_predicoes(modelMobileNet, val_ds)
resultados.append(calcular_metricas(y_true, y_pred, y_score, "MobileNetV2", tempo_MobileNet))

y_true, y_pred, y_score = obter_predicoes(modelResNet50, val_ds)
resultados.append(calcular_metricas(y_true, y_pred, y_score, "ResNet50", tempo_ResNet))

y_true, y_pred, y_score = obter_predicoes(modelVGG16, val_ds)
resultados.append(calcular_metricas(y_true, y_pred, y_score, "VGG16", tempo_VGG))

df, amostragem, pasta_imagens = salvar_resultados(resultados)

plotar_matrizes_confusao(
    [modelMobileNet, modelResNet50, modelVGG16],
    ["MobileNetV2", "ResNet50", "VGG16"],
    val_ds,
    pasta_imagens
)


plotar_roc_comparativo(

   [modelMobileNet, modelResNet50, modelVGG16],
   ["MobileNetV2", "ResNet50", "VGG16"],
    val_ds, pasta_imagens
)

plotar_roc_individual(
   [modelMobileNet, modelResNet50, modelVGG16],
   ["MobileNetV2", "ResNet50", "VGG16"],
    val_ds, pasta_imagens
)

plotar_acuracia_por_epoca()