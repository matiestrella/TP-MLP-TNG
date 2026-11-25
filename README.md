# TP2 - Clasificación de texto con un MLP

Clasificación de los 20 Newsgroups con un perceptrón multicapa (MLP) en PyTorch, partiendo del TP1 de regresión softmax.

## Contenido

`02_MLP_20Newsgroups.ipynb` tiene todo el TP:

- Carga de datos, preprocesamiento (stopwords, stemming, lematización) y TF-IDF.
- Clase `MLP` con cantidad de capas, neuronas y activación configurables.
- Búsqueda de hiperparámetros con Optuna (preprocesamiento, arquitectura, activación, optimizador Adam/RMSprop/SGD y sus parámetros) con pruning.
- Entrenamiento del mejor modelo y evaluación en test.
- Análisis de los trials: optimizadores, activaciones, tiempos, pruning y repetición con distintas semillas.
- Experimento de overfitting con redes de distinto tamaño.
- Comparación con la regresión softmax del TP1.

## Cómo correrlo

La notebook está pensada para Google Colab con GPU (Entorno de ejecución → Cambiar tipo de entorno → T4 GPU). Se corre entera de arriba hacia abajo. La primera celda instala lo que no viene en Colab.

Para correrla en otro lado:

```
pip install -r requirements.txt
```
