# Comparação de K-Means Sequencial e Paralelo em C com OpenMP e MPI

Este repositório implementa um algoritmo K-Means aplicado a uma base de dados de coordenadas de test pads, originalmente extraídas de imagens de placas de circuito impresso (PCB). O objetivo principal é realizar uma comparação entre as versões do código sequencial e paralelo com OpenMP e MPI.

O algoritmo realiza clustering de coordenadas e identifica regiões de pontos que possivelmente representam test pads cinzas. Esse tipo de análise pode ser utilizado para:

- **Classificação**: Identificação de test pads cinzas.
- **Detecção de Anomalias**: Identificação de test pads falsos, que não correspondem às características esperadas.
- **Agrupamento**: Descoberta de clusters de test pads cinzas.

## Sobre o Dataset

O conjunto de dados está em formato CSV e contém as seguintes colunas:

- **X e Y**: Coordenadas dos pixels nas imagens, representando a posição dos test pads.
- **R, G, B**: Valores de cor dos pixels, normalizados entre 0 e 255, para determinar a coloração.
- **Grey**: Indica pixels aproximadamente cinzas.

Esse dataset foi originalmente utilizado para identificar e agrupar uma grande quantidade de test pads (mais de 100 clusters) em uma abordagem de descoberta em duas etapas. O dataset pode ser encontrado aqui: [Printed Circuit Board Processed Image Dataset](https://archive.ics.uci.edu/dataset/990/printed+circuit+board+processed+image)

## Resultados obtidos na implenetação paralela do algoritmo
O algoritmo K-Means foi implementado em versões paralelizadas em uma abordagem utilizando apenas OpenMP e em uma abordagem híbrida utilizando OpenMP e MPI. Foram realizados testes em um ambiente em servidor Linux, com processador Intel de 4 núcleos e GPU Nvidia GT 1030 com 384 núcleos.

### Versão Sequencial do Algoritmo K-means
- **Tempo sequencial**: 161.055 segundos

### Versão OpenMP do Algoritmo K-means

**Resultados:**
- **1 thread**
  - Tempo: 170.659 segundos
  - Speedup: 0.94

- **2 threads**
  - Tempo: 89.955 segundos
  - Speedup: 1.79

- **4 threads**
  - Tempo: 46.946 segundos
  - Speedup: 3.43

- **8 threads**
  - Tempo: 49.461 segundos
  - Speedup: 3.25

### Versão Híbrida (MPI e OpenMP) do Algoritmo K-means

**Resultados:**
- **1 processo, 4 threads**
  - Tempo: 52.489 segundos
  - Speedup: 3.06

- **2 processos, 2 threads**
  - Tempo: 45.176 segundos
  - Speedup: 3.56

- **4 processos, sem threads**
  - Tempo: 44.293 segundos
  - Speedup: 3.63

## Uso
Dê git clone.

Em seguida, rode bash run.sh ou:

chmod +x run.sh

./run.sh



