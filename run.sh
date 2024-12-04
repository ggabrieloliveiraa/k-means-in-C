#!/bin/bash

# Caminho dos arquivos e parâmetros
DATA_FILE="./circuito.csv"
N=723552        # Número de instâncias
M=5             # Número de features
K=20            # Número de clusters
SEQUENTIAL_OUTPUT="src/results-sequencial.txt"
OPENMP_OUTPUT="src/results-openmp.txt"
OMP_MPI_OUTPUT="src/results-omp-mpi.txt"
CUDA_OUTPUT="src/results-cuda.txt"  # Novo arquivo de saída para CUDA
RESULTS_FILE="src/results.txt"

# Apagando o arquivo de resultados no início
> $RESULTS_FILE
> $OMP_MPI_OUTPUT
> $OPENMP_OUTPUT
> $SEQUENTIAL_OUTPUT
> $CUDA_OUTPUT  # Limpa o arquivo de saída da CUDA

# Exibindo as informações dos parâmetros de entrada
echo "Executando o programa K-means com os seguintes parâmetros:" | tee -a $RESULTS_FILE
echo " - Número de instâncias (n): $N" | tee -a $RESULTS_FILE
echo " - Número de features (m): $M" | tee -a $RESULTS_FILE
echo " - Número de clusters (k): $K" | tee -a $RESULTS_FILE
echo " - Arquivo de dados: $DATA_FILE" | tee -a $RESULTS_FILE
echo "Os tempos serão calculados com o programa 'time' do Linux." | tee -a $RESULTS_FILE
echo "Usaremos o valor 'real' do resultado do 'time'" | tee -a $RESULTS_FILE
echo

# Função para converter tempo do formato real "XmY.YYYs" para segundos
convert_to_seconds() {
    local time_str=$1
    # Extraímos o tempo da string de entrada, que deve conter algo como "0m4.081s"
    # Usamos grep para pegar apenas a linha com "real" e então extraímos o tempo.
    local real_time=$(echo "$time_str" | grep -oP 'real\s+\K\d+m\d+\.\d+s')

    # Agora convertemos o tempo "XmYs" para segundos
    echo "$real_time" | sed -E 's/([0-9]+)m([0-9.]+)s/\1 \2/' | awk '{print $1 * 60 + $2}'
}

# Função para calcular o speedup
calc_speedup() {
    local seq_time=$1
    local par_time=$2
    # Verificando se o tempo paralelo é zero para evitar erro de "divide by zero"
    if (( $(echo "$par_time == 0" | bc -l) )); then
        echo "Erro: Tempo paralelo não pode ser zero para o cálculo do speedup."
        return 1
    fi
    echo "scale=2; $seq_time / $par_time" | bc
}

# Compilação dos códigos
echo "Compilando o programa K-means sequencial..." | tee -a $RESULTS_FILE
gcc src/kmeans-sequencial.c -o src/kmeans-sequencial -lm
if [ $? -ne 0 ]; then
    echo "Erro ao compilar kmeans-sequencial.c"
    exit 1
fi
echo "Compilação do K-means sequencial concluída com sucesso!" | tee -a $RESULTS_FILE

echo "Compilando o programa K-means com OpenMP..." | tee -a $RESULTS_FILE
gcc -fopenmp src/kmeans-openmp.c -o src/kmeans-openmp -lm
if [ $? -ne 0 ]; then
    echo "Erro ao compilar kmeans-openmp.c"
    exit 1
fi
echo "Compilação do K-means com OpenMP concluída com sucesso!" | tee -a $RESULTS_FILE

echo "Compilando o programa K-means com OpenMP e MPI..." | tee -a $RESULTS_FILE
mpicc -fopenmp src/kmeans-omp-mpi.c -o src/kmeans-omp-mpi -lm
if [ $? -ne 0 ]; then
    echo "Erro ao compilar kmeans-omp-mpi.c"
    exit 1
fi
echo "Compilação do K-means com OpenMP e MPI concluída com sucesso!" | tee -a $RESULTS_FILE

echo "Compilando o programa K-means com CUDA..." | tee -a $RESULTS_FILE
# Substitua 'sm_60' pela arquitetura da sua GPU, por exemplo, 'sm_75' para Turing
nvcc -O3 src/kmeans-cuda.cu -o src/kmeans-cuda -lm
if [ $? -ne 0 ]; then
    echo "Erro ao compilar kmeans-cuda.cu"
    exit 1
fi
echo "Compilação do K-means com CUDA concluída com sucesso!" | tee -a $RESULTS_FILE

# Executando a versão sequencial e exibindo o tempo de execução
echo -e "\nExecutando o K-means sequencial..." | tee -a $RESULTS_FILE
SEQ_TIME=$( { time ./src/kmeans-sequencial "$DATA_FILE" "$N" "$M" "$K" "$SEQUENTIAL_OUTPUT"; } 2>&1 | tee >(grep "real" | awk '{print $2}') )
SEQ_TIME_SEC=$(convert_to_seconds "$SEQ_TIME")
echo "Tempo sequencial: $SEQ_TIME_SEC segundos" | tee -a $RESULTS_FILE

# Testando diferentes números de threads OpenMP (1, 2, 4, 8)
for threads in 1 2 4 8; do
    echo -e "\nExecutando o K-means com OpenMP usando $threads threads..." | tee -a $RESULTS_FILE
    export OMP_NUM_THREADS=$threads
    OPENMP_TIME=$( { time ./src/kmeans-openmp "$DATA_FILE" "$N" "$M" "$K" "$OPENMP_OUTPUT"; } 2>&1 | tee >(grep "real" | awk '{print $2}') )
    OPENMP_TIME_SEC=$(convert_to_seconds "$OPENMP_TIME")
    echo "Tempo OpenMP com $threads threads: $OPENMP_TIME_SEC segundos" | tee -a $RESULTS_FILE
    SPEEDUP=$(calc_speedup $SEQ_TIME_SEC $OPENMP_TIME_SEC)
    if [ $? -eq 0 ]; then
        echo "Speedup OpenMP com $threads threads: $SPEEDUP" | tee -a $RESULTS_FILE
    fi
done

# Testando a versão CUDA
echo -e "\nExecutando o K-means com CUDA..." | tee -a $RESULTS_FILE
CUDA_TIME=$( { time ./src/kmeans-cuda "$DATA_FILE" "$N" "$M" "$K" "$CUDA_OUTPUT"; } 2>&1 | tee >(grep "real" | awk '{print $2}') )
CUDA_TIME_SEC=$(convert_to_seconds "$CUDA_TIME")
echo "Tempo CUDA: $CUDA_TIME_SEC segundos" | tee -a $RESULTS_FILE
SPEEDUP_CUDA=$(calc_speedup $SEQ_TIME_SEC $CUDA_TIME_SEC)
if [ $? -eq 0 ]; then
    echo "Speedup CUDA: $SPEEDUP_CUDA" | tee -a $RESULTS_FILE
fi

# Testando as combinações de OpenMP e MPI
echo -e "\nExecutando o K-means com OpenMP e MPI..." | tee -a $RESULTS_FILE

# 1 processo com 4 threads
export OMP_NUM_THREADS=4
echo -e "\n"  # Linha em branco antes do tempo
MPI_1_PROC_4_THREADS_TIME=$( { time mpirun -np 1 ./src/kmeans-omp-mpi "$DATA_FILE" "$N" "$M" "$K" "$OMP_MPI_OUTPUT"; } 2>&1 | tee >(grep "real" | awk '{print $2}') )
MPI_1_PROC_4_THREADS_TIME_SEC=$(convert_to_seconds "$MPI_1_PROC_4_THREADS_TIME")
echo "Tempo OpenMP e MPI (1 processo, 4 threads): $MPI_1_PROC_4_THREADS_TIME_SEC segundos" | tee -a $RESULTS_FILE
SPEEDUP_1_PROC_4_THREADS=$(calc_speedup $SEQ_TIME_SEC $MPI_1_PROC_4_THREADS_TIME_SEC)
if [ $? -eq 0 ]; then
    echo "Speedup OpenMP e MPI (1 processo, 4 threads): $SPEEDUP_1_PROC_4_THREADS" | tee -a $RESULTS_FILE
fi

# 2 processos com 2 threads cada
export OMP_NUM_THREADS=2
echo -e "\n"  # Linha em branco antes do tempo
MPI_2_PROC_2_THREADS_TIME=$( { time mpirun -np 2 ./src/kmeans-omp-mpi "$DATA_FILE" "$N" "$M" "$K" "$OMP_MPI_OUTPUT"; } 2>&1 | tee >(grep "real" | awk '{print $2}') )
MPI_2_PROC_2_THREADS_TIME_SEC=$(convert_to_seconds "$MPI_2_PROC_2_THREADS_TIME")
echo "Tempo OpenMP e MPI (2 processos, 2 threads): $MPI_2_PROC_2_THREADS_TIME_SEC segundos" | tee -a $RESULTS_FILE
SPEEDUP_2_PROC_2_THREADS=$(calc_speedup $SEQ_TIME_SEC $MPI_2_PROC_2_THREADS_TIME_SEC)
if [ $? -eq 0 ]; then
    echo "Speedup OpenMP e MPI (2 processos, 2 threads): $SPEEDUP_2_PROC_2_THREADS" | tee -a $RESULTS_FILE
fi

# 4 processos sem threads
export OMP_NUM_THREADS=1
echo -e "\n"  # Linha em branco antes do tempo
MPI_4_PROC_NO_THREADS_TIME=$( { time mpirun -np 4 ./src/kmeans-omp-mpi "$DATA_FILE" "$N" "$M" "$K" "$OMP_MPI_OUTPUT"; } 2>&1 | tee >(grep "real" | awk '{print $2}') )
MPI_4_PROC_NO_THREADS_TIME_SEC=$(convert_to_seconds "$MPI_4_PROC_NO_THREADS_TIME")
echo "Tempo OpenMP e MPI (4 processos, sem threads): $MPI_4_PROC_NO_THREADS_TIME_SEC segundos" | tee -a $RESULTS_FILE
SPEEDUP_4_PROC_NO_THREADS=$(calc_speedup $SEQ_TIME_SEC $MPI_4_PROC_NO_THREADS_TIME_SEC)
if [ $? -eq 0 ]; then
    echo "Speedup OpenMP e MPI (4 processos, sem threads): $SPEEDUP_4_PROC_NO_THREADS" | tee -a $RESULTS_FILE
fi

echo -e "\nExecuções concluídas." | tee -a $RESULTS_FILE
echo "Resultados do K-means sequencial estão em $SEQUENTIAL_OUTPUT" | tee -a $RESULTS_FILE
echo "Resultados do K-means com OpenMP estão em $OPENMP_OUTPUT" | tee -a $RESULTS_FILE
echo "Resultados do K-means com CUDA estão em $CUDA_OUTPUT" | tee -a $RESULTS_FILE
echo "Resultados do K-means com OpenMP e MPI estão em $OMP_MPI_OUTPUT" | tee -a $RESULTS_FILE
echo "Todos os tempos e cálculos de speedup foram salvos em $RESULTS_FILE" | tee -a $RESULTS_FILE