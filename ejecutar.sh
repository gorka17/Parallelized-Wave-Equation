#!/bin/bash

SEQUENTIAL_EXEC=./secuencial_vis
MPI_EXEC=./paral_mpi
OMP_EXEC=./paral_omp
CUDA_EXEC=./paral_cuda
SIZE=4000
AMPLITUDE=900.0
MAX_TIME=2000
GRID=50
OUTPUT="speedup_results.csv"


# Compilar programas
# Secuencial
echo "Compilando versión secuencial..."
gcc secuencial_vis.c -o secuencial_vis -lm
# OpenMP
echo "Compilando versión paralela OpenMP..."
gcc paral_omp.c -o paral_omp -fopenmp -lm
# OpenMPI
echo "Compilando versión paralela OpenMPI..."
mpicc -o paral_mpi paral_mpi.c -lm
# Cuda
echo "Compilando versión paralela Cuda..."
nvc paral_cuda.c -o paral_cuda

# Encabezado
echo "Método,Procesos/Hilos,Tiempo(s),Speedup,Eficiencia(%)" > $OUTPUT

# ==== 1. Secuencial ====
echo "Ejecutando versión secuencial..."
START=$(date +%s.%N)
$SEQUENTIAL_EXEC $SIZE $AMPLITUDE $MAX_TIME $GRID
END=$(date +%s.%N)
T1=$(echo "$END - $START" | bc)
echo "Secuencial,1,$T1,1.0,100" >> $OUTPUT

# ==== 2. MPI ====
for P in 1 2 4 8 16
do
    echo "Ejecutando versión MPI con $P procesos..."
    START=$(date +%s.%N)
    mpirun -np $P $MPI_EXEC $SIZE $AMPLITUDE $MAX_TIME $GRID
    END=$(date +%s.%N)
    TP=$(echo "$END - $START" | bc)
    SPEEDUP=$(echo "scale=2; $T1 / $TP" | bc)
    EFFICIENCY=$(echo "scale=2; ($SPEEDUP / $P) * 100" | bc)
    echo "MPI,$P,$TP,$SPEEDUP,$EFFICIENCY" >> $OUTPUT
done

# ==== 3. OpenMP ====
for T in 1 2 4 8 16
do
    echo "Ejecutando versión OpenMP con $T hilos..."
    export OMP_NUM_THREADS=$T
    START=$(date +%s.%N)
    $OMP_EXEC $SIZE $AMPLITUDE $MAX_TIME $GRID
    END=$(date +%s.%N)
    TT=$(echo "$END - $START" | bc)
    SPEEDUP=$(echo "scale=2; $T1 / $TT" | bc)
    EFFICIENCY=$(echo "scale=2; ($SPEEDUP / $T) * 100" | bc)
    echo "OpenMP,$T,$TT,$SPEEDUP,$EFFICIENCY" >> $OUTPUT
done

# ==== 4. Cuda ====
echo "Ejecutando versión cuda..."
START=$(date +%s.%N)
$CUDA_EXEC $SIZE $AMPLITUDE $MAX_TIME $GRID
END=$(date +%s.%N)
TT=$(echo "$END - $START" | bc)
SPEEDUP=$(echo "scale=2; $T1 / $TT" | bc)
EFFICIENCY=$(echo "scale=2; ($SPEEDUP / $T) * 100" | bc)
echo "Cuda,RTX3060,$TT,$SPEEDUP,$EFFICIENCY" >> $OUTPUT

echo "Benchmark completo. Resultados guardados en $OUTPUT"