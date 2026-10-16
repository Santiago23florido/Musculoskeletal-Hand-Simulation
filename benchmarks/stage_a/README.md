# Stage A — corrección aprendida del blanqueamiento

Implementación independiente de una corrección aprendida del blanqueamiento.
El [resumen publicado](../../docs/benchmarks/stage_a_p1_summary.json) conserva
los resultados y el criterio de salida de la evaluación P1.

La arquitectura aplica `W_b = (I + U(z_b)V(z_b)^T) W_0`, con seis bases, rango 3,
MLP de mezcla y estado causal. El límite certificado de norma espectral es 0,8.
Se mantienen las calibraciones y las funciones originales de detección, compuerta
IQR, centroides y actualización de separación de `adapt_decomp`, fijado a
`a8f53736e297962670276bc123c137050b7056b6`. No es una reproducción del código
no disponible de HarmonICA.

El modelo principal usa ángulo de muñeca disponible (**Stage A-K**). Entrena con
KL gaussiano y consistencia del contraste `log(cosh)`, sobre secuencias. Los
detalles numéricos se recogen en los módulos y resultados enlazados más abajo.

**Resultado ejecutado: NO-GO.** Cinco entrenamientos y P1 completo, 100 registros
y 768 instancias por método. RoA medio: Stage A 57,83%, Pareto 79,25%; diferencias
en rampas de 5/10 s y escalera: −10,55/−20,74/−28,62 puntos. El modelo no supera
el criterio de salida. Se conserva el resultado sin volver a seleccionar mirando
el test.

## Reproducir sin sobrescribir los experimentos conservados

Desde la raíz, en PowerShell; los datos completos de las semanas 1–2 deben existir:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/farina/setup.ps1 -Fdsi
uv pip install --python .venv-benchmarks/Scripts/python.exe --requirement benchmarks/stage_a/requirements.lock.txt
$env:STAGE_A_OUTPUT = 'outputs/stage_a_reproduction'
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 6 prepare
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 audit
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 train pilot_cov128 --epochs 3 --samples 128 --loss-mode delta --validation-whitening-weight 1 --validation-contrast-weight 0.01
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 6 train full_fifo_squared --epochs 8 --samples 2000 --loss-mode squared --validation-whitening-weight 1 --validation-contrast-weight 0.01
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 train balanced_gru --epochs 10 --samples 2000 --whitening-weight 0.0001 --contrast-weight 0.1
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 train balanced_mlp --epochs 10 --samples 2000 --encoder mlp --whitening-weight 0.0001 --contrast-weight 0.1
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 train state3_gru --epochs 8 --samples 2000 --state-dim 3
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 select pilot_cov128 full_fifo_squared balanced_gru balanced_mlp state3_gru
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.run_stage_a --threads 2 evaluate outputs/stage_a_reproduction/selected.pt --name p1_complete
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.statistics --threads 2
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.stage_a.report
Remove-Item Env:STAGE_A_OUTPUT
```

Los nombres de entrenamiento y la selección se protegen contra sobrescritura.
Para una nueva exploración usa otra carpeta de salida. La evaluación se puede
reanudar con el mismo nombre si coinciden plan, código y checkpoint; se reutilizan
solo registros terminados. El script no selecciona otro modelo mirando el test.

El entrenamiento se hace en dos rampas del pool P1 (`sub-01`, 30 dB, 40 y 5 s)
y selecciona en la tercera (10 s), exclusivamente por pérdida sin etiquetas.
La calibración supervisada de los primeros cinco segundos se conserva. La
selección común usa pesos 0,0001 para KL y 0,1 para contraste. Las descargas reales
posteriores a calibración solo se abren para evaluar predicciones ya generadas.
Los dos primeros entrenamientos seleccionaron época con sus pesos originales
1/0,01, reproducidos por las opciones `--validation-...` de arriba. La selección
final entre los cinco candidatos vuelve a puntuarlos con el objetivo común.

La evaluación completa equivale a cinco particiones por sujeto. Para repartir
el cálculo se puede ejecutar `evaluate ... --name p1_sub01 --subjects sub-01`
y lo mismo para los otros cuatro sujetos, conservando idéntico checkpoint,
código, métodos e hilos. Después, `merge p1_sub01 p1_sub02 p1_sub03 p1_sub04 p1_sub05`
genera `evaluations/p1_complete/`. La unión rechaza solapamientos y coberturas
incompletas. La ejecución registrada usó como máximo tres procesos y dos hilos
por proceso.

## Archivos y resultados

- `data.py`: preparación de señales y características, sin cargar descargas reales.
- `model.py`: encoder, corrección invertible y pérdidas diferenciables.
- `objectives.py`: constantes KL del FIFO y estadísticas de calibración, con Cholesky.
- `train.py`, `selection.py`: secuencias, checkpoints y selección antes del test.
- `inference.py`: detector/SV originales con el nuevo blanqueamiento y acoplamiento.
- `evaluate.py`, `merge.py`: P1, controles, métricas, pruebas pareadas y cobertura.
- `statistics.py`: modelo mixto `método × condición + (1 | sujeto/registro)`.
- `report.py`: figuras regenerables desde los registros calculados.

Por defecto los artefactos van a `outputs/stage_a/`, ignorado por Git. Cada
entrenamiento guarda protocolo, semilla, hashes, copia de código, pérdidas y pesos.
La selección guarda el hash del candidato. Cada evaluación conserva código,
predicciones y puntuaciones por unidad, fases y tiempos. El Pareto publicado se
recalcula y debe coincidir con sus RoA guardados a `1e-10`.

El criterio de salida exige **+5 puntos en las rampas de 5 y 10 s y ninguna pérdida
en escalera**, sobre las 768 instancias del P1 completo. Una evaluación parcial
no puede marcarlo como superado. La inferencia con `W_0 x` precalculado conserva
el protocolo offline: sus tiempos no acreditan latencia de un sistema en vivo.

## Verificación

```powershell
.\.venv-benchmarks\Scripts\python.exe -m pytest -q benchmarks/stage_a/test_stage_a.py benchmarks/farina/test_metrics.py benchmarks/farina/test_fdsi.py benchmarks/farina/test_download.py
.\.venv-windows\Scripts\ruff.exe check benchmarks/stage_a
```

Las pruebas verifican equivalencia y gradientes contra cálculo denso, Cholesky,
invertibilidad, causalidad, la postura de referencia, acoplamiento Woodbury,
contraste contra los autores y el rechazo de un `go` basado en datos parciales.
La auditoría sobre señales reales del benchmark reproduce bit a bit el comparador
fijo antes de entrenar.
