# Pareto-MC

Desarrollo en CPU de una corrección residual de Pareto, con memoria por registro,
estado ciego y decisiones de disparo realimentadas. No es un modelo aprobado:
las puertas de validación y los criterios finales son independientes.

## Protocolo previo

Rama desde `review/robust-mvdr`, PR #1 abierto. Entrenamiento en sub-01–03;
validación en sub-04; sub-05 reservado. **Cambio frente a Stage B declarado antes
de entrenar:** todas las condiciones y los cuatro SNR se permiten en desarrollo.
No se abre sub-05 para ajustar ni para diagnosticar. El diario registra todas
las decisiones y desviaciones en
[`pareto_mc_journal.md`](../../docs/benchmarks/pareto_mc_journal.md).

La Fase 0 intentó iniciar el backend oficial NeuroMotion de
[MUniverse](https://github.com/dfarinagroup/muniverse), revisión
`94fbfab253f037e3b3cbfdbc6aed85b781f0ea49`. Docker no tiene su motor Linux
activo y Ubuntu no dispone de Singularity/Apptainer. No se generaron registros
confirmatorios; el alcance es exploratorio. Se conservan los comandos, errores
y hashes del Dockerfile y lanzador en `phase0.json`.

## Fundamento y límites

La columna vertebral conserva las operaciones de
[`adapt_decomp` a8f5373](https://github.com/imendezguerra/adapt_decomp/tree/a8f53736e297962670276bc123c137050b7056b6),
con
[`optim_muniverse_fdsi_pareto.yaml`](https://github.com/imendezguerra/adapt_decomp/blob/a8f53736e297962670276bc123c137050b7056b6/configs/adapt_configs/optim_muniverse_fdsi_pareto.yaml):
paso fijo de blanqueo `W -= eta_wh (Rz-I) W`, con FIFO, shrinkage y clipping de
la referencia; corrección de coordenadas wh→sv; contraste `tanh(y)` solo en
disparos fiables; QR; y centroides EMA (0,95). Tasas 0,04038249 y 0,002422225.
No se confunde la fórmula abreviada con todas las salvaguardas del código.

Esto se interpreta como seguimiento autoentrenado de plantillas blanqueadas.
El compromiso entre seguimiento de deriva y ruido en filtros adaptativos
motiva pasos dependientes del estado; no garantiza un paso universalmente
óptimo. Véanse [Widrow et al., 1975](https://doi.org/10.1109/PROC.1975.10036)
y [Kalman, 1960](https://doi.org/10.1115/1.3662552).
En ramp5s se repite la postura aproximadamente ocho veces: la memoria local
busca reutilizar filtros aprendidos del propio registro. No se presupone que
el morph entre sujetos de Γ haya funcionado.

La auditoría
[`fifo_audit`](../fifo_audit/README.md) muestra el riesgo de pérdida brusca de
amplitud frente a centroides con estado. La mezcla se limita **después de QR**;
`beta=0` omite toda operación adicional sobre el filtro. Se guardan filtros
efectivos `v=W.T@w` y centroides, con EMA por bin y confianza combinada de tasa,
separación e IQR. La recuperación es una regresión kernel ponderada
([Nadaraya, 1964](https://doi.org/10.1137/1109020)). No se inserta el lote actual
antes de recuperarlo, ni se comparte memoria entre registros.

El éxito previo de
[`spike_likelihood`](../spike_likelihood/README.md) motiva aprender la decisión.
Realimentarla cambia las observaciones de entrenamiento: las rondas agregadas
siguen el principio de
[DAgger, Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html).
Esto es una motivación metodológica, no una garantía de sus cotas para EMG.

C1 conserva GRU(120,24), añade media y log-varianza y usa NLL gaussiana. C3
tiene 159 parámetros y salidas acotadas; los multiplicadores empiezan en uno
y beta en cero. C4 extiende el MLP 19→8→1 a 22→8→1 (193 parámetros), con
fuente, confianza y velocidad estimada. Su última capa empieza en cero:
sin entrenar, las decisiones y la adaptación reproducen Pareto.

## C0 medido

En tres registros reales de sub-01 (staircase/ramp10s/ramp5s, 30 dB): **100 % de
muestras iguales**, cero diferencias de RoA frente al bucle original y también
frente a las salidas publicadas. El protocolo mantuvo el nombre conservador
**Pareto-rerun**. Medianas de lote: 38,05/38,48/38,12 ms; p95: 46,70/47,21/46,55
ms. **No pasa el objetivo de 22 ms** con cuatro hilos en esta máquina.
Estos tres registros son una prueba de equivalencia de entrenamiento; sus
medias no son una evaluación externa de 80 registros.

La interfaz operacional recibe EMG crudo y un índice secuencial. El filtrado
es `sosfilt` con estado; la extensión usa solo historial. El filtrado de la
referencia también es causal: "offline" describe la preparación del registro,
no un `filtfilt`. La media global de `Data` se evita en la interfaz nueva.
Las salidas tienen hasta un lote de latencia (100 ms); se permite la calibración
inicial suministrada de cinco segundos. Ángulos y disparos verdaderos se abren
solo en funciones separadas de supervisión/evaluación protegidas por sujeto.

## Reproducir

Cada ejecución utiliza un directorio nuevo; no se sobreescriben protocolos,
predicciones selladas ni resultados. El primer intento L0 en
`outputs/pareto_mc/L0` falló por una firma de constructor; se conserva.

```powershell
$env:PARETO_MC_OUTPUT = 'outputs/pareto_mc/reproduction'
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.phase0
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.run identity --threads 4
.venv-stage-b-gpu/Scripts/python.exe -m benchmarks.pareto_mc.run train-observer --threads 4
$env:OPENBLAS_NUM_THREADS = '1'
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.run identity-head --threads 4
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.run l1 --threads 4
.venv-benchmarks/Scripts/python.exe -m pytest benchmarks/pareto_mc -q
.venv-windows/Scripts/python.exe -m ruff check benchmarks/pareto_mc
.venv-windows/Scripts/python.exe -m ruff format --check benchmarks/pareto_mc
```

El entorno de entrenamiento es `.venv-stage-b-gpu`, RTX 4070; la inferencia y
sus tiempos se miden con `.venv-benchmarks` en CPU. El constructor original
emite errores Intel MKL `SLASWP` al estimar estadísticas KL por lotes en Windows;
esas estadísticas no escalan los pasos con `lr_mode=fixed`. La coincidencia
exacta de los tres controles se comprueba pese a esos mensajes. No se modifica
la dependencia fijada para silenciarlos.

Los diagnósticos por nivel incluyen amplitud verdadera con `amplitude_metrics`,
centroides, aciertos de memoria, control frente a velocidad ciega, unidades
con recall <0,5 y distribución de tiempos. Las fases usan `angle_phases`.
Los ciclos son excursiones angulares >1° que alcanzan ≥35°, con calibración
excluida; no se usan intervalos temporales fijos para fabricarlos.

## Ejecución de desarrollo

C1 terminó: 30 épocas, semilla 0, checkpoint por NLL de sub-04 en época 13;
NLL −1,87788 y RMSE angular descriptivo de 3,2407°. Pesos en
`outputs/pareto_mc/development_v2/C1/seed0/observer.pt`, con su manifest original.
Solo el objetivo de entrenamiento/evaluación contiene ángulos medidos.

El primer L1 (`development_v1`) se conserva, sustituido antes de seleccionar por
RoA/Jaccard. `development_v2` reutiliza entradas y un checkpoint completado,
verificados en `input_reuse.json`. La caché de entrenamiento conserva exactamente
arrays y etiquetas por registro; no recalcula desfases ni descomprime en cada
época. Sus archivos y procedencia tienen hashes y controles de sujeto.

Se fija Torch a cuatro hilos como en el C0 exacto. Dos hilos producían siete
diferencias en un registro por diferencias numéricas de BLAS. Con cuatro,
**C4 sin entrenar vuelve a dar cero diferencias en los tres registros reales**,
incluso con el observador entrenado y con una sola ventana para las estadísticas
KL que no se usan con paso fijo. No se confunden los tiempos bajo contención
de estas pruebas con una medición aislada de latencia final.

L1 agrega las rondas 0–3, con tres semillas por ronda. El checkpoint se elige por
BCE de validación; después, 40 ensayos TPE ajustan umbral y fuerza residual por
Jaccard a desfases de calibración. Los tres mejores se repiten en tres semillas,
seleccionando por mediana. Los parámetros de memoria todavía no están activos
y no se buscan hasta L2. La puerta se comprueba después de ejecutar el modelo
completo, con el evaluador original y predicciones previamente selladas.

Si la realimentación completa no pasa, hay un único cambio estructural
preregistrado: eventos añadidos solo pueden actualizar SV/centroides si su
posterior es ≥0,8 (los aceptados siguen siendo predicciones). Este umbral es
alcanzable con el residual acotado a cuatro; 0,9 no lo era para candidatos nuevos.
Si también falla, se conserva L0 y no se ejecutan L2–L4.

La retención de Pareto durante calibración se decide por lotes que comienzan
antes de 5 s; puede abarcar la fracción restante del último lote (~80 ms).
No se usan esos instantes como etiquetas en inferencia. La referencia abierta
conserva exactamente los cinco segundos iniciales de predicciones originales.

### L1 completado

La primera estructura supera la puerta; no se ejecuta la alternativa.
Validación sub-04 (20 registros, 124 instancias), con el evaluador original:

| Nivel | RoA global | ramp5s | ≥90 % | Precisión | Recall |
|---|---:|---:|---:|---:|---:|
| Pareto-rerun | 87,65 % | 73,88 % | 66,94 % | 96,06 % | 89,03 % |
| C4 abierto | 90,39 % | 81,77 % | 70,16 % | 95,70 % | 92,94 % |
| L1 cerrado | 91,24 % | 84,71 % | 72,58 % | 96,38 % | 93,01 % |

L1 gana +0,84 pp global y +2,94 pp en ramp5s frente al abierto. Reposo de
referencia 97,13 %, movimiento 90,48 %, mesetas alejadas 89,98 %. Primer ciclo
de ramp5s 65,89 %, ciclos 2–8 84,94 %. La latencia medida durante cuatro
ejecuciones simultáneas fue mediana 127,92 ms, p95 151,32 ms; no satisface
22 ms y no representa una medición aislada. Son resultados exploratorios de
desarrollo, sin evaluación externa de 80 registros ni confirmación nueva.

Detalles, puertas, limitaciones y hashes en el
[diario](../../docs/benchmarks/pareto_mc_journal.md) y
[resumen versionado](../../docs/benchmarks/pareto_mc_l1/pareto_mc_summary.json).

### Desarrollo de L2

`memory_loop.py` añade recuperación causal de filtros efectivos y sus centroides,
proyección al blanqueo actual, atenuación por incertidumbre y mezcla limitada
después de QR. En L2 solo existe la instantánea de calibración. Beta cero
reproduce exactamente Pareto en los tres controles reales y conserva todo el
estado de L1 en las pruebas de integración.

```powershell
# Requiere L0, C1 y L1 completados en PARETO_MC_OUTPUT.
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.memory_identity "$env:PARETO_MC_OUTPUT/C2_identity_v1"
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.shared_identity
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.memory_experiment L2 --workers 4
```

Se registran 45 ensayos, cinco por cada combinación estructural, y se repiten
los tres mejores con las tres semillas entrenadas de C4. La búsqueda comparte
solo preprocesado, blanqueo y C1 del mismo lote entre nueve trayectorias; una
prueba real exige igualdad bit a bit con la ejecución independiente. No se
consideran sus tiempos como latencia de inferencia. Después se reentrena C4 en
tres rondas DAgger y se predice de nuevo con el bucle CPU completo antes de
decidir la puerta. Todos los registros se sellan antes de abrir sus etiquetas.

El momento de bin permanece fijo en L2 porque no hay inserciones; se busca en
L3. El limitador acota el residual de memoria respecto a la propuesta Pareto.
Los centroides recuperados se conservan; los vivos mantienen su actualización
EMA. Estas decisiones y la única alternativa permitida tras un fallo están
declaradas en el diario. No hay resultado L2 hasta que exista su `gate.json`.

Perfil adicional de L1 con un registro a la vez: mediana **56,12 ms**, p95
**360,87 ms**; tampoco satisface los 22 ms. Las pruebas y mediciones iniciales
de C2 están en `docs/benchmarks/pareto_mc_l2_setup/`.

### L2 completado

La primera estructura supera la puerta, después de 45 ensayos, repeticiones
con tres semillas y tres rondas DAgger nuevas. Validación sub-04, 20 registros
y 124 instancias:

| Métrica | L1 | L2 |
|---|---:|---:|
| RoA global | 91,24 % | 91,66 % |
| RoA ramp5s | 84,71 % | 85,05 % |
| Instancias ≥90 % | 72,58 % | 75,81 % |
| Precisión | 96,38 % | 95,68 % |
| Recall | 93,01 % | 94,14 % |
| Reposo de referencia | 97,13 % | 97,54 % |
| Movimiento | 90,48 % | 90,94 % |

Pasa reposo ≥97 % y mejora movimiento frente a L1. La pérdida de precisión
frente a Pareto es 0,38 pp, dentro del margen de un punto. Ramp5s: ciclo 1
68,97 %, ciclos 2–8 85,01 %. Memoria activa en el 4,29 % de pares unidad/lote.

Tiempo CPU con cuatro registros simultáneos: mediana **86,16 ms**, p95
**105,51 ms**. No satisface 22 ms; queda pendiente un perfil aislado de L2.
Se conserva L2. Son resultados exploratorios de validación, sin evaluación
externa ni confirmación nueva. La puerta y los hashes están en
[el resultado versionado](../../docs/benchmarks/pareto_mc_l2_standard/gate.json).

### Desarrollo de L3

L3 activa inserciones fiables en los bins de memoria y conserva la calibración
inmutable. Parte de los tres checkpoints finales de L2 y añade tres rondas
DAgger; las trayectorias de L1 y L2 también forman parte de la agregación.
El momento de los bins se incluye ahora en TPE. Antes de buscar, una prueba
real exige igualdad bit a bit entre ejecución independiente y compartida,
incluyendo los bins aprendidos y su confianza.

```powershell
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.full_memory_identity "$env:PARETO_MC_OUTPUT/C2_full_memory_identity_v1"
.venv-benchmarks/Scripts/python.exe -m benchmarks.pareto_mc.full_memory_experiment --workers 4
```

La puerta exige global ≥92,6580 % y ciclos 2–8 de ramp5s al menos cinco puntos
por encima del ciclo 1. Si la primera estructura falla, solo se permite la
alternativa de confianza local registrada; un segundo fallo detiene la escalera.
