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
