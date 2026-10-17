# Stage B — MoCA con cinemática y observador ciego

Implementación independiente de morfología MUAP compartida y estado causal.
Primero se contrasta
**MoCA-K**, que recibe el ángulo observado, y después **MoCA-B**, que estima ese
estado a partir de EMG. Ambos comparten los mismos pesos de morfología: la
diferencia K–B permite evaluar el coste de inferir el estado. Los resultados se
registran en [el resumen publicado](../../docs/benchmarks/stage_b_summary.json).

## Resultado ejecutado: NO-GO

Ocho entrenamientos CUDA, pesos congelados, 320 predicciones guardadas antes
de puntuar y 80 registros externos con 591 instancias por método.

| Método | RoA global | Rampas 5 s | Instancias finales ≥90% |
| --- | ---: | ---: | ---: |
| Pareto publicado | 80,89% | 65,81% | 511 |
| Plantilla estática analítica | 45,56% | 39,17% | 577 |
| MoCA-K, ángulo observado | 36,97% | 33,34% | 573 |
| **MoCA-B, primario** | **37,26%** | **33,78%** | **573** |
| B sin shrinkage físico online | 26,11% | 25,41% | 130 |

B falla los tres umbrales de precisión: 37,26% global frente a 85%, 33,78% en
ramp5s frente a 80% y 0% de instancias con RoA global ≥90% frente al 70% exigido.
Su precisión media es 92,17% y el recall 39,30%: se pierden muchas descargas.
K tampoco mejora la separación, aunque recibe el estado real. El error angular
de B es 3,37° de media por registro después de calibración y 5,32° en ramp5s.

Sí hay recuperación al final: B alcanza 97,32% de RoA en los últimos cinco
segundos y 573 instancias ≥90%, con 74 ganadas y 12 perdidas respecto a Pareto.
La plantilla estática llega a 577; ese resultado no acredita una mejora de Γ.
El shrinkage físico mejora B en 11,15 puntos respecto a FIFO puro. La
corrección aprendida pierde 8,30 puntos respecto al control estático.

El núcleo CPU tiene mediana de medianas por registro de 10,92 ms y mediana
de percentiles 95 de 20,01 ms por lote. Son mediciones con hasta cuatro procesos
concurrentes, desde entradas ya proyectadas; la latencia completa queda sin
verificar. El entrenamiento GPU tarda 33,97–41,52 segundos por ejecución.

El [resumen](../../docs/benchmarks/stage_b_summary.json),
[métricas por instancia](../../docs/benchmarks/stage_b_per_unit.csv),
[análisis descriptivo](../../docs/benchmarks/stage_b_analysis.json) y
[auditoría](../../docs/benchmarks/stage_b_audit.json) conservan el resultado.
Las 591 filas Pareto reproducen exactamente el experimento anterior, incluidas
las métricas por fase. Los pesos y las predicciones no se ajustan tras este fallo.

![Resultados de Stage B en los mismos registros externos](../../docs/images/stage-b.png)

## Teoría implementada

Cada plantilla multicanal procede de las descargas de calibración inicial de
su unidad, durante cinco segundos. Se normaliza por su energía y se procesa
como un volumen de 64 muestras por 10×10 electrodos. Una CNN espaciotemporal
con dos bloques de [FiLM](https://arxiv.org/abs/1709.07871) calcula

\[
\widehat h_i(q)=h_{i,cal}+\operatorname{clip}_{\|\cdot\|_2\le2}
\{q\,\Gamma_\theta(h_{i,cal},q)\}.
\]

La salida inicial es la identidad y la corrección es cero para `q=0`.
Los 3.137 parámetros de Γ se comparten entre sujetos y unidades; no hay
vectores entrenables por identificador. El límite de energía, el decaimiento
de pesos y la perturbación de entradas regularizan el tamaño; no demuestran
generalización por sí mismos.

La forma predicha se convierte al orden de retardos del benchmark y se proyecta
en 512 ejes de calibración. El separador se calcula analíticamente:

\[
w_i=\frac{R^{-1}\widehat h_i}
{\sqrt{\widehat h_i^\top R^{-1}\widehat h_i}},\qquad
R=0.5R_{FIFO}+0.5\left(\sum_i\widehat\lambda_i\widehat h_i
\widehat h_i^\top+\sigma^2I\right)+\epsilon I.
\]

Para una plantilla fijada y covarianza positiva, esta dirección maximiza
`(wᵀh)²/(wᵀRw)`: al blanquear por R, Cauchy–Schwarz da `w ∝ R⁻¹h`.
La normalización usada fija `wᵀRw=1`. La propiedad solo vale para la plantilla
y covarianza estimadas; no asegura separación si las identidades están mal
estimadas. La aproximación y la generalización se contrastan experimentalmente.

Se usa Cholesky compartida, FIFO de diez lotes y tasa causal estimada a partir
de descargas aceptadas. La base PCA y las ganancias quedan fijas tras calibración.
Se conserva el detector SD original, sin respaldos Pareto posteriores a los
primeros cinco segundos ni nuevas unidades. La actualización utiliza el lote
actual y el pasado, con lotes de aproximadamente 100 ms. El preprocesamiento
original contiene filtrado offline: no se presenta como causal desde EMG cruda.

En K, `q=ángulo/40`. En B, una GRU de 10.537 parámetros recibe 120 variables:
100 log RMS de canales, 16 log autovalores de covarianza y cuatro estadísticas
de fuentes fijas del lote **anterior**. La normalización usa solo la calibración.
El ángulo es una etiqueta de entrenamiento, nunca una entrada del observador B.
Su salida es escalar; no consume velocidad real ni tiempo absoluto. Inicios
circulares aleatorios en entrenamiento reducen el uso del reloj del protocolo.

## Supervisión y diferencias respecto a la guía

FDSI contiene EMG limpia/ruidosa, ángulos y tiempos de disparo, pero no las
trayectorias MUAP originales del generador. Los objetivos de Γ son por ello
**promedios de EMG limpia activados por disparos verdaderos**, en nueve
vecindades de postura con un mínimo de veinte eventos. La alineación por unidad
usa solo la calibración. Los objetivos contienen error e interferencia; no se
denominan MUAPs verdaderos.

Esta variante usa etiquetas posteriores a calibración en sujetos de ajuste y
validación, a diferencia de los experimentos anteriores sin esas etiquetas de
entrenamiento. Se minimiza error de forma normalizado, más 0,1 de error en la
derivada temporal y 0,001 de energía de corrección. No se implementa optimización
conjunta por reconstrucción EMG/BCE ni peel-off opcional.

La covarianza se aproxima en 512 dimensiones. Γ congelada se evalúa en 129
posturas durante la inicialización y luego se interpola su proyección. Esta
tabla depende de calibración y pesos, sin señales futuras. La auditoría compara
interpolación con Γ directa. Se distingue el coste de crear tablas del núcleo
por lote. Capturar mucha varianza PCA no demuestra preservar toda la identidad.

## Protocolo y generalización

Se conservan los 80 registros y las 591 instancias calibradas de los experimentos
previos. Para cada sujeto externo `sub-02` a `sub-05`, el siguiente sujeto cíclico
es validación y los tres restantes entrenan; `sub-01` siempre entrena. Solamente
staircase, ramp40s y ramp10s entran en ajuste: SNR 30/20 para entrenar y 25 para
seleccionar. Ramp5s, ramp20s y 15 dB no entran en ajuste ni selección de pesos.
Cada modelo excluye las etiquetas de su sujeto externo. En validación cruzada,
un sujeto externo de una partición puede entrenar otra; no se afirma que ningún
archivo de etiquetas haya sido leído globalmente antes de inferencia.

Se fijan dos semillas, veinte épocas de Γ y treinta del observador, incluida la
época cero como candidata. Cada rama elige checkpoint por su objetivo de
validación, sin RoA externo. Se promedian plantillas de ambas semillas; cada
observador conserva su estado. Ocho entrenamientos y dieciséis checkpoints se
congelan antes de inferencia externa. Los cuatro métodos nuevos guardan sus
320 predicciones antes de iniciar la puntuación.

Comparadores: Pareto publicado, plantilla estática con inversa analítica,
MoCA-K, MoCA-B primario y ablación B sin shrinkage del modelo en la covarianza
online. Se conservan instancias y evaluador original: tolerancia de 2 ms y
alineación original por unidad. Las fases se alinean por separado, conforme al
trabajo previo. Se registran precisión, recall, F1, SIL, retención final y error
angular posterior a calibración.

El protocolo guarda hashes de código y PDF, una instantánea de fuentes, pesos
elegidos y manifiesto de predicciones. P1 ya fue visto en trabajos anteriores:
esta es transferencia exploratoria agrupada sobre el mismo generador, no una
confirmación independiente.

## Entrenamiento GPU e inferencia CPU

El entrenamiento usa **CUDA en la RTX 4070 Laptop de 8 GB**, PyTorch 2.7.0 con
CUDA 12.6, float32 sin AMP/TF32, semillas y algoritmos deterministas. El entorno
CPU anterior queda intacto. El intento parcial en CPU se conserva con fuentes,
protocolo e historiales en `outputs/stage_b_main/aborted_cpu_v1/`; sus pesos
no participan en la ejecución GPU. Se cambió de dispositivo antes de puntuar
modelos Stage B externos.

Instalación PowerShell con uv:

```powershell
uv venv .venv-stage-b-gpu --python .venv-benchmarks/Scripts/python.exe
uv pip install --python .venv-stage-b-gpu/Scripts/python.exe torch==2.7.0+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-stage-b-gpu/Scripts/python.exe -r benchmarks/stage_a/requirements.lock.txt
.venv-stage-b-gpu/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

La distribución CUDA sigue las
[instrucciones oficiales de PyTorch 2.7.0](https://pytorch.org/get-started/previous-versions/#v270).
Las otras dependencias conservan versiones previas. No se cambia el controlador
NVIDIA. `train` exige CUDA y falla si falta; evaluación carga pesos en CPU.

```powershell
# Para una salida nueva: una escritura de objetivos por sujeto.
foreach ($subject in 'sub-01','sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b.run targets --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b.run register
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  foreach ($trialSeed in 0,1) {
    .venv-stage-b-gpu/Scripts/python.exe -m benchmarks.stage_b.run train --subject $subject --seed $trialSeed
  }
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b.run freeze
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b.run audit
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b.run predict --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b.run seal
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b.run score
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b.run report
```

Destino por defecto: `outputs/stage_b_main`. `STAGE_B_OUTPUT` permite una salida
distinta conservando experimentos previos. No se registra encima de un protocolo
existente. Los resultados pequeños se copian a `docs/benchmarks/stage_b_*`;
datos y pesos grandes permanecen ignorados.

El [lock GPU](requirements-gpu.lock.txt) conserva las versiones instaladas.
Después de `report`, `python -m benchmarks.analyze_stage_b` exporta una
comparación descriptiva por fase, sujeto, condición/ruido y error de estado.
Ese análisis no cambia ni selecciona parámetros.

En la ejecución registrada, el primer sujeto se puntuó en serie; los otros
tres se distribuyeron con `python -m benchmarks.score_stage_b_partition sub-03`
y sus equivalentes. Ese programador restringe únicamente el bucle de sujetos
del evaluador congelado. Cada proceso verifica el manifiesto completo y guarda
los hashes de sus veinte resultados; el informe exige los ochenta registros.
El [registro de distribución](../../docs/benchmarks/stage_b_scoring_schedule.json)
conserva la programación. No cambia pesos, métricas ni umbrales.

Después de puntuar, `python -m benchmarks.diagnose_stage_b` examina dos registros
de entrenamiento/validación del modelo congelado para `sub-02`. Compara error
de forma con error de la plantilla central proyectada y puntúa los mismos
modelos sin reentrenar. Su [diagnóstico](../../docs/benchmarks/stage_b_training_diagnostic.json)
es descriptivo y no selecciona una nueva variante.

## Criterio científico

Para B, la guía pide RoA global ≥85%, ramp5s ≥80% y ≥70% de instancias con
RoA ≥90%. También exige ≤22 ms por lote de inferencia CPU completa y caída ≤10
puntos entre 8 y 40 grados/s. FDSI no distribuye rampas nuevas de 1/2,5 s ni sus
trayectorias verdaderas; no se sustituyen mediante remuestreo. El tiempo medido
excluye proyección y extracción de variables. Aunque pasen los tres umbrales
de precisión, el estado completo queda **incompleto** hasta ejecutar esas pruebas.
Si falla precisión, es **NO-GO**.

La auditoría comprueba que cambiar todos los ángulos reales deja B idéntico,
perturbar señales futuras conserva el prefijo tras preprocesamiento y la tabla
tiene menos del 1% de error relativo frente a Γ en las plantillas proyectadas
del registro de entrenamiento auditado. No demuestra causalidad del filtrado
offline ni generalización fuera del generador.
