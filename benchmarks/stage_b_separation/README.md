# Stage B: corrección de la geometría de separación

**Estado: piloto detenido en validación, antes de inferencia externa.** La
CNN reajustada reduce el error proyectado pero su riesgo de fuente pierde
frente a la identidad en las particiones completadas. Sus pesos no se
promueven. Continúa la [corrección por etiquetas de disparo](../spike_likelihood/README.md).

Corrige el desajuste observado en la [variante anterior](../stage_b/README.md),
donde mejorar NMSE de la forma completa empeoró su proyección al separador.
La nueva corrida conserva sus registros, plantillas, proyección, arquitectura
Γ y observadores congelados. Cambia la normalización y el objetivo de Γ;
los experimentos anteriores y sus resultados permanecen intactos.

## Derivación

Sea `x=h_i s_i+interferencia`, con covarianza positiva R. La dirección previa
`R⁻¹h/sqrt(hᵀR⁻¹h)` tiene respuesta propia `sqrt(hᵀR⁻¹h)`, que varía con postura
y covarianza. Eso permite cambios de escala aun con una plantilla correcta,
mientras el detector parte de centroides calibrados en una postura.

Se impone respuesta propia unitaria mediante **MVDR**:

\[
\min_w w^\top Rw\quad\text{sujeto a }w^\top h=1,
\qquad w=\frac{R^{-1}h}{h^\top R^{-1}h}.
\]

La solución se obtiene con multiplicadores de Lagrange. Conserva la dirección
de separación y fija la amplitud esperada del evento modelado. La ganancia
adicional se calibra una sola vez con los cinco segundos iniciales. Se sigue
usando el detector SD original. Fuentes: [Capon, 1969](https://doi.org/10.1109/PROC.1969.7278)
y [documentación de MVDR](https://www.mathworks.com/help/phased/ref/mvdrweights.html).
MVDR depende de la exactitud de la plantilla: la restricción se refiere al
modelo estimado y no garantiza respuesta unitaria sobre una unidad mal estimada.

Para entrenar, `h*` es la trayectoria estimada de EMG limpia y disparos verdaderos
de sujetos permitidos. C es la covarianza observada de EMG proyectada en una
vecindad de postura, y `R* = L*L*ᵀ` su versión con el mismo shrinkage físico.
Se penaliza la forma extendida **en la geometría de separación**:

\[
L_p=\frac{\|L_*^{-1}(\widehat h-h_*)\|^2}{\|L_*^{-1}h_*\|^2}.
\]

Este término positivo controla dirección y amplitud. A diferencia de NMSE
en 6.400 muestras de forma, pondera los errores después de la proyección y la
covarianza que utiliza el separador. Se añade riesgo de fuente diferenciable:

\[
L_s=\frac{(\widehat w-w_*)^\top C(\widehat w-w_*)}{w_*^\top Cw_*}
=\frac{\mathbb E[(x^\top\widehat w-x^\top w_*)^2]}
{\mathbb E[(x^\top w_*)^2]}.
\]

La igualdad vale para x centrada con covarianza C. Por ello se penaliza el
error real de la fuente respecto al separador maestro, incluidas contribuciones
de interferencia representadas en C. Se diferencia a través del modelo de
covarianza, Cholesky y normalización MVDR. No se identifica artificialmente
este riesgo con RoA: el maestro usa plantillas estimadas y puede equivocarse.

El objetivo fijo es `Lp + 0.25 Ls + 0.1 Lang + 0.01 Lwave + 0.001 Lsize`.
Lang compara direcciones blanqueadas; Lwave conserva una regularización débil
de forma completa. Mantener Lp evita que una solución local que suprima la
salida reduzca Ls sin aprender la identidad. La identidad exacta en q=0 y
el límite de corrección de norma dos permanecen.

## Protocolo y controles

Se reutilizan las cuatro particiones externas, dos semillas y 42 registros
permitidos de supervisión: staircase/ramp40s/ramp10s, SNR 30/20 para ajuste
y 25 para validación. Cada registro tiene nueve estados con objetivos para
todas sus unidades. Ningún sujeto externo entra en el ajuste de su modelo;
ramp5s/ramp20s/15 dB se reservan del ajuste y selección. Las covarianzas
agrupan muestras de entrenamiento/validación por postura; en inferencia se
conserva el FIFO causal de diez lotes después del preprocesamiento offline.

Se refina Γ durante veinte épocas en GPU, con AdamW, lr 0,0003, decaimiento
0,001, gradiente máximo uno y perturbación de entrada del 3%. Hay tres tipos
de candidatos: identidad sin corrección, pesos anteriores sin reajustar y
épocas de refinamiento. Se elige por **riesgo de fuente en validación**,
ponderado por unidades, sin seleccionar con RoA externo. El observador B
anterior queda congelado, con los mismos estados por semilla. No aumentan
los parámetros: Γ conserva 3.137 y el observador 10.537.

Los cuatro métodos nuevos son Γ corregida con estado B, la misma Γ con estado
K, Γ antigua con MVDR para aislar la normalización y plantilla estática con
MVDR para aislar el aprendizaje. Se incluyen Pareto y el B anterior como
referencias sobre las mismas 591 instancias. Las 320 predicciones nuevas se
guardan antes de la primera puntuación. La selección de pesos y las fuentes
quedan congeladas con hashes.

Los resultados anteriores motivaron la corrección y ya exponen este benchmark.
Las nuevas decisiones de selección usan entrenamiento/validación; aun así,
la transferencia medida sigue siendo exploratoria sobre el mismo generador.
Una confirmación independiente requiere datos nuevos. No se producen rampas
de 1/2,5 s mediante remuestreo ni se afirma latencia completa desde EMG cruda.

## Ejecución

Se usan el entorno CUDA `.venv-stage-b-gpu` y el entorno de evaluación CPU
`.venv-benchmarks` del experimento anterior. No se reinstalan dependencias.
La salida es `outputs/stage_b_separation`; `STAGE_B_SEPARATION_OUTPUT`
permite otra salida sin sobrescribir una corrida registrada.

```powershell
foreach ($subject in 'sub-01','sub-02','sub-03','sub-04','sub-05') {
  .venv-stage-b-gpu/Scripts/python.exe -m benchmarks.stage_b_separation.run prepare --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run register
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  foreach ($trialSeed in 0,1) {
    .venv-stage-b-gpu/Scripts/python.exe -m benchmarks.stage_b_separation.run train --subject $subject --seed $trialSeed
  }
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run freeze
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run audit
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run predict --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run seal
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run start-scoring
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run score --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.stage_b_separation.run report
```

Tras iniciar puntuación se pueden distribuir los cuatro sujetos en procesos
independientes. El informe exige ochenta registros y 591 instancias únicas
por método. Los resultados pequeños se exportan con prefijo
`docs/benchmarks/stage_b_separation_`; los arrays y pesos grandes se ignoran.

El [piloto geométrico](../stage_b_geometry/README.md) documenta la alternativa
posterior. Ambos pilotos se detuvieron en validación, sin evaluación externa.
