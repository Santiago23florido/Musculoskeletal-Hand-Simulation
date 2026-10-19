# Reparación de detección mediante verosimilitud de disparos

Tras el fallo de la [CNN de Stage B](../stage_b/README.md), esta variante aprende
la decisión que se evalúa: aceptar o rechazar candidatos de descarga de cada
unidad. Conserva las fuentes Pareto adaptativas y las fuentes fijas del mismo
benchmark como observaciones. **Repara el decodificador de eventos; no demuestra
haber aprendido las trayectorias verdaderas de MUAP.** Los pilotos de
[separación](../stage_b_separation/README.md) y
[geometría](../stage_b_geometry/README.md) documentan los intentos anteriores
a la supervisión directa de disparos.

## Resultado medido

Se completaron ocho entrenamientos CUDA y se congelaron los pesos y umbrales.
Las 160 predicciones se guardaron antes de puntuar los mismos 80 registros
externos y 591 instancias calibradas del estudio anterior.

| Método | RoA global | Rampas 5 s | Rampas 10 s | Recall | Final ≥90% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pareto | 80,89% | 65,81% | 77,14% | 83,16% | 511 |
| **Verosimilitud, primario** | **83,51%** | **72,33%** | **80,05%** | **87,74%** | **520** |
| Unión de rescate | 83,27% | 72,08% | 79,82% | 87,86% | 506 |
| Stage B morfológico anterior | 37,26% | 33,78% | 34,93% | 39,30% | 573 |

La mejora propia sobre Pareto es **+2,62 puntos globales**, **+6,52 en ramp5s**,
+2,91 en ramp10s y +2,70 a 15 dB. Mejoran los cuatro sujetos y las cinco
condiciones. Hay 502 instancias con mejora y 89 con empeoramiento. El intervalo
bootstrap descriptivo por sujeto de la mejora global es [+2,21; +2,98] puntos;
solo hay cuatro sujetos y entrenamiento compartido entre particiones.

La precisión media pasa de 91,47% a **91,10%**. Se gana recall con un coste de
falsos positivos. Las cuentas del evaluador suman +30.653 TP y +10.285 FP;
usan el desfase propio de cada método y no etiquetan independientemente cada
evento añadido. La eliminación aprendida aporta 0,24 puntos frente a conservar
todos los eventos anteriores mediante unión.

Al final se ganan 16 instancias ≥90% y se pierden siete: el saldo es +9,
pero no se preserva toda unidad que Pareto conservaba. El RoA final medio
sube de 87,99% a 95,25%. La mediana global baja ligeramente de 95,29% a 95,07%.
Se informa el coste sobre unidades fuertes junto a la mejora de las débiles.

Los objetivos completos de la guía siguen pendientes: 83,51% frente a 85%
global, 72,33% frente a 80% en ramp5s y 59,73% frente al 70% de instancias con
RoA global ≥90%. El estado formal del JSON conserva `no-go` para esos umbrales.
La conclusión positiva es una mejora medida de detección y transferencia en
este benchmark; la CNN de morfología anterior no se presenta como corregida.

Resultados: [resumen](../../docs/benchmarks/spike_likelihood_summary.json),
[CSV por instancia](../../docs/benchmarks/spike_likelihood_per_unit.csv),
[análisis descriptivo](../../docs/benchmarks/spike_likelihood_analysis.json),
[auditoría](../../docs/benchmarks/spike_likelihood_audit.json) y
[selección congelada](../../docs/benchmarks/spike_likelihood_locked.json).

![Comparación del decodificador corregido en los mismos registros externos](../../docs/images/spike-likelihood.png)

## Fundamento

El error cuadrático de una forma o de una fuente continua no fija el umbral
correcto de detección. Puede disminuir sin recuperar eventos. Aquí la etiqueta
es `y=1` cuando un candidato coincide con un disparo verdadero, dentro de la
tolerancia de 2 ms y usando solo el desfase medido en calibración inicial.

Se entrena log loss binaria:

\[
L=-\mathbb E[y\log p_\theta+(1-y)\log(1-p_\theta)].
\]

La esperanza condicional se minimiza en `p=P(y=1|z)`. Es una regla de puntuación
estrictamente propia, bajo su distribución de entrenamiento; véase
[Gneiting y Raftery, 2007](https://doi.org/10.1198/016214506000001437).
Eso justifica aprender probabilidades de disparo en lugar de reducir energía
de forma. No garantiza calibración fuera de esa distribución ni un RoA alto.
El umbral se selecciona en validación con la métrica de eventos, siguiendo la
distinción entre estimar probabilidades y elegir una decisión por F1/Jaccard
de [Lipton et al., 2014](https://arxiv.org/abs/1402.1892).

El modelo es una corrección de las razones de probabilidad de Pareto:

\[
\operatorname{logit}p=\ell_0+\eta\,4\tanh(f_\theta(z)),
\qquad \ell_0=\begin{cases}\log9&\text{evento Pareto}\\-\log9&\text{otro candidato.}\end{cases}
\]

`f` es un MLP de 19→8→1, con **169 parámetros** compartidos entre todas las
unidades y sujetos. La última capa empieza en cero. La corrección cero y
`η=0` reproducen exactamente Pareto, incluida su predicción inicial. El límite
del residual regulariza cuánto pueden cambiar las probabilidades iniciales.
Su valor inicial es un punto de referencia, no una afirmación de que 0,9/0,1
sean probabilidades empíricamente calibradas.

Los diecinueve datos de entrada describen amplitud normalizada por calibración,
forma temporal local, acuerdo entre las dos vistas, competencia entre fuentes,
SIL/tasa de calibración y seis estadísticas EMG del lote. **No hay ángulos,
velocidad real, reloj absoluto ni identificadores entrenables.** Se descartan
explícitamente las dos columnas cinemáticas del caché antiguo.

Los candidatos incluyen todos los eventos Pareto y máximos débiles de ambas
fuentes. La ventana de forma está limitada al lote actual de unos 100 ms.
La decisión final aplica supresión de máximos dentro del lote con la misma
distancia mínima del detector original. El preprocesamiento original sigue
siendo offline: la causalidad se refiere a bloques después de ese paso.

## Ajuste y generalización

Se conserva la validación externa por sujetos: cuatro sujetos externos
`sub-02` a `sub-05`, un sujeto de validación por partición y tres de ajuste,
siempre incluyendo `sub-01` en ajuste. Se usan staircase/ramp40s/ramp10s con
30/20 dB para entrenar y 25 dB para seleccionar. Ramp5s, ramp20s y 15 dB
quedan fuera de ajuste y selección de pesos/umbrales de su propia partición.

Se usan etiquetas posteriores a los primeros cinco segundos de los sujetos
de ajuste. No se reponderan clases ni se descartan negativos: se mantiene
su frecuencia observada. Se equilibran las unidades para evitar que una
fuente con muchos máximos de ruido domine la función objetivo. El desfase
por unidad procede de los cinco segundos de calibración; nunca de alinear
el registro de ajuste completo después de predecir.

El presupuesto es de diez épocas, dos semillas, AdamW con lr 0,001,
decaimiento 0,01, ruido de entrada 0,02 y gradiente máximo uno. Cada dos épocas
se valida. El candidato inicial de identidad también participa. Se selecciona
por Jaccard medio de unidades en validación, sobre los desfases de calibración,
con un conjunto fijo de siete umbrales. Después se promedian los logits de
ambas semillas y se elige `η∈{0,0.5,1}` y el umbral usando solo validación.

Los ocho modelos y las decisiones de conjunto se congelan antes de predecir
externos. Se guardan **160 predicciones**: decisión posterior y una ablación
que añade sus eventos conservando los originales. Las 160 se sellan antes
de abrir puntuación. Pareto y el Stage B anterior se incorporan como referencias
sobre las mismas **591 instancias de 80 registros**. La puntuación externa
usa el evaluador original completo, incluida su búsqueda de desfase original.

Este benchmark ya fue explorado y motivó la corrección. Las decisiones nuevas
se toman en entrenamiento/validación, pero el resultado sigue siendo
transferencia exploratoria sobre el mismo generador. El tamaño pequeño y la
ausencia de IDs no prueban por sí solos generalización independiente.

## Reproducción

Se reutilizan los entornos CUDA y CPU existentes. Datos, fuentes y cachés
anteriores permanecen congelados. La salida es `outputs/spike_likelihood`;
`SPIKE_LIKELIHOOD_OUTPUT` permite una salida distinta.

```powershell
foreach ($subject in 'sub-01','sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run prepare --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run register
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  foreach ($trialSeed in 0,1) {
    .venv-stage-b-gpu/Scripts/python.exe -m benchmarks.spike_likelihood.run train --subject $subject --seed $trialSeed
  }
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run freeze
.venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run audit
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run predict --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run seal
.venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run start-scoring
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run score --subject $subject
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.spike_likelihood.run report
```

Los sujetos se pueden preparar, predecir o puntuar en procesos independientes;
solo se distribuye la puntuación después de validar el manifiesto completo.
El informe requiere ochenta registros y 591 claves únicas por método.
Los resultados pequeños se copian con prefijo `docs/benchmarks/spike_likelihood_`.

El tiempo registrado comienza con los candidatos ya calculados y mide
verosimilitud en CPU, decisión y persistencia de salidas. Excluye extracción
de características y descomposición previa. Sigue pendiente el presupuesto
completo de 22 ms desde EMG cruda, las rampas verdaderas de 1/2,5 s y un
generador independiente. No se asigna artificialmente SIL de una sola fuente
a decisiones que combinan dos vistas.
