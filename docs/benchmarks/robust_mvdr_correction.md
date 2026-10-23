# Corrección del diagnóstico robust_mvdr y auditoría del régimen FIFO

## Resumen

**La propuesta robust_mvdr queda retirada.** El decodificador propuesto tiene un
defecto de régimen: al cambiar a la covarianza FIFO dentro de muestra con
normalización de respuesta unitaria, pierde amplitud frente al umbral del detector.
El sintético reproduce la pérdida y la comprobación en sub-01 da el **caso A**:
el control FIFO colapsa, mientras Stage B no. No se trata de un nuevo experimento
sobre sub-02 a sub-05; esos sujetos no se han vuelto a usar.

Excluir el bloque actual recupera solo parte del rendimiento en FDSI. **No se ha
establecido** que sea la causa principal de toda la diferencia con Stage B ni una
explicación exhaustiva del déficit restante. La hipótesis F1 sobre el déficit del
Stage B original sigue sin confirmar.

Fuentes reproducibles: [sintético y sensibilidad](fifo_audit_synthetic.json),
[sub-01](fifo_audit_sub01.json), [arnés](../../benchmarks/fifo_audit/harness.py) y
[lector de cachés](../../benchmarks/fifo_audit/fdsi_check.py). Los JSON incluyen
los hashes del código, las versiones y las métricas por unidad. No se ajustó
ningún parámetro ni se cambió el generador, el decodificador o la métrica original
del arnés al reanudar esta tarea.

## Qué pasó en el experimento publicado

El [CSV original](robust_mvdr_per_unit.csv) conserva 591 instancias por método.
El recall medio de los seis métodos nuevos está entre 5,2093% y 5,3024%; casi
todo procede de los disparos de calibración, copiados durante los primeros 5 s.
Para `static_subtract`, las 591 instancias tienen recall entre 3,5948% y 5,9286%;
para `static_fifo_mvdr`, entre 3,5948% y 5,9113%. El RoA medio del primario en
rampa es 0,0252%, en meseta final 0,0947% y en reposo de referencia 0,0943%.

La afirmación «los seis métodos están entre 3,6% y 5,9% en todas las instancias»
no es literal: `static_subtract_strong` tiene 30 instancias fuera del intervalo
redondeado y llega a 23,4095%; `static_loaded` llega a 6,9946%, MoCA-K a 6,2397%
y MoCA-B a 6,3218%. Estas excepciones no eliminan el colapso agregado. Se corrige
la descripción, sin editar el CSV ni otros resultados publicados.

## Mecanismo: tres factores y el umbral del detector

1. **Covarianza dentro de muestra.** El bloque actual entra antes de sintetizar
   su propio filtro. Son diez bloques de 204 muestras (aproximadamente 2040)
   para D=512, no una covarianza de 10 s. Esto reduce la amplitud en los disparos
   verdaderos incluso sin movimiento. La variante fuera de muestra incorpora
   el bloque solo después de calcular el filtro; todo lo demás se conserva.
2. **Normalización.** El filtro propuesto usa
   `A^-1 h / (h^T A^-1 h)` (respuesta unitaria); Stage B usa
   `A^-1 h / sqrt(h^T A^-1 h)` (potencia de salida). La misma dirección no implica
   la misma ganancia tras el cambio desde la calibración.
3. **Mezcla α.** Stage B mezcla α=0,5 del modelo con la covarianza empírica.
   En el pipeline realista con detector, esta mezcla supera al control
   `fifo_mvdr_output_power` con α=0 en los ocho casos sintéticos y en las tres
   condiciones de sub-01. Se retira la crítica anterior a la mezcla.

El detector real clasifica picos de `abs(source)^2` con el punto medio de los
centroides de señal y base. El centroide de disparos solo se actualiza con
disparos detectados y aceptados por la compuerta. Si la amplitud cae demasiado,
el detector puede quedarse sin eventos con los que adaptar ese centroide.
En el sintético estacionario a 30 dB, semilla 3, el umbral absoluto inicial está
entre 0,7072 y 0,7088. **Ese umbral absoluto no es un umbral universal en la razón
de amplitud mediana**: la distribución de picos, la adaptación y la historia
importan. Los umbrales no se han cambiado.

La primera tanda FIFO empieza a 5,080078 s por el tamaño de lote; las ventanas
solicitadas siguen empezando en 5 s y 10 s. El primer segundo 5–6 s incluye ese
pequeño tramo previo al cambio. El resto abarca 6 s hasta el final, con movimiento.

## Atribución sintética

Cada celda indica **RoA (%) / razón de amplitud**. La razón es la mediana de las
razones por unidad respecto a 1–5 s; no el cociente de dos medianas agregadas.
Se calibra el retardo de cada unidad buscando ±15 muestras y no se vuelve a
alinear la amplitud con datos posteriores. RoA usa `rate_of_agreement_paired`,
2 ms. Estacionarios puntuados desde 5 s y rampas desde 8 s.

Pool de 64 unidades, semilla 3, impulso 0,15, 25 s, ocho unidades seleccionadas
por energía blanqueada de calibración, extensión 10, PCA de 512 ejes a varianza
unidad, STA de radio 32 y centrado por lote. Las rampas bajan de 8 a 13 s,
mantienen hasta 18 s y regresan de 18 a 23 s. Los metadatos documentan la
interpolación de MUAP y la selección; no se modificó el montaje para acercar RoA.

| Variante | Estacionario 20 dB | Estacionario 30 dB | Rampa 0,94 | Rampa 0,91 |
| --- | --- | --- | --- | --- |
| stage_b_static | 99.637 / 0.8537 | 99.838 / 0.8689 | 92.699 / 0.7266 | 82.976 / 0.6903 |
| static_fifo_mvdr | 87.873 / 0.6727 | 93.283 / 0.6985 | 40.509 / 0.4709 | 36.069 / 0.4145 |
| static_subtract | 88.775 / 0.6783 | 92.129 / 0.6862 | 41.706 / 0.4678 | 34.469 / 0.4167 |
| fifo_mvdr_output_power | 96.980 / 0.8121 | 98.167 / 0.8246 | 81.761 / 0.6688 | 71.822 / 0.6315 |
| fifo_mvdr_out_of_sample | 99.446 / 0.9298 | 99.783 / 0.9491 | 73.312 / 0.6608 | 60.525 / 0.5859 |
| stage_b_out_of_sample | 99.729 / 0.9767 | 99.838 / 0.9890 | 93.972 / 0.8280 | 84.597 / 0.7812 |
| subtract_out_of_sample | 99.270 / 0.9342 | 99.692 / 0.9487 | 60.285 / 0.5797 | 49.771 / 0.5182 |

Una implementación independiente del revisor dio razones de amplitud menores
(0,61 frente a 0,77 a 30 dB) y RoA 0,56 para el FIFO real, de modo que la magnitud
depende del registro sintético y solo el mecanismo es reproducible. Son cifras
comunicadas por el revisor, no valores de referencia ni ejecuciones locales.
Aquí se obtuvieron 0,6985 frente a 0,8689 y RoA 0,9328 para ese control.

### Sensibilidad estacionaria: semillas 3, 4 y 5

Las siete variantes se ejecutaron en cada caso. La tabla resume los dos controles;
el JSON contiene las siete, sus métricas y las tablas por unidad. Los casos de
semilla 3 reutilizan exactamente los resultados principales: son ocho registros
únicos, no diez muestras independientes.

| Caso | FIFO: RoA (%) / razón | Stage B: RoA (%) / razón |
| --- | --- | --- |
| stationary_20db_seed3 | 87.873 / 0.6727 | 99.637 / 0.8537 |
| stationary_20db_seed4 | 83.605 / 0.6808 | 99.296 / 0.8614 |
| stationary_20db_seed5 | 86.639 / 0.6639 | 99.231 / 0.8481 |
| stationary_30db_seed3 | 93.283 / 0.6985 | 99.838 / 0.8689 |
| stationary_30db_seed4 | 91.169 / 0.6935 | 99.538 / 0.8676 |
| stationary_30db_seed5 | 88.621 / 0.6839 | 99.359 / 0.8626 |

### Evaluación de C1–C3

- **C1 se cumple: 8/8 casos únicos.** La razón de amplitud de FIFO es menor que
  la de Stage B. La menor diferencia Stage B − FIFO es 0.170418.
- **C2 se cumple: 24/24 comparaciones.** Las tres variantes fuera de muestra
  superan en razón a su contraparte. La menor ganancia de razón es 0.090905.
- **C3 no queda establecida como un corte común o universal.** Hay una asociación
  descriptiva entre amplitud baja y RoA baja, pero las tablas por unidad contienen
  excepciones importantes; no se ajustó curva ni se estimó un punto de corte.

En las 128 observaciones unidad–método–caso (no independientes), las 18 con razón
menor de 0,6 tienen RoA mediana 20.333%,
mientras las 50 con razón al menos 0,8 tienen mediana 99.547%.
Sin embargo, en la rampa 0,91 la unidad del pool 39 tiene razón 0,6201 y RoA 100%,
y la unidad 36 razón 0,6069 y RoA 98,19%. Por tanto, estos datos no demuestran que
pasar por debajo de 0,65–0,7 provoque siempre una pérdida permanente. El mecanismo
del detector permite ese fallo y FDSI lo reproduce, pero su frontera no queda
identificada por una mediana agregada. Las tablas completas están al final.

## Confirmación en sub-01

**Caso A**, con 41 instancias en tres registros existentes a 30 dB: staircase
(14), ramp40s (14) y ramp10s (13). Se omitieron ramp20s y ramp5s porque no existe
su caché. Se cargó la verdad exactamente como en el evaluador registrado:
`load_gt(...)[:, record["calibration"].gt_matched_indices]`.

Los controles usan ambos `predict` reales. El arnés reproduce sus fuentes con
diferencia máxima **0** y sus máscaras de disparos con igualdad **100%** en los
tres registros. El lector solo abre archivos existentes; no llama a `prepare`
ni dispone de una ruta para crear cachés de Stage B.

Tabla agregada: RoA media sobre las 41 instancias, no media sin ponderar de las
tres condiciones. Las razones son medianas sobre esas instancias.

| Variante | RoA desde 5 s (%) | RoA desde 10 s (%) | Razón 5–6 s | Razón desde 6 s |
| --- | --- | --- | --- | --- |
| stage_b_static | 42.8317 | 41.7003 | 0.7863 | 0.3635 |
| static_fifo_mvdr | 0.1018 | 0.0146 | 0.5157 | 0.1325 |
| static_subtract | 0.0978 | 0.0117 | 0.4750 | 0.1383 |
| fifo_mvdr_output_power | 20.9700 | 20.6619 | 0.7122 | 0.2970 |
| fifo_mvdr_out_of_sample | 22.8900 | 20.8725 | 0.9418 | 0.2609 |
| stage_b_out_of_sample | 44.1240 | 42.2784 | 0.9850 | 0.4412 |
| subtract_out_of_sample | 17.4558 | 16.0614 | 0.9124 | 0.2398 |

El control FIFO queda por debajo del 20% declarado como colapso y Stage B no:
0,1018% frente a 42,8317% desde 5 s; desde 10 s, 0,0146% frente a 41,7003%.
En el primer segundo, la razón de FIFO es 0,5157, claramente inferior a 0,7863
de Stage B y al intervalo orientativo 0,65–0,7. Se reproduce el fallo de amplitud
y detección en FDSI sin volver a usar los sujetos externos.

Excluir el bloque actual eleva FIFO a 22,8900%, pero deja **19,9417 puntos**
frente a Stage B. No recupera una RoA comparable en esta comprobación; no se fijó
un margen formal de equivalencia. El bloque actual es un factor causal de parte
de la pérdida, **no se ha establecido** que sea la causa principal de todo el
déficit en FDSI. Con solo normalización de potencia alcanza 20,9700%; Stage B
fuera de muestra obtiene 44,1240% (+1,2924 puntos). La mezcla, la normalización y
la historia del detector siguen importando. El factor residual específico de
FDSI no se ha aislado exhaustivamente y no se corrigió.

Las razones del resto del registro incluyen posturas cambiantes. Por ejemplo,
Stage B tiene razón agregada 0,3635 y mantiene 42,83% de RoA; no se puede tratar
esa mediana temporal como una atenuación fija frente a un detector inmutable.

### Amplitudes y RoA por registro

Amplitudes en unidades de fuente: mediana de las medianas por unidad. Entre
paréntesis figura la mediana de las razones respecto a calibración. La
calibración cubre 1–5 s; primer segundo 5–6 s; resto 6 s hasta el final.

| Condición | Variante | RoA 5 s (%) | RoA 10 s (%) | A calibración | A primer segundo (razón) | A resto (razón) |
| --- | --- | --- | --- | --- | --- | --- |
| staircase | stage_b_static | 46.262 | 44.892 | 7.6347 | 5.7193 (0.8000) | 1.5007 (0.3015) |
| staircase | static_fifo_mvdr | 0.071 | 0.000 | 7.6377 | 3.7812 (0.5176) | 0.6524 (0.1222) |
| staircase | static_subtract | 0.058 | 0.000 | 7.0327 | 3.5769 (0.4611) | 0.7477 (0.1285) |
| staircase | fifo_mvdr_output_power | 21.572 | 21.004 | 7.6377 | 5.2404 (0.7214) | 1.5563 (0.2643) |
| staircase | fifo_mvdr_out_of_sample | 22.371 | 20.038 | 7.6377 | 6.8648 (0.9597) | 1.3415 (0.2348) |
| staircase | stage_b_out_of_sample | 47.340 | 45.541 | 7.6347 | 7.1468 (0.9887) | 1.7656 (0.3620) |
| staircase | subtract_out_of_sample | 14.805 | 13.293 | 7.0327 | 6.9439 (0.8955) | 1.3418 (0.2097) |
| triangular-ramp40s | stage_b_static | 47.510 | 45.625 | 7.3062 | 5.4132 (0.7695) | 1.4411 (0.3233) |
| triangular-ramp40s | static_fifo_mvdr | 0.102 | 0.012 | 7.2844 | 3.2434 (0.5048) | 0.7069 (0.1232) |
| triangular-ramp40s | static_subtract | 0.106 | 0.012 | 7.3004 | 3.1801 (0.5189) | 0.7800 (0.1474) |
| triangular-ramp40s | fifo_mvdr_output_power | 21.125 | 20.298 | 7.2844 | 4.5024 (0.7105) | 1.4256 (0.2815) |
| triangular-ramp40s | fifo_mvdr_out_of_sample | 23.909 | 20.895 | 7.2844 | 6.2422 (0.9456) | 1.4095 (0.2370) |
| triangular-ramp40s | stage_b_out_of_sample | 48.367 | 45.782 | 7.3062 | 6.8964 (0.9702) | 1.7433 (0.3992) |
| triangular-ramp40s | subtract_out_of_sample | 18.450 | 16.181 | 7.3004 | 6.0312 (0.9745) | 1.4189 (0.2526) |
| triangular-ramp10s | stage_b_static | 34.099 | 34.037 | 6.4583 | 4.7048 (0.7997) | 1.4205 (0.4439) |
| triangular-ramp10s | static_fifo_mvdr | 0.135 | 0.033 | 6.4555 | 2.7302 (0.5208) | 0.6282 (0.1664) |
| triangular-ramp10s | static_subtract | 0.132 | 0.024 | 6.5174 | 2.4855 (0.4734) | 0.6732 (0.1431) |
| triangular-ramp10s | fifo_mvdr_output_power | 20.154 | 20.686 | 6.4555 | 3.9205 (0.7092) | 1.3665 (0.3551) |
| triangular-ramp10s | fifo_mvdr_out_of_sample | 22.352 | 21.747 | 6.4555 | 5.0013 (0.8780) | 1.2955 (0.3261) |
| triangular-ramp10s | stage_b_out_of_sample | 36.092 | 34.992 | 6.4583 | 5.8044 (0.9717) | 1.7347 (0.5395) |
| triangular-ramp10s | subtract_out_of_sample | 19.240 | 18.914 | 6.5174 | 4.5658 (0.8752) | 1.2250 (0.2925) |

## La sustracción no mejora aun corregida, en los casos comprobados

Sin el bloque actual, la sustracción obtiene 60,285% y 49,771% en las dos rampas
sintéticas; Stage B estático obtiene 92,699% y 82,976%. En sub-01 obtiene 17,456%
desde 5 s frente a 42,832% de Stage B y 22,890% de FIFO fuera de muestra; queda
por debajo de Stage B en las tres condiciones disponibles. Desde 10 s ocurre
lo mismo. La afirmación se limita a estos registros y estas variantes, sin
extrapolar a todas las posibles sustracciones o generadores.

En sub-01, la sustracción fuera de muestra suma 3426 falsos positivos, frente a
1676 de FIFO fuera de muestra; también tiene más falsos negativos (38944 frente
a 36463). Su peor RoA no se explica solo por falsos positivos. No se ajustó ρ,
loading, α ni ningún umbral después de observarlo.

## Auditorías válidas: F5 confirmado y F2 refutado

- **F5 confirmado.** En ramp5s, el ángulo a 10 s ya es 40°; la fracción media en
  movimiento de 0–10 s es 0,504053. La ventana triangular «first hold» incluye
  movimiento después de los 5 s de calibración. Se conserva la
  [auditoría de fases](robust_mvdr_phase_audit.json).
- **F2 refutado en el ámbito auditado.** La razón señal/ruido de objetivos en
  bins desplazados es 30,922377; projected_change_nmse es 0,989930 y el ruido
  proyectado 0,034695. El cambio postural es grande y fiable en esa geometría;
  no se sostiene que los objetivos estén dominados por ruido. Esto no demuestra
  que Γ pueda representarlos o generalizar. Se conserva la
  [auditoría de fiabilidad](robust_mvdr_target_reliability.json).

## frozen.py correcto

Los métodos estáticos no cargan modelos, por lo que `frozen.py` no interviene en
su colapso. Se mantiene el cargador validado, sus controles de procedencia y los
pesos originales. No se corrigió ni se entrenó nada en ese módulo.

## Consecuencias

- Las cuatro reglas de decisión del experimento original no son interpretables
  como pruebas del mecanismo: los seis métodos, incluido el control, comparten
  el colapso. Sus números siguen siendo resultados históricos válidos del código
  registrado; no se reescriben para hacer funcionar la hipótesis.
- Se retira robust_mvdr como propuesta de mejora. La crítica a la mezcla α
  tampoco se sostiene en el pipeline realista comprobado.
- F1 («el déficit del Stage B original se debe a autocancelación») pasa a
  hipótesis no confirmada. El defecto del nuevo decodificador no demuestra F1.
- El estudio original no representó el FIFO corto con detector; D=48 en el test
  extremo a extremo ocultó el fallo que sí se observa con D=512. La magnitud
  sintética es sensible al registro.
- Se priorizan la verosimilitud de disparos (83,51% en resultados existentes)
  y el seguimiento de plantilla con objetivos fiables. Stage B fuera de muestra
  queda como mejora opcional pequeña, no confirmada en datos nuevos.

## Qué no hacer

- No reejecutar robust_mvdr corregido ni Stage B fuera de muestra sobre sub-02
  a sub-05: ya se reutilizaron seis veces y no constituyen un test nuevo.
- No reentrenar Γ con el argumento de que sus objetivos están dominados por ruido.
- No modificar las predicciones, los resultados puntuados, el código registrado
  ni los parámetros para mejorar una cifra.

## Ejecución interrumpida y motivo

La primera ejecución se detuvo siguiendo la regla original: a 30 dB,
static_subtract dio 0,921290 de RoA, por debajo del control FIFO (0,932829),
contrariamente al orden comunicado inicialmente. Se interrumpió la cuarta
condición antes de escribir el JSON final; no se ejecutó FDSI ni se hizo commit.
Las dos equivalencias y la regresión D=512 pasaron entonces (3 pruebas en
33,98 s). Ese registro se conservó en `.git/fifo-audit-interrupted.md`.

La aclaración posterior descartó el orden y la magnitud de RoA como referencias:
el montaje sintético independiente era distinto, aunque el mecanismo coincidía.
Se reanudó conservando las nueve definiciones AST del generador, decodificador
y métricas, sin aproximarlas a ningún resultado. La sensibilidad y las tablas
por unidad se añadieron solo al informe. Las nuevas condiciones de parada son
fallo de equivalencia, cambio de hashes o caso C en sub-01; no ocurrió ninguna.

## Reproducción, verificación y tiempo

Desde la raíz, en Windows y CPU, con el checkout externo fijado en
`a8f53736e297962670276bc123c137050b7056b6`:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
.venv-benchmarks/Scripts/python.exe -m benchmarks.fifo_audit.harness --output outputs/fifo_audit_reproduction --threads 4
.venv-benchmarks/Scripts/python.exe -m benchmarks.fifo_audit.fdsi_check --subject sub-01 --output outputs/fifo_audit_reproduction --threads 4
```

El directorio de salida debe ser nuevo: los CLI no sobrescriben un JSON existente.
La ejecución publicada usó `--output docs/benchmarks`. `fdsi_check` rechaza
cualquier sujeto distinto de sub-01 y omite toda condición sin caché existente.
El CLI no entrena ni prepara características de Stage B.

Tiempo medido de los CLI: sintético **110.462 s** y sub-01
**268.462 s**; suma **378.924 s**
(6 min 18,924 s), con cuatro hilos por proceso. Los informes registran sus fechas
UTC. El tiempo de desarrollo, documentación y CI es adicional. La suite de
benchmarks pasó **105 pruebas, con 2 omisiones CUDA, en 11,75 s**; Ruff check y
format aprobaron los 144 archivos Python.

Se verifican los 18 hashes del protocolo, los 87 hashes de fuentes y artefactos
protegidos y la lista, tamaños y fechas de los 1449 archivos previos de Stage B;
no se añade ni elimina ninguno. Los hashes previamente archivados de sus JSON,
pesos y fuentes también se conservan. El cuerpo del informe original se mantiene
byte a byte debajo del aviso añadido. Ningún JSON/CSV `robust_mvdr_*` se edita.

Desviaciones y límites: dos condiciones de sub-01 omitidas por falta de caché;
excepciones al intervalo literal de recall descritas arriba; C3 no establece
un umbral universal; la exclusión del bloque no recupera Stage B en FDSI. Las
pruebas completas coincidieron parcialmente en tiempo con el CLI de sub-01;
sus tiempos son de reloj de esta ejecución CPU, sin pretensión de latencia en vivo.

## Anexo: tablas por unidad, sin ajuste de curva

Las columnas son razón de amplitud y RoA (%) de los dos controles. Las unidades
están emparejadas dentro de cada caso; los pools cambian entre semillas. Las
semillas 3 de la sensibilidad son los mismos estacionarios principales y no se
duplican en este anexo.

### stationary_20db

| Unidad | Pool | Razón FIFO | RoA FIFO (%) | Razón Stage B | RoA Stage B (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 38 | 0.7003 | 94.886 | 0.8706 | 99.432 |
| 1 | 39 | 0.7071 | 96.386 | 0.8745 | 100.000 |
| 2 | 37 | 0.6968 | 92.632 | 0.8572 | 99.474 |
| 3 | 36 | 0.6783 | 96.429 | 0.8601 | 100.000 |
| 4 | 33 | 0.6597 | 85.202 | 0.8458 | 100.000 |
| 5 | 31 | 0.6670 | 92.887 | 0.8503 | 100.000 |
| 6 | 19 | 0.6639 | 71.331 | 0.8499 | 98.976 |
| 7 | 28 | 0.6440 | 73.228 | 0.8140 | 99.213 |

### stationary_30db

| Unidad | Pool | Razón FIFO | RoA FIFO (%) | Razón Stage B | RoA Stage B (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 38 | 0.6972 | 95.455 | 0.8689 | 99.432 |
| 1 | 39 | 0.7042 | 96.386 | 0.8761 | 100.000 |
| 2 | 37 | 0.7078 | 94.737 | 0.8640 | 100.000 |
| 3 | 36 | 0.7024 | 97.449 | 0.8797 | 100.000 |
| 4 | 33 | 0.6869 | 94.170 | 0.8689 | 100.000 |
| 5 | 31 | 0.6808 | 90.795 | 0.8695 | 99.582 |
| 6 | 9 | 0.6998 | 91.049 | 0.8666 | 99.691 |
| 7 | 32 | 0.6672 | 86.222 | 0.8571 | 100.000 |

### ramp_094

| Unidad | Pool | Razón FIFO | RoA FIFO (%) | Razón Stage B | RoA Stage B (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 38 | 0.4795 | 18.792 | 0.7378 | 99.329 |
| 1 | 39 | 0.6547 | 100.000 | 0.8584 | 100.000 |
| 2 | 37 | 0.4622 | 21.875 | 0.7154 | 99.375 |
| 3 | 36 | 0.6403 | 98.193 | 0.8232 | 99.398 |
| 4 | 33 | 0.1355 | 15.789 | 0.1783 | 96.842 |
| 5 | 31 | 0.0531 | 15.920 | 0.0788 | 60.697 |
| 6 | 19 | 0.5705 | 46.988 | 0.7890 | 97.992 |
| 7 | 28 | 0.1341 | 6.512 | 0.1706 | 87.963 |

### ramp_091

| Unidad | Pool | Razón FIFO | RoA FIFO (%) | Razón Stage B | RoA Stage B (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 38 | 0.4021 | 14.094 | 0.6804 | 100.000 |
| 1 | 39 | 0.6201 | 100.000 | 0.8297 | 100.000 |
| 2 | 37 | 0.4270 | 14.375 | 0.7002 | 99.375 |
| 3 | 36 | 0.6069 | 98.193 | 0.7998 | 100.000 |
| 4 | 33 | 0.0866 | 15.263 | 0.1184 | 78.947 |
| 5 | 31 | 0.0422 | 12.935 | 0.1024 | 37.313 |
| 6 | 19 | 0.5429 | 28.112 | 0.7638 | 96.787 |
| 7 | 28 | 0.0994 | 5.581 | 0.1356 | 51.389 |

### stationary_20db_seed4

| Unidad | Pool | Razón FIFO | RoA FIFO (%) | Razón Stage B | RoA Stage B (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 33 | 0.6968 | 95.982 | 0.8658 | 100.000 |
| 1 | 34 | 0.6833 | 92.453 | 0.8634 | 99.528 |
| 2 | 32 | 0.6845 | 91.342 | 0.8603 | 98.701 |
| 3 | 30 | 0.6926 | 93.033 | 0.8734 | 98.770 |
| 4 | 35 | 0.6784 | 82.692 | 0.8625 | 100.000 |
| 5 | 39 | 0.6253 | 50.299 | 0.7966 | 99.401 |
| 6 | 23 | 0.6692 | 94.326 | 0.8449 | 99.645 |
| 7 | 38 | 0.6313 | 68.715 | 0.8138 | 98.324 |

### stationary_20db_seed5

| Unidad | Pool | Razón FIFO | RoA FIFO (%) | Razón Stage B | RoA Stage B (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 32 | 0.6852 | 93.913 | 0.8633 | 99.130 |
| 1 | 39 | 0.6719 | 88.166 | 0.8610 | 98.817 |
| 2 | 28 | 0.6818 | 91.732 | 0.8631 | 99.213 |
| 3 | 38 | 0.6550 | 83.708 | 0.8459 | 99.438 |
| 4 | 35 | 0.6651 | 78.218 | 0.8397 | 99.010 |
| 5 | 37 | 0.6572 | 86.316 | 0.8342 | 99.474 |
| 6 | 27 | 0.6626 | 91.221 | 0.8504 | 100.000 |
| 7 | 30 | 0.6627 | 79.835 | 0.8385 | 98.765 |

### stationary_30db_seed4

| Unidad | Pool | Razón FIFO | RoA FIFO (%) | Razón Stage B | RoA Stage B (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 33 | 0.7121 | 93.304 | 0.8752 | 100.000 |
| 1 | 34 | 0.6983 | 97.170 | 0.8781 | 100.000 |
| 2 | 35 | 0.6846 | 95.673 | 0.8696 | 100.000 |
| 3 | 32 | 0.6979 | 93.506 | 0.8634 | 98.701 |
| 4 | 30 | 0.7009 | 94.262 | 0.8773 | 98.770 |
| 5 | 39 | 0.6397 | 73.054 | 0.8283 | 100.000 |
| 6 | 23 | 0.6892 | 95.390 | 0.8655 | 99.645 |
| 7 | 29 | 0.6828 | 86.992 | 0.8638 | 99.187 |

### stationary_30db_seed5

| Unidad | Pool | Razón FIFO | RoA FIFO (%) | Razón Stage B | RoA Stage B (%) |
| --- | --- | --- | --- | --- | --- |
| 0 | 39 | 0.6671 | 82.840 | 0.8644 | 98.817 |
| 1 | 32 | 0.7055 | 96.957 | 0.8768 | 99.565 |
| 2 | 37 | 0.6711 | 91.579 | 0.8541 | 99.474 |
| 3 | 28 | 0.7020 | 95.669 | 0.8715 | 99.213 |
| 4 | 38 | 0.6520 | 79.213 | 0.8470 | 100.000 |
| 5 | 35 | 0.6800 | 84.158 | 0.8517 | 99.010 |
| 6 | 27 | 0.6977 | 90.076 | 0.8788 | 99.618 |
| 7 | 30 | 0.6879 | 88.477 | 0.8607 | 99.177 |
