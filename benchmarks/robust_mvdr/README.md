# Decodificación analítica robusta a la autocancelación

**Estado: retirado.** El cambio al régimen FIFO con normalización de respuesta
unitaria produce una pérdida de amplitud frente al umbral del detector; el
colapso se reproduce en sub-01. Excluir solo el bloque actual recupera parte de
la RoA, pero no se ha establecido una explicación exhaustiva del déficit restante.
La [corrección y auditoría CPU](../../docs/benchmarks/robust_mvdr_correction.md)
contiene las cifras, la sensibilidad y los límites de esta atribución.

El [informe original](../../docs/benchmarks/robust_mvdr_diagnostic.md) conserva
los resultados de 80 registros y 591 instancias: static_subtract obtuvo 5,22%.
Las cuatro reglas de decisión no son interpretables como prueba de F1, porque
el control comparte el defecto. No se cambió ningún código ejecutable registrado,
predicción ni resultado puntuado; no se reevalúan sub-02 a sub-05.

El resto documenta la propuesta histórica de la
[revisión independiente](../../docs/benchmarks/independent_review.md), retirada
tras la auditoría. No debe ejecutarse de nuevo sobre los sujetos externos.

## Hipótesis original, no confirmada

El filtro de respuesta unitaria propuesto, `w = R⁻¹h / (hᵀR⁻¹h)`, es un beamformer
MVDR/Capon. Stage B divide por `sqrt(hᵀR⁻¹h)`; la diferencia de amplitud importa.
`R` es la covarianza FIFO de los datos observados y **contiene la
unidad objetivo**. Cuando `h` no coincide con el MUAP real, MVDR suprime parte de
la propia unidad: es la autocancelación descrita por
[Cox, 1973](https://doi.org/10.1121/1.1913659) y tratada por el Capon robusto de
[Li, Stoica y Wang, 2003](https://doi.org/10.1109/TSP.2003.812831). La respuesta
a la unidad real cae por debajo del umbral calibrado en reposo y las demás unidades
siguen suprimidas. El resultado esperado es precisión alta y recall bajo, que es
el patrón medido en Stage B:

| Método Stage B | RoA | Precisión | Recall | Meseta final |
| --- | ---: | ---: | ---: | ---: |
| Pareto | 80,89% | 91,47% | 83,16% | 87,99% |
| Plantilla estática analítica | 45,56% | 91,39% | 47,85% | 97,50% |
| MoCA-B | 37,26% | 92,17% | 39,30% | 97,32% |

La meseta final, donde la plantilla vuelve a ser correcta, supera a Pareto. El
déficit aparece solo cuando la plantilla deja de coincidir.

## Evidencia sintética que fijó los parámetros

Se usó un generador *line-source* independiente de BioMime
([`synthetic/generator.py`](synthetic/generator.py)), con las leyes de longitud de
NeuroMotion, una rejilla de 10×10 y 20 dB. Se calibra en `s=1` y después se
mantienen 10 s en una longitud acortada `s₁`, sin movimiento, para excluir el
retardo de la covarianza. Se decodifican las 12 unidades más separables de unas 40
activas, con plantillas interpoladas entre el MUAP real de la postura (λ=0) y la
plantilla estática de calibración (λ=1). RoA con tolerancia de 2 ms; columnas:
umbral MVDR fijo 0,5 / umbral de dos medias sin etiquetas / mejor umbral.

| s₁, λ | `R` completa | Loading 0,1 | **ρ=0,5 + loading 0,03** | ρ=1 + loading 0,1 |
| --- | --- | --- | --- | --- |
| 0,94, λ=0 | 0,999 / 0,999 / 0,986 | 0,912 / 0,990 / 0,984 | 0,963 / 0,997 / 0,990 | 0,919 / 0,991 / 0,981 |
| 0,94, λ=0,5 | 0,864 / 0,906 / 0,943 | 0,833 / 0,923 / 0,947 | 0,869 / 0,968 / 0,971 | 0,843 / 0,918 / 0,949 |
| 0,94, λ=1 | 0,585 / 0,897 / 0,935 | 0,853 / 0,972 / 0,972 | 0,848 / 0,982 / 0,976 | 0,931 / 0,966 / 0,964 |
| 0,91, λ=0,5 | 0,674 / 0,722 / 0,858 | 0,728 / 0,744 / 0,861 | 0,773 / 0,790 / 0,876 | 0,759 / 0,758 / 0,864 |
| 0,91, λ=1 | 0,312 / 0,770 / 0,854 | 0,730 / 0,851 / 0,897 | 0,650 / 0,920 / 0,940 | 0,848 / 0,888 / 0,918 |

Con la covarianza **sin la unidad objetivo** (cota oráculo, no implementable),
la plantilla estática recupera 0,997 en `s₁=0,94` y 0,870 en `s₁=0,91` con el
umbral fijo. La separabilidad apenas se pierde (mejor umbral ≥0,85 en todas las
celdas): falla la amplitud de la respuesta frente a un umbral fijo. La mezcla α=0,5
del modelo de covarianza de Stage B deja el RoA con umbral fijo en 0,74–0,76 incluso
con plantilla exacta en este generador. El recall sigue en 0,99, así que la pérdida
son falsos positivos: el modelo de 12 unidades deja de suprimir a las no calibradas. El *blanking* de las propias detecciones apenas ayuda (0,677 frente a
0,585 en `s₁=0,94`, λ=1).

Se eligió **ρ=0,5 con loading 0,03** como método primario: en las nueve celdas
(`s₁` = 0,96, 0,94 y 0,91) es el mejor o queda a menos de 0,3 puntos del mejor con el
umbral de dos medias, el análogo más próximo a los centroides adaptativos del
detector. Con plantilla exacta cuesta 1,9–5,4 puntos con el umbral fijo y como
máximo 0,3 con el de dos medias. ρ=1 con loading 0,1
gana con umbral fijo ante desajustes grandes y se conserva como ablación
preregistrada. Los parámetros se fijaron solo con datos sintéticos, antes de abrir
cualquier registro FDSI en esta revisión.

## Filtro

Para cada unidad `j`, con su plantilla completa en los 73 retardos `C_j` (la fila
del retardo cero es exactamente `extended_template` de Stage B):

```text
A   = R + loading · media(eig(R)) · I
R_j = A − ρ · tasa_j · C_jᵀ C_j
w_j = R_j⁻¹ h_j / (h_jᵀ R_j⁻¹ h_j)
```

`R_j⁻¹` se obtiene con Woodbury sobre un único Cholesky de `A`. La matriz interna
`I − UᵀA⁻¹U` se recorta en sus autovalores a 0,05, de modo que la sustracción no
puede volver indefinida la covarianza aunque la plantilla sea errónea. La tasa se
inicializa con la calibración y se actualiza con el EMA de las descargas aceptadas
por el detector, como en Stage B.

## Métodos preregistrados

| Método | Estado | ρ | Loading | Propósito |
| --- | --- | ---: | ---: | --- |
| **`static_subtract`** (primario) | plantilla de calibración | 0,5 | 0,03 | ¿Basta corregir la autocancelación? |
| `static_subtract_strong` | plantilla de calibración | 1,0 | 0,1 | Ablación más agresiva |
| `static_loaded` | plantilla de calibración | 0 | 0,1 | Solo Capon robusto |
| `static_fifo_mvdr` | plantilla de calibración | 0 | 0,001 | Control: MVDR FIFO sin mezcla α |
| `moca_k_subtract` | morph congelado, ángulo medido | 0,5 | 0,03 | Valor del morph sin autocancelación |
| `moca_b_subtract` | morph y observador congelados | 0,5 | 0,03 | Igual, con estado ciego |

Se comparan con Pareto y, si están presentes y sin cambios, con las predicciones
selladas de Stage B (`stage_b_static_analytic`, `stage_b_moca_k`,
`stage_b_moca_b`) sobre las mismas 591 instancias.

Lectura declarada antes de predecir:

1. Si `static_subtract` supera a `stage_b_static_analytic` en recall y RoA,
   sobre todo en la fase angular `moving` y en ramp5s, la autocancelación era un
   cuello de botella real en FDSI.
2. Si `static_subtract ≈ static_fifo_mvdr`, la autocancelación no explica el
   déficit de FDSI y hay que revisar el diagnóstico.
3. `moca_*_subtract − static_subtract` mide el valor del morph congelado una vez
   eliminada la autocancelación. Si es ≤0, se interpreta junto con
   `target-reliability`.
4. No se espera superar a Pareto solo con este filtro. El siguiente paso sería
   colocar encima la [verosimilitud de disparos](../spike_likelihood/README.md) o
   un umbral adaptativo sin etiquetas.

**Limitación importante.** Los sujetos `sub-02`–`sub-05` ya se han puntuado en
cinco experimentos anteriores. Aunque aquí no se ajusta nada con ellos, el
resultado es **diagnóstico, no confirmatorio**. Una mejora debe confirmarse con
registros nuevos (otro generador o nuevas simulaciones NeuroMotion).

## Auditorías

- `phase-audit` lee solo los ángulos. Para cada registro informa el ángulo a 5 y
  10 s y la fracción en movimiento dentro de 0–10 s. La fase `first_hold` de las
  triangulares abarca 0–10 s (como `get_triangular_phases` de la referencia); si la
  fracción en movimiento es mayor que cero, esa etiqueta contiene movimiento.
  El evaluador añade además fases definidas por el ángulo: reposo de referencia
  tras la calibración, movimiento (>0,5°/s) y mesetas lejos de la referencia.
- `target-reliability` usa solo `sub-01`, sujeto de entrenamiento en todas las
  particiones de Stage B. Divide en mitades par/impar los objetivos STA por
  postura que entrenaron Γ y compara su ruido con el cambio respecto a la
  plantilla de calibración, en bruto y en la geometría de separación. Una razón
  cambio/ruido cercana a 1 significa que esos objetivos no contienen una señal de
  postura aprendible.

## Reproducción en CPU

Requisitos: el entorno `.venv-benchmarks` de los benchmarks anteriores, los datos
FDSI y las salidas de Stage B en `outputs/stage_b_main` (entradas en caché,
`protocol.json`, `locked.json` y los pesos `folds/*/seed*/{morph,observer}.pt`).
Sin esos pesos, define `ROBUST_MVDR_STATIC_ONLY=1` para **todas** las órdenes:
la elección forma parte de la especificación registrada.

### Compatibilidad con los pesos congelados

La publicación hizo opcional el hash del PDF en `Stage B register`, después de
congelar sus pesos. El cargador original rechaza ese cambio de procedencia.
`frozen.py` conserva el protocolo, las fuentes archivadas y los pesos anteriores:
verifica sus hashes, los dieciséis checkpoints y el código actual. La única
diferencia permitida es la expresión que hace opcional `guide_sha256` dentro de
`register`; el resto del árbol sintáctico debe ser idéntico. Cualquier cambio
numérico, de checkpoint o de protocolo se rechaza. Se instancian las clases
originales con sus parámetros exactos en CPU, sin entrenamiento. El nuevo
registro guarda toda esta procedencia, incluida la excepción de metadatos.

```powershell
.venv-benchmarks/Scripts/python.exe -m pytest benchmarks/robust_mvdr -q
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run phase-audit
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run target-reliability
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run register
foreach ($subject in 'sub-02','sub-03','sub-04','sub-05') {
  .venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run predict --subject $subject --threads 4
}
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run seal
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run score
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.run report
```

La salida es `outputs/robust_mvdr`; `ROBUST_MVDR_OUTPUT` permite otra. `register`
guarda los hashes del código propio y del código de Stage B del que depende, y
rechaza sobrescribir un protocolo. `predict` no abre disparos verdaderos externos.
`seal` exige las 80 × 6 predicciones y `score` verifica sus hashes antes de
puntuar; además comprueba que el evaluador reproduce el RoA guardado de Pareto.
Los sujetos pueden predecirse en procesos paralelos.

Coste: la síntesis con sustracción tarda unos 9 ms por lote de 100 ms con 8
unidades y D=512 (4 hilos, medido en Linux), además de los 8–11 ms del bucle de
Stage B. Stage B tardó una mediana de 51 s por registro con cuatro métodos
(1,1 h en total); aquí se esperan 1,5–3 min por registro con seis, es decir, unas
2–4 h para los 80 registros.

`report` escribe `report/summary.json` y `report/per_unit.csv`, con medias por
condición, SNR y fase (fija y angular), y diferencias frente a Pareto y a la
plantilla estática de Stage B. Copia los resultados pequeños con el prefijo
`docs/benchmarks/robust_mvdr_`.

El estudio sintético se reproduce sin datos externos:

```powershell
.venv-benchmarks/Scripts/python.exe -m benchmarks.robust_mvdr.synthetic.study all --output docs/benchmarks
```

Escribe `docs/benchmarks/robust_mvdr_synthetic_*.json`, que contienen las cifras de
las tablas anteriores. Las semillas están fijadas; con otro número de hilos BLAS el
redondeo puede cambiar una decisión aislada y mover alguna celda en ±0,001.
