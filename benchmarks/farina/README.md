# Referencias de Farina para descomposición de EMG durante movimiento

La reproducción FDSI de las **semanas 1–2** del informe de investigación tiene
un [registro independiente](../../docs/benchmarks/fdsi_weeks_1_2.md). El comando
`setup.ps1 -Fdsi` prepara sus dependencias; `reproduce_fdsi.py` comprueba los
100 registros y recalcula el RoA de los cuatro comparadores de la tabla 2 desde
los resultados publicados. El estado y los resultados de esa ejecución se
documentan en el registro, separados del tutorial descrito a continuación.
La ejecución del **18 de octubre de 2026** ha superado el criterio: **400 resultados
recalculados**, 768 instancias por método, ninguna diferencia con los RoA guardados
y los 28 valores de la tabla 2 dentro de ±1 punto (máxima diferencia: 0,0544).

Investigación y ejecución local: **16 de octubre de 2026**. El resultado alcanzado
es la reproducción del ejemplo público del método adaptativo. **La reproducción
completa de ambos artículos sigue pendiente**, por las limitaciones de acceso
descritas abajo. Las cifras observadas están en
[el registro de ejecución](../../docs/benchmarks/farina_tutorial_summary.json).

## Los dos métodos

| Método e implementación | Movimiento, datos y ruido | Acceso comprobado |
| --- | --- | --- |
| **HarmonICA / Quasi-Linear ICA**, Clarke et al., [arXiv:2406.19581v2](https://arxiv.org/html/2406.19581v2), 9 mayo 2026. Compensador invertible de bajo rango, rango 3 en la configuración principal, separación ICA y extracción secuencial. | NeuroMotion + BioMime mediante MUniverse. Barridos sinusoidales y triangulares de **±32,5° o ±65°**, 0,3 Hz, 50% MVC, **20/25 dB SNR**. Registros de 10 s a 2048 Hz; rejillas de 70 y 320 canales. Test: 21 pools, 5 sujetos simulados, 336 registros; además 80 de validación. | El [repositorio indicado en el artículo](https://github.com/AlexKClarke/harmonica_2026) devuelve **404**, tanto con Git como con la API de GitHub. No aparece en la lista pública del autor. No se obtuvo su código ni el benchmark específico modificado. |
| **Adaptive EMG decomposition**, Mendez Guerra, Barsakcioglu y Farina, [JNE 2024, 21 046023](https://doi.org/10.1088/1741-2552/ad5ebf), [código oficial](https://github.com/imendezguerra/adapt_decomp). Adaptación del blanqueamiento, vectores de separación y detección de picos, con criterios KL y curtosis. | El artículo simula **−40° a +40°**, por separado flexión y extensión desde 0°, pasos de 10°, mesetas de 30 s y rampas de 10 s (**1°/s**). 100 motoneuronas del FDS, excitación constante del 15%, rejilla 10×32, 2048 Hz. **Ruido gaussiano de media cero, SNR de 30 dB**. Diez remuestreos × dos condiciones = 20 contracciones. | Código y un ejemplo de flexión disponibles. El artículo indica que los datos completos se facilitan bajo petición a los autores. |

Son métodos de separación ciega/no supervisada que recuperan **descargas y trenes
de unidades motoras**. Eso no equivale automáticamente a recuperar una biblioteca
de formas de onda MUAP con error conocido. La versión v1 de HarmonICA (2024) es
un trabajo anterior al protocolo de bajo rango de v2: sus datos, objetivos y
resultados no deben intercambiarse.

## De dónde vienen las señales

En el protocolo adaptativo, **NeuroMotion** combina el modelo de reclutamiento y
frecuencia de descarga de Fuglevand, un modelo musculoesquelético de miembro
superior en **OpenSim**, y **BioMime**, un modelo generativo de conductor de volumen.
Los cambios de longitud de fibra informan la profundidad y velocidad de conducción
bajo una hipótesis de volumen muscular constante. BioMime genera/modifica los
MUAPs según la postura; las descargas se combinan con esos MUAPs para sintetizar
el EMG y después se añade ruido. Fuente: sección 2.1 del
[artículo completo en Imperial](https://spiral.imperial.ac.uk/server/api/core/bitstreams/875e0afe-7025-4350-aee2-6feee0672b71/content).

Esta reproducción usa las señales ruidosas distribuidas por los autores; **no
genera datos nuevos con MyoHand**. El valor 30 dB procede del protocolo y del README
del archivo: no se midió de nuevo, porque el ejemplo no incluye su señal limpia.
El código de descomposición no recibe las etiquetas de descargas reales durante
la adaptación; estas se emplean para evaluar.

El [archivo oficial del ejemplo](https://doi.org/10.5281/zenodo.22880910) contiene
190 s de EMG, 320 canales y los trenes reales simulados de 100 neuronas (56 activas).
La calibración MATLAB suministrada contiene **18 unidades**, usa los primeros
30 s y un factor de extensión de **3**. Se reutiliza esa calibración; no se ha
reproducido su estimación original en MATLAB. La variable `muaps` del HDF5 tiene
el primer eje vacío: la biblioteca completa de MUAPs simulados **no está incluida**.
Los MUAPs estimados que contiene el MAT no sustituyen esa verdad de referencia.

El mismo repositorio publica un **benchmark FDSI posterior**, diferente de los 20
registros originales: 5 sujetos × 5 condiciones × 4 SNR = 100 registros, 100
canales, muñeca de 0° a 40°, SNR **30/25/20/15 dB**, con señales limpias y ruidosas.
Incluye escalera y rampas triangulares de 40/20/10/5 s. Está disponible como
[datos, 10,35 GB](https://doi.org/10.5281/zenodo.22882346) y
[salidas precalculadas, 10,75 GB](https://doi.org/10.5281/zenodo.22882323).
Los cuadernos distinguen selección/tuning por RoA supervisado y por criterios
no supervisados; esos resultados no deben presentarse bajo la misma etiqueta.
No se descargaron ni se ejecutaron esos 100 casos en la reproducción del tutorial
del 16 de octubre. La fase FDSI posterior se documenta en el registro enlazado
al principio.

## Resultado ejecutado y configuración del comparador

Se fijó `adapt_decomp` a **8ffeb66ebd30bab9b5167a3fc485c2330a81da0f**, la versión
anterior a la reescritura de septiembre de 2026. El cuaderno de esa revisión
incluye los siguientes resultados guardados. Se ejecutó su algoritmo sin cambios,
con CPU, 4 hilos, semilla 0, `compute_loss=False` y ambos métodos desde copias
independientes de la misma calibración.

| RoA en los 190 s completos | Cuaderno: media / mediana | Ejecución local: media / mediana |
| --- | --- | --- |
| Adaptación completa | 90,25% / 98,58% | **90,2497% / 98,5839%** |
| Todo fijo (`adapt_wh=False`, `adapt_sv=False`, `adapt_sd=False`) | Rotulado como referencia: 44,31% / 39,37% | **37,7210% / 37,0300%** |
| Filtros fijos, detección adaptativa (`False`, `False`, `True`) | Comparador guardado: 44,31% / 39,37% | **44,3092% / 39,3668%** |

La adaptación coincide con las cifras redondeadas del cuaderno. El valor guardado
para el comparador **se reproduce al adaptar solo la detección**, aunque las celdas
actuales del cuaderno desactivan las tres banderas. Se aisló esa condición aplicando
el detector original a las proyecciones de la ejecución fija, sin recalcular ni
modificar los filtros. El [registro de esa comprobación](../../docs/benchmarks/farina_detection_ablation.json)
conserva la configuración y las diferencias numéricas. Esto reproduce las cifras
del comparador; no demuestra con certeza el estado histórico del kernel que las
guardó. La condición con todo fijo permanece separada y conserva su resultado.

También se comprobó que EMG del MAT y los primeros 30 s del HDF5 son idénticos,
y que emparejar desde los picos MATLAB o desde la ejecución fija inicial produce
las mismas 18 identidades y retardos. No se optimizó contra las etiquetas de la
ventana de evaluación para obtener esas coincidencias.

Para la comparación futura se calculan además métricas independientes en
**30–190 s**, excluyendo la calibración de la puntuación. Se mantienen las
identidades y retardos estimados en calibración, con tolerancia ±2 ms y coincidencias
uno a uno:

| Método | RoA medio | F1 medio |
| --- | --- | --- |
| Fijo | 27,2799% | 39,0309% |
| Adaptativo | 87,6349% | 91,2043% |

El cuaderno realinea cada par al evaluar el registro completo; la evaluación
independiente congela ese retardo en calibración. Por eso las dos evaluaciones
tienen nombres y archivos distintos. RoA = TP/(TP+FP+FN), mientras
F1 = 2TP/(2TP+FP+FN): **no son porcentajes intercambiables**.
El código legado calcula su media de centrado sobre todo el registro; esta
ejecución conserva ese comportamiento y no acredita un procesamiento causal
completo. Los tiempos locales tampoco sustituyen el benchmark de hardware del
artículo.

![RoA real por postura: mediana e intervalo intercuartílico](../../docs/images/farina-tutorial.png)

El artículo de 2024 reporta, sobre su conjunto completo, RoA mediano de
**98,10% frente a 35,86%**, precisión **99,92% frente a 62,61%** y sensibilidad
**98,21% frente a 36,46%** (adaptativo frente a fijo, sección 3.1). **Esas cifras
agregadas no se han reproducido aquí**. Tampoco se han reproducido NMSE, PNR,
resultados de los nueve participantes ni las tablas de HarmonICA v2.

## Repetir la ejecución en Windows

Desde la raíz del repositorio:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/farina/setup.ps1
.\.venv-benchmarks\Scripts\python.exe benchmarks/farina/download.py tutorial
.\.venv-benchmarks\Scripts\python.exe benchmarks/farina/run_tutorial.py --threads 4
.\.venv-benchmarks\Scripts\python.exe benchmarks/farina/run_detection_ablation.py
.\.venv-benchmarks\Scripts\python.exe benchmarks/farina/plot_results.py
.\.venv-benchmarks\Scripts\python.exe -m pytest -q benchmarks/farina/test_metrics.py
```

`setup.ps1` obtiene el código oficial en `external/farina/`, crea un checkout
legado separado y sincroniza `.venv-benchmarks` con el lock de dependencias de este
directorio. No cambia `pyproject.toml`, `uv.lock` ni el entorno de MyoHand. El lock
se validó con Python 3.12.14 en Windows y PyTorch 2.7.0+cpu; no es una copia exacta
del entorno histórico de los autores.

El descargador verifica el checksum publicado por Zenodo antes de extraer,
comprueba rutas, permite reanudar por bloques y conserva el ZIP. El ejemplo ocupa
1,64 GB de descarga y una cantidad similar descomprimida. Los archivos se
guardan en `data/benchmarks/farina/`. El código externo, datos y entornos quedan
**ignorados por Git**. Las licencias originales permanecen en sus checkouts; no se
ha incorporado el algoritmo externo a `hand_sim`.

Las salidas locales están en `outputs/farina/tutorial_legacy/`:

- `fixed.npz`, `adapted.npz`: picos, fuentes y tiempos calculados en esta máquina.
- `summary.json`: configuración, versiones, resultados observados y diferencias.
- `scores.csv`: TP/FP/FN, RoA, precisión, sensibilidad y F1 por unidad y ventana.
- `evaluation_reference.npz`: unidades y retardos congelados, etiquetas y ángulo.
- `comparison.png`: figura regenerable a partir de las puntuaciones reales.
- `fixed_filters_adaptive_detection.npz` y `detection_ablation.json`: comprobación
  separada del comparador guardado en el cuaderno.

El programa de ejecución sobrescribe sus salidas; usa `--output` para conservar
otra ejecución. La copia pequeña de `summary.json` y la figura incluidas en
`docs/` registran la ejecución descrita aquí, sin versionar las señales descargadas.

## Comparar una variante

Para esta primera prueba de **seguimiento de las mismas unidades calibradas**,
exporta `spikes`, una matriz binaria `[389120, 18]`, en un NPZ con el orden de
columnas de la calibración común. No ajustes hiperparámetros usando las etiquetas
de los 160 s evaluados. La CLI reutiliza el mismo evaluador para referencia y
variante:

```powershell
.\.venv-benchmarks\Scripts\python.exe benchmarks/farina/evaluate_predictions.py outputs/farina/tutorial_legacy/adapted.npz --output outputs/farina/adapted_scores.csv
.\.venv-benchmarks\Scripts\python.exe benchmarks/farina/evaluate_predictions.py outputs/mi_variante.npz --output outputs/farina/variant_scores.csv
```

Si la variante descubre unidades adicionales, se necesita otro protocolo de
emparejamiento y cobertura sobre todas las unidades reales activas: este contrato
de 18 columnas no mide descubrimiento ni permite atribuir resultados a las 100
neuronas del pool completo. Tampoco basta este ejemplo para concluir generalización
entre sujetos, músculos, velocidades o niveles de ruido.

Para completar la reproducción solicitada faltan el código y el conjunto específico
de HarmonICA v2, los 20 registros originales con sus calibraciones/protocolo completo
y, para verificar las formas de onda, la biblioteca MUAP simulada ausente.
No se contactó a los autores. [sources.json](sources.json) conserva los enlaces,
revisiones y comprobaciones de acceso.
