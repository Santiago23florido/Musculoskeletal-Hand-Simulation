# Pareto-MC — diario de desarrollo

## Declaración previa al entrenamiento (21 de octubre de 2026)

- Rama `feat/pareto-mc`, desde `review/robust-mvdr` (`0613abe`); PR #1 abierto.
- Entrenamiento: sub-01, sub-02, sub-03; validación: sub-04; sub-05 reservado.
- Cambio explícito frente a Stage B: se permiten las cinco condiciones y todos
  los SNR (30, 25, 20, 15 dB) de entrenamiento y validación.
- Inferencia: solo EMG crudo, calibración inicial y estado pasado. El ángulo
  entra exclusivamente en supervisión y evaluación angular.
- Pareto publicado utiliza preprocesado offline. C0 usará el preprocesado causal
  con estado del commit fijado y se contrastará con una nueva ejecución causal:
  referencia **Pareto-rerun**. No se atribuirán sus diferencias a C1–C4.
- Las puertas se deciden solo en validación. Máximo tres iteraciones por nivel,
  con como máximo un cambio estructural tras un fallo; si vuelve a fallar, parada.
- Resultados, protocolos y predicciones nuevos en `outputs/pareto_mc`; los
  experimentos anteriores permanecen intactos. Sin push ni PR sin autorización.

## L0 — hipótesis registrada

Un bucle propio que conserva las operaciones, el orden, la precisión y el estado
de Pareto reproduce su ejecución causal. Tres registros de sub-01: staircase,
ramp10s y ramp5s, 30 dB. Puerta: ≥99,9 % de muestras iguales y diferencia máxima
de RoA por unidad ≤0,01 puntos. Se guardan y sellan las predicciones antes de
abrir las etiquetas para comprobar RoA. Resultado pendiente de ejecución.

### L0 — resultado y decisión

Puerta **superada** en los tres registros: 100 % de muestras iguales y diferencia
máxima de RoA 0,0000 puntos, tanto frente al método original como al Pareto
publicado. RoA 88,7598 / 61,1219 / 51,0367 % (solo estas tres pruebas sub-01).
Medianas 38,0525 / 38,4831 / 38,1205 ms; p95 46,7016 / 47,2096 / 46,5453 ms.
El criterio final de latencia de 22 ms **no está satisfecho**.

Se conserva «Pareto-rerun», nombre preregistrado aunque también coincida con
las salidas publicadas. Corrección de la interpretación inicial: el filtro
offline de la referencia también usa `sosfilt` causal; la media global de
preparación se cancela por centrado de lote. El bucle nuevo evita esa dependencia.

Incidencias: primer protocolo conservado en `outputs/pareto_mc/L0` fallido por
`from_calibration` sin `cbss_config`; corregido usando el constructor explícito
con todos los tensores de calibración. Nueva ejecución completa e inmutable en
`outputs/pareto_mc/development_v1/L0`. El constructor original emite errores MKL
SLASWP al calcular estadísticas KL; no afecta al paso fijo y se conserva la
coincidencia exacta. Ruff se ejecuta desde `.venv-windows`: `uv run` encontró
una `.venv` local de otra plataforma y falló antes de ejecutar Ruff.

Fase 0 fallida: intento real de contenedor oficial, Docker Linux apagado;
sin Singularity/Apptainer en Ubuntu. Sin datos confirmatorios. Se conserva el
informe `outputs/pareto_mc/development_v1/phase0.json`. No se sustituye FDSI
por el generador analítico de FIFO.

## L1 — hipótesis previa al entrenamiento

El clasificador residual 22→8→1, reentrenado con candidatos agregados de sus
propias trayectorias, puede conservar la ganancia abierta al realimentar SV y
centroides. La calibración inicial permanece Pareto, los retardos de etiquetas
se estiman solo en sus cinco segundos y la tolerancia es 2 ms.
C1 se ajusta antes de C4 por NLL en los sujetos autorizados; su velocidad
estimada entra en C4. Memoria y controlador no modifican filtros en L1.
Puerta: RoA global de validación ≥ verosimilitud abierta y ramp5s ≥ abierta+1 pp.
No se procede a L2 si falla la puerta tras el único cambio estructural permitido.

### C1 — entrenamiento cerrado

30 épocas, AdamW lr=0,001, weight decay=0,01, gradiente limitado a uno, semilla 0,
GRU(120,24) y dos salidas. Checkpoint seleccionado en época 13 por NLL gaussiana
de sub-04: **−1,8778805**, RMSE angular descriptivo **3,2407°**. Se usaron los 60
registros de entrenamiento y 20 de validación declarados, todos los SNR/condiciones.
GPU real: RTX 4070 Laptop GPU. SHA256 de pesos:
`9826803063c82af10c306e1d4e445972f9564e55778ed3ef2d3a9a1f120ba3fa`.
El código de entrada del observador recibe solo 120 rasgos de EMG; el ángulo
está separado como objetivo. Se preservan protocolo, fuentes, historia y pesos.

### L0 — validación descriptiva

20 registros / 124 instancias de sub-04: RoA global 87,6526 %, ramp5s 73,8773 %,
66,9355 % de instancias ≥90 %, precisión 96,0571 %, recall 89,0306 %.
Fases angulares: reposo de referencia 92,4553 %, movimiento 86,7152 %,
mesetas alejadas 90,6892 %. Ramp5s: ciclo 1 58,9060 %, ciclos 2–8 72,1521 %.
Predicciones originales verificadas en L0, copiadas y selladas antes de puntuar;
se conservan amplitudes por segundo, trayectorias de centroides e instancias
perdidas por condición. No es una evaluación externa de los 80 registros.

### L1 — ejecución inicial sustituida antes de decidir ninguna puerta

La versión `development_v1` preparó los 80 conjuntos de candidatos y completó
la semilla 0 de C4 abierto (10 épocas, checkpoint con BCE de validación 0,0539051).
No se inspeccionó Jaccard/RoA de C4 ni ninguna puerta en esta versión.
Se detuvo durante la semilla 1 por dos motivos de implementación:

1. Las épocas repetían descompresión, carga de calibración y etiquetado. La
   nueva caché conserva exactamente los arrays y etiquetas, con hashes y acceso
   protegido por sujeto. Un test exige igualdad exacta y evita volver a abrir GT.
2. La prueba real de C4 identidad con **dos** hilos dio siete diferencias en
   ramp5s frente al C0 calculado con **cuatro** (99,9997287 % de muestras iguales).
   Se fija el mismo número de hilos, cuatro, para mantener la referencia numérica.
   Los otros dos controles habían dado cero diferencias. No se adjudica a C4 una
   diferencia numérica de BLAS provocada por cambiar los hilos.

También se corrigió, antes de evaluar la estructura alternativa, su guardia de
confianza de 0,9 a **0,8**: con el residual acotado a cuatro y prior −log(9),
un candidato nuevo no puede superar sigmoid(4−log(9))≈0,8585. El cambio se basa
en ese límite algebraico, no en métricas externas.

Se inicia un protocolo nuevo en `development_v2`, con 505 archivos de entrada
inmutables reutilizados por enlaces y verificados por SHA256. Se conservan
íntegros los resultados y snapshots de v1; no se reescriben sus archivos.
Se reutiliza solo la semilla 0 **completada**, cuyo algoritmo, datos y etiquetas
no cambian. Las semillas incompletas se entrenan de nuevo. Pendiente de puerta L1.

### L1 — resultado cerrado (22 de octubre de 2026)

La ejecución terminó el 21 de octubre a las 18:04 UTC y se revisó al reanudar.
La primera estructura, realimentación completa, **supera la puerta**. No procede
ejecutar la alternativa de realimentación restringida. Las tres rondas DAgger
están completas; se seleccionó la semilla 1 de la ronda 3, umbral 0,5900584852
y fuerza residual 0,9935271327. SHA256 del checkpoint:
`337a16d0d470804a33cbef2ee2fdd2179453e370bf8d9d40b428a8faf63e22ca`.

Validación sub-04, 20 registros y 124 instancias, evaluador original tras sello:

| Nivel | Global | ramp5s | Instancias ≥90 % | Precisión | Recall |
|---|---:|---:|---:|---:|---:|
| L0 Pareto-rerun | 87,6526 | 73,8773 | 66,9355 | 96,0571 | 89,0306 |
| C4 abierto | 90,3938 | 81,7689 | 70,1613 | 95,7008 | 92,9442 |
| L1 cerrado | 91,2360 | 84,7084 | 72,5806 | 96,3830 | 93,0115 |

Ganancias de L1 respecto al abierto: global +0,8422 pp, ramp5s +2,9396 pp
(la puerta exige ≥0 y ≥1, respectivamente). Fases angulares de L1: reposo de
referencia 97,1309 %, movimiento 90,4811 %, mesetas alejadas 89,9788 %.
Ramp5s: ciclo 1 65,8899 %, ciclos 2–8 84,9446 %. Unidades perdidas: staircase 2,
ramp10s 1, ramp20s 0, ramp40s 0, ramp5s 1. Memoria inactiva; controlador identidad.
Se conservan amplitudes verdaderas por segundo y centroides en las predicciones
y puntuaciones selladas; figura en `docs/images/pareto-mc-l1-diagnostics.png`.

Mediana 127,9194 ms/lote, p95 151,3212 ms, con cuatro registros simultáneos y
cuatro hilos Torch por registro. Esta medida incluye todo el bucle CPU y está
afectada por contención; no es una medida aislada. El objetivo ≤22 ms continúa
incumplido y requiere una medición final aislada. Las metas de exactitud se
alcanzan **solo en validación de desarrollo**; no hay todavía evaluación externa
de 80 registros ni confirmación nueva. Sub-05 permanece sin abrir.

Decisión: conservar L1 y continuar a L2 con instantánea de calibración únicamente.
Referencia de su puerta: reposo ≥97 % y movimiento ≥90,4810521752 %.
Artefactos compactos versionables en `docs/benchmarks/pareto_mc_l1/`;
los protocolos, pesos, ensayos y predicciones completos permanecen en
`outputs/pareto_mc/development_v2/`.

Verificación al cerrar L1: `pytest benchmarks -q`, 126 pasadas y 2 omitidas;
Ruff check y format limpios en 173 archivos. Fuentes registradas de L1 y todos
los archivos históricos protegidos verificados por SHA256 antes de exportar.

## L2 — hipótesis registrada antes de buscar en validación

Recuperar gradualmente el filtro de calibración al volver a un estado ciego
cercano a cero puede evitar la pérdida de unidades y de centroides en reposo.
Solo existe el bin inicial, inmutable y con confianza uno: no se insertan filtros
de movimiento. La recuperación ocurre después de C1 y antes de C4; el filtro
recuperado se proyecta al blanqueo actual y mezcla después del paso SV de Pareto
para afectar el siguiente lote. Se guardan también sus centroides, pero los
centroides vivos siguen la EMA de L1: no se restauran abruptamente.

La mezcla residual pasa por la QR y tiene límite relativo κ respecto a la
propuesta Pareto de ese lote. Este límite controla la contribución nueva de
memoria, no modifica el paso Pareto en la ruta identidad. La varianza ciega
atenúa beta por `exp(-0.5*var/h²)`; solo se mezcla con masa de recuperación
≥umbral. El diagnóstico de aciertos mide la fracción de pares unidad/lote con
mezcla no nula. Las decisiones de C4 siguen realimentando centroides y SV.

Búsqueda: las nueve combinaciones bins={8,16,32} × h={0,03;0,06;0,12}, con
45 ensayos completos (cinco por estructura), TPE continuo para umbral
[0,5;0,95], κ [0,02;0,3] y beta_max [0,1;0,8]. El momento de bin se fija en
0,95: es inactivo mientras no hay inserciones y no se simula una búsqueda
inefectiva. Se activa su búsqueda [0,8;0,99] en L3. El umbral de confianza se
aplica a la recuperación en L2 y, además, a la inserción en L3.

Cada ensayo predice todos los registros de sub-04 y sella las salidas antes
del evaluador original. Se buscan los parámetros con el C4 de semilla 0 de L1;
los tres mejores se repiten con sus semillas entrenadas 0/1/2 y se elige la
mediana del objetivo 0,5 global + 0,5 ramp5s, penalizando cinco veces la pérdida
de precisión más allá de un punto frente a Pareto. No se abren sujetos externos.

Después de elegir memoria se reentrena C4 desde identidad, tres semillas por
ronda y tres rondas DAgger nuevas. Se agregan candidatos de las cuatro rondas
de L1 y las rondas nuevas completadas. Se usa la misma receta AdamW y selección
por BCE y Jaccard de validación de L1. Solo la ejecución completa final decide
la puerta: reposo ≥97 % y movimiento ≥90,4810521752 %.

Si falla, el único cambio estructural adicional será multiplicar beta por el
cuadrado de la confianza local (tasa, separación e IQR), conservando la EMA de
centroides. Se registra el análisis del fallo antes de ejecutarlo. Si tampoco
pasa, se conserva L1 y se detiene la escalera. No se ejecutan L3/L4 por omisión.

Desviaciones explícitas: 45 ensayos continuos repartidos entre las nueve
estructuras, no 45 por estructura; momento fijo por ser inactivo en L2;
limitación del residual de memoria respecto a Pareto; centroides recuperados
conservados como diagnóstico, sin sustitución directa de los centroides vivos.
Sub-05 sigue reservado. Resultados L2 pendientes.

### C2 — controles previos completados

Tres registros reales de sub-01 (staircase/ramp10s/ramp5s, 30 dB), observador
entrenado y memoria con beta=0: **cero diferencias** respecto a los disparos
sellados de L0 en los tres. Además, las pruebas de integración verifican la
igualdad de fuentes, blanqueo, filtros, FIFO y centroides frente a L1, el límite
después de QR y la independencia respecto al EMG futuro con memoria activa.
Protocolo y salidas en `outputs/pareto_mc/development_v2/C2_identity_v1/`.

Perfil CPU del L1 retenido, un único registro sub-01 ramp5s 30 dB y cuatro
hilos: mediana **56,1218 ms**, p95 **360,8740 ms**. No se abrieron etiquetas.
El objetivo de 22 ms tampoco pasa en esta ejecución aislada; el p95 incluye
las pausas observadas de la máquina y no se descartan lotes lentos.

### L2 — reutilización exacta del cálculo común antes de la búsqueda

La búsqueda TPE usa ask/tell en cinco grupos de nueve ensayos. Dentro de cada
lote de EMG se calculan una vez el preprocesado, el blanqueo y C1, que dependen
solo de la calibración y del EMG, y se usan en las nueve trayectorias antes de
avanzar al lote siguiente. Filtros SV, centroides, FIFO del detector, decisiones
C4 y memoria son independientes. La optimización no mide latencia de despliegue;
las rondas DAgger y la puerta final ejecutan el bucle CPU completo habitual.

Prueba real previa, sub-01 ramp5s 30 dB, primeros 30 s: tres cabezas entrenadas,
tres estructuras y ganancias de memoria distintas. **Igualdad bit a bit** entre
ejecución independiente y cálculo compartido para disparos, fuentes, centroides,
confianza IQR, C1, memoria, filtros, blanqueo y FIFO. Se exigió memoria activa;
no se abrieron etiquetas. Resultado en `C2_shared_identity_v1/result.json`.
Una prueba adicional interrumpe TPE con nueve ensayos en marcha y verifica que
reanudar conserva los mismos parámetros y termina exactamente los 45 ensayos
(cinco por estructura), con selección por semilla mediana. No se descartan los
ensayos en marcha ni se reinicia la búsqueda con otra semilla.

Verificación de la implementación L2 antes de lanzar búsqueda: 137 pruebas
pasadas y 2 omitidas en `pytest benchmarks -q`; Ruff check y format limpios
en 185 archivos. Los resultados L1 siguen congelados y sin cambios.

### L2 — ejecución iniciada

22 de octubre, 09:03 CEST: búsqueda nativa Windows iniciada desde el commit
`3bb3a68`, cuatro procesos, Torch cuatro hilos por proceso y OpenBLAS uno.
Protocolo y estudio en `outputs/pareto_mc/development_v2/L2/standard/`;
salida en `l2_search_v1.stdout.log` y errores en `l2_search_v1.stderr.log`,
ambos dentro de `outputs/pareto_mc/development_v2/`. La búsqueda continúa en
segundo plano. Aún no existe una decisión de puerta L2. No se debe relanzar
otro proceso sobre el mismo estudio mientras siga activo.

### L2/standard — resultado automático

RoA global 91.6580 %, ramp5s 85.0519 %, precisión 95.6784 %. Reposo 97.5422 %, movimiento 90.9428 %. Puertas: {'reference_rest': True, 'moving': True}. Superada: True. Diagnósticos y sello en `C:\Users\USER\Documents\AppsPlayGround\Musculoskeletal-Hand-Simulation\outputs\pareto_mc\development_v2\L2\standard`.

### L2 — cierre revisado (23 de octubre de 2026)

Finalizó el 22 de octubre a las 17:57 CEST. La primera estructura **superó ambas
puertas**; no se ejecuta la alternativa. Se completaron 45 ensayos de memoria,
repetición de los tres mejores con tres semillas y las tres rondas nuevas de
DAgger, con tres semillas por ronda. El checkpoint final es ronda 3, semilla 2:
`b9f9df948b25670e03e84ab157645168f80d0781ae4ba9f06df58324d7018ad5`.

Validación sub-04: 20 registros, 124 instancias, predicciones selladas y evaluador
original. Todos los valores de exactitud de la tabla son porcentajes.

| Métrica | L1 | L2 |
|---|---:|---:|
| RoA global | 91,2360 | 91,6580 |
| RoA ramp5s | 84,7084 | 85,0519 |
| Instancias con RoA ≥90 % | 72,5806 | 75,8065 |
| Precisión | 96,3830 | 95,6784 |
| Recall | 93,0115 | 94,1446 |
| Reposo de referencia | 97,1309 | 97,5422 |
| Movimiento | 90,4811 | 90,9428 |
| Meseta alejada | 89,9788 | 91,0045 |
| Ramp5s ciclo 1 | 65,8899 | 68,9742 |
| Ramp5s ciclos 2–8 | 84,9446 | 85,0129 |

Puertas: reposo 97,5422 % ≥97 %; movimiento 90,9428 % ≥90,4811 %.
Global +0,4220 pp y ramp5s +0,3435 pp frente a L1. La precisión cae 0,7045 pp
frente a L1 y 0,3787 pp frente a Pareto-rerun: dentro del margen de un punto.
Unidades perdidas por condición: staircase 1, ramp10s 1, ramp20s 0, ramp40s 0,
ramp5s 1. La memoria modifica el filtro en el 4,2875 % de pares unidad/lote.

Configuración: 8 bins, h=0,03, confianza mínima 0,6597845, κ=0,1819790,
beta_max=0,6090293, momento inactivo 0,95, solo instantánea de calibración.
Umbral final de C4 0,5126006 y fuerza residual 0,9711128. Los centroides
continúan su EMA; no se restauran abruptamente desde memoria.

CPU: mediana **86,1557 ms**, p95 **105,5095 ms**, con cuatro registros simultáneos
y cuatro hilos Torch cada uno. Esta medición incluye todo el bucle; no es un
perfil aislado y **no satisface 22 ms**. Las condiciones de contención no permiten
atribuir la diferencia temporal frente a L1 a una mejora de implementación.

Se conservan protocolo, búsqueda, repeticiones y puerta en
`docs/benchmarks/pareto_mc_l2_standard/`, y diagnósticos completos en las salidas
selladas. Sigue siendo desarrollo exploratorio; sub-05 no se abrió y no existe
evaluación externa ni confirmación nueva. Decisión: conservar L2 y continuar
a L3. Su puerta exigirá global ≥92,6580081443 % y, en ramp5s, ciclos 2–8
al menos cinco puntos por encima del ciclo 1.
