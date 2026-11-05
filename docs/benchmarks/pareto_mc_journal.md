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

## L3 — hipótesis previa a la búsqueda

La memoria deja de limitarse al filtro de calibración. Se insertan filtros
efectivos y centroides de otros estados solo con suficiente confianza local,
después de recuperar y procesar el lote actual; el bin de calibración continúa
inmutable. Reutilizar posturas visitadas podría mejorar los ciclos posteriores
sin reiniciar bruscamente los centroides vivos, que conservan su EMA.

El punto de partida es el C4 final de **L2**, con sus semillas 0/1/2 de ronda 3,
y C1 congelado. El protocolo de L3 identifica expresamente esos checkpoints;
no atribuye los pesos iniciales a L1. Se comprueba primero en EMG de entrenamiento
que el cálculo compartido reproduce bit a bit las trayectorias independientes
con inserciones en bins no iniciales y memoria activa, sin abrir etiquetas.

Se mantienen las nueve estructuras y los 45 ensayos. En L3 también se busca
el momento de bin [0,8;0,99], que ahora sí es activo. Se repiten los tres mejores
con las tres semillas y se elige la mediana. Después se agregan las trayectorias
selladas de L1, L2 y tres rondas nuevas de L3 para reentrenar C4 en GPU.
La generación de trayectorias y la medición final continúan en CPU; no se ha
portado el bucle de búsqueda a GPU.

Puertas finales sobre sub-04: RoA global ≥92,6580081443 % (L2 +1 pp) y
RoA de ramp5s en ciclos 2–8 ≥ciclo 1 +5 pp. La única alternativa si falla es
atenuar beta por el cuadrado de la confianza local; si también falla, se conserva
L2 y se detiene antes de L4. Todos los parámetros y decisiones se registran
antes de predecir; los 20 registros finales se sellan antes de puntuar.
Sub-05 permanece sin abrir durante desarrollo. Resultados L3 pendientes.

### L3 — controles previos

EMG real de entrenamiento sub-01 ramp5s 30 dB, primeros 30 s, tres cabezas
finales de L2 y tres configuraciones de memoria completa. Las trayectorias
independientes y compartidas coinciden bit a bit en disparos, fuentes, C1,
centroides, IQR, blanqueo, SV, FIFO y todos los filtros/centroides/confianzas
de memoria. Se poblaron 7/14/29 bins no iniciales, con memoria activa en
77,44/62,20/29,21 % de pares unidad/lote. No se abrieron etiquetas.
Protocolo y resultado en `C2_full_memory_identity_v1/`; controles superados.

Antes del lanzamiento de L3: 139 pruebas pasadas y 2 omitidas en benchmarks;
Ruff check y format limpios en 188 archivos. Las fuentes registradas de L2,
los 234 archivos históricos protegidos, los cuatro sellos de sus rollouts
y el hash de su checkpoint final se verificaron al cerrar ese nivel.

### L3 — pausa solicitada por el usuario (23 de octubre de 2026)

L3 se inició a las 09:49 CEST y se detuvo a las 09:51 CEST por petición
expresa del usuario. Se detuvieron el coordinador y sus cuatro procesos de
inferencia, incluida la redirección del entorno Windows. No quedan procesos
de L3 en ejecución. La búsqueda conserva sus nueve ensayos marcados RUNNING
y 36 WAITING para reanudarlos, sin clasificarlos como fallidos o completados.
Todavía no había registros de predicción completados ni puntuaciones de L3.
SQLite pasa su comprobación de integridad. Los controles de identidad de
memoria completa sí estaban terminados y superados.

Estado conservado en `docs/benchmarks/pareto_mc_l3_setup/pause.json` y en las
salidas locales; L2 sigue siendo el último nivel aceptado. La publicación y
fusión solicitadas después no reanudan L3. Solo se retomará por una nueva
instrucción del usuario, con el mismo directorio y las fuentes registradas.

## Diagnóstico previo a reanudar L3 — registro (2 de noviembre de 2026)

Rama `exp/pareto-mc-l2-ablation`, desde `main` (`0732f9c`). L3 sigue en pausa y
no se modifica ninguna fuente registrada de L1, L2 ni L3: solo se añaden
`l2_ablation.py` y sus pruebas, con salidas nuevas en `L2_ablation/` y
`L3_screen/`.

Motivo, a partir de artefactos ya sellados: en 36 de los 45 ensayos TPE de L2 la
memoria no mezcló ningún filtro, y sus cinco ensayos por estructura dieron
métricas idénticas. Con la cabeza de L1, activar la memoria subió el reposo de
95,4 % a 97,4 %, pero no el RoA global (90,90 % frente a 90,91 %). La ganancia
L1→L2 (+0,42 pp) llegó junto con tres rondas DAgger nuevas, sin ablación que
separe ambos efectos. La puerta de ciclos de L3 ya la cumple L2 (ciclos 2–8
85,01 % frente a ciclo 1 68,97 %).

**Hipótesis:** la ganancia L1→L2 se debe sobre todo al reentrenamiento de C4, no
a la memoria. Diseño factorial sobre los 20 registros de sub-04, ya usados como
validación: cabezas L2 de ronda 3 (semillas 0/1/2) × {sin memoria, solo
confianza a C4, solo mezcla de filtros, L2 completo}, más la cabeza L1 sin
memoria y con mezcla. Catorce trayectorias por registro con el cálculo común
compartido, verificado bit a bit en `C2_shared_identity_v1`.

**Control previo a abrir etiquetas:** L2 completo con semilla 2 debe reproducir
bit a bit los disparos, estados, memoria y centroides sellados de L2; la cabeza
L1 sin memoria, los de L1. Si no ocurre, los contrastes serán solo descriptivos.

**Reglas fijadas antes de ejecutar:** bootstrap pareado por clústeres
(condición × unidad motora real, 2000 réplicas, semilla 2026). La memoria se
acredita solo si el IC 95 % de su efecto total sobre el RoA global queda por
encima de cero. Si no se acredita y el efecto del reentrenamiento es mayor o
igual, la ganancia se atribuye sobre todo a C4 y la premisa de L3 queda marcada
como no respaldada por L2. Se informa si las cabezas L2 sin memoria cumplen el
reposo ≥97 %. No se selecciona ningún parámetro ni cambia el nivel aceptado.

**Cribado sin etiquetas:** (1) tasa de uso exacta de la memoria de L2 para las
nueve estructuras y umbrales de 0,50 a 0,95 en las 60 grabaciones de
entrenamiento, calculada sin rollout a partir de los estados de C1 y validada
contra las 45 trazas selladas de la búsqueda L2; (2) rollouts de memoria
completa (L3) en sub-01–03 a 30 y 20 dB, con la cabeza de búsqueda de L3,
umbrales 0,6 y 0,85, κ=0,16, beta_max=0,45 y momento 0,895. Una estructura se
marca inactiva si su tasa queda por debajo del 1 % en ambos umbrales.

**Propuesta de puerta, solo retrospectiva:** global ≥ nivel previo +1 pp con IC
por encima de cero; Δ(ciclos 2–8) ≥ Δ(ciclo 1); ciclos 6–8 no inferiores al
nivel previo. Se evalúa sobre L2 frente a L1 y no se aplica a L3 sin decisión
del usuario. Sub-05 permanece sin abrir. Resultados pendientes.

### Diagnóstico L2 — resultados (2 de noviembre de 2026)

Ejecución de 10:37 a 11:26 CEST, cuatro procesos, cuatro hilos Torch y OpenBLAS
uno. **Reproducción bit a bit superada** antes de abrir etiquetas: L2 completo
con semilla 2 y la cabeza L1 sin memoria coinciden en los 20 registros con
disparos, estados, memoria, centroides y recuentos sellados de L2 y L1.

El cálculo sin rollout de la tasa de uso coincide con las 45 trazas de la
búsqueda L2 en 5.439.195 pares unidad/lote (0 discrepancias). En las 60
grabaciones de entrenamiento solo se activan 8 bins con h=0,03 (2,0–5,7 % según
el umbral) y 8 bins con h=0,06 (hasta 5,7 %). Con 8 bins y h=0,12 solo hay uso
con umbral 0,5 (1,9 %), y 16 bins con h=0,03 no pasa del 0,05 %. Las otras cinco
estructuras no mezclan en ningún lote de entrenamiento con umbrales ≥0,5. Tres
de ellas no alcanzan 0,5 ni siquiera en la postura de referencia (masa máxima
0,19–0,36); las otras dos llegan a 0,60–0,61 solo en una ventana muy estrecha
alrededor de q=0.

Medias de las tres semillas L2 en sub-04 (porcentajes):

| Variante | Global | ramp5s | Reposo | Movimiento | Ciclo 1 | Ciclos 2–8 | Ciclos 6–8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cabeza L2, sin memoria | 91,08 | 85,62 | 94,99 | 90,42 | 69,53 | 86,02 | 87,91 |
| Solo confianza a C4 | 91,16 | 85,75 | 96,08 | 90,46 | 69,53 | 86,05 | 87,95 |
| Solo mezcla de filtros | 91,02 | 84,95 | 97,24 | 90,26 | 69,56 | 84,88 | 86,62 |
| L2 completo | 91,05 | 85,01 | 97,54 | 90,29 | 69,56 | 84,91 | 86,68 |
| Cabeza L1, sin memoria | 91,24 | 84,71 | 97,13 | 90,48 | 65,89 | 84,94 | 88,17 |

Efectos con IC 95 % del bootstrap registrado (57 clústeres):

- **Memoria total:** global −0,03 pp [−0,26; +0,23]; reposo +2,55 [+0,60; +5,50];
  ramp5s ciclos 2–8 −1,11 [−1,90; −0,42]; ciclos 6–8 −1,23 [−2,57; +0,07].
- **Mezcla de filtros:** reposo +1,86 [+0,34; +4,26]; ramp5s −0,71 [−1,40; −0,02];
  ciclos 2–8 −1,14 [−1,91; −0,46]. Con la cabeza L1, global −0,25 [−0,43; −0,08]
  y ciclos 6–8 −2,47 [−4,14; −1,00].
- **Confianza como entrada de C4:** global +0,06 [+0,02; +0,11]; reposo +0,69.
- **Reentrenamiento de C4** (L2 sin memoria frente a L1 sin memoria): global
  −0,16 [−0,96; +0,52]; ciclo 1 +3,64 [+1,88; +5,88]; recall +1,32 [+0,95; +1,74];
  precisión −1,66 [−2,51; −0,95]; reposo −2,14 [−3,97; −0,71].
- **L2 aceptado frente a L1:** global +0,42 [−0,03; +0,90]; ≥90 % +3,23
  [+0,77; +6,56]; ciclo 1 +3,08 [+0,64; +5,89]; ciclos 6–8 −1,51 [−3,27; −0,04].

Reglas preregistradas: **la memoria no se acredita** (su IC global incluye
cero). La regla de atribución al reentrenamiento tampoco se cumple: su
estimación global (−0,16) no supera a la de la memoria (−0,03). **Las cabezas L2
sin memoria no cumplen el reposo ≥97 %** (94,99 % de media): el criterio de
reposo de L2 depende de la mezcla.

Observación posterior, no preregistrada: L2 completo obtiene 91,20 / 90,29 /
91,66 % global con las semillas 0/1/2; su media (91,05 %) queda por debajo del
91,24 % de la cabeza L1. La ganancia aceptada de +0,42 pp corresponde a la
semilla seleccionada y es menor que la dispersión entre semillas (1,37 pp). Lo
que sí cambia de forma robusta es el punto de operación: más recall y mejor
primer ciclo con el reentrenamiento; menos precisión y reposo, que la mezcla de
calibración recupera a costa de los ciclos posteriores.

Puerta retrospectiva (L2 frente a L1): la puerta de ciclos registrada para L3 se
cumple de forma trivial, y las tres condiciones propuestas fallan (global +0,42
con IC que incluye cero; Δciclos 2–8 − Δciclo 1 = −3,0 pp; ciclos 6–8 −1,51 pp).
La propuesta discrimina; la registrada no.

Implicación para L3: su hipótesis es que mezclar filtros guardados mejora los
ciclos repetidos, pero en L2 la mezcla los empeora. No se cambia ningún nivel
aceptado; reanudar L3 tal como está registrado, modificarlo o sustituirlo
queda a decisión del usuario. Artefactos compactos en
`docs/benchmarks/pareto_mc_l2_ablation/`; predicciones, sellos y puntuaciones en
`outputs/pareto_mc/development_v2/L2_ablation/`. El cribado de memoria completa
de L3 sigue en curso.

### Uso de la salida de C1 — análisis posterior (2 de noviembre de 2026)

Análisis no preregistrado, de solo lectura, sobre los estados, eventos y
disparos sellados de L2 en sub-04, ya puntuados. No genera predicciones nuevas.

- **Posición:** q·40 sigue al ángulo medio por lote con r=0,968 (RMSE 3,15°;
  ramp5s 3,74°) y separa perfectamente reposo cercano (<5°) de posturas
  alejadas (≥20°): AUC 1,000.
- **Velocidad:** la entrada de C4 es |Δq|/0,1 s y no distingue movimiento de
  reposo (AUC 0,518; Spearman 0,07 con la velocidad real; mediana 7,9°/s en
  reposo y 8,2°/s en movimiento). La velocidad real no supera 8°/s. Con Δq sobre
  0,5/1/2 s el AUC sube a 0,60/0,68/0,78.
- **Varianza:** la desviación predicha no se relaciona con el error (Spearman
  −0,01). La atenuación de memoria por varianza actúa como una compuerta sin
  información. Con h=0,03 (1,2°) y bins de 16/32 (2,7°/1,3°), la memoria es más
  fina que la resolución de C1.
- **Uso en C4:** fijar la velocidad o la masa de memoria cambia como mucho el
  0,015 % de 7,04 millones de decisiones candidatas. C4 no recibe q.
- **Dónde están los errores:** RoA de L2 de 96,98 % cerca de la referencia
  frente a 86,73 % a ≥20°, y de 94,67 % en reposo frente a 88,84 % en
  movimiento.

Conclusión: la información fiable de C1, la postura, no llega a las decisiones
ni a la adaptación; lo que llega, la velocidad a 0,1 s y la varianza, es ruido.
Cualquier uso nuevo de q requiere un protocolo propio antes de ejecutar.

## P1 — postura de C1 como entrada de C4, lazo abierto (registro)

Pregunta, a petición del usuario: ¿mejora la postura ciega de C1 las decisiones
de C4? Es una prueba barata en lazo abierto, previa a cualquier nivel cerrado.
Módulo nuevo `posture_c4.py` con salidas en `P1_posture_c4/`; no se modifica
ninguna fuente registrada.

**Brazos.** `base22` usa las 22 entradas de L2. `posture24` añade q en el lote de
decisión y q − q(hace 2 s); ambas son causales y salen solo del EMG. Los datos
son los de la ronda final de C4 en L2: L1 abierto, rondas L1 1–3 y rondas L2 1–3
(sub-01–03). Las etiquetas usan desfases de calibración y la receta es AdamW
idéntica, con semillas 0/1/2, selección de época por BCE y TPE de umbral/fuerza
sobre los candidatos de la ronda 3 de sub-04, como en L2.

**Controles.** Antes de usar la postura se comprueba que la velocidad guardada
en cada candidato coincide con la de C1 en su lote: exacta en rollouts y con
tolerancia 1e-4 en las características abiertas recalculadas. La semilla 0 de
`base22` debe reproducir bit a bit la semilla 0 de L2; si lo hace, las semillas
1–2 reutilizan los checkpoints de L2, y si no, se reentrenan. La cabeza L2
aceptada, decidida en lazo abierto sobre sus propios candidatos, debe devolver
sus disparos sellados.

**Evaluación.** Las decisiones en lazo abierto se toman sobre los candidatos
sellados de la ejecución final de L2, que no se usan para entrenar ni para
seleccionar. Se sellan antes del evaluador original.

**Reglas fijadas antes de ejecutar:**
- Primaria: RoA global medio de las semillas, `posture24` − `base22`, con
  bootstrap pareado por clústeres (condición × unidad real, 2000 réplicas). La
  postura ayuda solo si el IC 95 % queda por encima de cero.
- De apoyo: BCE de validación menor en las tres semillas.
- Mecanismo: la ganancia a ≥20° supera a la ganancia cerca de la referencia.
- Solo si se cumplen la primaria y la de apoyo se propone un nivel cerrado
  registrado. No se despliega nada ni cambia el nivel aceptado.

Sesgo conocido: C1 está dentro de muestra en los sujetos de entrenamiento; en
sub-04 su q es fuera de muestra. Sub-05 permanece sin abrir.

### P1 — resultados (2 de noviembre de 2026)

**Controles superados.** Las 460 fuentes de candidatos coinciden con la velocidad
de C1 de su lote con discrepancia 0, también las características abiertas
recalculadas. `base22` con semilla 0 reproduce la semilla 0 de L2 bit a bit:
mismo checkpoint, historial y SHA-256. Su TPE elige el mismo ensayo 23,
parámetros, repeticiones y semilla 2. Por la regla registrada, las semillas
1–2 de `base22` reutilizan los checkpoints de L2. La cabeza L2 aceptada,
decidida en lazo abierto sobre sus candidatos finales, reproduce sus disparos
sellados con 0 muestras distintas en los 20 registros.

**Incidencias de ejecución, sin cambio de protocolo.** Con el cribado de L3 en
paralelo y 3,9 GB de RAM libre, la semilla 1 de `posture24` pasó más de 30 min
sin terminar una época: lectura por fallos de página a 24 MB/s. Se detuvo el
cribado con 18 de 30 registros completos y sin archivos huérfanos, para
reanudarlo después. La semilla 1 se reinició en paralelo con la 2; el
entrenamiento es determinista.

BCE de validación, mejor época: `base22` 0,051148 / 0,051640 / 0,051337;
`posture24` 0,050590 / 0,050473 / 0,050903 (−1,09 / −2,26 / −0,85 %).
TPE de `posture24`: umbral 0,4446 y fuerza 0,9089 (en `base22`, 0,5126 y 0,9711).

Decisiones en lazo abierto sobre los candidatos finales sellados de L2, con el
evaluador original. Medias de tres semillas (porcentajes):

| Brazo | Global | ramp5s | Reposo | Movimiento | Ciclo 1 | Ciclos 2–8 | <5° | ≥20° | Precisión | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base22` | 91,53 | 85,10 | 97,53 | 90,80 | 69,54 | 85,01 | 96,86 | 86,59 | 95,45 | 94,21 |
| `posture24` | 92,01 | 86,38 | 98,20 | 91,30 | 71,96 | 86,33 | 98,36 | 86,92 | 95,31 | 95,10 |

Diferencias con IC 95 % (57 clústeres): global **+0,49 [+0,08; +0,88]**; ramp5s
+1,28 [+0,21; +2,53]; ciclo 1 +2,41 [+1,13; +3,61]; ciclos 2–8 +1,32
[+0,11; +2,76]; reposo +0,67 [+0,16; +1,34]; movimiento +0,51 [+0,06; +0,93];
recall +0,89 [+0,57; +1,26]; precisión −0,14 [−0,56; +0,26]; ≥90 % −1,08
[−4,21; +1,90]. Cada semilla de `posture24` (92,14 / 92,02 / 91,88 %) supera a
cada semilla de `base22` (91,64 / 91,29 / 91,66 %). Las unidades perdidas no
cambian (3). Congelar la postura cambia el 0,16–0,23 % de las decisiones
candidatas, unas diez veces más que la velocidad.

Reglas preregistradas: **la postura mejora C4 en lazo abierto** (IC primario
por encima de cero) y la BCE baja en las tres semillas, así que se cumple la
condición para proponer un nivel cerrado. **El mecanismo previsto no se
cumple:** la ganancia a ≥20° (+0,33 [−0,23; +0,87]) es menor que cerca de la
referencia (+1,50 [+0,63; +2,57]). La postura no rescata las unidades que
desaparecen lejos de la referencia; mejora sobre todo las decisiones cerca del
reposo y en el primer ciclo rápido.

Límites: es lazo abierto, sin realimentación de las decisiones a la
adaptación. Es un único sujeto de validación, con umbral ajustado sobre otros
candidatos de los mismos registros. No hay evaluación externa. El nivel
aceptado sigue siendo L2; el nivel cerrado con postura requiere su propio
protocolo y decisión del usuario. Artefactos en
`docs/benchmarks/pareto_mc_p1_posture/`. El cribado de L3 se reanudó al
terminar P1.

### Cribado de memoria completa de L3 — resultados (2 de noviembre de 2026)

Rollouts sin etiquetas del bucle de L3 (escritura de memoria tras la
calibración) en los 30 registros de sub-01–03 a 30 y 20 dB, con la cabeza de
búsqueda de L3 (semilla 0 de L2), κ=0,16, beta_max=0,45 y momento 0,895, para
las nueve estructuras y umbrales 0,6 y 0,85. Se ejecutó de 11:36 a 20:28 con
una pausa durante el entrenamiento de P1; los 18 brazos tienen sus 30 registros
sellados y no se abrió ninguna etiqueta. Las fuentes registradas no cambiaron.

Tasa de mezcla: fracción de pares unidad/lote con beta > 0, calibración
incluida (allí la mezcla es cero). Porcentajes, media de los 30 registros:

| Estructura | Umbral 0,6 | Umbral 0,85 |
|---|---:|---:|
| 8 bins, h=0,03 | 92,80 | 76,23 |
| 8 bins, h=0,06 | 89,59 | 65,28 |
| 8 bins, h=0,12 | 78,88 | 54,40 |
| 16 bins, h=0,03 | 90,46 | 66,84 |
| 16 bins, h=0,06 | 77,65 | 51,67 |
| 16 bins, h=0,12 | 74,15 | 41,94 |
| 32 bins, h=0,03 | 81,87 | 55,67 |
| 32 bins, h=0,06 | 73,60 | 44,65 |
| 32 bins, h=0,12 | 69,18 | 36,82 |

Regla preregistrada: **ninguna estructura es inactiva**. Todas mezclan en todos
los registros; el mínimo por registro es 22,3 % (32 bins, h=0,12, umbral 0,85).
Casi todos los nodos reciben filtros: 7 de 7 nodos fuera de la referencia con
8 bins, unos 14 de 15 con 16 y 29 de 31 con 32. La mezcla es más frecuente con
menos bins, núcleos más estrechos y umbral bajo, y algo menor a 20 dB (hasta
7 puntos con umbral 0,85). El cribado no excluye ninguna estructura del espacio
de búsqueda de L3.

Lectura posterior, no preregistrada: con memoria completa el bucle mezcla
filtros en la mayoría de los lotes, frente al 0–6 % de la memoria de solo
calibración de L2. Esa mezcla parcial ya bajó los ciclos 2–8 de ramp5s en
L2 (−1,14 pp); que una mezcla casi permanente los mejore queda sin respaldo.
Además, la memoria guarda filtros que ya incluyen la mezcla, así que puede
reforzarse a sí misma (hipótesis). El cribado no mide el efecto sobre el RoA.
Reanudar L3, modificarlo o sustituirlo sigue a decisión del usuario.
Artefactos en `docs/benchmarks/pareto_mc_l3_screen/`; predicciones y sellos en
`outputs/pareto_mc/development_v2/L3_screen/`.

### Siguiente nivel: ¿se aprovecha la información de movimiento? (4 de noviembre de 2026)

Diagnóstico posterior, de solo lectura, sobre los candidatos finales sellados de L2
en sub-04 y las decisiones selladas de P1, ya puntuadas. No genera predicciones.

- **Cobertura y oráculo no informan aquí.** Hay 593 candidatos por unidad y
  segundo frente a 11,7 descargas reales; un tren desplazado 37 muestras queda
  cubierto el 99,95 % de las veces. Aceptar los candidatos etiquetados (97,75 %)
  no es un techo alcanzable.
- **C4 ordena bien, pero no lo bastante.** El AUC dentro de cada unidad es 0,985
  de mediana (0,983 a ≥20°, percentil 10 0,961). Con unos 50 candidatos por
  descarga, ese 1,5 % restante decide el RoA.
- **Los umbrales adaptativos rinden poco.** Con desplazamientos de logit elegidos
  con etiquetas (la base reproduce las decisiones de P1 con 0 diferencias),
  `posture24` semilla 0 pasa de 92,14 % a 93,03 % por unidad y a 93,74 % por
  unidad y bin de 5°.
- **Importa la velocidad, no la dirección.** Flexión 91,3 % frente a extensión
  91,4 %; a ≥25° el RoA baja de 95,7 % (ramp40s) a 70,6 % (ramp5s), señal del
  retraso de la adaptación.

Lectura: el límite está en el orden de los candidatos dentro de cada unidad. C4
juzga cada candidato solo, sin ver los que compiten en el mismo lote ni la
historia reciente de la unidad; el movimiento entra solo como q y su
desplazamiento. No se repiten Stage A ni Stage B (morfología MUAP y filtros
analíticos, NO-GO); lo que funcionó fue supervisar las decisiones.

**Familias generativas.** Un VAE o una GAN no tienen aquí un papel claro: no hay
escasez de datos etiquetados, el ruido no es el cuello de botella y el generador
de MUAP o filtros ya falló en Stage B. Se prueba en cambio un modelo generativo
del tren de descargas: difusión discreta enmascarada (flow matching discreto con
fuente de máscara) sobre las etiquetas conjuntas del lote y del lote anterior.

**Pilotos sin registrar** (semilla 0, 60 trayectorias de la ronda 3 de L2, tres
épocas; RoA rápido de validación en la ronda 3 de sub-04, el mejor de los
umbrales 0,35/0,45/0,55):

| Cabeza | BCE validación | RoA rápido | ramp5s |
|---|---:|---:|---:|
| Forma P1, 24→8→1 | 0,0538 | 90,04 | 82,5 |
| `wide24`, MLP 24→64→64→1 | 0,0515 | 90,96 | 84,0 |
| `set24`, atención en el lote | 0,0336 | 92,15 | 86,4 |
| `setrec24`, más GRU entre lotes | 0,0345 | 92,28 | 87,2 |
| `mdiff24`, primera versión | 0,0323 | 90,34 | 84,5 |
| `mdiff24`, rediseñada | 0,0310 | 91,49 | 86,0 |

La primera versión generativa guardaba el logit de cada candidato en el paso en
que se revelaba; esos logits no eran comparables entre pasos y la precisión cayó.
Se sustituyó antes del registro por una lectura en la que cada candidato ve las
etiquetas de los demás, nunca la suya, y las del lote anterior. Ningún brazo se
eliminó por el piloto. La validación oscila mucho entre épocas (`set24`: 0,042,
0,047, 0,034).

### P2 — cabezas C4 con contexto, registro (4 de noviembre de 2026)

Prueba en lazo abierto con los datos, etiquetas, 24 entradas, semillas, selección
de época y TPE de P1. Solo cambia la cabeza: `wide24` (control de capacidad,
bucle i.i.d. de P1, 10 épocas), `set24`, `setrec24` y `mdiff24` (tres épocas,
BPTT truncado con el estado GRU llevado entre bloques de hasta 64 lotes). Todas
parten del prior de Pareto. Decisiones sobre los candidatos finales sellados de
L2, selladas antes del evaluador original; P1 `posture24` es la referencia.
Ablaciones de `setrec24` y `mdiff24`: sin historia (contexto GRU a cero) y con la
postura congelada en la mediana del registro.

Reglas: una cabeza mejora C4 solo si el IC 95 % de su diferencia de RoA global
frente a `posture24` (media de semillas, bootstrap por condición × unidad real)
queda por encima de cero. La historia ayuda solo si `setrec24` supera a `set24` y
a su ablación sin historia; usa la postura solo si supera a su ablación con
postura congelada; el entrenamiento generativo ayuda solo si `mdiff24` supera a
`setrec24`. Se propone un nivel en lazo cerrado con la mejor cabeza que mejore
solo si su BCE de validación queda por debajo de la de `posture24` en las tres
semillas. Sesgo conocido: sub-04 es el único sujeto de validación y ya se usó en
P1 y en el diagnóstico anterior. Sub-05 sigue sin abrir.

### P2 — resultados (4 de noviembre de 2026)

**Controles superados.** Los 16 tests de `test_context_c4.py` pasan:
- prior de Pareto al inicio;
- causalidad ante perturbaciones futuras;
- invariancia al tamaño de bloque;
- independencia entre unidades;
- un candidato nunca ve su propia etiqueta.

Las 76 fuentes registradas no cambiaron, y tampoco las de P1, L2_ablation y
L3_screen. Las decisiones se sellaron antes del evaluador, y `score` comprobó
el sello. Los cuatro entrenamientos corrieron en paralelo en la RTX 4070, de
12:56 a 19:23 UTC. Con la GPU compartida, cada época tardó unos 3,7 min en
`wide24`, 35–38 min en `set24` y `setrec24`, y 19–57 min en `mdiff24`. La
cadena de selección, decisión y puntuación terminó a las 19:59.

BCE de validación en la mejor época:

| Brazo | Semilla 0 | Semilla 1 | Semilla 2 |
|---|---:|---:|---:|
| `wide24` | 0,0480 | 0,0473 | 0,0479 |
| `set24` | 0,0215 | 0,0229 | 0,0207 |
| `setrec24` | 0,0188 | 0,0166 | 0,0182 |
| `mdiff24` | 0,0217 | 0,0213 | 0,0215 |
| `posture24` (P1) | 0,0506 | 0,0505 | 0,0509 |

La BCE de `mdiff24` se mide con su decodificación real por etiquetas, así que
es comparable con las demás. La validación oscila entre épocas: en `setrec24`,
la mejor época es la 1 en las semillas 0 y 2.

Umbral y fuerza elegidos por TPE:

| Brazo | Umbral | Fuerza |
|---|---:|---:|
| `wide24` | 0,386 | 0,836 |
| `set24` | 0,447 | 0,711 |
| `setrec24` | 0,377 | 0,697 |
| `mdiff24` | 0,602 | 0,655 |

Decisiones en lazo abierto sobre los candidatos finales sellados de L2, con el
evaluador original. Medias de tres semillas (porcentajes; «perdidas» cuenta
unidades):

| Brazo | Global | ramp5s | ≥90 % | Precisión | Recall | Movimiento | Pausa lejos | ≥20° | Ciclo 1 | Perdidas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base22` (P1) | 91,53 | 85,10 | 74,73 | 95,45 | 94,21 | 90,80 | 90,80 | 86,59 | 69,54 | 3,0 |
| `posture24` (P1) | 92,01 | 86,38 | 73,66 | 95,31 | 95,10 | 91,30 | 90,94 | 86,92 | 71,96 | 3,0 |
| `wide24` | 92,75 | 88,21 | 77,96 | 95,79 | 95,62 | 92,09 | 91,32 | 88,10 | 74,16 | 2,7 |
| `set24` | 93,06 | 88,23 | 78,23 | 95,68 | 95,99 | 92,45 | 91,46 | 88,74 | 74,28 | 2,0 |
| `setrec24` | 94,06 | 89,55 | 81,99 | 96,96 | 96,52 | 93,50 | 93,49 | 89,78 | 75,77 | 2,0 |
| `mdiff24` | 94,19 | 89,64 | 83,33 | 97,47 | 96,00 | 93,64 | 93,60 | 90,07 | 75,54 | 2,0 |

Diferencias de RoA global frente a `posture24`, con IC 95 % (57 clústeres):

| Brazo | Global | ramp5s | ≥20° | Movimiento | Pausa lejos |
|---|---|---|---|---|---|
| `wide24` | **+0,74 [+0,46; +1,05]** | +1,82 [+0,91; +2,88] | +1,18 [+0,64; +1,79] | — | — |
| `set24` | **+1,04 [+0,67; +1,47]** | +1,85 [+0,94; +2,86] | +1,82 [+1,16; +2,54] | — | — |
| `setrec24` | **+2,04 [+1,18; +3,18]** | +3,16 [+1,39; +5,45] | +2,85 [+1,86; +4,03] | +2,20 [+1,30; +3,40] | +2,55 [+0,06; +7,90] |
| `mdiff24` | **+2,18 [+1,37; +3,24]** | +3,25 [+1,73; +5,17] | +3,15 [+2,11; +4,33] | +2,34 [+1,47; +3,45] | +2,66 [+0,24; +7,63] |

Cerca de la referencia (<5°), `setrec24` no cambia: +0,04 [−0,17; +0,29].

En `setrec24`, el ciclo 1 de ramp5s sube +3,81 [+0,28; +7,35], la
precisión +1,65 [+0,35; +3,43], el recall +1,42 [+0,91; +1,99] y la fracción
de unidades ≥90 % +8,3 [+3,8; +13,5]. Se pierde una unidad menos (de 3 a 2).

**Reglas preregistradas:**
- **Las cuatro cabezas mejoran C4 en lazo abierto.** Su IC primario queda por
  encima de cero.
- **La historia ayuda.** `setrec24` supera a `set24` en +1,00 [+0,38; +1,94]
  (en pausa lejos +2,03 [+0,14; +6,74]) y a su ablación sin historia.
- **`setrec24` usa la postura.** Congelarla cuesta −12,5 [−15,7; −9,9].
- **El entrenamiento generativo no ayuda.** `mdiff24` frente a `setrec24` da
  +0,14 [−0,16; +0,45]. Cambia recall por precisión: precisión +0,51
  [+0,14; +0,91], recall −0,52 [−0,74; −0,33].
- **Se cumple la condición para proponer un nivel cerrado.** La mejor cabeza
  por estimación es `mdiff24`, y las tres semillas tienen BCE por debajo de
  `posture24`.

**Lectura de las ablaciones.** «Sin historia» y «postura congelada» se aplican
solo en inferencia, sobre modelos entrenados con esas entradas. Miden
dependencia, no lo que lograría un modelo entrenado sin ellas. Sin historia,
el RoA cae a 39–44 %, y cambia el 63–90 % de las descargas propias. Para el
valor real de la historia, la comparación correcta es `setrec24` frente a
`set24`. La dependencia de la postura varía mucho entre semillas. Congelarla
cambia el 6,7 / 15,0 / 46,7 % de las descargas en `setrec24` y el
6,7 / 7,3 / 26,5 % en `mdiff24`. En la semilla 2 de `setrec24`, el RoA cae a
65,2 %. Es una fragilidad para el lazo cerrado, donde q depende de las propias
decisiones a través de C1.

**Contrastes posteriores, no registrados** (mismo bootstrap; en
`docs/benchmarks/pareto_mc_p2_context/posthoc_vs_wide24.json`):
- `set24` frente a `wide24`: +0,30 [+0,06; +0,59];
- `setrec24` frente a `wide24`: +1,30 [+0,63; +2,30];
- `mdiff24` frente a `wide24`: +1,44 [+0,80; +2,31].

Reparto aproximado de los +2,0 pp de `setrec24` sobre P1, en medias de semilla
y no aditivo en sentido estricto:
- capacidad: unos +0,7;
- contexto del lote: unos +0,3;
- historia y trayectoria de postura: unos +1,0.

**Respuesta a la pregunta del nivel.** La información de movimiento estaba
infrautilizada. En P1, q como entrada por candidato apenas mejoraba lejos de la
referencia (+0,33 [−0,23; +0,87] a ≥20°). Con historia y trayectoria de
postura, la ganancia se concentra justo ahí:
- ≥20°: +2,85;
- movimiento: +2,20;
- pausa lejos: +2,55;
- ramp5s: +3,16.

Ese es el mecanismo que P1 esperaba y no encontró: compensar el retraso de la
adaptación cuando el movimiento es rápido o lejano. Parte de la mejora es
capacidad, porque la cabeza 24→8→1 de P1 se quedaba corta.

**Modelo generativo.** La difusión discreta enmascarada (flow matching discreto
con fuente de máscara) empata con su gemela discriminativa en RoA, con BCE
peor (0,0215 frente a 0,0179). Cuesta un 50 % más por época y tres pasadas por
lote al decodificar, y en lazo cerrado añade otra realimentación: las
etiquetas decididas del lote anterior. No se probaron un VAE ni una GAN, por
las razones del diagnóstico anterior.

Límites:
- es lazo abierto;
- sub-04 es el único sujeto de validación, ya usado en P1 y en el diagnóstico;
- no hay evaluación externa, y sub-05 sigue sin abrir.

El nivel aceptado sigue siendo L2. El nivel cerrado con cabeza de contexto
necesita su propio protocolo y la decisión del usuario. Por coste y por
número de realimentaciones, se propone `setrec24` como brazo principal y
`mdiff24` solo como secundario. Artefactos en
`docs/benchmarks/pareto_mc_p2_context/`.
