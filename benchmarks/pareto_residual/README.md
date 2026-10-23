# Residual de filtros sobre Pareto: transferencia y fallos de seguimiento

Experimento posterior al NO-GO de Stage A. Implementa una corrección pequeña
sobre el sistema Pareto completo, con selección sin etiquetas y validación
separada por sujeto. El resultado está en el [resumen publicado](../../docs/benchmarks/pareto_residual_summary.json).

**Resultado ejecutado:** 81,6896% frente a 80,8885% de Pareto sobre los mismos
80 registros externos, con mejora en los cuatro sujetos. Rampa 5 s: +2,8224
puntos; 10 s: +0,7230; 15 dB: +1,0960. **NO-GO:** no alcanza +5 puntos en
rampas rápidas y termina con 501 frente a 511 instancias ≥90% en la fase final.
Ocho modelos y 240 archivos de predicciones quedan conservados;
no se reajustan con el test.

## Fundamento y cambio respecto al primer Stage A

Pareto conserva íntegros su blanqueamiento, acoplamiento y actualización SV/SD.
Para cada unidad calibrada, sus fuentes adaptativas y las del filtro inicial son
dos observaciones alineadas por la identidad establecida durante calibración:

`y_P = w_P,b^T x`, `y_C = w_C^T x`.

La salida corregida es

`y_H = y_P + a_phi(f_b,j) (y_C - y_P)`, con `|a_phi| <= 0,15`.

Por linealidad, equivale exactamente a aplicar el filtro
`w_H = w_P,b + a_phi(f_b,j) (w_C - w_P,b)` dentro de cada lote. El residual se
aplica a los filtros efectivos, después del blanqueamiento, para conservar
el circuito de Pareto. **No se presenta como otra ejecución del Stage A original
ni como una corrección aprendida de rango tres en el espacio de electrodos.**

El coeficiente positivo devuelve parte de la respuesta hacia la calibración;
el negativo permite extrapolar la dirección de cambio respecto a ella.
Si el filtro varía localmente de forma aproximadamente lineal con el movimiento,
esa extrapolación puede compensar una parte del retraso de adaptación. Al volver
a la postura de referencia, la vista de calibración puede aportar una unidad
que el seguimiento adaptativo haya perdido. Son hipótesis contrastables:
la aproximación local y la recuperación de unidades no están garantizadas.

Sobre la observación aumentada `(y_P, y_C)`, la transformación es

`T = [[I-D, D], [0,I]]`, `D=diag(a)`.

Entonces `||T-I||_2 = sqrt(2) max|a| <= 0,21214 < 1`,
`sigma_min(T) >= 0,78786` y `T` es invertible. La cota se verifica en pruebas.
Es una propiedad de la transformación aumentada, no una garantía de RoA, de
invertibilidad de una mezcla EMG no cuadrada ni de mejora fuera de muestra.
La afirmación es **condicional al coeficiente y al estado ya calculados**.
Como el coeficiente depende de las fuentes actuales, esta cota de matriz no
demuestra que el mapa no lineal completo, incluyendo el cálculo del coeficiente,
sea globalmente invertible o tenga el mismo límite para su Jacobiano.
Inicializar a cero la última capa reproduce exactamente Pareto.

La red contiene **97 parámetros**: 10 características → 8 unidades `tanh` → un
coeficiente acotado. Los mismos pesos se aplican a cada unidad y registro.
No hay identificadores de sujeto, número de unidad ni pesos por electrodo.
Los RMS y amplitudes se normalizan por la calibración propia; un promedio
entre unidades aporta contexto invariante a su orden. La salida es equivariante
a permutar las unidades y acepta cantidades distintas de unidades. La historia
causal está en los centroides y FIFO del detector, no en un embedding por sujeto.

**Entradas cinemáticas (aclaración de la revisión del 19 de octubre de 2026).**
Dos de las diez características son el **ángulo de muñeca medido**, `q = ángulo/40`,
y su velocidad normalizada, leídos de `*_angle.npz` en `data.py`. Esta corrección
es por tanto una variante con cinemática observada (tipo K), no una corrección
ciega: no es comparable sin más con Pareto, que no usa el ángulo. El rescate de
plantillas hereda estos pesos. La verosimilitud de disparos descarta
explícitamente esas dos columnas.

La arquitectura residual y la representación por conjuntos se apoyan en
[He et al., identidad residual](https://arxiv.org/abs/1603.05027) y
[Zaheer et al., Deep Sets](https://arxiv.org/abs/1703.06114). Esas publicaciones
justifican patrones de arquitectura; no prueban el éxito de esta propuesta en EMG.
El blanqueamiento, contraste y adaptación de referencia proceden de
[Mendez Guerra et al.](https://doi.org/10.1088/1741-2552/ad5ebf).

## Entrenamiento y control de sobreajuste

Se usan las fuentes publicadas de Pareto, generadas por su circuito adaptativo
completo, y las fuentes del comparador fijo. El código de preparación solo
extrae fuentes predichas, predicciones y correspondencias iniciales; no usa sus
RoA históricos como características ni objetivos. Es un replay exacto de una
rama base congelada: el residual nunca modifica retrospectivamente su estado.

El detector del residual sí actualiza centroides y FIFO con las fuentes
corregidas. Se ejecuta igual durante ajuste, validación e inferencia. La detección
dura y el estado se desacoplan del gradiente; se diferencia a través de las
fuentes actuales y la red compartida. Se actualizan pesos cada 32 lotes y se
evalúa pérdida cada cuatro. No se afirma diferenciación completa a través del
detector ni a través de la adaptación SV de la rama Pareto.

Objetivo fijado antes del test:

`L = 0,1 contraste + forma + 0,2 duplicación + 0,1 desplazamiento + 0,02 tamaño + 0,01 suavidad`.

- Contraste: error normalizado por calibración de `log(cosh)` en picos confiables,
  con la penalización original para unidades inactivas.
- Forma: consistencia de la respuesta temporal de la fuente alrededor de picos
  confiables con su respuesta de calibración, normalizada en energía e invariante
  al signo arbitrario de ICA. Es una
  plantilla de fuente, **no la trayectoria MUAP multicanal real**.
- Duplicación: correlación al cuadrado entre fuentes distintas.
- Desplazamiento, tamaño y suavidad: penalizan alejarse de Pareto y oscilar entre
  lotes. La calibración inicial de cinco segundos conserva corrección cero.

Las pérdidas de forma e identidad añaden restricciones al contraste; siguen
siendo proxies y pueden fallar. No hay selección por RoA ni una promesa teórica
de que menor pérdida implique mejores descargas identificadas.

Cuatro particiones externas reservan sucesivamente `sub-02..05`. El siguiente
sujeto, en orden cíclico, es validación; los otros tres entrenan. `sub-01`, usado
para ajustar el Pareto publicado, queda siempre en entrenamiento. Así se evita
presentarlo como sujeto desconocido para la rama base.

Entrenamiento: escalera, rampas 40/10 s, SNR 30/20 dB. Validación: esas mismas
condiciones a 25 dB, en otro sujeto. Evaluación: los veinte registros del sujeto
externo, incluidos **15 dB y rampas de 5/20 s**, ausentes del ajuste y selección.
La rampa de 5 s prueba velocidad más alta; la de 20 s, interpolación.

Se fijan dos semillas, cinco épocas máximas, AdamW, tasa 0,003 y decay 0,01.
La época cero es candidata. Se exige al menos 0,5% de mejora en el proxy de
validación y no empeorarlo más de 0,01 en ninguna condición; de lo contrario se
conserva la corrección cero. Se promedian los coeficientes de ambas semillas;
no se elige la semilla mirando el test. Todos los candidatos de las cuatro
particiones se congelan antes de abrir etiquetas para puntuar una sola.

Los cien registros ya se exploraron en el trabajo anterior. La separación
impide entrenar esta red con el sujeto externo, pero el estudio sigue siendo
**exploratorio**, con cuatro sujetos externos y una sola familia de simulación.
Una confirmación independiente requiere registros nuevos reservados.

## Brechas que se evalúan

El comparador principal usa **los mismos 80 registros y unidades externas**,
no el 79,25% calculado sobre otros cien registros. Se informa:

- RoA medio, mediana y porcentaje de unidades ≥90%, por sujeto, condición y SNR.
- Rampas rápidas: objetivo ≥+5 puntos tanto en 5 s como en 10 s.
- Escalera: ninguna pérdida de RoA.
- Seguimiento: RoA de primera fase, fase central y últimos cinco segundos;
  ninguna pérdida en la fase final ni en el número de instancias ≥90% al final.
- Aceptación adicional del usuario: mejora estricta del promedio global y
  ausencia de pérdida a 15 dB. Esas condiciones se registran en `acceptance.json`
  antes de puntuar los sujetos externos, sin alterar selección ni pesos.
- Precisión, recall, F1 y SIL por unidad, dos semillas separadas, Wilcoxon con
  Holm e intervalos bootstrap por sujeto.

Superar solo el promedio no da `go`. El umbral de +5 puntos es un criterio de
avance, no demuestra por sí solo que la brecha de movimientos rápidos haya
desaparecido; también se examinan errores y unidades perdidas. Recuperación de unidades no calibradas,
nuevos reclutamientos, otras anatomías y velocidades fuera de estos datos no
se acreditan con este experimento.

## Ejecución reproducible

Se utiliza `.venv-benchmarks` y los datos de las semanas 1–2. Para repetir sin
sobrescribir, define primero `$env:PARETO_RESIDUAL_OUTPUT = 'outputs/residual_repeat'`.

```powershell
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.pareto_residual.run --threads 2 prepare
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.pareto_residual.run register
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.assess_residual --declare
foreach ($testSubject in 'sub-02','sub-03','sub-04','sub-05') {
    foreach ($trialSeed in 0,1) {
        .\.venv-benchmarks\Scripts\python.exe -m benchmarks.pareto_residual.run --threads 2 train --test $testSubject --seed $trialSeed
    }
}
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.pareto_residual.run freeze
foreach ($testSubject in 'sub-02','sub-03','sub-04','sub-05') {
    .\.venv-benchmarks\Scripts\python.exe -m benchmarks.pareto_residual.run --threads 2 evaluate --test $testSubject
}
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.pareto_residual.run report
.\.venv-benchmarks\Scripts\python.exe -m benchmarks.assess_residual
.\.venv-benchmarks\Scripts\python.exe -m pytest -q benchmarks/pareto_residual/test_residual.py --basetemp=outputs/test_residual
```

Las predicciones se guardan antes de abrir las etiquetas. Los modelos y caches
quedan en `outputs/pareto_residual/`. El origen de cada fuente tiene hash; el
registro de protocolo conserva código, particiones y objetivos. Una corrección
cero debe reproducir exactamente las descargas publicadas en los cien registros.
Los tiempos del residual sobre fuentes precalculadas excluyen el coste de la
rama Pareto y no demuestran latencia completa en vivo.
