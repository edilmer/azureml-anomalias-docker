# README — Detección de anomalías con Azure ML (Demo + Docker)

En este proyecto se propone y demuestra una solución de **detección de anomalías** (picos atípicos) usando **Azure Machine Learning** para entrenar y publicar un modelo, y una **aplicación web** en Docker para consumir el endpoint, graficar y exportar resultados.

---

## ⚙️ Requisitos previos

- **Azure**: suscripción activa (puede ser Azure for Students).
- **Azure Machine Learning**: un **Workspace** creado.
- **Modelo publicado** como **Managed Online Endpoint** (tienes un endpoint como  
  `https://anomalias-endpoint.<region>.inference.ml.azure.com/score` y una **clave**).
- **Docker** instalado (Windows/Mac/Linux).
- (Opcional) **Python 3.11+** si quieres correr sin Docker.

---

## 🚀 Instalación y ejecución (rápida)

### 1) Clona o descarga este repositorio (carpeta de la app)
Asegúrate de tener estos archivos:

```
app.py
requirements.txt
Dockerfile
templates/
  └─ index.html
```

### 2) Construye la imagen Docker
```bash
docker build -t anomaly-detection-app .
```

### 3) Ejecuta el contenedor
> Cambia los valores de `AML_ENDPOINT` y `AML_KEY` por los tuyos.

**Linux/macOS:**
```bash
docker run --name anomaly-app -p 5001:5000   -e AML_ENDPOINT="https://anomalias-endpoint.eastus2.inference.ml.azure.com/score"   -e AML_KEY="TU_CLAVE_DEL_ENDPOINT"   -e ANOM_SCORE_MIN=0.8   -e ATYPICAL_MODE=intersect   anomaly-detection-app
```

**Windows PowerShell (usa comillas simples para evitar escapes):**
```powershell
docker run --name anomaly-app -p 5001:5000 `
```

Abre el navegador en **http://localhost:5001**

### 4) Uso de la app
1. Sube un archivo **.csv / .xlsx / .data** con columnas tipo:
   - `Hourly_Date`, `VolCorrected` (o `VolUnCorrected`) y las **features** usadas por el endpoint:  
     `VolUnCorrected, Pressure, Temperature, MaxPressure, MinPressure, MaxFlow, MinFlow, MaxTemp, MinTemp, VolCorrected_lag1, VolCorrected_lag2, VolCorrected_lag24, VolCorrected_rollmean3, VolCorrected_rollmean6, VolCorrected_rollmean24`.
2. (Opcional) Escribe la **etiqueta de sitio** (ej. `103`).
3. Presiona **Procesar y graficar**.
4. Descarga las anomalías detectadas en **Excel** o **PDF** con los botones.

---

## 🔧 Variables de configuración

| Variable            | Descripción                                                                                     | Valor por defecto         |
|---------------------|-------------------------------------------------------------------------------------------------|---------------------------|
| `AML_ENDPOINT`      | URL del endpoint REST de Azure ML                                                               | *(obligatorio)*           |
| `AML_KEY`           | Clave (Bearer) del endpoint                                                                     | *(obligatorio)*           |
| `ANOM_SCORE_MIN`    | Umbral mínimo del **anomaly_score** del modelo para considerar una detección (0.0 = desactivar) | `0.8`                     |
| `ATYPICAL_MODE`     | Cómo marcar puntos: `intersect` (modelo ∧ IQR), `series` (solo IQR), `model` (solo modelo)      | `intersect`               |

---

## 🧹 Operaciones con Docker

```bash
# Ver contenedores
docker ps -a

# Parar y borrar el contenedor (si existe)
docker stop anomaly-app 2>/dev/null || true
docker rm   anomaly-app 2>/dev/null || true

# (Opcional) borrar imagen
docker rmi anomaly-detection-app 2>/dev/null || true

# Reconstruir limpio
docker build --no-cache -t anomaly-detection-app .

# Ejecutar nuevamente
docker run --name anomaly-app -p 5001:5000 anomaly-detection-app
```

---

# 📚 Desarrollo del proyecto

A continuación, la guía para documentar y presentar la solución,

## 1. Análisis de los requerimientos

### a) Empresa, necesidades, requerimientos y restricciones
**Empresa (ficticia/real):** *Distribuidora de gas “EnerGas”*  
**Necesidad:** detectar **picos anómalos** en el **volumen corregido** por sitio y hora, para:
- detectar fugas/fraudes, fallas de sensor y eventos operativos.
- alertar rápidamente, priorizando cuadrillas y evitando pérdidas.

**Requerimientos:**
- Entrenamiento con histórico multianual (min. 12 meses).
- Inferencia **near-real time** (minutos) vía endpoint REST.
- Disponibilidad ≥ 99%, trazabilidad y monitoreo de drift.
- Exportación de anomalías (CSV/XLSX/PDF) y visualización.
- Seguridad: autenticación por clave, red restringida.

**Restricciones:**
- Datasets con valores faltantes/cambios de sensor.
- Ventanas de mantenimiento → spikes esperados (no alertar).
- Presupuesto acotado (estudiantes / POC).

### b) Alternativas de solución
- **Estadística clásica** (control charts, z-score, Tukey IQR): simple, explica, pero sensible a estacionalidad.
- **ML no supervisado**:
  - **Isolation Forest** (actual): robusto, rápido, buen baseline.
  - **LOF / One-Class SVM**: sensibles a escala/parametrización.
  - **Autoencoders**: prometen, pero requieren más computo y curado.
- **Azure AutoML Anomaly Detection**: automatiza búsqueda de modelos, fácil de operar.

**Selección:** **Isolation Forest** + escalado + features de *lags/rolling* + **filtro atípico** vía **Tukey (IQR)** en la app para “picos”.

### c) Propuesta de pipeline de Azure Machine Learning
(Referencias de componentes:  
https://docs.microsoft.com/es-es/azure/machine-learning/component-reference/component-reference)

**Componentes/Jobs:**
1. **Ingesta** (Azure Data Lake/Blob Storage) → `mltable`.
2. **Preparación** (component): limpieza, *feature engineering* (lags, rolling).
3. **Entrenamiento** (component): IsolationForest + scaler, métricas, persistencia de modelo (`.pkl`).
4. **Registro de modelo** en **Model Registry**.
5. **Despliegue** a **Managed Online Endpoint** (Aci/Managed).
6. **Monitoreo**: Application Insights + Data Drift Monitor.
7. **Re-entrenamiento** (schedule semanal/mensual) con **Pipelines**.


### d) Cálculo aproximado de costos
Usa la calculadora: https://azure.microsoft.com/es-es/pricing/calculator/

- **Almacenamiento** (Blob/ADLS): 100–500 GB/mes → **5–15 USD/mes**.
- **Compute entrenamiento** (Aml compute Standard_DS11_v2): 10–20 h/mes → **15–35 USD/mes**.
- **Endpoint online** (Managed, 1 instancia Standard_F2s): **40–80 USD/mes**.
- **Application Insights** + **Log Analytics**: **5–20 USD/mes**.
> **POC** típico: **60–150 USD/mes** (según región/uso). Optimiza con *autoscaling* y horarios.

---

## 2. Propuesta de diseño (arquitectura)

### a) Diagrama (texto)
```
[Fuentes de datos (SCADA/CSV)] --> [Blob Storage/ADLS] --> [Azure ML Workspace]
                                              |                  |
                                              |          +------v--------+
                                              |          |  Data Prep    | (component)
                                              |          +------v--------+
                                              |                 |
                                              |          +------v--------+
                                              |          |  Train (IF)   | (component)
                                              |          +------v--------+
                                              |                 |
                                              |          [Model Registry]
                                              |                 |
                                              |          +------v--------+
                                              |          | Online Endpoint|
                                              |          +------v--------+
                                              |                 ^
                                              |                 |
                                 [Web App Docker (esta)]  -----+
```

### b) Flujo de trabajo
1. **Carga** de históricos al **Blob/ADLS**.
2. **Pipeline**: *prep → train → register*.
3. **Deploy** del modelo como **endpoint REST**.
4. **App web** consume el endpoint, grafica y exporta.
5. **Monitoreo** de logs / drift. **Retrain** periódico.

### c) Componentes (descripción)
- **Azure ML Workspace**: orquesta recursos y experimentos.
- **Compute Cluster**: ejecución de *jobs* de entrenamiento/prep.
- **DataStore/Blob**: datasets históricos.
- **Pipelines & Components**: pasos reproducibles (prep, train, eval).
- **Model Registry**: versionado de modelos.
- **Managed Online Endpoint**: inferencia REST gestionada.
- **Web App Docker**: interfaz de usuario para subir archivos, filtrar, graficar y exportar.

---

## 3. Implementación de un DEMO

### 3.1 Entrenamiento y despliegue en Azure ML (resumen)
1. Sube datos (Blob/ADLS).
2. Crea **job de preparación**: lags/rolling, limpieza.
3. Crea **job de entrenamiento**: IsolationForest + scaler (guarda `model.pkl`).
4. **Registra** el modelo → **Model Registry**.
5. **Despliega** en **Managed Online Endpoint** (obtén **URL** y **clave**).
6. Prueba el endpoint en el portal (pestaña **Probar**).

### 3.2 App web (este repositorio)
- **Docker build** y **run** (ver sección “Instalación y ejecución”).
- Subir CSV/XLSX/Data con `Hourly_Date`, `VolCorrected`/`VolUnCorrected` y features.
- (Opcional) **filtrar** por sitio y **rango de fechas**.
- Ver **gráficas** (completa + zoom) y **descargar** anomalías (Excel/PDF).

**Notas de la app:**
- **Ordena** por fecha y **elimina** filas inválidas (NaN en features) antes de inferir (como en Colab).
- Marca **solo datos atípicos** (modo configurable: `intersect`/`series`/`model`).

---

## 4. Presentación de la solución (15 min)

Estructura sugerida:
1. **Contexto y requerimientos** (3–4 min): empresa, problema, KPIs, restricciones.
2. **Diseño y arquitectura** (4–5 min): diagrama, componentes, flujo etc.
3. **Demo** (6–7 min):  
   - Datos de prueba → app web  
   - Gráficas (completa + zoom)  
   - Exportación Excel/PDF  
   - Monitoreo y futuras alertas
4. **Costos & siguientes pasos** (1–2 min): optimizaciones, autoscaling, retraining, integración con Power BI/Alertas.

---

## 🧩 Estructura del repo (sugerida)

```
.
├─ app.py
├─ requirements.txt
├─ Dockerfile
├─ templates/
│  └─ index.html
└─ README.md  ← (este)
```

---

## ✅ Checklist final

- [ ] Workspace de Azure ML creado.
- [ ] Modelo entrenado, registrado y **endpoint online** activo.
- [ ] `AML_ENDPOINT` y `AML_KEY` configurados al correr Docker.
- [ ] App accesible en `http://localhost:5001`.
- [ ] Gráficas correctas y descargas funcionando.
- [ ] Documento de **análisis/diseño** listo para presentar.
