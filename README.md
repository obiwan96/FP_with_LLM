# MDAF-based Explainable Failure Prediction Framework for 5G/6G Networks

This repository contains the experimental codebase used in our research on **MDAF-based Explainable Failure Prediction for 5G/6G Network Functions**.  
The framework enables real-time multi-domain data collection, network failure prediction, and explainable AI (XAI) reasoning using **SHAP** and **LLMs**.

---

## 🧠 Overview

The system operates in a 5G/6G testbed that integrates:
- **OpenAirInterface (OAI)** – Core Network functions (AMF, SMF, UPF, etc.)
- **UERANSIM** – RAN emulation (gNodeB, UE)
- **Kubernetes** – Deployment and orchestration platform
- **Prometheus / Loki / InfluxDB** – Telemetry collection and time-series data management

The framework collects resource and log metrics from all domains, stores them in InfluxDB, and applies machine learning models for fault prediction and explainable reasoning.

---

## 🧩 Architecture
```
 ┌──────────────────────────────┐
 │      5G/6G Testbed           │
 │  (OAI Core, UERANSIM RAN)    │
 └──────────────┬───────────────┘
                │
                ▼
        [Kubernetes Pods]
          └─ POD_management.py
                │
                ▼
        Prometheus & Loki
          └─ Prome_helper.py
                │
                ▼
       [MDAF Collector Layer]
         └─ MDAF_Collector.py → InfluxDB
                │
                ▼
      [ML & XAI Analysis Layer]
        ├─ failure_prediction.py
        ├─ learning_helper.py
        └─ XAI.py
```
---

## 📁 Repository Structure

| File | Description |
|------|--------------|
| `secret.py` | Contains configuration and authentication details for communication with the testbed (Prometheus, Loki, InfluxDB, Kubernetes API, etc.). Users should fill in their own credentials and endpoints. |
| `slo-violation-check-ue` /| Contains python and dockerfile for *(SLO Violation Check UE)* |
| `POD_management.py` | Handles creation, deployment, and lifecycle management of **UE Pods** in the Kubernetes cluster. *(UE Manager)* |
| `Prome_helper.py` | Libraries to handle with  **Prometheus** and **InfluxDB**. |
| `MDAF_Collector.py` | Interfaces with **Prometheus** and **InfluxDB**, collects resource metrics and logs from RAN/Core/Application domains. *(Acts as the MDAF Collector)* |
| `learning_helper.py` | Provides helper functions for dataset preprocessing, normalization, and model training utilities. |
| `failure_prediction.py` | Runs the core **failure prediction experiments**, loads data from InfluxDB, trains ML models (LSTM, GRU, CNV-GRU, Attention-GRU), and evaluates results. *(MDAF-AI-Analyzer: Fault Prediction Module)*|
| `XAI.py` | Explain the prediction results and suggest the solution based on **SHAP+LLM**. *(MDAF-AI-Analyzer: Root Cause Analysis Module)*|
| `README.md` | Project documentation. |

---

## ⚙️ Environment Requirements

This repository assumes a pre-deployed **5G/6G testbed** with the following components:

- **OpenAirInterface (OAI)**: Core network
- **UERANSIM**: gNodeB / UE emulator
- **Kubernetes (v1.33+)**
- **Prometheus** and **Loki** for metrics/logs
- **InfluxDB** for time-series storage
- **Python 3.9+** with the following major dependencies:
  ```bash
  pip install pandas numpy torch influxdb_client prometheus_api_client kubernetes shap openai

## 🚀 Quick Start

1. Clone the repository
    ```bash
    git clone https://github.com/obiwan96/MDAF-Failure-Prediction.git
    cd MDAF-Failure-Prediction
    ```
1.  Set up environment
    Ensure OAI, UERANSIM, Prometheus, Loki, and InfluxDB are operational.
    Create a Kubernetes namespace for 5G test components (e.g., oai).

1. Edit secret configuration

    ```python
    # secret.py
    prometheus_ip = ''
    loki_ip = ''
    ollama_ip = ''
    InDB_info = {
        'url' : "http://localhost:8086",
        'token' : "",
        'org' : ""
    }
    error_alert_info = {
        'url':'',
        'body': {"chat_id":"", "text": 'Error occured!' },
        'headers' : { 'Content-Type': 'application/json' }
        }
    ```

1. Run data collector (MDAF-Collector)
    ```bash
    python3 MDAF_Collector.py
    ```

1. Run failure prediction experiments
    ```bash
    python3 failure_prediction.py
    ```

1. Perform SHAP + LLM-based explanation:
    ```bash
    python3 XAI.py
    ```
## 🧠 Explainable AI Integration

This project integrates:

 - **SHAP (SHapley Additive exPlanations)**: Feature 
 attribution for model interpretability.

 - **LLMs** (via Ollama or OpenAI API):
Converts SHAP results and logs into human-readable causal explanations.

Example output:
```markdown
Top contributing features:
1. AMF CPU Usage
2. SMF Session Drop
3. PDU Session Delay

LLM Explanation:
"High CPU load and frequent session drops indicate AMF overload.
Scaling or process isolation is recommended."
```

## 📊 Experimental Notes

 - The codebase includes modules for:

 - Data collection via Prometheus / Loki APIs

 - Real-time InfluxDB write and query operations

 - Time-series model training and evaluation

 - Explainability via SHAP + LLM

Several scripts contain Korean comments (auto-generated by GPT).
These will be cleaned and standardized in future releases.

## 📚 Citation

If you use this code in your research, please cite our paper:
```
Sukhyun Nam, Wonseok Choi, James Won-Ki Hong, "MDAF-based Explainable Failure Prediction for Network Functions in 5G/6G", 2026 IEEE/IFIP Network Operations and Management Symposium (NOMS 2026), Rome, Italy, 18-22 May, 2026. (Accepted to appear)
```


## 📬 Contact

For questions or collaboration:

 - POSTECH DPNM Lab

 - Website: http://dpnm.postech.ac.kr

 - Contact: obiwan96@postech.ac.kr
