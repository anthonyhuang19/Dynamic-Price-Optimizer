graph TD
    A[1. Data Sources<br/>(Raw Data)] --> B[2. Data Pipeline<br/>(src/data/)]
    B -- Processed Data: data.csv --> C[3. ML Model Development<br/>(src/model/)]
    C -- Trained Model & Scaler --> D[4. Model Serving<br/>(src/serving/)]
    D -- Predictions & Monitoring Data --> E[5. Monitoring<br/>(src/serving/monitor.py)]
    E -- (Feedback Loop / Retraining Triggers) --> C
    D -- User Interactions --> D
    D --> F[6. Deployment<br/>(Dockerfile)]

    subgraph Data Pipeline
        B1(load.py)
        B2(preprocessing.py)
        B3(feature.py)
        B4(eda.py)
        B -.- B1
        B -.- B2
        B -.- B3
        B -.- B4
    end

    subgraph ML Model Development
        C1(train_evaluation.py)
        C2(model.pkl, scaler.pkl)
        C3(MLflow: mlruns/)
        C4(src/model/images/)
        C -.- C1
        C -.- C2
        C -.- C3
        C -.- C4
    end

    subgraph Model Serving
        D1(predict.py)
        D2(app.py)
        D3(templates/, static/)
        D -.- D1
        D -.- D2
        D -.- D3
    end

    subgraph Monitoring
        E1(monitoring.log)
        E -.- E1
    end

    subgraph Deployment
        F1(Containerization)
        F2(Production Env.)
        F -.- F1
        F -.- F2
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bfb,stroke:#333,stroke-width:2px
    style D fill:#fbe,stroke:#333,stroke-width:2px
    style E fill:#ffc,stroke:#333,stroke-width:2px
    style F fill:#cfc,stroke:#333,stroke-width:2px
