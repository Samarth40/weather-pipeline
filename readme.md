# Weather Forecast AI Pipeline

**Built by Samarth Shinde**

Welcome to the Weather Forecast AI Pipeline! This project is an end-to-end Machine Learning Operations (MLOps) system designed to automate the collection of weather data, perform accurate predictions, monitor for model drift, and trigger automatic retraining. 

## Key Features

- **Automated Data Collection:** Scheduled tasks to pull both historical and daily forecast data, integrating seamlessly with Azure Cloud services.
- **Robust ML Model Service:** A Flask-based API for inference that actively detects drift and orchestrates model retraining.
- **Advanced Training Pipeline:** Utilizes Support Vector Regression (SVR) and MLflow to meticulously track experiments and model performance.
- **Interactive Web Dashboard:** A Streamlit application for visualizing weather predictions and monitoring model health in real-time.

## Architecture

This project is built using a cloud-native architecture heavily utilizing Azure services for scalability and reliability.

1. **Data Ingestion:** Scheduled Azure Functions pull data from the Open-Meteo API and store it in Azure Blob Storage.
2. **Event-Driven Processing:** Event Grid triggers Azure Container Instances whenever new data arrives.
3. **Model Management:** MLflow tracks all model experiments, parameters, and metrics.
4. **Serving & Visualization:** The Flask API serves predictions to our interactive Streamlit dashboard.

## Getting Started

### Prerequisites

Ensure you have Python installed. You can install all base dependencies by running:

```bash
pip install -r requirements.txt
```

You'll also need to navigate into each component's directory (`azure_function`, `ml_model`, `weather_app`) and install their specific `requirements.txt`.

### Local Execution

1. **Start Azure Functions:**
   ```bash
   cd azure_function
   func start
   ```

2. **Start the ML Model API:**
   ```bash
   cd ml_model
   python app.py
   ```

3. **Launch the Dashboard:**
   ```bash
   cd weather_app
   python streamlit_app.py
   ```

## CI/CD and Deployment

This project uses fully automated GitHub Actions for Continuous Integration and Continuous Deployment.
- **Azure Functions** are deployed automatically on commit.
- **The ML API** is built into a Docker container, pushed to Azure Container Registry (ACR), and deployed to Azure Container Instances (ACI).

## License & Attribution

Developed and maintained by **Samarth Shinde**. 
Feel free to explore the code, experiment with the pipeline, and contribute!