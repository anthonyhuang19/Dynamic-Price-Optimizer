import logging
from prometheus_client import Summary, Counter, start_http_server
import time

# Initialize Prometheus metrics
REQUEST_COUNT = Counter('prediction_requests_total', 'Total number of prediction requests')
PREDICTION_LATENCY = Summary('prediction_latency_seconds', 'Latency of prediction requests')
ERROR_COUNT = Counter('prediction_error_total', 'Total number of prediction errors')

# Set up logging
logging.basicConfig(filename="data/monitoring/monitoring.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def log_request_metrics(input_features, prediction, latency):
    """
    Logs the input features, prediction result, and latency.
    """
    logging.info(f"Request: {input_features} | Prediction: {prediction} | Latency: {latency}s")

def log_error(error_message):
    """
    Logs the error message when a prediction fails.
    """
    logging.error(f"Error: {error_message}")

# Start Prometheus metrics server
start_http_server(8001)  # Exposes the Prometheus metrics at port 8001

# Timer decorator for latency monitoring
@PREDICTION_LATENCY.time()
def process_request(input_features):
    """
    Simulates a request processing.
    Here, you can put your prediction code (e.g., FastAPI endpoint logic).
    """
    start_time = time.time()
    try:
        # Simulate prediction processing
        prediction = 100  # Replace with actual prediction function call
        latency = time.time() - start_time
        REQUEST_COUNT.inc()  # Increment the request counter
        log_request_metrics(input_features, prediction, latency)
        return prediction
    except Exception as e:
        ERROR_COUNT.inc()  # Increment the error counter on failure
        log_error(str(e))
        raise

