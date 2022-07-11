from typing import Tuple, Optional, Dict
import flwr as fl
from flwr.common.typing import Metrics
import tensorflow as tf


# def get_eval_fn(model):
#     """Return an evaluation function for server-side evaluation."""

#     # Load data and model here to avoid the overhead of doing it in `evaluate` itself
#     (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

#     # Use the last 5k training examples as a validation set
#     x_val, y_val = x_train[45000:50000], y_train[45000:50000]

#     # The `evaluate` function will be called after every round
#     def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, Dict]]:
#         model.set_weights(weights)  # Update model with the latest parameters
#         loss, accuracy = model.evaluate(x_val, y_val)
#         return loss, {"accuracy": accuracy}

#     return evaluate

# # Load and compile model for server-side parameter evaluation
# model = tf.keras.applications.NASNetMobile((32, 32, 3), classes=10, weights=None)
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


# # Create strategy
# strategy = fl.server.strategy.FedAvg(
#     # ... other FedAvg arguments
#     eval_fn=get_eval_fn(model),
# )

# Start Flower server
fl.server.start_server(
    server_address="192.168.68.141:8080",
    config={"num_rounds": 1},
    # strategy=strategy
)
