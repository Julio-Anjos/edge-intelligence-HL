# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower server app."""


from logging import INFO
from typing import Dict, Optional, Tuple

from numpy import void

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.combiner.client_manager import ClientManager, SimpleClientManager
from flwr.combiner.grpc_server.grpc_server import start_grpc_server
from flwr.combiner.history import History
from flwr.combiner.server import Server
from flwr.combiner.strategy import FedAvg, Strategy
import time
from .grpc_client.connection import grpc_connection
from .grpc_client.message_handler import handle2

import grpc

DEFAULT_SERVER_ADDRESS = "[::]:8071"


def start_combiner(  # pylint: disable=too-many-arguments
    combiner_address: str = DEFAULT_SERVER_ADDRESS,
    server_address: str = "[::]:8080",
    server: Optional[Server] = None,
    config: Optional[Dict[str, int]] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    force_final_distributed_eval: bool = False,
    certificates: Optional[Tuple[bytes, bytes, bytes]] = None,
) -> void:
    """Start a Flower server using the gRPC transport layer.

    Arguments
    ---------
        combiner_address: Optional[str] (default: `"[::]:8080"`). The IPv6
            address of the server.
        server: Optional[flwr.combiner.Server] (default: None). An implementation
            of the abstract base class `flwr.combiner.Server`. If no instance is
            provided, then `start_server` will create one.
        config: Optional[Dict[str, int]] (default: None). The only currently
            supported values is `num_rounds`, so a full configuration object
            instructing the server to perform three rounds of federated
            learning looks like the following: `{"num_rounds": 3}`.
        strategy: Optional[flwr.combiner.Strategy] (default: None). An
            implementation of the abstract base class `flwr.combiner.Strategy`.
            If no strategy is provided, then `start_server` will use
            `flwr.combiner.strategy.FedAvg`.
        client_manager: Optional[flwr.combiner.ClientManager] (default: None)
            An implementation of the abstract base class `flwr.combiner.ClientManager`.
            If no implementation is provided, then `start_server` will use
            `flwr.combiner.client_manager.SimpleClientManager`.
        grpc_max_message_length: int (default: 536_870_912, this equals 512MB).
            The maximum length of gRPC messages that can be exchanged with the
            Flower clients. The default should be sufficient for most models.
            Users who train very large models might need to increase this
            value. Note that the Flower clients need to be started with the
            same value (see `flwr.client.start_client`), otherwise clients will
            not know about the increased limit and block larger messages.
        force_final_distributed_eval: bool (default: False).
            Forces a distributed evaluation to occur after the last training
            epoch when enabled.
        certificates : Tuple[bytes, bytes, bytes] (default: None)
            Tuple containing root certificate, server certificate, and private key to
            start a secure SSL-enabled server. The tuple is expected to have three bytes
            elements in the following order:

                * CA certificate.
                * server certificate.
                * server private key.

    Returns
    -------
        hist: flwr.combiner.history.History. Object containing metrics from training.

    Examples
    --------
    Starting an insecure server:

    >>> start_server()

    Starting a SSL-enabled server:

    >>> start_server(
    >>>     certificates=(
    >>>         Path("/crts/root.pem").read_bytes(),
    >>>         Path("/crts/localhost.crt").read_bytes(),
    >>>         Path("/crts/localhost.key").read_bytes()
    >>>     )
    >>> )
    """
    initialized_server, initialized_config = _init_defaults(
        server, config, strategy, client_manager
    )

    # Start gRPC server
    grpc_server = start_grpc_server(
        client_manager=initialized_server.client_manager(),
        server_address=combiner_address,
        max_message_length=grpc_max_message_length,
        certificates=certificates,
    )
    num_rounds = initialized_config["num_rounds"]
    ssl_status = "enabled" if certificates is not None else "disabled"
    msg = f"Flower server running ({num_rounds} rounds), SSL is {ssl_status}"
    log(INFO, msg)

    start_client(server_address,initialized_server)

    # hist = _fl(
    #     server=initialized_server,
    #     config=initialized_config,
    #     force_final_distributed_eval=force_final_distributed_eval,
    # )

    # Stop the gRPC server
    grpc_server.stop(grace=1)

    return


def _init_defaults(
    server: Optional[Server],
    config: Optional[Dict[str, int]],
    strategy: Optional[Strategy],
    client_manager: Optional[ClientManager],
) -> Tuple[Server, Dict[str, int]]:
    # Create server instance if none was given
    if server is None:
        if client_manager is None:
            client_manager = SimpleClientManager()
        if strategy is None:
            strategy = FedAvg()
        server = Server(client_manager=client_manager, strategy=strategy)

    # Set default config values
    if config is None:
        config = {}
    if "num_rounds" not in config:
        config["num_rounds"] = 1

    return server, config



def start_client(
    server_address: str,
    server: Server,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[bytes] = None,
) -> None:
    """Start a Flower Client which connects to a gRPC server.

    Parameters
    ----------
        server_address: str. The IPv6 address of the server. If the Flower
            server runs on the same machine on port 8080, then `server_address`
            would be `"[::]:8080"`.
        client: flwr.client.Client. An implementation of the abstract base
            class `flwr.client.Client`.
        grpc_max_message_length: int (default: 536_870_912, this equals 512MB).
            The maximum length of gRPC messages that can be exchanged with the
            Flower server. The default should be sufficient for most models.
            Users who train very large models might need to increase this
            value. Note that the Flower server needs to be started with the
            same value (see `flwr.combiner.start_server`), otherwise it will not
            know about the increased limit and block larger messages.
        root_certificates: bytes (default: None)
            The PEM-encoded root certificates as a byte string. If provided, a secure
            connection using the certificates will be established to a
            SSL-enabled Flower server.

    Returns
    -------
        None

    Examples
    --------
    Starting a client with insecure server connection:

    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>> )

    Starting a SSL-enabled client:

    >>> from pathlib import Path
    >>> start_client(
    >>>     server_address=localhost:8080,
    >>>     client=FlowerClient(),
    >>>     root_certificates=Path("/crts/root.pem").read_bytes(),
    >>> )
    """
    while True:
        sleep_duration: int = 0
        with grpc_connection(
            server_address,
            max_message_length=grpc_max_message_length,
            root_certificates=root_certificates,
        ) as conn:
            receive, send = conn

            while True:
                server_message = receive()
                client_message, sleep_duration, keep_going = handle2(server ,server_message)
                send(client_message)
                if not keep_going:
                    break
        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            break
        # Sleep and reconnect afterwards
        log(
            INFO,
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)