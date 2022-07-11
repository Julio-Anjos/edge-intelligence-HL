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
"""Handle server messages by calling appropriate client methods."""


from logging import INFO
from flwr.common.logger import log

from typing import Tuple
from unittest import result

from pytest import param

from flwr.client.client import Client, has_get_properties
from flwr.combiner.server import Server
from flwr.common import serde, typing
from flwr.proto.transport_pb2 import ClientMessage, Reason, ServerMessage

import grpc

# pylint: disable=missing-function-docstring


class UnknownServerMessage(Exception):
    """Signifies that the received message is unknown."""

def handle2(server: Server ,server_msg: ServerMessage) -> Tuple[ClientMessage, int, bool]:
    """Handle incoming messages from the server.

    Parameters
    ----------
    client : Client
        The Client instance provided by the user.

    Returns
    -------
    client_message: ClientMessage
        The message comming from the server, to be processed by the client.
    sleep_duration : int
        Number of seconds that the client should disconnect from the server.
    keep_going : bool
        Flag that indicates whether the client should continue to process the
        next message from the server (True) or disconnect and optionally
        reconnect later (False).
    """
    field = server_msg.WhichOneof("msg")
    if field == "reconnect":
        server.disconnect_all_clients()
        disconnect_msg, sleep_duration = _reconnect(server_msg.reconnect)
        return disconnect_msg, sleep_duration, False
    if field == "properties_ins":
        return _get_properties(server_msg.properties_ins), 0, True
    if field == "get_parameters":
        return _get_parameters(server._get_initial_parameters()), 0, True
    if field == "fit_ins":
        return _fit(server.fit(server_msg)), 0, True
    if field == "evaluate_ins":
        return _evaluate(server.evaluate(server_msg)), 0, True
    raise UnknownServerMessage()


def _reconnect(
    reconnect_msg: ServerMessage.Reconnect,
) -> Tuple[ClientMessage, int]:
    # Determine the reason for sending Disconnect message
    reason = Reason.ACK
    sleep_duration = None
    if reconnect_msg.seconds is not None:
        reason = Reason.RECONNECT
        sleep_duration = reconnect_msg.seconds
    # Build Disconnect message
    disconnect = ClientMessage.Disconnect(reason=reason)
    return ClientMessage(disconnect=disconnect), sleep_duration


def _get_properties(
    properties_msg: ServerMessage.PropertiesIns
) -> ClientMessage:
    # If client does not override get_properties, don't call it
    properties_res = typing.PropertiesRes(
        status=typing.Status(
            code=typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED,
            message="Client does not implement get_properties",
        ),
        properties={},
    )
    properties_res_proto = serde.properties_res_to_proto(properties_res)
    return ClientMessage(properties_res=properties_res_proto)


def _get_parameters(parameters: typing.ParametersRes) -> ClientMessage:
    parameters_res_proto = serde.parameters_res_to_proto(parameters)
    return ClientMessage(parameters_res=parameters_res_proto)


def _fit(results: Tuple[typing.Parameters, int]) -> ClientMessage:
    # Serialize fit result
    log(INFO, "Refine the provided weights using the locally held dataset.")
    parameters, num_examples = results
    fit_res = typing.FitRes(
            parameters=parameters,
            num_examples=num_examples,
            metrics={},
        )
    fit_res_proto = serde.fit_res_to_proto(fit_res)
    return ClientMessage(fit_res=fit_res_proto)


def _evaluate(results: Tuple[float, int]) -> ClientMessage:
    loss, num_examples = results
    evaluate_res = typing.EvaluateRes(
            loss=loss,
            num_examples=num_examples,
            metrics={},
        )
    # Serialize evaluate result
    evaluate_res_proto = serde.evaluate_res_to_proto(evaluate_res)
    return ClientMessage(evaluate_res=evaluate_res_proto)
