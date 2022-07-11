import flwr as fl


# Start Flower server
fl.combiner.start_combiner(
    combiner_address="[::]:8050",
    server_address="[::]:8070"
)
