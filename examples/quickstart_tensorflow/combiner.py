import flwr as fl


# Start Flower server
fl.combiner.start_combiner(
    combiner_address="192.168.68.141:8070",
)
