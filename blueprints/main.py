class Context:
    """One experiment environment shared by all comparison methods."""

    def __init__(
        self,
        seed=None,
        task=None,
        partitions=None,
        test_data=None,
        counts=None,
        num_users=None,
        model_factory=None,
        initial_model_state=None,
        model_bits=None,
        num_rbs=None,
        distances=None,
        channel_gain=None,
        downlink_gain=None,
        interference=None,
        downlink_interference=None,
        powers=None,
        packet_errors=None,
        uplink_rates=None,
        downlink_rates=None,
        total_delays=None,
        energies=None,
        feasible=None,
        delay_s=None,
        energy_j=None,
        pmax_w=None,
        rounds=None,
        local_epochs=None,
        learning_rate=None,
        batch_size=None,
        device=None,
        loss=None,
        optimizer_states=None,
    ):
        self.seed = seed
        self.task = task

        # Shared data environment
        self.partitions = partitions
        self.test_data = test_data
        self.counts = counts
        self.num_users = num_users

        # Shared model environment
        self.model_factory = model_factory
        self.initial_model_state = initial_model_state
        self.model_bits = model_bits

        # Shared wireless environment
        self.num_rbs = num_rbs
        self.distances = distances
        self.channel_gain = channel_gain
        self.downlink_gain = downlink_gain
        self.interference = interference
        self.downlink_interference = downlink_interference
        self.powers = powers
        self.packet_errors = packet_errors
        self.uplink_rates = uplink_rates
        self.downlink_rates = downlink_rates
        self.total_delays = total_delays
        self.energies = energies
        self.feasible = feasible

        # Shared constraints
        self.delay_s = delay_s
        self.energy_j = energy_j
        self.pmax_w = pmax_w

        # Shared FL training environment
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.loss = loss
        self.optimizer_states = optimizer_states

# data processing
def generate_synthetic_data():
    pass

def load_mnist_data():
    pass

def get_partitioned_data():
    pass

# models
class RegressionFNN:
    pass

class MNISTFNN:
    pass

class MNISTCNN:
    pass

# FL algorithms
def fl_one_round(
    context,
    global_model,
    selected_users,
    assigned_rbs,
    selected_errors,
    rng,
    round_index=0,
    aggregation="fedavg",
    verbose=False,
    scheme=None,
):
    """선택된 클라이언트들을 대상으로 local training을 수행하고, 그 결과를 서버로 전송하여 모델을 업데이트한다."""
    pass

def baseline_a(context, verbose=False):
    """FL-aware user selection + random RB allocation"""

def baseline_b(context, verbose=False):
    """random user selection + random RB allocation"""

def baseline_c(context, verbose=False):
    """Wireless-only baseline c; mode is selected by training.baseline_c_mode."""

def proposed_algorithm(context, verbose=False):
    """
    FL-aware user selection + Hungarian RB allocation.
    1. Analyze the expedcted convergence of the federated learning based on (17)
    2. Find the optimal transmit power of each user over each RB using (22)
    3. Solve the optimization problem (23) using a standard Hungarian algorithm and (24)
    4. Update the global model using the selected users and RBs.
    """
# utils
def build_contexts(yaml_file) -> list[Context]:
    """Build a contexts from a yaml file."""


# figures
# 각 figure마다 context sweep을 만들어서 결과를 저장하고, 저장된 결과를 바탕으로 figure을 그린다.
def figure_3(contexts):
    pass

def figure_4(contexts):
    pass

def figure_5(contexts):
    pass

def figure_6(contexts):
    pass

def figure_7(contexts):
    pass

def figure_8(contexts):
    pass

def figure_9(contexts):
    pass

def figure_10(contexts):
    pass

# main
def main():
    pass

if __name__ == "__main__":
    main()
