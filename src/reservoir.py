
import numpy as np

from brian2 import (
    Network,
    NeuronGroup,
    Synapses,
    SpikeMonitor,
    TimedArray,
    defaultclock,
    start_scope,
    seed as brian_seed,
    ms,
    mV,
    prefs
)


def run_reservoir_pair(
    signal,
    weight_scale,
    seed_value,
    n_neurons=200,
    bin_ms=10.0,
    dt_ms=0.1,
    connectivity=0.1
):
    """
    Simulate two matched recurrent LIF reservoirs.

    One reservoir has its original initial conditions.
    The second receives a small initial voltage perturbation.

    Returns
    -------
    rates_a : ndarray
        Binned firing rates for the first reservoir.
    divergence_index : float
        Normalized RMS difference between the two reservoirs.
    mean_firing_rate : float
        Mean firing rate across both reservoirs.
    """

    start_scope()
    prefs.codegen.target = "numpy"

    brian_seed(seed_value)
    defaultclock.dt = dt_ms * ms

    rng = np.random.default_rng(seed_value)

    duration_s = len(signal) * bin_ms / 1000.0

    # Explicitly inject the same external input into both networks.
    input_drive = (17.0 + 5.0 * signal) * mV

    input_signal = TimedArray(
        input_drive,
        dt=bin_ms * ms
    )

    # LIF neuron equations.
    equations = """
    dv/dt = (
        -(v - (-65*mV)) + I_syn + I_ext
    ) / (20*ms) : volt (unless refractory)

    dI_syn/dt = -I_syn / (5*ms) : volt

    I_ext = input_signal(t) * input_gain : volt

    input_gain : 1
    """

    # Two matched neuronal populations.
    neurons_a = NeuronGroup(
        n_neurons,
        equations,
        threshold="v > -50*mV",
        reset="v = -70*mV",
        refractory=5*ms,
        method="euler",
        namespace={"input_signal": input_signal}
    )

    neurons_b = NeuronGroup(
        n_neurons,
        equations,
        threshold="v > -50*mV",
        reset="v = -70*mV",
        refractory=5*ms,
        method="euler",
        namespace={"input_signal": input_signal}
    )

    # Identical heterogeneous input gains.
    input_gains = rng.uniform(
        0.75,
        1.25,
        n_neurons
    )

    neurons_a.input_gain = input_gains
    neurons_b.input_gain = input_gains

    # Identical initial membrane potentials,
    # except for a small perturbation in reservoir B.
    initial_v = rng.uniform(
        -65.0,
        -55.0,
        n_neurons
    )

    neurons_a.v = initial_v * mV
    neurons_b.v = (initial_v + 0.01) * mV

    neurons_a.I_syn = 0 * mV
    neurons_b.I_syn = 0 * mV

    # Recurrent connections in reservoir A.
    synapses_a = Synapses(
        neurons_a,
        neurons_a,
        model="w : 1",
        on_pre="I_syn_post += w*mV"
    )

    synapses_a.connect(
        condition="i != j",
        p=connectivity
    )

    # Copy the exact same topology to reservoir B.
    source_indices = np.asarray(
        synapses_a.i[:],
        dtype=int
    )

    target_indices = np.asarray(
        synapses_a.j[:],
        dtype=int
    )

    synapses_b = Synapses(
        neurons_b,
        neurons_b,
        model="w : 1",
        on_pre="I_syn_post += w*mV"
    )

    synapses_b.connect(
        i=source_indices,
        j=target_indices
    )

    # 80% excitatory and 20% inhibitory neurons.
    n_exc = int(0.8 * n_neurons)

    excitatory_mask = source_indices < n_exc

    weights = np.empty(
        len(source_indices),
        dtype=float
    )

    weights[excitatory_mask] = rng.uniform(
        0.15,
        0.35,
        excitatory_mask.sum()
    )

    weights[~excitatory_mask] = -rng.uniform(
        0.30,
        0.55,
        (~excitatory_mask).sum()
    )

    # Sweep recurrent coupling strength.
    weights *= weight_scale

    synapses_a.w = weights
    synapses_b.w = weights

    # Record spikes.
    monitor_a = SpikeMonitor(neurons_a)
    monitor_b = SpikeMonitor(neurons_b)

    network = Network(
        neurons_a,
        neurons_b,
        synapses_a,
        synapses_b,
        monitor_a,
        monitor_b
    )

    print(
        f"Simulating coupling scale: {weight_scale:.2f}"
    )

    network.run(
        len(signal) * bin_ms * ms
    )

    # Convert spike trains into binned firing-rate matrices.
    time_edges = np.arange(
        len(signal) + 1
    ) * bin_ms

    neuron_edges = np.arange(
        n_neurons + 1
    )

    counts_a, _, _ = np.histogram2d(
        np.asarray(monitor_a.t / ms),
        np.asarray(monitor_a.i),
        bins=(time_edges, neuron_edges)
    )

    counts_b, _, _ = np.histogram2d(
        np.asarray(monitor_b.t / ms),
        np.asarray(monitor_b.i),
        bins=(time_edges, neuron_edges)
    )

    bin_seconds = bin_ms / 1000.0

    rates_a = counts_a / bin_seconds
    rates_b = counts_b / bin_seconds

    # Ignore the first 10% as a washout period.
    washout = max(
        1,
        int(0.10 * len(signal))
    )

    rates_a_eval = rates_a[washout:]
    rates_b_eval = rates_b[washout:]

    # Normalized perturbation sensitivity.
    difference_rms = np.sqrt(
        np.mean(
            (rates_a_eval - rates_b_eval) ** 2
        )
    )

    activity_rms = np.sqrt(
        0.5 * (
            np.mean(rates_a_eval ** 2)
            +
            np.mean(rates_b_eval ** 2)
        )
    )

    divergence_index = float(
        difference_rms / (activity_rms + 1e-8)
    )

    # Average firing rate across both networks.
    total_spikes = (
        len(monitor_a.i) + len(monitor_b.i)
    )

    mean_firing_rate = (
        total_spikes
        / (2 * n_neurons * duration_s)
    )

    return (
        rates_a,
        divergence_index,
        mean_firing_rate
    )
