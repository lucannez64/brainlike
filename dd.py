from brian2 import (
    start_scope, NeuronGroup, PoissonGroup, Synapses,
    SpikeMonitor, ms, mV, Hz, run, clip
)
import numpy as np
import itertools

def run_binary_function_learning(f, name, n_bits=2):
    start_scope()

    # Parameters
    n_input = n_bits
    n_output = 2
    tau_m = 10 * ms
    E_L = -70 * mV
    V_th = -50 * mV
    V_reset = -65 * mV
    refractory = 5 * ms

    # STDP
    tau_pre = 20 * ms
    tau_post = 20 * ms
    A_pre = 0.01
    A_post = -A_pre * tau_pre / tau_post * 1.05
    w_max = 1.0

    # Training settings
    dur = 100 * ms
    n_epochs = 50
    r_high = 50 * Hz
    r_low = 1 * Hz
    r_teacher = 100 * Hz
    test_dur = 200 * ms

    # Build network
    input_group = PoissonGroup(n_input, rates=0 * Hz)
    teacher = PoissonGroup(n_output, rates=0 * Hz)

    eqs = '''
    dv/dt = (E_L - v) / tau_m : volt (unless refractory)
    '''
    output = NeuronGroup(
        n_output, eqs,
        threshold='v>V_th', reset='v=V_reset',
        refractory=refractory, method='exact'
    )
    output.v = E_L

    # Teacher → output (one‐to‐one, large weight)
    S_t = Synapses(
        teacher, output,
        model='w_t:1',
        on_pre='v_post += w_t*mV'
    )
    S_t.connect(j='i')
    S_t.w_t = 20.0

    # Input → output, plastic STDP synapses
    stdp_eqs = '''
    dpre/dt = -pre / tau_pre   : 1 (event-driven)
    dpost/dt = -post / tau_post: 1 (event-driven)
    '''
    S = Synapses(
        input_group, output,
        model=stdp_eqs + '\nw:1',
        on_pre='''
            v_post += w*mV
            pre = 1.0
            w = clip(w + A_pre*post, 0, w_max)
        ''',
        on_post='''
            post = 1.0
            w = clip(w + A_post*pre, 0, w_max)
        '''
    )
    S.connect(True)
    S.w = 'rand() * w_max'

    # Monitor
    spike_mon = SpikeMonitor(output)

    # All possible input patterns
    patterns = list(itertools.product([0, 1], repeat=n_bits))

    # Training
    for epoch in range(n_epochs):
        for bits in patterns:
            # set input rates
            inp_rates = np.array(
                [r_high if b else r_low for b in bits]
            ) * Hz
            input_group.rates = inp_rates

            # teacher drives correct output neuron
            y = int(f(bits))
            teach_rates = np.array(
                [r_teacher if i == y else 0.0
                 for i in range(n_output)]
            ) * Hz
            teacher.rates = teach_rates

            run(dur)

    # turn off teacher
    teacher.rates = np.zeros(n_output) * Hz

    # Testing
    print(f"Results for {name}:")
    correct = 0
    for bits in patterns:
        inp_rates = np.array(
            [r_high if b else r_low for b in bits]
        ) * Hz
        input_group.rates = inp_rates

        before = np.copy(spike_mon.count[:])
        run(test_dur)
        diff = spike_mon.count[:] - before
        pred = int(np.argmax(diff))
        target = int(f(bits))
        ok = 'OK' if pred == target else 'ERR'
        if pred == target:
            correct += 1
        print(f" Input {bits} → target {target}, pred {pred} {ok}")
    print(f"Accuracy: {correct}/{len(patterns)}\n")


if __name__ == '__main__':
    funcs = {
        'AND': lambda b: b[0] & b[1],
        'XOR': lambda b: b[0] ^ b[1]
    }
    for name, func in funcs.items():
        run_binary_function_learning(func, name)
