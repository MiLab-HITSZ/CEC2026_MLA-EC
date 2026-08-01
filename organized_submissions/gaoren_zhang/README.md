# Structured Active-Label Attack

Entrant: Gaoren Zhang

The method searches structured spatial and DCT directions while concentrating
the query budget on active target labels. It includes adaptive step sizes,
label memory, restarts, and a rescue phase.

`main.py` is the experiment entry point. It supports `MLAE_CEC_DATA_DIR`,
`MLAE_START_INDEX`, `MLAE_END_INDEX`, `MLAE_RESULT_SUFFIX`, and
`MLAE_CUDA_DEVICE` environment variables. The algorithm module must be copied
or linked into the benchmark framework's `attack_algorithm` package before the
entry point is executed.
