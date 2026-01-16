\# The Space Between Words – Code Repository

The following appendices are **implemented in this repository and omitted from the paper for brevity**:


\- Appendix C – Frequency analysis: `scripts/appendix\_C\_frequency\_trends.py`

\- Appendix D – Correlation heatmap: `scripts/appendix\_D\_heatmap.py`

\- Appendix E – Pairwise embedding comparisons: `scripts/appendix\_E\_pairwise\_comparisons.py`

\- Appendix F – Normalized frequency \& relative value: `scripts/appendix\_F\_normalized\_frequency.py`

\- Appendix H – Robustness checks: `scripts/appendix\_H\_robustness\_checks.py`

## Training parameters

All Word2Vec models were trained using a consistent set of hyperparameters:

```text
vector_size: 100        # dimensionality of word vectors
window: 10              # context window size
min_count: 2            # minimum frequency for inclusion
workers: half CPU cores # parallel worker threads (minimum 1)
epochs: 15              # training iterations
sg: 1                   # skip-gram architecture

# The-Space-Between-Words-GitHub-Repo