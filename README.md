\# The Space Between Words – Code Repository

The following appendices are **implemented in this repository and omitted from the paper for brevity**:


\- Frequency analysis: `scripts/appendix\_C\_frequency\_trends.py`

\- Correlation heatmap: `scripts/appendix\_D\_heatmap.py`

\-  Pairwise embedding comparisons: `scripts/appendix\_E\_pairwise\_comparisons.py`

\- Normalized frequency \& relative value: `scripts/appendix\_F\_normalized\_frequency.py`

\- Bootstrap Analysis: `scripts/appendix\_G\_bootstrap\_analysis.py`

\- Control Analysis: `scripts/appendix\_H\_control\_analysis.py`

\- Semantic Axis: `scripts/vector_arithmetic_agency_axis.py`

## Notes
Apologies for the mess! I've included this repo for transparency. The scripts correspond to what was originally pseudocode within the appendices of an early draft of the paper, and so it has retained the appendix labels from that previous version. 

It is not very **pretty**, but the time has passed for it to be so. It merely reflects the research workflow used to generate the reported results. It is not intended as a polished software package. Regardless, this should get someone where they need to eproduce the analyses described in the paper, should they want to.


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
