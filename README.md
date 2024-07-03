### What is it?
This is a Python implementation (accelerated by some Rust code) of the Sketchy CGAL method from https://arxiv.org/pdf/1912.02949 for large scale semidefinite programs (SDP).

### How to install
- [Install rust](https://www.rust-lang.org/tools/install) on your machine;
- Clone this repo to your machine;
- Run `pip install .` from the cloned repo under your virtual environment.

### Examples
 - `./examples/random_sdp.py` -- a small scale random SDP program solved by this implementation and compared with CVXPY solution;
 - `./examples/maxcut.py` -- a maxcut problem for [GSet](https://web.stanford.edu/~yyye/yyye/Gset/), one can run an example simply executing this script with the number of a graph from the GSet as the command line argument, e.g. `./examples/maxcut.py 67`.