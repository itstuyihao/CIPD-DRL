# CIPD-DRL for Multi-Rate WLANs

Based on my [first project](https://github.com/itstuyihao/CIPD-802.11), I extend my CIPD backoff mechanism from a single-rate to a multi-rate scenario by using a **sequence transduction model with multi-agent DRL** framework to improve the overall performance. However, because this project is still under consideration by the journal, I provide only part of the scripts for concept demonstration purposes in this repository. I will provide full access to the code once this project is accepted by the journal.

## License
Proprietary. Evaluation-only. You may run this software solely for personal and internal evaluation purposes. No permission is granted to copy, modify, distribute, or create derivative works without written permission from the author [itstuyihao@gmail.com](mailto:itstuyihao@gmail.com). See [LICENSE](LICENSE).

## Main Concept
```
   DRL Agent (Python, model.py)                  Environment (C++, cipd_drl)
 +------------------------------------+        +-----------------------------------+
 | acts:  ACT / PREDICT requests      |        | serves:  TCP 127.0.0.1:5555       |
 | learns: STORE / LEARN calls        |        | sim: WLAN multi-rate DCF via CIPD |
 +------------------+-----------------+        +------------------+----------------+
                    |                                              ^
                    |    TCP socket (observations -> agent,        |
                    |     actions/experiences -> env)              |
                    +------------------------------->--------------+
```

Notes:
- `cipd_drl` is the C++ WLAN/CIPD simulator (compiled binary); as a **server** listening on port 5555.
- `model.py` hosts the Bi-LSTM dueling DQN agents; as a **client** connecting to 127.0.0.1:5555, sampling actions, storing transitions, and training.
- Outputs (e.g., output_sta*.txt) are written under `static/exp1`; you can plot with `CIPD-802.11/tools/generate_plot.sh static/exp1 Image` via [here](https://github.com/itstuyihao/CIPD-802.11/blob/main/tools/generate_plot.sh).

## How to Run
1) Run the environment (terminal 1):
```bash
./cipd_drl   # C++ WLAN/CIPD simulator listening on 5555
```

2) Run the DRL agent (terminal 2):
```bash
python3 model.py 127.0.0.1 5555   # connects to the env and trains
```
   - The agent connects to the env on TCP port 5555 and exchanges ACT/PREDICT/STORE/LEARN.

3) Outputs go to `static/exp1`.
