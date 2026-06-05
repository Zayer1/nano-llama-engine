# RoPE Upgrade Task Tracker

- [ ] Delete `self.positional_embedding` from `NanoGPT` in `01_nano_gpt.py`
- [ ] Implement `apply_rotary_emb` in `01_nano_gpt.py`
- [ ] Inject RoPE into `CausalSelfAttention.forward()` in `01_nano_gpt.py`
- [ ] Document RoPE math in `volume_1_numpy_math/MATH.md`
- [ ] Document RoPE upgrade in `volume_2_pytorch_automaton/README.md`
- [ ] Verify execution of `01_nano_gpt.py` forward pass test
- [ ] Create Walkthrough
