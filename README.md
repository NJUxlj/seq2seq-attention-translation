# Sequence to Sequence Attention
this file reproduce the seq2seq attention brought up by this paper [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://peerj.com/articles/cs-2607/code.zip)


# File
`seq2seq_translation.py` is a implementation of the Soft-Attention with a standard encoder-decoder model implemented using the GRU

`seq2seq_translation_2.py` is a modified the version of soft-attention with a slightly different implementation and a multi-layer GRU
# Run
```shell
pip install torch
python seq2seq_translation.py
```


# Results
![image](https://github.com/user-attachments/assets/29ed9470-ebbe-4ba9-b869-76042ebca124)
![image](https://github.com/user-attachments/assets/ed0fe54d-50a1-41d3-b558-e0538db83902)


# Citation
```bibtex
@article{bahdanau2014neural,
  title={Neural machine translation by jointly learning to align and translate},
  author={Bahdanau, Dzmitry},
  journal={arXiv preprint arXiv:1409.0473},
  year={2014}
}
```
