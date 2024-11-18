# dx-privacy-text
This repository contains the code used in the experiments of the paper titled "$d_x$-Privacy for Text and the Curse of Dimensionality".

## Requirements
- conda must be installed to set up the python environment
- The code leverages [cupy](https://cupy.dev/) which requires a GPU and a CUDA driver enabled

## How to run
### Setup the python environment
1. Install the dependencies with `conda env create -n dx-privacy-text --file dx-privacy-text.yml`
2. Activate the conda environment `conda activate dx-privacy-text`
3. Make a directory to host the vocabularies and write its absolute path to config.py.

### Download and pre-process the vocabularies
1. In the folder created above, download the following files:
    - glove.6B.zip from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/). After extraction, you should have four files: glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt and glove.6B.300d.txt
    - glove.twitter.27B.zip from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/). After extraction, you should have four files: glove.twitter.27B.25d.txt, glove.twitter.27B.50d.txt, glove.twitter.27B.100d.txt, glove.twitter.27B.200d.txt
    - wiki.en.vec from [https://fasttext.cc/docs/en/pretrained-vectors.html](https://fasttext.cc/docs/en/pretrained-vectors.html) under "English: text".
    - GoogleNews-vectors-negative300.bin.gz from [https://code.google.com/archive/p/word2vec/](https://code.google.com/archive/p/word2vec/). Look for "The archive is available here:". After extraction, you should have one file GoogleNews-vectors-negative300.bin.

2. Run the PreProcessing.ipynb files contained in each of the code folders named fasttext, glove and word2vec.

### Execute the experiments
In each of the code folders mentionned above there are two files:
- NeighborDistances.ipynb generates the results used in Table 2, which involves computing distances between words and their k-th neighbor within the vocabulary. The code in the glove folder contains an additional section to generate the data for Figure 6.
- CloseNeighborsDxFrequencies.ipynb generates the results used in Figure 2, which involves applying $d_x$-privacy to random words and identifying the rank (i.e., the "k" in k-th neighbor) of the word which was chosen as the replacement by the mechanism. The code in the glove and word2vec folders additionaly include the post-processing fix proposed in the paper and used for Figure 8.

Note that the glove folder contains an additional file named textSanitization.ipynb which was used to produce the sanitization example of a small paragraph in Section 2.

## References in the code:
- (Feyisetan et al, 2020): O. Feyisetan, B. Balle, T. Drake, and T. Diethe, “Privacy- and utility-preserving textual analysis via calibrated multivariate perturbations,” in Proceedings of the 13th international conference on web search and data mining, in WSDM ’20. New York, NY, USA: Association for Computing Machinery, 2020, pp. 178–186. doi: 10.1145/3336191.3371856.

- (Qu et al., 2021): C. Qu, W. Kong, L. Yang, M. Zhang, M. Bendersky, and M. Najork, “Natural language understanding with privacy-preserving BERT,” in Proceedings of the 30th ACM international conference on information & knowledge management, in CIKM ’21. New York, NY, USA: Association for Computing Machinery, 2021, pp. 1488–1497. doi: 10.1145/3459637.3482281.