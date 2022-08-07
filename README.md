# About
Part-of-speech (POS) tagging is a popular Natural Language Processing process which refers to categorizing words in a text (corpus) in correspondence with a particular part of speech, depending on the definition of the word and its context. Part-of-speech tags describe the characteristic structure of lexical terms within a sentence or text, therefore, we can use them for making assumptions about semantics.

A part-of-speech (POS) task was used to validate or invalidate the general view that Conditional Random Fields
(CRF) perform better than Hidden Markov Models (HMM). The testing was performed as a Part of
Speech (POS) sequence labelling task. This project presents
the data pre-processing activities and the steps taken in training and evaluating the respective HMM
and CRF models. A summary of the results and concluding remarks are presented below.

# Results and Conclusion Summary
From the metrics gathered during the testing and evaluation phases the following observations could be made:

1. The HMM is significantly faster to train than the CRF
2. The CRF is significantly faster than the HMM in making inferences/predictions
3. The CRF produced more accurate inferences than the HMM

My research validates the belief that Linear Chain CRFs are better
than HMMs, especially when looking at sequence labelling tasks where an emphasis is placed on
classification rather than generating samples. Apart from the lengthy training times, LCCRFs proved to
be a better model for sequence labelling tasks. 

_(For a full breakdown of my analysis, please refer to the report.)_
