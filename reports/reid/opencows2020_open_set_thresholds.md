# Open-set threshold calibration

Method: sweep similarity thresholds over FAISS top-1 score and maximize balanced score between accepting correct matches and rejecting incorrect matches.

## prototype
- Total queries: 496
- Correct (no threshold): 388
- Incorrect (no threshold): 108
- Mean score correct: 0.9846197627868849
- Mean score incorrect: 0.9726903162620686
- Recommended threshold: 0.9927988052368164
- Balanced score: 0.6311569301260023
- Accepted correct rate: 0.3790322580645161
- False reject rate: 0.4032258064516129
- False accept rate: 0.04838709677419355

## all_vectors
- Total queries: 496
- Correct (no threshold): 418
- Incorrect (no threshold): 78
- Mean score correct: 0.9989943773837752
- Mean score incorrect: 0.9974410167107215
- Recommended threshold: 0.9995235204696655
- Balanced score: 0.6481106612685561
- Accepted correct rate: 0.4657258064516129
- False reject rate: 0.37701612903225806
- False accept rate: 0.04032258064516129
