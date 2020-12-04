# A neural network based model for polypharmacy side effects prediction
Polypharmacy is a type of treatment that involves the concurrent use of multiple medications. Drugs may interact when they are used simultaneously. Polypharmacy side effects are represented as drug-drug interactions (DDIs). Understanding and mitigating polypharmacy side effects is critical for patient safety and health. Since the known polypharmacy side effects are rare and they are not detected in clinical trials, various computational methods are developed to model such polypharmacy side effects. 
## Usage
We propose Neural Network-based method for Polypharmacy Side effects prediction (NNPS). By using drug-protein interactions, and drug-drug interactions information, NNPS applies novel feature vectors for a given drug pair to predict 964 polypharmacy side effects. 
### Running the code
The setup for our problem is outlined in `NNPS.py`. It uses a simple neural network with 964 side effects. Run the code as following:

```
$ python3 NNPS.py
```

The polypharmacy side effect datasets on Datasets folder is available and ready to use. 
## Requirements
Work on python 3 and recent versions of Keras, Tensorflow, sklearn, networkx, seaborn, pandas, numpy and scipy are required.
