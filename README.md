# Disco Defense Project 

### Abstract

Query-based black-box attack algorithms can compute imperceptible adversarial perturbations to misguide learned models, relying _only_ on model outputs. The success of these attack algorithms poses a significant problem, especially for Machine Learning as a Service (MLaaS) providers. Our study explores a new approach to _obfuscate_ information from an attacker. To craft an adversarial example, attacks exploit the relationship between successive responses to queries to optimize a perturbation. Our idea to attempt to _obfuscate_ this relationship is to randomly select a model from a _diverse_ set of models to respond to each query. Effectively, this randomization of models violates the attacker’s assumption of _model parameters_ remaining unaltered between queries to extract information. _What is unclear is, if model randomization leads to sufficient obfuscation to confuse attacks or how best to build such a method._ This study seeks answers to these questions. Our theoretical analysis proves this approach consistently increases robustness. Extensive experiments across 7 state-of-the-art attacks and all major perturbation norms ($l_\infty, l_2, l_0$), including adaptive variants, confirm its effectiveness. Importantly, our findings reveal a new avenue for investigating robust methods against black-box attacks, offering theoretical understandings and a practical implementation pathway.

### Evaluate our models and defense
1. Download our _pre-trained_ __model sets__ using _SVGD+_ for evaluation:
   - [MNIST](https://drive.google.com/file/d/1wnuHtcC7wwnP6iH6LDRTkIOcZ2GD0NEt/view?usp=drive_link) (40 models)
   - [CIFAR-10](https://drive.google.com/file/d/1u1gwsa2gf6ZZDmVFnvE9us0q5Zff-nb8/view?usp=drive_link) (10 models)
   - [STL-10](https://drive.google.com/file/d/1GdCD8TWWsjJjsPWiQBlAB3y1Xw4Kejgf/view?usp=drive_link) (10 models)
2. Follow our Notebooks to load datasets and models for clean accuracy check
3. Our _pre-trained_ __single models__:
   - [MNIST](https://drive.google.com/file/d/1nvDBn9WNS7fnKlnYv2wNdPiiHS-VIVIv/view?usp=drive_link)
   - [CIFAR-10](https://drive.google.com/file/d/1MFBI_UrgqPy3nX2PVbC4G8cpxhgzU7nL/view?usp=drive_link)
   - [STL-10](https://drive.google.com/file/d/1sLI-pDZR5jIBZn38QHV5AGiiaN0DEYOW/view?usp=drive_link)
4. Evaluate our defense: (to be released)

### Training model using SVGD framework
Our method is developed based on the code base SVGD [Source](https://github.com/baogiadoan/IG-BNN). 

### ToDo
1. Training code
2. Evaluation code

### References
1. 
