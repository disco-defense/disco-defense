# Disco Defense Project 

### Abstract

Deep neural networks are misguided by simple-to-craft, imperceptible adversarial perturbations to inputs. Now, it is possible to craft such perturbations solely using model outputs and black-box attack algorithms. These algorithms compute adversarial examples by iteratively querying a model and inspecting responses. Attacks success in near information vacuums pose a significant challenge for developing mitigations. We investigate a new idea for a defense driven by a fundamental insight—to compute an adversarial example, attacks depend on the relationship between successive responses to queries to optimize a perturbation. Therefore, to obfuscate this relationship, we investigate randomly sampling a model from a set to generate a response to a query. Effectively, this model randomization violates the attacker's expectation of the unknown parameters of a model to remain static between queries to extract information to guide the search toward an adversarial example. It is not immediately clear if model randomization can lead to sufficient obfuscation to confuse query-based black-box attacks or how such a method could be built. Our theoretical analysis proves model randomization always increases resilience to query-based black-box attacks. We demonstrate with extensive empirical studies using 6 state-of-the-art attacks under all three perturbation objectives ($l_\infty, l_2, l_0$) and adaptive attacks, our proposed method injects sufficient uncertainty through obfuscation to yield a highly effective defense.

### Evaluate our models
1. Download our pre-trained models for evaluation:
   - [MNIST](https://drive.google.com/file/d/1wnuHtcC7wwnP6iH6LDRTkIOcZ2GD0NEt/view?usp=drive_link)
   - [CIFAR-10](https://drive.google.com/file/d/1u1gwsa2gf6ZZDmVFnvE9us0q5Zff-nb8/view?usp=drive_link)
   - [STL-10](https://drive.google.com/file/d/1GdCD8TWWsjJjsPWiQBlAB3y1Xw4Kejgf/view?usp=drive_link)
2. Follow our Notebooks to load datasets and models for clean accuracy check

### ToDo
1. Training code
2. Evaluation code

### References
1. 
