## Project Description:
This project, submitted for Colgate University's Computational Neuroscience course (NEUR374A), aims to model the synaptic downscaling hypothesis, a proposed mechanism by which memory consolidation occurs during slow-wave sleep (SWS) (Sullivan and de Sa, 2008).

## Model Design:
A synaptic-downscaling "asleep" cycle was incorporated into a multilayer feed-forward neural network that engaged in supervised vowel-consonant classification, alternating between 200 iterations of "awake cycles" and 5 iterations of a "sleep cycle," wherein training is frozen and weights were normalized. These cycles repeated for a total of 1000 iterations.

## Results:
In the "sleepy model", vowel target probabilities are closer to 1 and significantly greater than those of the normal model (two-tailed independent samples t-test,  _t_(8) = 13.8064, _p_ =  7.3160 x 10<sup>-7</sup>) (all of which are above the 0.7 threshold utilized in class). This indicates that, on average, this network is better able to identify vowels when it sleeps. The consonant target probabilities of the network with sleep are also closer to 0 in comparison to the network when sleep-deprived. For simulations lasting 1000 iterations, modeling the synaptic downscaling effect in a deep learning neural network model increased its ability to learn to identify vowels faster and identify them more accurately in comparison to the same model without sleep implemented.

# Full Paper:
To read more about our "sleepy" model, read my final class paper:


[Sleeping_Our_Way_To_Success_Emma_Slupik.pdf](https://github.com/user-attachments/files/26589336/Sleeping_Our_Way_To_Success_Emma_Slupik.pdf)
