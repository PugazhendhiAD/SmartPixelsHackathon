## Hackathon Project: Effects of Radiation Damage on SmartPixels
Look at the config files in `config/`. They define three datasets: `baseline` with no radiation damage, and `370fb` and `1100fb` with radiation damage corresponding to an integrated luminosity of 370 and 1100 inverse fb, respectively. The goal is to train a neural network that performs well on all three scenarios. In particular, we want to evaluate the following approaches on all three datasets :
- a model trained exclusively on baseline (this should perform worst)
- three individual models: one trained on each radiation level (this should perform best)
- a model trained on a mixture of all three radition levels
- a mdoel trained on all radiation levels, where an additional feature indicates the level radiation
The question is: How much better is option 4 w.r.t. 3 and how close to option 2 do we get?