# RAMP starting kit on Melanoma detection

_Authors: Cyril Equilbec, Jean Babin, Victorien Gimenez, Vincent Jacob_

Skin diseases are very common, especially melanoma and require full attention. Indeed, the statistics are striking : Melanoma skin cancer is the 5th most common cancer in the UK, accounting for 4% of all new cancer cases (2015). 1 person dies of melanoma every hour in North America, 1 out of 5 people in the USA will eventually contract a skin cancer during their lifetime and an interesting fact motivating our challenge : 70% of skin cancers are first detected by the patients themselves or their relatives.

Skin cancer is a major problem but unfortunately underestimated : too few people regularly consult a dermatologist, which can be understood because getting an appointment can be really long.

However, several initiatives have been launched to simplify the prevention of this disease, including the Molescope smartphone application that allows the user to take a picture of your mole with a special physical device and share it with a dermatologist. This is a great initiative but it can be restrictive as you have to buy an extra device and wait for the dermatologist to send its feedback.

This is why we propose to use data science to build a strong classifier able to distinguish between benign moles, moles suspicious enough to require a profesionnal point of view for further examinations and already dangerous moles which require immediate attention. This classifier should be used as a part of a fully automated smartphone application which basically will allow the user to take a snapshot of one of his mole and get instant feedback.

A good prediction could lead to a significant improvement in the prevention of melanoma.

#### Set up

Open a terminal and

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
  ```
  
2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [dedicated notebook](Melanoma_starting_kit.ipynb).

#### Help
Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](http:www.ramp.studio) ecosystem.
