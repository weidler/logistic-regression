# Logistic Regression

### Requirements
Required packages are listed in requirements.txt. Install them into your virtual environment with

    pip install -r requirements.txt
   
### Usage

    usage: evaluate.py [-h] [--dataset {iris,monk}]
                   [--features {0,1,2,3} [{0,1,2,3} ...]] [--exploration]
                   [--performance] [--decision-boundary] [--no-plot] [--safe]

    optional arguments:
      -h, --help            show this help message and exit
      --dataset {iris,monk}
                            the dataset to be used, either iris or monk
      --features {0,1,2,3} [{0,1,2,3} ...]
                            features used when dataset is iris
      --exploration         whether to plot the pairplot
      --performance         whether to plot performance measures
      --decision-boundary   whether to plot/save the decision boundary
      --no-plot             deactivate plotting for the decision boundary
      --safe                activate saving of decision boundary plot