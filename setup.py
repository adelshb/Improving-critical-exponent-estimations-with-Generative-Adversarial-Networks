from setuptools import setup, find_packages

setup(
   name='Improving critical exponent estimations with Generative Adversarial Networks',
   packages=find_packages(), 
   install_requires=['wheel', # necessary on lexicon
                     'cython', # necessary for pandas on lexicon
                     'tensorflow',
                     'numpy',
                     'matplotlib',
                     'seaborn',
                     'sklearn',
                     'scipy==1.5.4', # the specific version is because lexicon
                                     # only has python < 3.7 installed 
                     'tqdm',
                     'jupyter',
                     'pandas',
                    ])
