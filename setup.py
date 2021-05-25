from setuptools import setup, find_packages

setup(
   name='Improving critical exponent estimations with Generative Adversarial Networks',
   packages=find_packages(), 
   install_requires=['tensorflow',
                     'numpy',
                     'matplotlib',
                     'seaborn',
                     'sklearn',
                     'scipy==1.5.4',
                     'tqdm',
                     'jupyter',
                     'pandas',
                    ])
