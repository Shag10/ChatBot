# ChatBot
Python based Chatbot using the concept of neural networks.
# Requirements :-
1. Python 3.x  
2. Anaconda  
3. Tensorflow version 1.x  
4. nltk module  
5. tflearn module
# Links :-
### Pycharm-  
https://download-cf.jetbrains.com/python/pycharm-community-2020.1.1.exe  
### Anconda(64-bit)-  
https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe
# Command to create conda environment-  
conda create -n myenv python  
or  
conda env create -f environment.yml   "Through yml file"  
After  
activate myenv  "After creating environment"
# Commands to install modules through command line(cmd):-  
### Tensorflow(version 1.15)-  
pip install tensorflow==1.15  
or  
conda install -c conda-forge tensorflow==1.15  '''"Through conda prompt"'''
### nltk-  
pip install nltk  
or  
conda install -c conda-forge nltk  '''"Through conda prompt"'''
### tflearn-  
pip install tflearn  
or  
conda install -c conda-forge tflearn  '''"Through conda prompt"'''
# Points to remember :-
1. Install the tensorflow < 2.0 as, up to the date of commit, the version 2.0 or above does not support tflearn module.  
2. If you have created the environment on anaconda, make sure to install every module within the environment.
