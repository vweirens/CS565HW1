# CS565HW1

## Overview
Homework submission for assignment 1: Code a Bayesian Network for a Car Diagnostic Tool.

This repository has two primary files: config.txt and Weirens_Vaughn_CS565_HW1.py.

This repository uses classes and functions defined in the aima-python repository: https://github.com/aimacode/aima-python


## Running the code.

The python file may be run from the command line by typing py -m Weirens_Vaughn_CS565_HW1, after which the program will prompt an input from the user.
The format for this user input is: 'node_name' = True/False/true/false.
Example: IW = true 
The spaces between the node name and the value are required, otherwise the query will not be properly read into the file.

The file will read the user's prompt and will compute the joint probability distribution for all conditions provided by the user. If the users prompt does not contain any relevant queries, the function will return 1, as every set of conditions for the joint probability distribution matches the completely independent event.

If the user inputs multiple values for the same node (e.g. 'IW = true, IW = false'), only the first value will be read

## Changing the probabilities
The probabilities are defined in the config.txt file within this repository. If a user wishes to try the code with different values, they may be modified. The file format for config.txt is as follows:
IW,
B(IW=T), B(IW=F),
SM(IW=T), SM(IW=F),
R(B=T), R(B=F),
I(B=T), I(B=F),
G,
S(I=T, SM=T, G=T), S(TTF), S(TFT), S(FTT), S(TFF), S(FTF), S(FFT), S(FFF),
M(S=T), M(S=F)

