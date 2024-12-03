%COMP-NEURO PROJECT!!! YIPPEEE

clear all
clc

% this is a test

% for synaptic scaling (Sullivan & Sa, 2008)
beta_n = 0.01; %time constant
y = 0.01; % γ is a constant between zero and 1 that controls the time course of the accumulation.
C_targ = 10; % in Hz. Target value to compare with chemical concentration
alpha = 1; % learning rate
num_updates = 100; % number of updates between sleep cycles
num_iter = 200; % number of iterations per sleep cycle
total_iter = 45000; % total number of iterations

% Ct =γzt +(1−γ)Ct−1



