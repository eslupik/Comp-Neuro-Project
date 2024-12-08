%COMP-NEURO PROJECT!!! YIPPEEE

clear all
clc


% for synaptic scaling (Sullivan & Sa, 2008)
beta_n = 0.01; %time constant
y = 0.01; % γ is a constant between zero and 1 that controls the time course of the accumulation.
C_targ = 10; % in Hz. Target value to compare with chemical concentration
alpha = 1; % learning rate
num_updates = 100; % number of updates between sleep cycles
num_iter = 200; % number of iterations per sleep cycle
total_iter = 45000; % total number of iterations

% Ct =γzt +(1−γ)Ct−1
%% Deep Learning Class Code:
clear all

%%%Load LETTERBLOCK
load('LetterStimuli.mat');

StimH = 30; %image pixel row dimension
StimW = 30; %image pixel column dimension
NumPatterns = 26;
TargetPatterns = [1;5;9;15;21]; %Select target letters (vowels)
PatternSet = 1; %Select font type (1 of 30)

NetworkInput = LETTERBLOCK(:,:,PatternSet); %training letters
NetworkInput = NetworkInput - mean(NetworkInput); %Remove the mean so the network can't use it
TargetTags = zeros(NumPatterns,1);
TargetTags(TargetPatterns,1) = 1; %Desired response array
TargetBias = 10; %Bias factor for target item trials only (multiply to error)

%Array to track network reponse for each letter over iterations
TrackPatternClassification = zeros(1, NumPatterns);

TrainingIterations = 1000;
Sig_b = 1;
Sig_Hmax = 0;

%Define number of nodes for each layer
NumNodes_L1 = 900;
NumNodes_L2 = 90;
NumNodes_L3 = 1;

%Add bias node to NetworkInput for each letter (all 1's)
NetworkInput((NumNodes_L1+1),:) = ones(1, NumPatterns);

%Learning Rate
Lrate = 0.000005;

%Ensure summed weights fall within -5 to +5
L2_W = (5./NumNodes_L1);
L3_W = (5./NumNodes_L2);

%Layer 2 Weights (weights for bias node included)
LAYER2_W = (rand(NumNodes_L1+1,NumNodes_L2).* (L2_W.*2))-L2_W;

%Layer 3 Weights
LAYER3_W = (rand(NumNodes_L2,1).* (L3_W.*2))-L3_W;

%BEGIN TRAINING
for TrainCycle = 1:TrainingIterations
    
    ESUM = 0; %Variable to hold total error for each trial

    for PatternCycle = 1:NumPatterns

        %LAYER 1 OPERATION (PASSIVE)
        L1 = NetworkInput(:,PatternCycle);

        %LAYER 2 OPERATIONS (ACTIVE)
        L2_WgtSum = [];
        L2_Eval = [];
        for L2loop = 1:NumNodes_L2
            L2_WgtSum(L2loop, 1) = sum(L1.*LAYER2_W(:,L2loop));
            L2_Eval(L2loop,1) = 1./(1+exp(-Sig_b.*(L2_WgtSum(L2loop,1)-Sig_Hmax)));
        end

        %LAYER 3 OPERATIONS (ACTIVE)
        L3_WgtSum = [];
        L3_Eval = [];
        L3_WgtSum = sum(L2_Eval.*LAYER3_W);
        L3_Eval = 1./(1+exp(-Sig_b.*(L3_WgtSum-Sig_Hmax)));

        TrackPatternClassification(1,PatternCycle) = L3_Eval; %Record output response for each pattern

        %CALC OUTPUT ERROR FOR EACH PATTERN
        PATT_ERROR = TargetTags(PatternCycle, 1)-L3_Eval;

        %APPLY TARGET BIAS
        if (TargetTags(PatternCycle, 1) == 1)
            PATT_ERROR = PATT_ERROR .* TargetBias;
        end

        %ACCUMULATE OUTPUT ERROR OVER ALL LETTERS FOR EACH TRIAL
        ESUM = ESUM + abs(PATT_ERROR);

        %LAYER 2 WEIGHT ADJUSTMENTS
        L2_Slope = L2_Eval .* (1-L2_Eval);
        L3_Slope = L3_Eval .* (1-L3_Eval);
        dOutdW = [];
        for L2loop = 1:NumNodes_L2
            dOutdW(:,1) = L1 .* L2_Slope(L2loop,1).* LAYER3_W(L2loop,1).*L3_Slope;
            LAYER2_W(:,L2loop) = LAYER2_W(:,L2loop) + ((dOutdW(:,1) .* PATT_ERROR) .* Lrate);
        end

        %LAYER3 WEIGHT ADJUST
        dOutdW = [];
        dOutdW(:,1) = L2_Eval.*L3_Slope;
        LAYER3_W(:,1) = LAYER3_W(:,1) + ((dOutdW(:,1) .* PATT_ERROR) .* Lrate);

    end %End Pattern Cycle
    
    TRACK_ERROR(TrainCycle,1) = ESUM;

    subplot(1,2,1);
    plot(TRACK_ERROR);

    subplot(1,2,2);
    bar(TrackPatternClassification)

    pause(0.01);

end








