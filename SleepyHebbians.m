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

%% Problem 1: Setting your neuron variables

%Fast Spiking Neuron Parameters: (Used for all nodes)
a = 0.2;
b = 0.2;
c = -65;
d = 2;

NoiseScalar = 15;

t_start = 1;
t_stop = 500;

PreSynNodes = 100;
PostSynNodes = 1;
LearningTrials = 200;

%Learning Rate Constants
LRScalar = 0.3;
t_const = 40;
 

%Part II: Create Pre-Syn Stimulus
StimTime = 26;
Stimulus = zeros(1, t_stop);
Stimulus(1, StimTime:t_stop) = 1.8 + (2.2-1.8) .* rand(1, t_stop-StimTime+1);


%PartIII: Initialize Wij
Wij = (ones(PostSynNodes, PreSynNodes) .* 0.4);

% WijInitial = Wij; %Saving the weights prior to training for plotting purposes

%% Problem 3: Train the Synaptic Weights

 PostSynFiringRates = zeros(1, LearningTrials);

for LearnTrial = 1:LearningTrials
    
    %Update Lrate for each trial
    Lrate = LRScalar .* exp(-(LearnTrial)./t_const);

    %Generate spiking activity of presynaptic nodes:
    PreSynSpikeTrains = IzhikevichSpikes(a,b,c,d,t_start,t_stop,PreSynNodes,NoiseScalar,Stimulus);

    %Model the PSP response of each post-synaptic node: Modified Spike Response Model
    %Alpha Function
    %Constants:
    A = 1;
    t_peak = 3;
    N_peak = t_peak .* exp(-1);
    
    %Time info (ms):
    tStart = 0;
    tEnd = 50;
    
    %Initializing counters/accumulators
    PSPmodel = [];
    
    PSP_t_rec = [];
    PSP_t_count = 0;
    %Generating an excitatory PSP model
    for t = tStart:tEnd
        PSP_t_count = PSP_t_count + 1;
        PSP_t_rec(1, PSP_t_count) = t;
        PSP =  A .* ((t .* exp(-t ./ t_peak)) ./ N_peak); %Alpha function (Excitatory)
        PSPmodel(1, PSP_t_count) = PSP;
    end

    SpikeTriggeredPSP_rec = zeros(PreSynNodes, t_stop);
    for PreSynNeuron = 1:PreSynNodes
        SpikeConv = conv(PreSynSpikeTrains(PreSynNeuron, :), PSPmodel, 'full'); %Convolving with an excitatory PSP model
        SpikeTrigPSPs = SpikeConv(1, 1:t_stop); %Assigning only first 500 elements of function's output to a variable
    
        SpikeTriggeredPSP_rec(PreSynNeuron,:) = SpikeTrigPSPs .* Wij(PostSynNodes, PreSynNeuron);
    end

    TotalPostSynPSP = sum(SpikeTriggeredPSP_rec);

    %Generating Spike Train for the Post-Synaptic Node
    PostSynSpikeTrain = IzhikevichSpikes(a,b,c,d,t_start,t_stop,1,NoiseScalar,TotalPostSynPSP);
    

    %Tracking Firing rate for each trial
    PostSynFiringRates(1, LearnTrial) = sum(PostSynSpikeTrain);

    %Update weights via STDP
    hLTtime = 25; %Half of the window size
    PostSpikeTime = hLTtime+1; %26 reference val
    tau_LTP = 10;
    tau_LTD = 10.8; %Bias towards LTD
    LTDScalar = 1.05;
    
    for PreSynNeuron = 1:PreSynNodes
        SpikeCount = 0;
        STDPw_rec = []; %Storing the changes, gets emptied each time
        for t = hLTtime:t_stop-hLTtime %Going through both post and pre
            if(PostSynSpikeTrain(1,t) == 1)
                SpikeCount = SpikeCount+1; %Used as an index later
                PreSynWindow = PreSynSpikeTrains(PreSynNeuron,t-hLTtime:t+hLTtime); %Gets the chunk of the presynaptic spike train relative to t (the post-syn spike)
                PreSpikeTimes = find(PreSynWindow == 1);
                tD = PostSpikeTime - PreSpikeTimes;
                if (isempty(tD))
                    STDPw = 0;
                else
                    Mu_tD = mean(tD);
                    if(Mu_tD >= 0)
                        STDPw = exp(-Mu_tD ./ tau_LTP);
                    
                    else
                        STDPw = -exp(Mu_tD ./ tau_LTD) .* LTDScalar; %Bias towards LTD
                    end
                end
                STDPw_rec(1, SpikeCount) = STDPw;
            end
        end
            DeltaWij = Lrate .* mean(STDPw_rec);
            Wij(PostSynNodes, PreSynNeuron) = Wij(PostSynNodes, PreSynNeuron) + DeltaWij; %Updating synaptic weight
    end
    Wij(Wij < 0) = 0; %Using logical indexing to replace negative values with 0

end

%% Problem 4: Plot the Results:

figure,
subplot(2, 1, 1);
scatter(1:LearningTrials, PostSynFiringRates, 10, 'filled');
xlabel('Learning Trial #');
ylabel('Post-Synaptic Firing Rate');

subplot(2, 1, 2);
bar(1:PreSynNodes, Wij);
xlabel('Pre-Synaptic Neuron #');
ylabel('W_i_j');
sgtitle('Results of Spike-time Dependent Plasticity Learning (Bias Towards LTD)');

%% Problem 2: Generate Izhikevich Spike Trains for each Learning Trial

function IZS = IzhikevichSpikes(a,b,c,d,t_start,t_stop,NumNodes,NoiseScalar,I_ext)

AP_Max = 30;
SpikeTrain = [];

for Neuron = 1:NumNodes
    Vm = -70;
    u = b.*Vm;
    for t = t_start:t_stop
        if (I_ext(1,t) ~= 0)
            Vm = Vm+0.5.*((((0.04.*Vm^2)+(5.*Vm)+140)-u)+((NoiseScalar.*randn)+I_ext(1,t))); %Biological noise is only used when I_ext is not zero
            Vm = Vm+0.5.*((((0.04.*Vm^2)+(5.*Vm)+140)-u)+((NoiseScalar.*randn)+I_ext(1,t)));
        else
            Vm = Vm+0.5.*(((0.04.*Vm^2)+(5.*Vm)+140)-u);
            Vm = Vm+0.5.*(((0.04.*Vm^2)+(5.*Vm)+140)-u);
        end
        u = u+(a.*((b.*Vm)-u));
        
        if(Vm >= AP_Max)
            Vm = c;
            u = u + d;
            SpikeTrain(Neuron,t) = 1;
        else
            SpikeTrain(Neuron,t) = 0;
        end
    end
end

IZS = SpikeTrain;

end








