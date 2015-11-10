%% Very simple and intuitive neural network implementation

function [W,U] = neural2(X,Yval)
% DATA SETS; demo file

Y = zeros(length(Yval), 3);


Y(:,1)=Yval==1;
Y(:,2)=Yval==2;
Y(:,3)=Yval==3;
Y(:,4)=Yval==4;
Y(:,5)=Yval==5;
Y(:,6)=Yval==6;

n = 6;
nbrOfNodes = 3;
nbrOfEpochs = 1000;

% Initialize matrices with random weights 0-1
W = rand(nbrOfNodes, length(X(1,:)));
U = rand(length(Y(1,:)),nbrOfNodes);

m = 0; figure; hold on; e = size(X);

while m < nbrOfEpochs

    % Increment loop counter
    m = m + 1;

    % Iterate through all examples
    for i=1:e(1)
        % Input data from current example set
        I = X(i,:).';
        D = Y(i,:).';

        % Propagate the signals through network
        H = f(W*I);
        O = f(U*H);

        % Output layer error
        delta_i = O.*(1-O).*(D-O);

        % Calculate error for each node in layer_(n-1)
        delta_j = H.*(1-H).*(U.'*delta_i);

        % Adjust weights in matrices sequentially
        U = U + n.*delta_i*(H.');
        W = W + n.*delta_j*(I.');
    end

    RMS_Err = 0;

    % Calculate RMS error
    for i=1:e(1)
        D = Y(i,:).';
        I = X(i,:).';
        RMS_Err = RMS_Err + norm(D-f(U*f(W*I)),2);
    end
    
    y = RMS_Err/e(1);
    plot(m,log(y),'*');

end


function x = f(x)
x = 1./(1+exp(-x));