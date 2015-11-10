%% Very simple and intuitive neural network implementation

function [W,U, Err] = neural(X,Yval, lambda, n, M, XV, YV)
% DATA SETS; demo file

Y = zeros(length(Yval), 3);


Y(:,1)= Yval==1;
Y(:,2)=Yval==2;
Y(:,3)=Yval==3;

YV2 = zeros(length(YV), 3);


YV2(:,1)= YV==1;
YV2(:,2)=YV==2;
YV2(:,3)=YV==3;

nbrOfNodes=M; 
nbrOfEpochs = 200;

% Initialize matrices with random weights 0-1
W = rand(nbrOfNodes, length(X(1,:)));
U = rand(length(Y(1,:)),nbrOfNodes);

m = 0;  e = size(X);

while m < nbrOfEpochs

    % Increment loop counter
    m = m + 1;

    % Iterate through all examples
    for i=1:e(1)
        % Input data from current example set
        I = X(i,:).';
        D = Yval(i,:).';
        D2=Y(i,:).'; 
        % Propagate the signals through network
        H = f(W*I);
        O = f(U*H);

        % Output layer error
        delta_i = O.*(1-O).*(D2./O - (1-D2)./(1-O));

        % Calculate error for each node in layer_(n-1)
        delta_j = H.*(1-H).*(U.'*delta_i);

        % Adjust weights in matrices sequentially
        U = U + n.*delta_i*(H.') - lambda*U;
        W = W + n.*delta_j*(I.') - lambda*W;
    end

    
    
 
 
    
    Err = 0;

    % Calculate RMS error
    for i=1:e(1)
        D = YV(i,:).';
        I = XV(i,:).';
        D2= YV2(i,:).'; 
        
        Err = Err + D2'*log(f(U*f(W*I))) + (1-D2')*log(1-f(U*f(W*I)));
    end
    
    y = Err/e(1);
    %plot(m,y,'*');

end


function x = f(x)
x = 1./(1+exp(-x));