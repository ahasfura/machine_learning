%% Very simple and intuitive neural network implementation
%
%  Carl Löndahl, 2008
%  email: carl(dot)londahl(at)gmail(dot)com
%  Feel free to redistribute and/or to modify in any way

function [W,U, y] = neuralremy1(l1,l2,n1,n2)
% DATA SETS; demo file
%[Attributes, Classifications] = mendez;
%disp(Classifications);
name = 'toy_multiclass_2';
%name = 'mnist';
d = importdata(strcat('/hw3_resources/',name,'_test.csv'));
d1 = importdata(strcat('/hw3_resources/',name,'_validate.csv'));
blah = ones(size(d,1),1);
blah1 = ones(size(d1,1),1);
d = [blah d];
d1 = [blah1 d1];
disp(size(d));
disp(size(d1));
Classifications = d(:,size(d,2));
Classificationsvalidate = d1(:,size(d,2));
%disp(Classifications);
disp(max(Classifications));
Attributes = d(:,1:size(d,2)-1);
Attributesvalidate = d1(:,1:size(d1,2)-1);
%disp(length(Attributes(:,1)));
%sleep(1000);
%disp(Classifications);
%n = .0026;
for lambda = l1:l2
    for nodes = n1:n2
nbrOfNodes = nodes;
nbrOfEpochs = 800;

% Initialize matrices with random weights 0-1
W = rand(nbrOfNodes, length(Attributes(1,:)));
%W = rand(nbrOfNodes, length(Attributes(:,1)));
%U = rand(length(Classifications(1,:)),nbrOfNodes);
U = rand(max(Classifications),nbrOfNodes);
m = 0; %figure;;
alpha = 10;
%hold on; 
e = size(Attributes);

while m < nbrOfEpochs
    %n = 1 /(m+alpha)^1.5;
    %n = .026;
    n= .1/(m+50)^.6; 
    % Increment loop counter
    m = m + 1;
    %disp(m);
    % Iterate through all examples
    for i = 1:e(1)
        % Input data from current example set
        I = Attributes(i,:).';
        D = Classifications(i,:).';
        D2 = zeros(max(Classifications),1);
        D2(D) = 1;
        %disp(I);
        % Propagate the signals through network
        H = f(W*I);
        O = f(U*H);
        
        % w = 1st weights
        %disp('O');
       % disp(O);
        % Output layer error
        %delta_i = O.*(1-O).*(D2-O); % must fix this part
        delta_i = O.*(1-O).*(D2./O - (1-D2)./(1-O));
        % Calculate error for each node in layer_(n-1)
        delta_j = H.*(1-H).*(U.'*delta_i);

        % Adjust weights in matrices sequentially
        U = U + n.*(delta_i*(H.')- 10^(lambda)*U);%/norm(U,'fro'));
        W = W + n.*(delta_j*(I.')- 10^(lambda)*W);%;/norm(W,'fro'));
        %disp(norm(n.*(delta_j*(I.')- 10^(lambda)*W)));
    end
    %disp(U);

    % Calculate RMS error
end
    
    Err = 0;
    Err1 = 0 ;
for i = 1:e(1)
    D = Classifications(i,:)';
    D2 = zeros(max(Classifications),1);
    D2(D) = 1;
    I = Attributes(i,:).';
        %disp('tus');
        %disp(f(U*f(W*I)));
    Err = Err + D2'*log(f(U*f(W*I))) + (1-D2)'*log(1-f(U*f(W*I)));
end
correct = 0 ;
Y = zeros(size(Attributesvalidate,1),1);
for i = 1:size(Attributesvalidate,1)
    D = Classificationsvalidate(i,:)';
    D2 = zeros(max(Classificationsvalidate),1);
    D2(D) = 1;
    I1 = Attributesvalidate(i,:).';
      %disp('tus');
        %disp(f(U*f(W*I)));
    [x,y] = max(f(U*f(W*I1)));
    Y(i) = y;
    %disp(f(U*f(W*I1)));
    %disp(y);
    %disp(D);
    %disp('D');
    if y == D
        correct = correct+1;
    end
    Err1 = Err1 + D2'*log(f(U*f(W*I1))) + (1-D2)'*log(1-f(U*f(W*I1)));
    
end



subplot(1,2,1);
scatter(Attributesvalidate(:,1),Attributesvalidate(:,2),5,Y);
subplot(1,2,2);
scatter(Attributesvalidate(:,1),Attributesvalidate(:,2),5,Classificationsvalidate);


disp('percent correct is' );
disp(correct/size(Attributesvalidate,1));
    %Err1 = l*norm(W,'fro') + .001*norm(U,'fro') + Err;
    %z  = Err/e(1);
    %y  = Err1/e(1);
    disp('lambda');
    disp(10^lambda);
    disp('nodes');
    disp(nodes);
    disp(Err1);
end
end
    %plot(m,log(y),'*');
    %plot(m,log(z),'o');


function x = f(x)
x = 1./(1+exp(-x));