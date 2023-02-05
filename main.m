close all;
clear all;
clc;

cvx_clear;
%cvx_solver mosek  %Uncomment if mosek is not the default solver

%%Loading the dataset
T = readtable('Data_for_UCI_named.csv'); 
T.p1 = [];
[n, m] = size(T);
n1 = 1000; %10% of the dataset is considered
rng('default');
index_1 = randperm(n1);
T_new = T(index_1(1:n1),:); %1000 examples are randomly drawn
%Converting the labels to '0' and '1'
for i = 1:n1
    if string(T_new{i,m}) == 'unstable'
        T_new{i,m} = num2cell(1);
    else 
        T_new{i,m} = num2cell(0);
    end
end

%% Splitting the dataset into training set and test set
labels = T_new(:,m);
M = table2array(T_new(:,1:(m-1)));
y = cell2mat(table2array(labels));
F = 0.8; %80% of the dataset is considered for training, examples are randomly extracted
A_train = M(index_1(1:round(F*n1)),:);
y_train = y(index_1(1:round(F*n1)));
l_train = length(y_train);
A_train = [A_train ones(l_train,1)];
A_test = M(index_1(round(F*n1)+1:end),:);
y_test = y(index_1(round(F*n1)+1:end));
l_test = length(y_test);
A_test = [A_test ones(l_test,1)];

%% Centralized approach
A = - (((2*y_train - 1)*ones(1,m)).*A_train);
[n_A, m_A] = size(A);
lambda = 1e-5;
C = [eye(11), zeros(11,1); zeros(1,12)];
cvx_begin quiet
variable b(m_A)
minimize(ones(1,n_A)*log(1+exp(A*b)) + lambda*norm(C*b,1))
cvx_end

%% Model Evaluation (centralized approach)
treshold_logistic = 0.5;
%Training set accuracy
z_train = A_train*b;
h_train = 1.0./(1.0+exp(-z_train));
y_train_pred = h_train > treshold_logistic;
accuracy_train = mean(double((y_train_pred == y_train)));
v = ['Accuracy_train: ', num2str(accuracy_train)];
disp(v);
%Test set accuracy
z_test = A_test*b;
h_test = 1.0./(1.0+exp(-z_test));
y_test_pred = h_test > treshold_logistic;
accuracy_test = mean(double((y_test_pred == y_test)));
v1 = ['Accuracy_test: ', num2str(accuracy_test)];
disp(v1);
%Precision, Recall and F-score
true_unstable = sum(double(y_test == 1));
predicted_unstable = sum(double(y_test_pred == 1));
true_positive = sum(double(y_test == 1) .* double(y_test_pred == 1));
precision = true_positive/predicted_unstable; %Precision
v2 = ['Precision: ', num2str(precision)];
disp(v2);
recall = true_positive/true_unstable; %Recall
v3= ['Recall: ', num2str(recall)];
disp(v3);
f_score = (2*precision*recall)/(precision + recall); %F-score
v4= ['F-score: ', num2str(f_score)];
disp(v4);

%% Distributed approach trough formulation by ADMM and splitting by examples
N_W = 20; %Number of workers
num_data = n_A/N_W;
A_D = zeros(num_data,m_A,N_W);
for i = 1:N_W
      A_D(:,:,i) = A((1+(num_data*(i-1))):num_data*i,:);
end
N_ITER = 200;
rho = 1;
treshold = (lambda)/(rho*N_W);
X = zeros(m_A,N_W);
Z = zeros(m_A,1);
U = zeros(m_A,N_W);
accuracy_dist = zeros(1,N_ITER);
f_score_dist = zeros(1,N_ITER);
for j = 1:N_ITER
    for i=1:N_W
        cvx_begin quiet
        variable b_d(m_A)
        minimize(ones(1,num_data)*log(1+exp(A_D(:,:,i)*b_d)) + (rho/2)*sum_square(b_d - Z + U(:,i)))
        cvx_end
        X(:,i) = b_d;
    end
    m_x = mean(X,2);
    m_u = mean(U,2);
    m_x_u = m_x + m_u;
    Z = wthresh(m_x_u,'s',treshold);
    for i=1:N_W
       U(:,i) = U(:,i) + X(:,i) - Z;
    end  
    z_dist = A_test*Z;
    h_dist = 1.0./(1.0+exp(-z_dist));
    y_pred_dist = h_dist > treshold_logistic;
    accuracy_dist(j) = mean(double((y_pred_dist == y_test)));
    predicted_unstable_dist = sum(double(y_pred_dist == 1));
    true_positive_dist = sum(double(y_test == 1) .* double(y_pred_dist == 1));
    precision_dist = true_positive_dist/predicted_unstable_dist;
    recall_dist = true_positive_dist/true_unstable;
    f_score_dist(j) = (2*precision_dist*recall_dist)/(precision_dist + recall_dist);
end

%% Model Evaluation (distributed approach)
figure;
accuracy_centr = accuracy_test*ones(1,N_ITER);
x_axis = 1:N_ITER;
plot(x_axis,accuracy_centr,'b','LineWidth',1);
hold on;
grid on;
box on;
plot(x_axis,accuracy_dist,'r','LineWidth',1);
xlabel('Number of iterations','FontWeight','bold');
ylabel('Accuracy','FontWeight','bold');
title('Accuracy (Test set)');
legend('Centralized','Distributed','Location','southeast');
ylim([0.73,0.82]);
figure;
f_score_centr = f_score*ones(1,N_ITER);
plot(x_axis,f_score_centr,'b','LineWidth',1);
hold on;
grid on;
box on;
plot(x_axis,f_score_dist,'r','LineWidth',1);
xlabel('Number of iterations','FontWeight','bold');
ylabel('F-score','FontWeight','bold');
title('F-score (Test set)');
legend('Centralized','Distributed','Location','southeast');
ylim([0.815,0.86]);