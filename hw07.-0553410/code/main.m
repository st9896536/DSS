clear;
warning off; 
load Bubble.mat;
Y = Bubble(:,2);
n = 10000;
sr = 0.01;
mr = 0.01;
g = 10;

data = myLPPLGA(Y,n,sr,mr,g);
plotData = LPPL(data.A,data.B,data.C,data.tc,data.beta,data.w,data.phi)';

% draw actual price with blue color.
% draw LPPL with red color.
timeSequence = [1:data.tc-1];
plot(timeSequence,Y(1:data.tc-1),'b',timeSequence,exp(plotData) ,'r'); 
hold on;
plot([data.tc data.tc], [min([Y(1:data.tc-1)-30; exp(plotData(1:data.tc-1))]) max([Y(1:data.tc-1); exp(plotData(1:data.tc-1))])]+30, 'g');
legend('actual','predict');

line('XData',5,'YData',100);
title('adjusted closing stock price of 3M Co.','FontSize',18)
xlabel('time');
ylabel('price');

plot(Y)
title('adjusted closing stock price of 3M Co.','FontSize',18)
legend('actual');
xlabel('time');
ylabel('price');