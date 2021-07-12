
clc             % 清屏

clear all;      % 清除内存以便加快运算速度

close all;      % 关闭当前所有figure图像

warning off;    % 屏蔽没有必要的警告

SamNum=20;      % 输入样本数量为20

TestSamNum=20;  % 测试样本数量也是20

ForcastSamNum=6;% 预测样本数量为2

HiddenUnitNum=8;% 中间层隐节点数量取8

InDim=2;        % 网络输入维度为3

OutDim=1;       % 网络输出维度为2

 

% 原始数据

% 端电压

sqddy=[12.503 12.531 12.497 12.447 12.350 12.308 12.254 12.215 12.107 12.071 11.947 11.899 11.796 11.718 11.609 11.513 11.397 11.238 11.138 10.971];

% 内阻

sqnz=[1006.67 1008.16 1009.25 1011.55 1013.62 1015.42 1018.81 1021.75 1025.81 1028.82 1040.63 1044.04 1043.73 1052.94 1057.07 1066.34 1072.37 1078.45 1088.13 1100.6];

% 寿命值

smz=[1.0024 0.9991 0.9953 0.9927 0.9880 0.9833 0.9810 0.9735 0.9672 0.9603 0.9529 0.9448 0.9335 0.9256 0.9173 0.9078 0.8892 0.8754 0.8554 0.8393];
 

p=[sqddy;sqnz];  % 输入数据矩阵

t=[smz];         % 目标数据矩阵

[SamIn,minp,maxp,tn,mint,maxt]=premnmx(p,t); % 原始样本对（输入和输出）初始化

 

rand('state',sum(100*clock));   % 依据系统时钟种子产生随机数

NoiseVar=0.01;                  % 噪声强度为0.01（添加噪声的目的是为了防止网络过度拟合）

Noise=NoiseVar*randn(2,SamNum); % 生成噪声

SamOut=tn+Noise;                % 将噪声添加到输出样本上

 

TestSamIn=SamIn;                % 这里取输入样本与测试样本相同，因为样本容量偏少

TestSanOut=SamOut;              % 也取输出样本与测试样本相同

 

MaxEpochs=50000;                % 最多训练次数为50000

lr=0.035;                       % 学习速率为0.035     

E0=0.65*10^(-3);                % 目标误差为0.65*10^(-3)

W1=0.5*rand(HiddenUnitNum,InDim)-0.1;% 初始化输入层与隐含层之间的权值

B1=0.5*rand(HiddenUnitNum,1)-0.1;% 初始化输入层与隐含层之间的权值

W2=0.5*rand(OutDim,HiddenUnitNum)-0.1;% 初始化输出层与隐含层之间的权值

B2=0.5*rand(OutDim,1)-0.1;% 初始化输出层与隐含层之间的权值

 

ErrHistory=[];  % 给中间变量预先占据内存

for i=1:MaxEpochs          

    HiddenOut=logsig(W1*SamIn+repmat(B1,1,SamNum)); % 隐含层网络输出

    NetworkOut=W2*HiddenOut+repmat(B2,1,SamNum);   %输出层网络输出

    Error=SamOut-NetworkOut;  % 实际输出与网络输出之差

    SSE=sumsqr(Error);   % 能量函数(误差平方和)

    ErrHistory=[ErrHistory SSE];

    if SSE<E0,break,end  % 如果达到误差要求则跳出学习循环

    

    % 以下6行是BP网络最核心的程序

    % 它们是权值（阙值）依据能量函数负梯度下降原理所做的每一步动态调整

    Delta2=Error;

    Delta1=W2'*Delta2.*HiddenOut.*(1-HiddenOut);

    % 对输出层与隐含层之间的权值和阙值进行修正

    dW2=Delta2*HiddenOut';

    dB2=Delta2*ones(SamNum,1);

    % 对输入层与隐含层之间的权值和阙值进行修正

    dW1=Delta1*SamIn';

    dB1=Delta1*ones(SamNum,1);

    

    W2=W2+lr*dW2;

    B2=B2+lr*dB2;

    

    W1=W1+lr*dW1;

    B1=B1+lr*dB1;

    

end

 

HiddenOut=logsig(W1*SamIn+repmat(B1,1,TestSamNum));  % 隐含层输出最终结果

NetworkOut=W2*HiddenOut+repmat(B2,1,TestSamNum);     % 输出层输出最终结果

a=postmnmx(NetworkOut,mint,maxt);                    % 还原网络输出层的结果

x=1:20;                                         % 时间轴刻度

newk=a(1,:);                                         % 网络输出客运量

% newh=a(2,:);                                         % 网络输出货运量

figure;

subplot;plot(x,newk,'r-o',x,smz,'b--+');  % 绘制公路客运量对比图

legend('网络输出寿命值','实际寿命值');

xlabel('样本编号'); ylabel('寿命值');

title('源程序神经网络客运量学习和测试对比图');

 

% subplot(2,1,2);plot(x,newh,'r-o',x,glhyl,'b--+');  % 绘制公路货运量对比图
% 
% legend('网络输出货运量','实际货运量');
% 
% xlabel('年份'); ylabel('货运量/万人');
% 
% title('源程序神经网络货运量学习和测试对比图');

 

% 利用训练好的数据进行预测

% 当用训练好的网络对新数据pnew进行预测时，也应做相应的处理

pnew=[10.822 10.729 10.564 10.409 10.278 10.093 

   1109.9 1119.69 1132.95 1147.54 1159.75 1177.83];   % 2010年和2011年的相关数据

pnewn=tramnmx(pnew,minp,maxp);  %利用原始输入数据的归一化参数对新数据进行归一化

HiddenOut=logsig(W1*pnewn+repmat(B1,1,ForcastSamNum));  % 隐含层输出预测结果

anewn=W2*HiddenOut+repmat(B2,1,ForcastSamNum);          % 输出层输出预测结果

% 把网络预测得到的数据还原为原始的数量级

format short

anew=postmnmx(anewn,mint,maxt)


