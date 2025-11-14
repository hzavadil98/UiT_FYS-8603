
%load('toyDataThreeWay.mat')
C = {rand(12, 100), rand(8,100)};
paramstruct = struct('iprint', false);
out = DJIVEMainJP(C, paramstruct) ;
DJIVEAngleDiagnosticJP(C, "Dataname", out, 556, "Demo")