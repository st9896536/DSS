function out = bubbleReg(tc, beta, w, phi)
    out = [];
    for i = 1 : tc-1
        out = [out; 1, (tc-i)^beta, (tc-i)^beta*(cos(w*log(tc-i)+phi))];
    end
end