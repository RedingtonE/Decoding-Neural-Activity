allrun = sort([highruntimes{3}]);
cpos = diff(allrun);

start = microscopeTime([allrun(1) allrun(find(cpos > 1)+1)]);
ends = microscopeTime([allrun(find(cpos > 1)) allrun(end)]);
epochsnt = cat(1, start', ends');

allrun = sort([highruntimes{2}]);
cpos = diff(allrun);

start = microscopeTime([allrun(1) allrun(find(cpos > 1)+1)]);
ends = microscopeTime([allrun(find(cpos > 1)) allrun(end)]);
epochst = cat(1, start', ends');

allepoch = cat(2, epochsnt, epochst);
allrun = sort([highruntimes{:}]);
hmmtestspikesmooth = zeros(size(F));
hmmtestspike = zeros(size(F));
for fn = 1:length(troughLocation)
    relT = troughLocation{fn};
    relP = peakPosition{fn}';
    for nn = 1:length(relT)
        if ismember(relT(nn), allrun)
        hmmtestspikesmooth(relT(nn):relP(nn), fn) = 1;
        hmmtestspike(relT(nn), fn) = 1;
        end
    end
    tlen(fn) = mean(relP-relT);
end

placecells = find(pvalues(:, 2) < 0.05 | pvalues(:, 3) < 0.05);

%%
allrun = sort([highruntimes{:}]);
cpos = diff(allrun);

start = microscopeTime([allrun(1) allrun(find(cpos > 1)+1)]);
ends = microscopeTime([allrun(find(cpos > 1)) allrun(end)]);
epochs = cat(1, start', ends');

hmmtestspikesmooth = zeros(size(F));
hmmtestspike = zeros(size(F));
for fn = 1:length(troughLocation)
    relT = troughLocation{fn};
    relP = peakPosition{fn}';
    for nn = 1:length(relT)
        if ismember(relT(nn), allrun)
        hmmtestspikesmooth(relT(nn):relP(nn), fn) = 1;
        hmmtestspike(relT(nn), fn) = 1;
        end
    end
    tlen(fn) = mean(relP-relT);
end

placecells = find(pvalues < 0.05);

