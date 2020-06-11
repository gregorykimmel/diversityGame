
## Code updated from RandomWalk_2020-04-01.jl

using Pkg;
Pkg.activate(".")

using Random
using Plots
using Statistics

using Dates

using Serialization

using DataFrames

using CSV

using Query

## Create initial conditions
   # input: number of cells, initial clones, will randomly assign initial frequency 

function makeInitialPayoff(clones::Int64, fillPayoff::Vector, freqSel::Bool=false)
    # ... is vararg splat, takes an array of values and passes them to a function as if 1st, 2nd, 3rd
    # converts an array into arguments

    if freqSel
      payoff = fillPayoff.*rand(clones,clones)
    else
      payoff = hcat(fill(fillPayoff, (clones,1))...)
    end
    
    return payoff::Matrix
end


function initialize(cellCount::Int64, numClones::Int64, fillPayoff::Vector, freqSel::Bool=false)
    # initialize what each cell is 
    
    totalCells = rand(1:numClones,cellCount); # assigning clones to each spot
    payoff = makeInitialPayoff(numClones, fillPayoff, freqSel)
    
    return (totalCells, payoff); 
end

function calcFreq(uniqueClones::Vector{Int64}, perClone::Vector{Int64})
    # calculates frequency of each unique clone at current time point
    
    ### ASK JAMES FOR A WORD DESCRIPTION OF X->X=i ...
    
    cts = [count(x->x==i,perClone) for i in uniqueClones]
    return cts/sum(cts)
end

## Create payoff matrix -- add row/column

function makeNewClone(mtx::Matrix, parent::Int64, freqSel::Bool=false)
    
    ## add or delete depending on if new clone is added or removed from population
    # to add, make a new matrix of [n+1, n+1], loop through to fill in old matrix and add new matrix
    
    # note, should be a square matrix, x==y --- use for error handling
    
    (x,y) = size(mtx)
    newMtx = zeros(x+1,x+1)   # creates an empty matrix of zeros
    newMtx[1:x,1:x] = mtx     # fills in the original matrix
    
    if freqSel
        # FREQUENCY SELECTION
        # returns clone that will produce the new clone
        # perturb by the mean with std proportional to mean
        newMtx[x+1,1:x] = mtx[parent,:].*(1 .+ 0.01*randn(x))      
        ## fill last elements of row 
        # calculate the mean of the rows to fill in for the new clone 
        rowMean = mean(newMtx[1:x,1:x],dims=2)
        for i in 1:x
            newMtx[i,x+1] = rowMean[i]
        end
    else
        # CONSTANT SELECTION 
        newOne = mtx[parent,1]*(1+0.01*randn()) # all the same across the row, 
        newMtx[x+1,:] = fill(newOne,x+1)   # fill in constant selection for new species
        for i in 1:x
              newMtx[i,x+1] = newMtx[i,x]
        end
    end 
    
        newMtx[end,end] = newMtx[parent,end]*(1+0.01*randn()) 
        # println(newMtx)
    return newMtx/maximum(newMtx)
end

## Create payoff matrix -- remove row/column

function deleteClone(mtx::Matrix{Float64}, remove::Int64)
    
    ## add or delete depending on if new clone is added or removed from population
    # to delete, just delete row and column of interest
    
    return mtx[1:end .!= remove, 1:end .!= remove]
end


function calcAvgFitness(payoff::Matrix{Float64}, freq::Vector{Float64})
    # multiply the frequencies by the payoff matrix to calculate the average fitness   
    # xT*A*x
    
    expectedPayoff = payoff*freq
    
    # return (transpose(freq)*Epayoff, Epayoff) 
    return (freq'*expectedPayoff, expectedPayoff) 
end

function probReplication(payoff::Matrix{Float64}, freq::Vector{Float64}, intensity::Float64)
    
    # calculating the probability that a clone replicates
    # inputs: payoff matrix (Matrix)
    #         frequency of the clones (Vector)
    #         selection intensity (Real)
    # output: probability 
    
    (avg, expectedPayoff) = calcAvgFitness(payoff,freq);

    # Species rate of production of offspring
    probRep = exp.(intensity*(expectedPayoff)).*freq  #  .- avg
    # @show probRep

    # Compute CDF
    cs = cumsum(probRep)
    prob = cs/cs[end]
    
    # returns grouping of which proportion will be replicating when random number
    return prob
end

function calcqD(freq::Vector{Float64}, res::Int64=1000)
  ## good default resoluion: 1000
  ## TO DO: add if q == 1 correction 

  q = exp10.(range(-2.0, stop=2.0, length=res));
  qD = sum(freq.^q', dims=1) .^ (1 ./ (1 .- q'))
  return (qD[:], q)
end

function runSim(initialPayoff::Vector{Float64}, totGen::Int64, cellCount::Int64, numClones::Int64, intensity::Float64, mutProb::Float64, howOftenCalcQD::Int64, resQD::Int64=1000, freqSelOn::Bool=false)
    
    # inputs: 
    #   initialPayoff::Vector{Float64} - fill vector 
    #   totGen::Int64 - number of generations to loop over
    #   cellCount::Int64 - total number of cells/total population size
    #   numClones::Int64 - number of distinct clones
    #   intensity::Float64 - selection intensity
    #   mutProb:: Float64 - probability a clone mutates 
    #   delay::Int64 - after delay steps, start calculating qD  !!!! TO DO !!!!
    #   howOftenCalcQD::Int64 - calculate qD every howOftenCalcQD timesteps
    #   resQD::Int64 - resolution at which qD is calculated, defaults to 1000 
    
    ## to collect for Muller plots
    # time points
    # population size at time points ---> frequency at the timepoint
    # edge list (who mutates (parent) to produce new clone)
    
    # Create the initial payoff matrix, based on cellCount & numClones
    # initialize the initial conditions
    # make individual cells with which clone they are and the initial payoff matrix
    (cells, payoff) = initialize(cellCount, numClones, initialPayoff, freqSelOn);
    totClones = numClones;
    uniqueClones = Array(1:numClones)
    numClonesTrack = zeros(Int,totGen);
    numClonesTrack[1] = numClones;
    
    # create array for collecting --- ASK JAMES IF THIS CAN BE IMPROVED ---- ask James what means 
    qDarray = Vector{Float64}[]

    # calculate the frequency of each clone
    freq = calcFreq(uniqueClones,cells)
    Mullerfreq = freq
    qD, qRange = calcqD(freq);
    push!(qDarray, qD)
    whoDied = Tuple{Int, Float64}[]
    
    ## make first entry of data for Muller plot data collection
    MullerData = DataFrame(generation=[], population=[], edgelist=[])
    push!(MullerData, [1, floor.(Int,cellCount*freq), []])

    for gen in 2:totGen

        # Step 1: calculate prob of each clone to be replicating
        # Step 2: find which clone replicates (random number, see who select)
        # Step 3: decide if mutate (rand num)
        # Step 4: decide how much mutates by (offspring close to num in payoff)
       
        edgeData = []
        
        # get a cumulative probability distribution (essentially CDF) for probability of replication
        CDF = probReplication(payoff, freq, intensity) 

        # decide which clone replicates
        ####### FIXME ###### (When only single clone )
        roll = rand()
        
        whichOneRep = []
        whichOneRepIndex = []
        try
            # looking for the first time we exceed the role, if so mutates
            whichOneRepIndex = findall(y->(y>=roll),CDF)[1]
            whichOneRep = uniqueClones[whichOneRepIndex]
        catch
            @show freq, payoff, uniqueClones,CDF
            @error("what??")
        end
        
       # println("which one reps ", whichOneRep)
        #once replication occurs, does it mutated
        if (mutProb > rand())
            ## true, add to payoff matrix
            # if you mutate, update payoff, update array of cells and add new cell to total num of clones
            payoff = makeNewClone(payoff, whichOneRepIndex, freqSelOn) 
            numClones +=1
            totClones +=1
            whoReplacesDead = totClones
            push!(uniqueClones, whoReplacesDead)
            edgeData = [whichOneRep,totClones]
        else
            ## false - no mutation
            whoReplacesDead = whichOneRep
        end
                    
        # randomly pick who dies (randomly pick a clone, replace with new clone)
        whichCellDies = rand(1:cellCount)
        cells[whichCellDies] = whoReplacesDead
        
        # check to see if a clone is lost --> 0 is a lost clone
        # calculate the frequency of each clone to identify 0 clone
        freq = calcFreq(uniqueClones,cells)
        Mullerfreq = calcFreq(Array(1:totClones),cells);
                
        # check if we lost a clone, if yes       
        locZeros = findall(x->x==0,freq)
        # if clone died, record it in whoDied and remove from payoff
        if length(locZeros) > 0
            
            deadClone = locZeros[1]
            # record which clone died and its payoff
            push!(whoDied, (uniqueClones[deadClone], payoff[deadClone,1]))     
            # delete dead clone from payoff, freq, and number of clones
            payoff = deleteClone(payoff, deadClone)
            deleteat!(freq,deadClone)
            deleteat!(uniqueClones,deadClone)
            numClones -= 1
            
        end
        
        numClonesTrack[gen] = numClones; 
        if gen % howOftenCalcQD == 0
          qD, _ = calcqD(freq);
          push!(qDarray, qD)
        end
        
        ### append Muller plot collection data
        # generation number; population size; edge list (parent, child)
        
        #popvector = cellCount*freq;
        push!(MullerData, [gen, floor.(Int,cellCount*Mullerfreq), edgeData])
        
    end
   # currently only returning first column of payoff matrix because all rows are the same
    return (numClonesTrack=numClonesTrack, totClones=totClones, freq=freq, payoff=payoff[:,1], whoDied=whoDied, qD=qDarray, q=qRange, MD=MullerData)
    
end

    # inputs: 
    #   initialPayoff::Vector{Float64} - fill vector 
    #   totGen::Int64 - number of generations to loop over
    #   cellCount::Int64 - total number of cells/total population size
    #   numClones::Int64 - number of distinct clones
    #   intensity::Float64 - selection intensity
    #   mutProb:: Float64 - probability a clone mutates 
    #   howOftenCalcQD::Int64 - calculate qD every howOftenCalcQD timesteps
    #   resQD::Int64 - resolution at which qD is calculated, defaults to 1000 
    #   freqSelOn::Bool = tells whether on freq dependent selection or constant selection

# outputs: (
#       numClonesTrack=numClonesTrack, 
#       totClones=totClones, 
#       freq=freq, 
#       payoff=payoff[:,1], 
#       whoDied=whoDied, 
#       qD=qDarray, q=qRange
# )

numClones = 1;
mut = 0.5;
cellCount = 100;
initialPayoff = rand(numClones);
gen=10;
selInt = 10.0;
howOften = 1;
resolution = 1;
freqSelType = true;

test2 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);



function fillInMullerTable(df::DataFrame)
    
    totClNum = length(df[end,:population])
    gen = df[end,:generation]
    
    newTable = zeros(Int,gen,totClNum);
    tmp = []
    
    for i in 1:gen
        tmp = length(df[i,:population])
        newTable[i, 1:tmp] = df[i,:population]
    end
    
    return newTable
end

newMullerPopulation = fillInMullerTable(test2.MD)

edgeImportant = @from i in test2.MD begin
    @where i.edgelist != []
    @select {gen=i.generation,i.edgelist}
    @collect DataFrame
end

pairedlist = edgeImportant[!,:edgelist];
parentlist=zeros(Int,size(newMullerPopulation)[2])
for j in 1:length(pairedlist)
     parentlist[pairedlist[j][2]] = pairedlist[j][1]
end

clonelist = collect(1:length(pairedlist)+1);

edgeDF = DataFrame(child=clonelist, parent=parentlist)

genDF = DataFrame(newMullerPopulation)

using CSV

CSV.write("MullerTest_GenDF_test3_2020-06-03.csv", genDF)
CSV.write("MullerTest_EdgeDF_test3_2020-06-03.csv", edgeDF)

using RCall



@rlibrary EvoFreq

R"install.packages('contrib.url')"

R"install.packages('devtools', source='https://cloud.r-project.org')"

R"library(devtools)"











function calcDivDF(q, qD)
    # create a dataframe with top diversity metrics we look at: lowq, highq, deltaqD, IPq, IP slope
    
    # step 1: create empty vectors
    lowTmp = zeros(length(qD))
    highTmp = zeros(length(qD))
    deltaTmp = zeros(length(qD))
    IPtmp = zeros(length(qD),2)
    
    # loop through all diversity curves generated in simulation
    for i in 1:length(qD)
        lowTmp[i] = qD[i][1]    # record low q diversity, q=0.01
        highTmp[i] = qD[i][end] # record high q diversity, q=100
        deltaTmp[i] = qD[i][1] - qD[i][end]  # calculate delta qD diversity
        IPq, IPm = reportInflection(q, qD[i]) # find and calculated IP q and slope
        IPtmp[i,1] = IPq; # inflection point q and slope to respective locations
        IPtmp[i,2] = IPm;
    end
    
    # compile together into a single dataframe
    dfDiv = DataFrame(lowQ=lowTmp, highQ=highTmp, deltaqD=deltaTmp, IPq=IPtmp[:,1], IPslope=IPtmp[:,2])
    
    return dfDiv; 
    
end


## inflection point calculation supporting functions

function findInflection(q::Array, qD::Array)
    approxLogDeriv = q[2:end].*diff(qD)./diff(q);
    return approxLogDeriv; 
end

function findInflectionLocal(approx::Array)
    return argmin((approx))
end

function reportInflection(q::Array, qD::Array)
    diffVector = findInflection(q, qD)
    inflectPt = findInflectionLocal(diffVector)
    slopeInflectPt = abs(diffVector[inflectPt])
    qInflecPt = q[inflectPt+1]
    return (qInflecPt, slopeInflectPt)
end

numClones = 1;
cellCount = 1000;
initialPayoff = rand(numClones);
gen=500000;
howOften = 10000;
resolution = 10000;
freqSelType = false;

mut = 2.0e-3;
selInt = 10.0;
resConstSel = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSel = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOut1 = string("SimResults_Constant-Sel_Rep-1_selInt-10_mutRate-2e-3_gen-500k_",Dates.today(),".dat")
open(fileOut1, "w") do fp
    serialize(fp, resConstSel)
end
fileOut2 = string("SimResults_Freq-Sel_Rep-1_selInt-10_mutRate-2e-3_gen-500k_",Dates.today(),".dat")
open(fileOut2, "w") do fp
    serialize(fp, resFreqSel)
end

# plot qD curve: 
qdPlot = plot(resConstSel.q,resConstSel.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot, resFreqSel.q,resFreqSel.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot, string("DiversityCurves_selInt-10_mutRate-2e-3_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot = histogram(resConstSel.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.7, grid=false)
histogram!(hPlot, resFreqSel.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.7)
savefig(hPlot, string("Histogram-of-Clones_selInt-10_mutRate-2e-3_",Dates.today(),".pdf"))

constSelDivDF = calcDivDF(resConstSel.q, resConstSel.qD)
first(constSelDivDF,3)

fileOutDF1 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-10_mutRate-2e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1,constSelDivDF)

freqSelDivDF = calcDivDF(resFreqSel.q, resFreqSel.qD)
first(freqSelDivDF,3)

fileOutDF2 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-10_mutRate-2e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2,freqSelDivDF)

mut = 2.0e-3;
selInt = 5.0;
resConstSel_2en3mut_int5 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSel_2en3mut_int5 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutA = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-5_mutRate-2e-3_gen-500k_",Dates.today(),".dat")
open(fileOutA, "w") do fp
    serialize(fp, resConstSel_2en3mut_int5)
end
fileOutB = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-5_mutRate-2e-3_gen-500k_",Dates.today(),".dat")
open(fileOutB, "w") do fp
    serialize(fp, resFreqSel_2en3mut_int5)
end

constSelDivDF_2en3mut_int5 = calcDivDF(resConstSel_2en3mut_int5.q, resConstSel_2en3mut_int5.qD)
freqSelDivDF_2en3mut_int5 = calcDivDF(resFreqSel_2en3mut_int5.q, resFreqSel_2en3mut_int5.qD)

fileOutDF1A = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-5_mutRate-2e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1A,constSelDivDF_2en3mut_int5)

fileOutDF2B = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-5_mutRate-2e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2B,freqSelDivDF_2en3mut_int5)

# plot qD curve: 
qdPlot2 = plot(resConstSel_2en3mut_int5.q,resConstSel_2en3mut_int5.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot2, resFreqSel_2en3mut_int5.q,resFreqSel_2en3mut_int5.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot2, string("./results/Moran-process/DiversityCurves_selInt-5_mutRate-2e-3_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot2 = histogram(resConstSel_2en3mut_int5.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.7, grid=false)
histogram!(hPlot2, resFreqSel_2en3mut_int5.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.7)
savefig(hPlot2, string("Histogram-of-Clones_selInt-5_mutRate-2e-3_",Dates.today(),".pdf"))

mut = 2.0e-1;
resConstSelHiMu_2en1mut_int5 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSelHiMu_2en1mut_int5 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutHiMuA = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-5_mutRate-2e-1_gen-500k_",Dates.today(),".dat")
open(fileOutHiMuA, "w") do fp
    serialize(fp, resConstSelHiMu_2en1mut_int5)
end
fileOutHiMuB = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-5_mutRate-2e-1_gen-500k_",Dates.today(),".dat")
open(fileOutHiMuB, "w") do fp
    serialize(fp, resFreqSelHiMu_2en1mut_int5)
end

constSelDivDF_2en1mut_int5 = calcDivDF(resConstSelHiMu_2en1mut_int5.q, resConstSelHiMu_2en1mut_int5.qD)
freqSelDivDF_2en1mut_int5 = calcDivDF(resFreqSelHiMu_2en1mut_int5.q, resFreqSelHiMu_2en1mut_int5.qD)

fileOutDF1HiMuA = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-5_mutRate-2e-1_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1HiMuA,constSelDivDF_2en1mut_int5)

fileOutDF2HiMuB = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-5_mutRate-2e-1_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2HiMuB,freqSelDivDF_2en1mut_int5)

# plot qD curve: 
qdPlot3 = plot(resConstSelHiMu_2en1mut_int5.q,resConstSelHiMu_2en1mut_int5.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot3, resFreqSelHiMu_2en1mut_int5.q,resFreqSelHiMu_2en1mut_int5.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot3, string("./results/Moran-process/DiversityCurves_selInt-5_mutRate-2e-1_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot3 = histogram(resConstSelHiMu_2en1mut_int5.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.7, grid=false)
histogram!(hPlot3, resFreqSelHiMu_2en1mut_int5.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.7)
savefig(hPlot3, string("Histogram-of-Clones_selInt-5_mutRate-2e-1_",Dates.today(),".pdf"))

selInt = 10.0;
resConstSelHiMu_2en1mut_int10 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSelHiMu_2en1mut_int10 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutHiMuA2 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-10_mutRate-2e-1_gen-500k_",Dates.today(),".dat")
open(fileOutHiMuA2, "w") do fp
    serialize(fp, resConstSelHiMu_2en1mut_int10)
end
fileOutHiMuB2 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-10_mutRate-2e-1_gen-500k_",Dates.today(),".dat")
open(fileOutHiMuB2, "w") do fp
    serialize(fp, resFreqSelHiMu_2en1mut_int10)
end

constSelDivDF_2en1mut_int10 = calcDivDF(resConstSelHiMu_2en1mut_int10.q, resConstSelHiMu_2en1mut_int10.qD)
freqSelDivDF_2en1mut_int10 = calcDivDF(resFreqSelHiMu_2en1mut_int10.q, resFreqSelHiMu_2en1mut_int10.qD)

fileOutDF1HiMuA2 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-10_mutRate-2e-1_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1HiMuA2,constSelDivDF_2en1mut_int10)

fileOutDF2HiMuB2 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-10_mutRate-2e-1_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2HiMuB2,freqSelDivDF_2en1mut_int10)

# plot qD curve: 
qdPlot4 = plot(resConstSelHiMu_2en1mut_int10.q,resConstSelHiMu_2en1mut_int10.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot4, resFreqSelHiMu_2en1mut_int10.q,resFreqSelHiMu_2en1mut_int10.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot4, string("./results/Moran-process/DiversityCurves_selInt-10_mutRate-2e-1_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot4 = histogram(resConstSelHiMu_2en1mut_int10.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.7, grid=false)
histogram!(hPlot4, resFreqSelHiMu_2en1mut_int10.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.7)
savefig(hPlot4, string("Histogram-of-Clones_selInt-10_mutRate-2e-1_",Dates.today(),".pdf"))

mut = 5e-4;
sel = 10.0; 
resConstSelLoMu2 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSelLoMu2 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutLoMuA = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-10_mutRate-5e-4_gen-500k_",Dates.today(),".dat")
open(fileOutLoMuA, "w") do fp
    serialize(fp, resConstSelLoMu2)
end
fileOutLoMuB = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-10_mutRate-5e-4_gen-500k_",Dates.today(),".dat")
open(fileOutLoMuB, "w") do fp
    serialize(fp, resFreqSelLoMu2)
end

constSelDivDF_5en4mut_int10 = calcDivDF(resConstSelLoMu2.q, resConstSelLoMu2.qD)
freqSelDivDF_5en4mut_int10 = calcDivDF(resFreqSelLoMu2.q, resFreqSelLoMu2.qD)

fileOutDF1LoMu = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-10_mutRate-5e-4_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1LoMu,constSelDivDF_5en4mut_int10)

fileOutDF2LoMu = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-10_mutRate-5e-4_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2LoMu,freqSelDivDF_5en4mut_int10)

# plot qD curve: 
qdPlotLoMu2 = plot(resConstSelLoMu2.q,resConstSelLoMu2.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotLoMu2, resFreqSelLoMu2.q,resFreqSelLoMu2.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlotLoMu2, string("./results/Moran-process/DiversityCurves_selInt-10_mutRate-5e-4_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlotLoMu2 = histogram(resConstSelLoMu2.numClonesTrack, bins=:scott, normalize=:probability, color=:red, linealpha=0.2, opacity=0.6, grid=false)
histogram!(hPlotLoMu2, resFreqSelLoMu2.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, linealpha=0.2, opacity=0.6, legend=false)
savefig(hPlotLoMu2, string("./results/Moran-process/Histogram-of-Clones_selInt-10_mutRate-5e-4_",Dates.today(),".pdf"))

mut = 5e-4;
sel = 5.0; 
resConstSelLoMu2B = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSelLoMu2B = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutLoMuAB = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-5_mutRate-5e-4_gen-500k_",Dates.today(),".dat")
open(fileOutLoMuAB, "w") do fp
    serialize(fp, resConstSelLoMu2B)
end
fileOutLoMuBB = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-5_mutRate-5e-4_gen-500k_",Dates.today(),".dat")
open(fileOutLoMuBB, "w") do fpB
    serialize(fpB, resFreqSelLoMu2B)
end

constSelDivDF_5en4mut_int5 = calcDivDF(resConstSelLoMu2B.q, resConstSelLoMu2B.qD)
freqSelDivDF_5en4mut_int5 = calcDivDF(resFreqSelLoMu2B.q, resFreqSelLoMu2B.qD)

fileOutDF1LoMuB = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-5_mutRate-5e-4_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1LoMuB,constSelDivDF_5en4mut_int5)

fileOutDF2LoMuB = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-5_mutRate-5e-4_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2LoMuB,freqSelDivDF_5en4mut_int5)

# plot qD curve: 
qdPlotLoMu2B = plot(resConstSelLoMu2B.q,resConstSelLoMu2B.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotLoMu2B, resFreqSelLoMu2B.q,resFreqSelLoMu2B.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlotLoMu2B, string("./results/Moran-process/DiversityCurves_selInt-5_mutRate-5e-4_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlotLoMu2B = histogram(resConstSelLoMu2B.numClonesTrack, bins=:scott, normalize=:probability, color=:red, linealpha=0.2, opacity=0.6, grid=false)
histogram!(hPlotLoMu2B, resFreqSelLoMu2B.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, linealpha=0.2, opacity=0.6, legend=false)
savefig(hPlotLoMu2B, string("./results/Moran-process/Histogram-of-Clones_selInt-5_mutRate-5e-4_",Dates.today(),".pdf"))

mut = 5e-2;
sel = 10.0; 
resConstSelHiMu2 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSelHiMu2 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutHiMu2A = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-10_mutRate-5e-2_gen-500k_",Dates.today(),".dat")
open(fileOutHiMu2A, "w") do fp
    serialize(fp, resConstSelHiMu2)
end
fileOutHiMu2B = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-10_mutRate-5e-2_gen-500k_",Dates.today(),".dat")
open(fileOutHiMu2B, "w") do fp
    serialize(fp, resFreqSelHiMu2)
end

constSelDivDF_5en2mut_int10 = calcDivDF(resConstSelHiMu2.q, resConstSelHiMu2.qD)
freqSelDivDF_5en2mut_int10 = calcDivDF(resFreqSelHiMu2.q, resFreqSelHiMu2.qD)

fileOutDF1HiMu = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-10_mutRate-5e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1HiMu,constSelDivDF_5en2mut_int10)

fileOutDF2HiMu = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-10_mutRate-5e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2HiMu,freqSelDivDF_5en2mut_int10)

# plot qD curve: 
qdPlotHiMu2 = plot(resConstSelHiMu2.q,resConstSelHiMu2.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotHiMu2, resFreqSelHiMu2.q,resFreqSelHiMu2.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlotHiMu2, string("./results/Moran-process/DiversityCurves_selInt-10_mutRate-5e-2_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlotHiMu2 = histogram(resConstSelHiMu2.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlotHiMu2, resFreqSelHiMu2.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlotHiMu2, string("./results/Moran-process/Histogram-of-Clones_selInt-10_mutRate-5e-2_",Dates.today(),".pdf"))

mut = 5e-2;
sel = 5.0; 
resConstSelHiMu2B = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSelHiMu2B = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutHiMu2AB = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-5_mutRate-5e-2_gen-500k_",Dates.today(),".dat")
open(fileOutHiMu2AB, "w") do fp
    serialize(fp, resConstSelHiMu2B)
end
fileOutHiMu2BB = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-5_mutRate-5e-2_gen-500k_",Dates.today(),".dat")
open(fileOutHiMu2BB, "w") do fp
    serialize(fp, resFreqSelHiMu2B)
end

constSelDivDF_5en2mut_int5 = calcDivDF(resConstSelHiMu2B.q, resConstSelHiMu2B.qD)
freqSelDivDF_5en2mut_int5 = calcDivDF(resFreqSelHiMu2B.q, resFreqSelHiMu2B.qD)

fileOutDF1HiMuB = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-5_mutRate-5e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1HiMuB,constSelDivDF_5en2mut_int5)

fileOutDF2HiMuB = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-5_mutRate-5e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2HiMuB,freqSelDivDF_5en2mut_int5)

# plot qD curve: 
qdPlotHiMu2B = plot(resConstSelHiMu2B.q,resConstSelHiMu2B.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotHiMu2B, resFreqSelHiMu2B.q,resFreqSelHiMu2B.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlotHiMu2B, string("./results/Moran-process/DiversityCurves_selInt-5_mutRate-5e-2_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlotHiMu2B = histogram(resConstSelHiMu2B.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlotHiMu2B, resFreqSelHiMu2B.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlotHiMu2B, string("./results/Moran-process/Histogram-of-Clones_selInt-5_mutRate-5e-2_",Dates.today(),".pdf"))

mut = 1e-3;
sel = 10.0; 
resConstSelMiMu = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSelMiMu = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutMiMuA = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-10_mutRate-1e-3_gen-500k_",Dates.today(),".dat")
open(fileOutMiMuA, "w") do fp
    serialize(fp, resConstSelMiMu)
end
fileOutMiMuB = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-10_mutRate-1e-3_gen-500k_",Dates.today(),".dat")
open(fileOutMiMuB, "w") do fp
    serialize(fp, resFreqSelMiMu)
end

constSelDivDF_1en3mut_int10 = calcDivDF(resConstSelMiMu.q, resConstSelMiMu.qD)
freqSelDivDF_1en3mut_int10 = calcDivDF(resFreqSelMiMu.q, resFreqSelMiMu.qD)

fileOutDF1MiMu = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-10_mutRate-1e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1MiMu,constSelDivDF_1en3mut_int10)

fileOutDF2MiMu = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-10_mutRate-1e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2MiMu,freqSelDivDF_1en3mut_int10)

# plot qD curve: 
qdPlotMiMu = plot(resConstSelMiMu.q,resConstSelMiMu.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotMiMu, resFreqSelMiMu.q,resFreqSelMiMu.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlotMiMu, string("./results/Moran-process/DiversityCurves_selInt-10_mutRate-1e-3_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlotMiMu = histogram(resConstSelMiMu.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlotMiMu, resFreqSelMiMu.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlotMiMu, string("./results/Moran-process/Histogram-of-Clones_selInt-10_mutRate-1e-3_",Dates.today(),".pdf"))

mut = 1e-3;
sel = 5.0; 
resConstSelMiMu2 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSelMiMu2 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutMiMuA2 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-5_mutRate-1e-3_gen-500k_",Dates.today(),".dat")
open(fileOutMiMuA2, "w") do fp
    serialize(fp, resConstSelMiMu2)
end
fileOutMiMuB2 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-5_mutRate-1e-3_gen-500k_",Dates.today(),".dat")
open(fileOutMiMuB2, "w") do fp
    serialize(fp, resFreqSelMiMu2)
end

constSelDivDF_1en3mut_int5 = calcDivDF(resConstSelMiMu2.q, resConstSelMiMu2.qD)
freqSelDivDF_1en3mut_int5 = calcDivDF(resFreqSelMiMu2.q, resFreqSelMiMu2.qD)

fileOutDF1MiMu2 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-5_mutRate-1e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1MiMu2,constSelDivDF_1en3mut_int5)

fileOutDF2MiMu2 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-5_mutRate-1e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2MiMu2,freqSelDivDF_1en3mut_int5)

# plot qD curve: 
qdPlotMiMu2 = plot(resConstSelMiMu2.q,resConstSelMiMu2.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotMiMu2, resFreqSelMiMu2.q,resFreqSelMiMu2.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlotMiMu2, string("./results/Moran-process/DiversityCurves_selInt-5_mutRate-1e-3_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlotMiMu2 = histogram(resConstSelMiMu2.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlotMiMu2, resFreqSelMiMu2.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlotMiMu2, string("./results/Moran-process/Histogram-of-Clones_selInt-5_mutRate-1e-3_",Dates.today(),".pdf"))

mut = 1e-2;
sel = 10.0; 
resConstSelMiMu3 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSelMiMu3 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutMiMuA3 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-10_mutRate-1e-2_gen-500k_",Dates.today(),".dat")
open(fileOutMiMuA3, "w") do fp
    serialize(fp, resConstSelMiMu3)
end
fileOutMiMuB3 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-10_mutRate-1e-2_gen-500k_",Dates.today(),".dat")
open(fileOutMiMuB3, "w") do fp
    serialize(fp, resFreqSelMiMu3)
end

constSelDivDF_1en2mut_int10 = calcDivDF(resConstSelMiMu3.q, resConstSelMiMu3.qD)
freqSelDivDF_1en2mut_int10 = calcDivDF(resFreqSelMiMu3.q, resFreqSelMiMu3.qD)

fileOutDF1MiMu3 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-10_mutRate-1e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1MiMu3,constSelDivDF_1en2mut_int10)

fileOutDF2MiMu3 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-10_mutRate-1e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2MiMu3,freqSelDivDF_1en2mut_int10)

# plot qD curve: 
qdPlotMiMu3 = plot(resConstSelMiMu3.q,resConstSelMiMu3.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotMiMu3, resFreqSelMiMu3.q,resFreqSelMiMu3.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlotMiMu3, string("./results/Moran-process/DiversityCurves_selInt-10_mutRate-1e-2_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlotMiMu3 = histogram(resConstSelMiMu3.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlotMiMu3, resFreqSelMiMu3.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlotMiMu3, string("./results/Moran-process/Histogram-of-Clones_selInt-10_mutRate-1e-2_",Dates.today(),".pdf"))

mut = 1e-2;
sel = 5.0; 
resConstSelMiMu4 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSelMiMu4 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutMiMuA4 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-5_mutRate-1e-2_gen-500k_",Dates.today(),".dat")
open(fileOutMiMuA4, "w") do fp
    serialize(fp, resConstSelMiMu4)
end
fileOutMiMuB4 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-5_mutRate-1e-2_gen-500k_",Dates.today(),".dat")
open(fileOutMiMuB4, "w") do fp
    serialize(fp, resFreqSelMiMu4)
end

constSelDivDF_1en2mut_int5 = calcDivDF(resConstSelMiMu4.q, resConstSelMiMu4.qD)
freqSelDivDF_1en2mut_int5 = calcDivDF(resFreqSelMiMu4.q, resFreqSelMiMu4.qD)

fileOutDF1MiMu4 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-5_mutRate-1e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1MiMu4,constSelDivDF_1en2mut_int5)

fileOutDF2MiMu4 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-5_mutRate-1e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2MiMu4,freqSelDivDF_1en2mut_int5)

# plot qD curve: 
qdPlotMiMu4 = plot(resConstSelMiMu4.q,resConstSelMiMu4.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotMiMu4, resFreqSelMiMu4.q,resFreqSelMiMu4.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlotMiMu4, string("./results/Moran-process/DiversityCurves_selInt-5_mutRate-1e-2_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlotMiMu4 = histogram(resConstSelMiMu4.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlotMiMu4, resFreqSelMiMu4.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlotMiMu4, string("./results/Moran-process/Histogram-of-Clones_selInt-5_mutRate-1e-2_",Dates.today(),".pdf"))

mut = 5e-3;
sel = 10.0; 
resConstSelMiMu5 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSelMiMu5 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutMiMuA5 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-10_mutRate-5e-3_gen-500k_",Dates.today(),".dat")
open(fileOutMiMuA5, "w") do fp
    serialize(fp, resConstSelMiMu5)
end
fileOutMiMuB5 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-10_mutRate-5e-3_gen-500k_",Dates.today(),".dat")
open(fileOutMiMuB5, "w") do fp
    serialize(fp, resFreqSelMiMu5)
end

constSelDivDF_5en3mut_int10 = calcDivDF(resConstSelMiMu5.q, resConstSelMiMu5.qD)
freqSelDivDF_5en3mut_int10 = calcDivDF(resFreqSelMiMu5.q, resFreqSelMiMu5.qD)

fileOutDF1MiMu5 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-10_mutRate-5e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1MiMu5,constSelDivDF_5en3mut_int10)

fileOutDF2MiMu5 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-10_mutRate-5e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2MiMu5,freqSelDivDF_5en3mut_int10)

# plot qD curve: 
qdPlotMiMu5 = plot(resConstSelMiMu5.q,resConstSelMiMu5.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotMiMu5, resFreqSelMiMu5.q,resFreqSelMiMu5.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlotMiMu5, string("./results/Moran-process/DiversityCurves_selInt-10_mutRate-5e-3_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlotMiMu5 = histogram(resConstSelMiMu5.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlotMiMu5, resFreqSelMiMu5.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlotMiMu5, string("./results/Moran-process/Histogram-of-Clones_selInt-10_mutRate-5e-3_",Dates.today(),".pdf"))

mut = 5e-3;
sel = 5.0; 
resConstSelMiMu6 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSelMiMu6 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutMiMuA6 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-5_mutRate-5e-3_gen-500k_",Dates.today(),".dat")
open(fileOutMiMuA6, "w") do fp
    serialize(fp, resConstSelMiMu6)
end
fileOutMiMuB6 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-5_mutRate-5e-3_gen-500k_",Dates.today(),".dat")
open(fileOutMiMuB6, "w") do fp
    serialize(fp, resFreqSelMiMu6)
end

constSelDivDF_5en3mut_int5 = calcDivDF(resConstSelMiMu6.q, resConstSelMiMu6.qD)
freqSelDivDF_5en3mut_int5 = calcDivDF(resFreqSelMiMu6.q, resFreqSelMiMu6.qD)

fileOutDF1MiMu6 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-5_mutRate-5e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1MiMu6,constSelDivDF_5en3mut_int5)

fileOutDF2MiMu6 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-5_mutRate-5e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2MiMu6,freqSelDivDF_5en3mut_int5)

# plot qD curve: 
qdPlotMiMu6 = plot(resConstSelMiMu6.q,resConstSelMiMu6.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotMiMu6, resFreqSelMiMu6.q,resFreqSelMiMu6.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlotMiMu6, string("./results/Moran-process/DiversityCurves_selInt-5_mutRate-5e-3_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlotMiMu6 = histogram(resConstSelMiMu6.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlotMiMu6, resFreqSelMiMu6.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlotMiMu6, string("./results/Moran-process/Histogram-of-Clones_selInt-5_mutRate-5e-3_",Dates.today(),".pdf"))

mut = 5e-4;
sel = 1.0; 
resConstSel7 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSel7 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutA7 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-1_mutRate-5e-4_gen-500k_",Dates.today(),".dat")
open(fileOutA7, "w") do fp
    serialize(fp, resConstSel7)
end
fileOutB7 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-1_mutRate-5e-4_gen-500k_",Dates.today(),".dat")
open(fileOutB7, "w") do fp
    serialize(fp, resFreqSel7)
end

constSelDivDF_5en4mut_int1 = calcDivDF(resConstSel7.q, resConstSel7.qD)
freqSelDivDF_5en4mut_int1 = calcDivDF(resFreqSel7.q, resFreqSel7.qD)

fileOutDF1_7 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-1_mutRate-5e-4_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1_7,constSelDivDF_5en4mut_int1)

fileOutDF2_7 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-1_mutRate-5e-4_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2_7,freqSelDivDF_5en4mut_int1)

# plot qD curve: 
qdPlot7 = plot(resConstSel7.q,resConstSel7.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot7, resFreqSel7.q,resFreqSel7.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot7, string("./results/Moran-process/DiversityCurves_selInt-1_mutRate-5e-4_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot7 = histogram(resConstSel7.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlot7, resFreqSel7.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlot7, string("./results/Moran-process/Histogram-of-Clones_selInt-1_mutRate-5e-4_",Dates.today(),".pdf"))

mut = 5e-3;
sel = 1.0; 
resConstSel8 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSel8 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutA8 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-1_mutRate-5e-3_gen-500k_",Dates.today(),".dat")
open(fileOutA8, "w") do fp
    serialize(fp, resConstSel8)
end
fileOutB8 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-1_mutRate-5e-3_gen-500k_",Dates.today(),".dat")
open(fileOutB8, "w") do fp
    serialize(fp, resFreqSel8)
end

constSelDivDF_5en3mut_int1 = calcDivDF(resConstSel8.q, resConstSel8.qD)
freqSelDivDF_5en3mut_int1 = calcDivDF(resFreqSel8.q, resFreqSel8.qD)

fileOutDF1_8 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-1_mutRate-5e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1_8,constSelDivDF_5en3mut_int1)

fileOutDF2_8 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-1_mutRate-5e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2_8,freqSelDivDF_5en3mut_int1)

# plot qD curve: 
qdPlot8 = plot(resConstSel8.q,resConstSel8.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot8, resFreqSel8.q,resFreqSel8.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot8, string("./results/Moran-process/DiversityCurves_selInt-1_mutRate-5e-3_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot8 = histogram(resConstSel8.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlot8, resFreqSel8.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlot8, string("./results/Moran-process/Histogram-of-Clones_selInt-1_mutRate-5e-3_",Dates.today(),".pdf"))

mut = 5e-2;
sel = 1.0; 
resConstSel9 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSel9 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutA9 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-1_mutRate-5e-2_gen-500k_",Dates.today(),".dat")
open(fileOutA9, "w") do fp
    serialize(fp, resConstSel9)
end
fileOutB9 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-1_mutRate-5e-2_gen-500k_",Dates.today(),".dat")
open(fileOutB9, "w") do fp
    serialize(fp, resFreqSel9)
end

constSelDivDF_5en2mut_int1 = calcDivDF(resConstSel9.q, resConstSel9.qD)
freqSelDivDF_5en2mut_int1 = calcDivDF(resFreqSel9.q, resFreqSel9.qD)

fileOutDF1_9 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-1_mutRate-5e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1_9,constSelDivDF_5en2mut_int1)

fileOutDF2_9 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-1_mutRate-5e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2_9,freqSelDivDF_5en2mut_int1)

# plot qD curve: 
qdPlot9 = plot(resConstSel9.q,resConstSel9.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot9, resFreqSel9.q,resFreqSel9.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot9, string("./results/Moran-process/DiversityCurves_selInt-1_mutRate-5e-2_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot9 = histogram(resConstSel9.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlot9, resFreqSel9.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlot9, string("./results/Moran-process/Histogram-of-Clones_selInt-1_mutRate-5e-2_",Dates.today(),".pdf"))

mut = 1e-2;
sel = 1.0; 
resConstSel10 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSel10 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutA10 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-1_mutRate-1e-2_gen-500k_",Dates.today(),".dat")
open(fileOutA10, "w") do fp
    serialize(fp, resConstSel10)
end
fileOutB10 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-1_mutRate-1e-2_gen-500k_",Dates.today(),".dat")
open(fileOutB10, "w") do fp
    serialize(fp, resFreqSel10)
end

constSelDivDF_1en2mut_int1 = calcDivDF(resConstSel10.q, resConstSel10.qD)
freqSelDivDF_1en2mut_int1 = calcDivDF(resFreqSel10.q, resFreqSel10.qD)

fileOutDF1_10 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-1_mutRate-1e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1_10,constSelDivDF_1en2mut_int1)

fileOutDF2_10 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-1_mutRate-1e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2_10,freqSelDivDF_1en2mut_int1)

# plot qD curve: 
qdPlot10 = plot(resConstSel10.q,resConstSel10.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot10, resFreqSel10.q,resFreqSel10.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot10, string("./results/Moran-process/DiversityCurves_selInt-1_mutRate-1e-2_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot10 = histogram(resConstSel10.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlot10, resFreqSel10.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlot10, string("./results/Moran-process/Histogram-of-Clones_selInt-1_mutRate-1e-2_",Dates.today(),".pdf"))

mut = 1e-3;
sel = 1.0; 
resConstSel11 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSel11 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutA11 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-1_mutRate-1e-3_gen-500k_",Dates.today(),".dat")
open(fileOutA11, "w") do fp
    serialize(fp, resConstSel11)
end
fileOutB11 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-1_mutRate-1e-3_gen-500k_",Dates.today(),".dat")
open(fileOutB11, "w") do fp
    serialize(fp, resFreqSel11)
end

constSelDivDF_1en3mut_int1 = calcDivDF(resConstSel11.q, resConstSel11.qD)
freqSelDivDF_1en3mut_int1 = calcDivDF(resFreqSel11.q, resFreqSel11.qD)

fileOutDF1_11 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-1_mutRate-1e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1_11,constSelDivDF_1en3mut_int1)

fileOutDF2_11 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-1_mutRate-1e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2_11,freqSelDivDF_1en3mut_int1)

# plot qD curve: 
qdPlot11 = plot(resConstSel11.q,resConstSel11.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot11, resFreqSel11.q,resFreqSel11.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot11, string("./results/Moran-process/DiversityCurves_selInt-1_mutRate-1e-3_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot11 = histogram(resConstSel11.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlot11, resFreqSel11.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlot11, string("./results/Moran-process/Histogram-of-Clones_selInt-1_mutRate-1e-3_",Dates.today(),".pdf"))

mut = 5e-4;
sel = 100.0; 
resConstSel12 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSel12 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutA12 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-100_mutRate-5e-4_gen-500k_",Dates.today(),".dat")
open(fileOutA12, "w") do fp
    serialize(fp, resConstSel12)
end
fileOutB12 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-100_mutRate-5e-4_gen-500k_",Dates.today(),".dat")
open(fileOutB12, "w") do fp
    serialize(fp, resFreqSel12)
end

constSelDivDF_5en4mut_int100 = calcDivDF(resConstSel12.q, resConstSel12.qD)
freqSelDivDF_5en4mut_int100 = calcDivDF(resFreqSel12.q, resFreqSel12.qD)

fileOutDF1_12 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-100_mutRate-5e-4_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1_12,constSelDivDF_5en4mut_int100)

fileOutDF2_12 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-100_mutRate-5e-4_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2_12,freqSelDivDF_5en4mut_int100)

# plot qD curve: 
qdPlot12 = plot(resConstSel12.q,resConstSel12.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot12, resFreqSel12.q,resFreqSel12.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot12, string("./results/Moran-process/DiversityCurves_selInt-100_mutRate-5e-4_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot12 = histogram(resConstSel12.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlot12, resFreqSel12.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlot12, string("./results/Moran-process/Histogram-of-Clones_selInt-100_mutRate-5e-4_",Dates.today(),".pdf"))

mut = 5e-3;
sel = 100.0; 
resConstSel13 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSel13 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutA13 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-100_mutRate-5e-3_gen-500k_",Dates.today(),".dat")
open(fileOutA13, "w") do fp
    serialize(fp, resConstSel13)
end
fileOutB13 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-100_mutRate-5e-3_gen-500k_",Dates.today(),".dat")
open(fileOutB13, "w") do fp
    serialize(fp, resFreqSel13)
end

constSelDivDF_5en3mut_int100 = calcDivDF(resConstSel13.q, resConstSel13.qD)
freqSelDivDF_5en3mut_int100 = calcDivDF(resFreqSel13.q, resFreqSel13.qD)

fileOutDF1_13 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-100_mutRate-5e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1_13,constSelDivDF_5en3mut_int100)

fileOutDF2_13 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-100_mutRate-5e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2_13,freqSelDivDF_5en3mut_int100)

# plot qD curve: 
qdPlot13 = plot(resConstSel13.q,resConstSel13.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot13, resFreqSel13.q,resFreqSel13.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot13, string("./results/Moran-process/DiversityCurves_selInt-100_mutRate-5e-3_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot13 = histogram(resConstSel13.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlot13, resFreqSel13.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlot13, string("./results/Moran-process/Histogram-of-Clones_selInt-100_mutRate-5e-3_",Dates.today(),".pdf"))

mut = 5e-2;
sel = 100.0; 
resConstSel14 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSel14 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutA14 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-100_mutRate-5e-2_gen-500k_",Dates.today(),".dat")
open(fileOutA14, "w") do fp
    serialize(fp, resConstSel14)
end
fileOutB14 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-100_mutRate-5e-2_gen-500k_",Dates.today(),".dat")
open(fileOutB14, "w") do fp
    serialize(fp, resFreqSel14)
end

constSelDivDF_5en2mut_int100 = calcDivDF(resConstSel14.q, resConstSel14.qD)
freqSelDivDF_5en2mut_int100 = calcDivDF(resFreqSel14.q, resFreqSel14.qD)

fileOutDF1_14 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-100_mutRate-5e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1_14,constSelDivDF_5en2mut_int100)

fileOutDF2_14 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-100_mutRate-5e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2_14,freqSelDivDF_5en2mut_int100)

# plot qD curve: 
qdPlot14 = plot(resConstSel14.q,resConstSel14.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot14, resFreqSel14.q,resFreqSel14.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot14, string("./results/Moran-process/DiversityCurves_selInt-100_mutRate-5e-2_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot14 = histogram(resConstSel14.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlot14, resFreqSel14.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlot14, string("./results/Moran-process/Histogram-of-Clones_selInt-100_mutRate-5e-2_",Dates.today(),".pdf"))

mut = 1e-2;
sel = 100.0; 
resConstSel15 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSel15 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutA15 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-100_mutRate-1e-2_gen-500k_",Dates.today(),".dat")
open(fileOutA15, "w") do fp
    serialize(fp, resConstSel15)
end
fileOutB15 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-100_mutRate-1e-2_gen-500k_",Dates.today(),".dat")
open(fileOutB15, "w") do fp
    serialize(fp, resFreqSel15)
end

constSelDivDF_1en2mut_int100 = calcDivDF(resConstSel15.q, resConstSel15.qD)
freqSelDivDF_1en2mut_int100 = calcDivDF(resFreqSel15.q, resFreqSel15.qD)

fileOutDF1_15 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-100_mutRate-1e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1_15,constSelDivDF_1en2mut_int100)

fileOutDF2_15 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-100_mutRate-1e-2_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2_15,freqSelDivDF_1en2mut_int100)

# plot qD curve: 
qdPlot15 = plot(resConstSel15.q,resConstSel15.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot15, resFreqSel15.q,resFreqSel15.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot15, string("./results/Moran-process/DiversityCurves_selInt-100_mutRate-1e-2_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot15 = histogram(resConstSel15.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlot15, resFreqSel15.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlot15, string("./results/Moran-process/Histogram-of-Clones_selInt-100_mutRate-1e-2_",Dates.today(),".pdf"))

mut = 1e-3;
sel = 100.0; 
resConstSel16 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,freqSelType);
resFreqSel16 = runSim(initialPayoff,gen,cellCount,numClones,selInt,mut,howOften,resolution,true);

fileOutA16 = string("./results/Moran-process/SimResults_Constant-Sel_Rep-1_selInt-100_mutRate-1e-3_gen-500k_",Dates.today(),".dat")
open(fileOutA16, "w") do fp
    serialize(fp, resConstSel16)
end
fileOutB16 = string("./results/Moran-process/SimResults_Freq-Sel_Rep-1_selInt-100_mutRate-1e-3_gen-500k_",Dates.today(),".dat")
open(fileOutB16, "w") do fp
    serialize(fp, resFreqSel16)
end

constSelDivDF_1en3mut_int100 = calcDivDF(resConstSel16.q, resConstSel16.qD)
freqSelDivDF_1en3mut_int100 = calcDivDF(resFreqSel16.q, resFreqSel16.qD)

fileOutDF1_16 = string("./results/Moran-process/SimResults_DIVERSITY_Constant-Sel_Rep-1_selInt-100_mutRate-1e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF1_16,constSelDivDF_1en3mut_int100)

fileOutDF2_16 = string("./results/Moran-process/SimResults_DIVERSITY_Freq-Dep-Sel_Rep-1_selInt-100_mutRate-1e-3_gen-500k_",Dates.today(),".csv")
CSV.write(fileOutDF2_16,freqSelDivDF_1en3mut_int100)

# plot qD curve: 
qdPlot16 = plot(resConstSel16.q,resConstSel16.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot16, resFreqSel16.q,resFreqSel16.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5)
savefig(qdPlot16, string("./results/Moran-process/DiversityCurves_selInt-100_mutRate-1e-3_",Dates.today(),".pdf"))

# histogram of number of clones at a time
hPlot16 = histogram(resConstSel16.numClonesTrack, bins=:scott, normalize=:probability, color=:red, opacity=0.6, linealpha=0.1, grid=false)
histogram!(hPlot16, resFreqSel16.numClonesTrack, bins=:scott, normalize=:probability, color=:blue, opacity=0.6, linealpha=0.1, legend=false)
savefig(hPlot16, string("./results/Moran-process/Histogram-of-Clones_selInt-100_mutRate-1e-3_",Dates.today(),".pdf"))

yrange = (0,160)

qdPlotLoMu2 = plot(resConstSelLoMu2.q,resConstSelLoMu2.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotLoMu2, resFreqSelLoMu2.q,resFreqSelLoMu2.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlotLoMu2, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-10_mutRate-5e-4_",Dates.today(),".pdf"))

qdPlotLoMu2B = plot(resConstSelLoMu2B.q,resConstSelLoMu2B.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotLoMu2B, resFreqSelLoMu2B.q,resFreqSelLoMu2B.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlotLoMu2B, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-5_mutRate-5e-4_",Dates.today(),".pdf"))

qdPlotHiMu2 = plot(resConstSelHiMu2.q,resConstSelHiMu2.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotHiMu2, resFreqSelHiMu2.q,resFreqSelHiMu2.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlotHiMu2, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-10_mutRate-5e-2_",Dates.today(),".pdf"))

qdPlotHiMu2B = plot(resConstSelHiMu2B.q,resConstSelHiMu2B.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotHiMu2B, resFreqSelHiMu2B.q,resFreqSelHiMu2B.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlotHiMu2B, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-5_mutRate-5e-2_",Dates.today(),".pdf"))

qdPlotMiMu = plot(resConstSelMiMu.q,resConstSelMiMu.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotMiMu, resFreqSelMiMu.q,resFreqSelMiMu.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlotMiMu, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-10_mutRate-1e-3_",Dates.today(),".pdf"))

qdPlotMiMu2 = plot(resConstSelMiMu2.q,resConstSelMiMu2.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotMiMu2, resFreqSelMiMu2.q,resFreqSelMiMu2.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlotMiMu2, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-5_mutRate-1e-3_",Dates.today(),".pdf"))

qdPlotMiMu3 = plot(resConstSelMiMu3.q,resConstSelMiMu3.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotMiMu3, resFreqSelMiMu3.q,resFreqSelMiMu3.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlotMiMu3, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-10_mutRate-1e-2_",Dates.today(),".pdf"))

qdPlotMiMu4 = plot(resConstSelMiMu4.q,resConstSelMiMu4.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotMiMu4, resFreqSelMiMu4.q,resFreqSelMiMu4.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlotMiMu4, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-5_mutRate-1e-2_",Dates.today(),".pdf"))

qdPlotMiMu5 = plot(resConstSelMiMu5.q,resConstSelMiMu5.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotMiMu5, resFreqSelMiMu5.q,resFreqSelMiMu5.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlotMiMu5, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-10_mutRate-5e-3_",Dates.today(),".pdf"))

qdPlotMiMu6 = plot(resConstSelMiMu6.q,resConstSelMiMu6.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlotMiMu6, resFreqSelMiMu6.q,resFreqSelMiMu6.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlotMiMu6, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-5_mutRate-5e-3_",Dates.today(),".pdf"))

qdPlot7 = plot(resConstSel7.q,resConstSel7.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot7, resFreqSel7.q,resFreqSel7.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlot7, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-1_mutRate-5e-4_",Dates.today(),".pdf"))

qdPlot8 = plot(resConstSel8.q,resConstSel8.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot8, resFreqSel8.q,resFreqSel8.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlot8, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-1_mutRate-5e-3_",Dates.today(),".pdf"))

qdPlot9 = plot(resConstSel9.q,resConstSel9.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot9, resFreqSel9.q,resFreqSel9.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlot9, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-1_mutRate-5e-2_",Dates.today(),".pdf"))

qdPlot10 = plot(resConstSel10.q,resConstSel10.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot10, resFreqSel10.q,resFreqSel10.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlot10, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-1_mutRate-1e-2_",Dates.today(),".pdf"))

qdPlot11 = plot(resConstSel11.q,resConstSel11.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot11, resFreqSel11.q,resFreqSel11.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlot11, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-1_mutRate-1e-3_",Dates.today(),".pdf"))

qdPlot12 = plot(resConstSel12.q,resConstSel12.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot12, resFreqSel12.q,resFreqSel12.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlot12, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-100_mutRate-5e-4_",Dates.today(),".pdf"))

qdPlot13 = plot(resConstSel13.q,resConstSel13.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot13, resFreqSel13.q,resFreqSel13.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlot13, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-100_mutRate-5e-3_",Dates.today(),".pdf"))

qdPlot14 = plot(resConstSel14.q,resConstSel14.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot14, resFreqSel14.q,resFreqSel14.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlot14, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-100_mutRate-5e-2_",Dates.today(),".pdf"))

qdPlot15 = plot(resConstSel15.q,resConstSel15.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot15, resFreqSel15.q,resFreqSel15.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlot15, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-100_mutRate-1e-2_",Dates.today(),".pdf"))

qdPlot16 = plot(resConstSel16.q,resConstSel16.qD,xaxis=:log,legend=:none, color_palette=:reds, grid=false)
plot!(qdPlot16, resFreqSel16.q,resFreqSel16.qD,xaxis=:log,legend=:none, color_palette=:blues, opacity=0.5, ylim=yrange)
savefig(qdPlot16, string("./results/Moran-process/DiversityCurves_FixedRange_selInt-100_mutRate-1e-3_",Dates.today(),".pdf"))

resConstSel16

resConstSel16.freq

resConstSel16.numClonesTrack

resConstSel16.whoDied


