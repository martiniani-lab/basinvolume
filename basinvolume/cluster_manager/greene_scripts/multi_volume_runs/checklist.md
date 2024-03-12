# Run Checklist

Manual Checklist for Monte Carlo runs. This ensures that each stage is reasonable before the next stage is run. Ideally there should exist some manual checking here, especially because Parallel tempering can be very fragile. But some of these could possibly be automated

## Jammed Packing

[ ] Packing folder structure is reasonable

## kmax

[ ] Previous stage did not fail
[ ] kmax did not explode, and is within reason

## kmin

[ ] Previous stage did not fail
[ ] kmin is within reason and msd did not explode

## Parallel tempering

[ ] Enough memory is allocated if this failed
[ ] Previous stage did not fail
[ ] Check that minimizations match results (for small subset if everything takes too long)
[ ] Time is reasonable for choice of minimizer

## Inner sphere

[ ] Previous stage did not fail

## Analysis

[ ] Enough memory is allocated if this failed last time
[ ] There is reasonable overlap between the histograms of previous runs
[ ] the time series looks equilibrated
[ ] The volumes look reasonable
