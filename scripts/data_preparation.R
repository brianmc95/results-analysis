#######################################################################################################
# Brian McCarhy
# b.mccarthy@cs.ucc.ie
# This is designed to read OMNeT++ CSV files, split them into their constituent parts
# Such as scalars, vectors, parameters and such and then tidy them so as to be prepared for graphing in 
# subsequent stages
#######################################################################################################

library(tidyverse)

omnetpp_df <- read_csv("data/raw_data/WithBeaconing_e2e_delay.csv") %>%
  
  # Split the run line into it's constituent parts
  separate(run, c("config_name", "repetition", "date", "time", "process_id"),sep="-") %>%
    
  # Split module into it's constituent parts.
  separate(module, c("network", "node", "layer"), sep="\\.", extra="merge", fill="right") %>%
  
  unite(date, time, col="datetime", sep="-") %>%
  
  # Remove the node[] from the node number, just makes things more readable
  mutate_at("node", ~gsub("node", "", .)) %>%
  mutate_at("node", ~gsub("\\[", "", .)) %>%
  mutate_at("node", ~gsub("\\]", "", .))

types = split(omnetpp_df, omnetpp_df$type) 

all_keep <- c("config_name", "repetition", "datetime")

params_keep <- c("attrname", "attrvalue")
#par_keep    <- c("attrname", "attrvalue")
run_keep    <- c("process_id")
attr_keep   <- c("name")

results_keep <- c("network", "node", "layer", "name")
sca_keep     <- c("value")
vec_keep     <- c("vectime", "vecvalue")
hist_keep    <- c("count", "mean", "stddev", "min", "max", "binedges", "binvalues")
    
params   <- types$param   %>% select(one_of(all_keep, params_keep))
runnattr <- types$runattr %>% select(one_of(all_keep, run_keep, params_keep))
attr     <- types$attr    %>% select(one_of(all_keep, attr_keep, params_keep))

vectors    <- types$vector    %>% select(one_of(all_keep, results_keep, vec_keep))
histograms <- types$histogram %>% select(one_of(all_keep, results_keep, hist_keep))
scalars    <- types$scalar    %>% select(one_of(all_keep, results_keep, sca_keep))
    

rm(omnetpp_df, types, all_keep, params_keep, run_keep, results_keep, sca_keep, vec_keep, hist_keep, attr_keep)

# rm (params, runnattr, attr, vectors, histograms, scalars)
