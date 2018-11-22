# This is designed to read OMNeT++ CSV files, split them into their constituent parts
# Such as scalars, vectors, parameters and such and then tidy them so as to be prepared for graphing in 
# subsequent stages

library(tidyverse)

omnetpp_df <- read_csv("data/raw_data/wh_wave.csv") %>%
  separate(run, c("configuration", "run", "date", "time", "unsure"),sep="-")

types = split(omnetpp_df, omnetpp_df$type) 

par_run_drops <- c("type", "module", "name", "value")
sca_vec_his_drops <- c("type", "attrname", "attrvalue")
    
params     = types$param %>% select(-one_of(par_run_drops))
runnattr   = types$runattr %>% select(-one_of(par_run_drops))
vectors    = types$vector %>% select(-one_of(sca_vec_his_drops))
histograms = types$histogram %>% select(-one_of(sca_vec_his_drops))

scalars <- types$scalar %>% select(-one_of(sca_vec_his_drops)) %>%
  separate(module, c("simulation", "node", "layer"), sep="\\.", extra="merge", fill="right") %>%
    mutate_at("node", ~gsub("node", "", .)) %>%
      mutate_at("node", ~gsub("\\[", "", .)) %>%
        mutate_at("node", ~gsub("\\]", "", .))

rm(omnetpp_df, types, params, runnattr, scalars, par_run_drops, sca_vec_his_drops)
